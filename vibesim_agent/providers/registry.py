"""Model selection and execution without provider-name branches."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing
from typing import Any

from pydantic import SecretStr

from .base import AgentRequest, Model, Provider, Selection


class ProviderRegistry:
    def __init__(self, secrets: Mapping[str, SecretStr] | None = None):
        self._providers: dict[str, Provider] = {}
        self._secrets = dict(secrets or {})

    def register(self, provider: Provider) -> None:
        if provider.provider_id in self._providers:
            raise ValueError(f"provider already registered: {provider.provider_id}")
        self._models(provider)
        self._providers[provider.provider_id] = provider

    def provider(self, provider_id: str) -> Provider:
        try:
            return self._providers[provider_id]
        except KeyError:
            raise ValueError(f"unknown provider: {provider_id}") from None

    @staticmethod
    def _models(provider: Provider) -> dict[str, Model]:
        models = provider.catalog()
        indexed = {model.model_id: model for model in models}
        if len(indexed) != len(models):
            raise ValueError(f"duplicate model IDs in provider: {provider.provider_id}")
        if provider.settings.model not in indexed:
            raise ValueError(
                f"default model absent from provider: {provider.provider_id}"
            )
        return indexed

    def select(
        self,
        provider_id: str,
        model_id: str | None = None,
        *,
        effort: str | None = None,
        service_tier: str | None = None,
    ) -> Selection:
        provider = self.provider(provider_id)
        try:
            model = self._models(provider)[model_id or provider.settings.model]
        except KeyError:
            raise ValueError(f"unknown model for provider: {provider_id}") from None
        selected_effort = (
            effort
            if effort is not None
            else (
                provider.settings.effort
                if provider.settings.effort in model.efforts
                else model.default_effort
            )
        )
        selected_tier = (
            service_tier if service_tier is not None else provider.settings.service_tier
        )
        if selected_effort not in model.efforts:
            raise ValueError(f"unsupported effort for model: {model.model_id}")
        if selected_tier not in model.service_tiers:
            raise ValueError(f"unsupported service tier for model: {model.model_id}")
        return Selection(
            provider_id, model, selected_effort, selected_tier, provider.session_scope
        )

    def available(self, provider_id: str) -> bool:
        return self.provider(provider_id).credentials.available(self._secrets)

    def catalog(self) -> list[dict[str, Any]]:
        result = []
        for provider in self._providers.values():
            models = self._models(provider)
            result.append(
                {
                    "id": provider.provider_id,
                    "label": provider.label,
                    "available": self.available(provider.provider_id),
                    "models": [
                        {
                            "id": model.model_id,
                            "label": model.label,
                            "efforts": list(model.efforts),
                            "defaultEffort": model.default_effort,
                            "serviceTiers": list(model.service_tiers),
                        }
                        for model in models.values()
                    ],
                }
            )
        return result

    async def run(self, request: AgentRequest) -> AsyncIterator[dict[str, Any]]:
        provider = self.provider(request.selection.provider_id)
        current = self.select(
            provider.provider_id,
            request.selection.model.model_id,
            effort=request.selection.effort,
            service_tier=request.selection.service_tier,
        )
        if current != request.selection:
            raise ValueError(
                "model capabilities or session scope changed; select again"
            )
        if request.session_id is not None and not current.model.resumable:
            raise ValueError("selected model does not support session resume")
        if not self.available(provider.provider_id):
            raise ValueError(
                f"credentials unavailable for provider: {provider.provider_id}"
            )
        async with aclosing(provider.adapter.run(request)) as events:
            async for event in events:
                yield event
