"""Legacy browser model-picker projection of the explicit provider registry."""

from fastapi import APIRouter, HTTPException

from ..services.conversation import ConversationService


def _environment_names(names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not name.startswith("inline:")]


def catalog_router(conversations: ConversationService) -> APIRouter:
    router = APIRouter(prefix="/api/agent/v1")

    @router.get("/codex-backends")
    def catalog():
        registry = conversations.providers
        if registry is None or conversations.default_runtimes is None:
            raise HTTPException(503, "provider catalog is not configured")
        models, families = [], []
        for item in registry.catalog():
            provider = registry.provider(item["id"])
            families.append(
                {
                    "id": provider.provider_id,
                    "label": provider.label,
                    "runner": provider.adapter.adapter_id,
                    "available": item["available"],
                    "requiredEnvironment": _environment_names(
                        provider.credentials.all_secrets
                    ),
                    "credentialEnvironmentAlternatives": _environment_names(
                        provider.credentials.any_secrets
                    ),
                }
            )
            for model in item["models"]:
                models.append(
                    {
                        **model,
                        "family": provider.provider_id,
                        "familyLabel": provider.label,
                        "runner": provider.adapter.adapter_id,
                        "available": item["available"],
                        "defaultServiceTier": "default",
                    }
                )
        defaults = conversations.browser_runtimes({})
        return {
            "models": models,
            "families": families,
            "defaults": conversations.runtime_projection(defaults),
        }

    return router
