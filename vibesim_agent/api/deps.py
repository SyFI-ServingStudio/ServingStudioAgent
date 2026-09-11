"""HTTP authentication dependencies built from explicit application authority."""

from secrets import compare_digest

from fastapi import Header, HTTPException
from pydantic import SecretStr

from ..services.capabilities import Capability, CapabilityRegistry, bearer_token


def capability_dependency(capabilities: CapabilityRegistry):
    async def require_capability(
        authorization: str | None = Header(default=None),
    ) -> Capability:
        capability = capabilities.authorize(bearer_token(authorization))
        if capability is None:
            raise HTTPException(
                401, "missing, expired, or invalid managed-run capability"
            )
        return capability

    return require_capability


def token_dependency(token: SecretStr):
    configured = token.get_secret_value()

    async def require_token(authorization: str | None = Header(default=None)) -> None:
        if configured and not compare_digest(
            (authorization or "").encode(), ("Bearer " + configured).encode()
        ):
            raise HTTPException(401, "missing or invalid bearer token")

    return require_token
