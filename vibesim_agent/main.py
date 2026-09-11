"""Explicit ASGI composition; importing this module prepares no runtime state."""

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.catalog import catalog_router
from .api.conversations import (
    conversation_collection_router,
    conversation_index_router,
    conversation_router,
)
from .services.conversation import ConversationService
from .services.turn import TurnService


def create_app(
    turns: TurnService,
    *,
    conversations: ConversationService | None = None,
    prepare_workspace: Callable[[str], str] | None = None,
    cleanup_conversation: Callable[[str, str], Awaitable[None]] | None = None,
    startup: Callable[[], Awaitable[object]] | None = None,
    shutdown: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        try:
            if startup is not None:
                await startup()
            yield
        finally:
            if shutdown is None:
                await turns.close()
            else:
                await shutdown()

    app = FastAPI(lifespan=lifespan)
    conversations = conversations or ConversationService(turns.storage)
    if conversations.workspaces is not None:
        app.include_router(conversation_index_router(conversations))
    app.include_router(catalog_router(conversations))
    app.include_router(
        conversation_collection_router(
            conversations, prepare_workspace=prepare_workspace
        )
    )
    app.include_router(
        conversation_router(
            turns, conversations, cleanup_conversation=cleanup_conversation
        )
    )
    return app
