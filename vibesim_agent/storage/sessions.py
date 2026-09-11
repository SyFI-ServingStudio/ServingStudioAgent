"""Persist opaque CLI sessions; compatibility is supplied by provider selection."""

from dataclasses import dataclass

from ..domain.roles import Role
from .database import Database


@dataclass(frozen=True)
class Session:
    role: Role
    provider_id: str
    session_scope: str
    session_id: str


class Sessions:
    def __init__(self, database: Database):
        self.database = database

    def save(self, conversation_id: str, session: Session) -> None:
        if not session.provider_id or not session.session_scope or not session.session_id:
            raise ValueError("session identity must not be empty")
        with self.database.connect(write=True) as connection:
            connection.execute(
                """INSERT INTO agent_sessions
                   (conversation_id, role, provider_id, session_scope, session_id)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(conversation_id, role) DO UPDATE SET
                     provider_id = excluded.provider_id,
                     session_scope = excluded.session_scope,
                     session_id = excluded.session_id""",
                (conversation_id, session.role.value, session.provider_id,
                 session.session_scope, session.session_id),
            )

    def list(self, conversation_id: str) -> tuple[Session, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT role, provider_id, session_scope, session_id FROM agent_sessions "
                "WHERE conversation_id = ? ORDER BY role", (conversation_id,)
            ).fetchall()
        return tuple(Session(Role(row["role"]), row["provider_id"], row["session_scope"], row["session_id"])
                     for row in rows)

    def compatible(self, conversation_id: str, scopes: dict[Role, str]) -> dict[Role, str]:
        return {session.role: session.session_id for session in self.list(conversation_id)
                if session.session_scope == scopes.get(session.role)}
