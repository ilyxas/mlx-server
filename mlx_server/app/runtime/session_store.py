from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from mlx_server.app.core.schemas import ChatMessage


@dataclass
class SessionState:
    session_id: str
    messages: list[ChatMessage] = field(default_factory=list)
    model_id: Optional[str] = None
    prompt_cache: Any = None


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def reset(self, session_id: str) -> SessionState:
        session = SessionState(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def set_model(self, session_id: str, model_id: str) -> None:
        session = self.get_or_create(session_id)
        session.model_id = model_id


session_store = SessionStore()
