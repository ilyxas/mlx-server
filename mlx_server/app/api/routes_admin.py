from __future__ import annotations

from fastapi import APIRouter

from mlx_server.app.core.schemas import ResetSessionRequest, ResetSessionResponse
from mlx_server.app.services.chat_service import chat_service

router = APIRouter()


@router.post("/session/reset", response_model=ResetSessionResponse)
def reset_session(body: ResetSessionRequest) -> ResetSessionResponse:
    chat_service.reset_session(body.sessionId)
    return ResetSessionResponse(ok=True, sessionId=body.sessionId, cleared=True)
