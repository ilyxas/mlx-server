from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from mlx_server.app.api.errors import AppError
from mlx_server.app.core.schemas import ChatRequest, ChatResponse
from mlx_server.app.runtime.mlx_runtime import mlx_runtime
from mlx_server.app.services.chat_service import chat_service
from mlx_server.app.services.model_service import model_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def post_chat(body: ChatRequest) -> ChatResponse:
    return chat_service.chat(
        session_id=body.sessionId,
        messages=body.messages,
        params=body.params,
    )


@router.post("/chat/stream")
def post_chat_stream(body: ChatRequest) -> StreamingResponse:
    # Validate model is loaded before initiating the stream so we can return
    # a proper error response instead of a 200 with an empty/error body.
    if not mlx_runtime.is_loaded:
        raise AppError("NO_MODEL_LOADED", "No model is currently loaded.")

    def token_stream():
        for token in chat_service.stream_chat(
            session_id=body.sessionId,
            messages=body.messages,
            params=body.params,
        ):
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")
