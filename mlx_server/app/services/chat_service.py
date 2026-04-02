from __future__ import annotations

from typing import Iterator

from mlx_server.app.api.errors import AppError
from mlx_server.app.core.schemas import ChatMessage, ChatResponse, GenerationParams, ModelInfo, UsageInfo
from mlx_server.app.runtime.mlx_runtime import mlx_runtime
from mlx_server.app.runtime.model_registry import model_registry
from mlx_server.app.runtime.session_store import SessionState, session_store
from mlx_server.app.services.model_service import model_service


def _require_model() -> ModelInfo:
    info = model_service.get_current_model()
    if info is None:
        raise AppError("NO_MODEL_LOADED", "No model is currently loaded.")
    return info


class ChatService:
    def _ensure_session_cache(self, session: SessionState, model_info: ModelInfo) -> None:
        """Ensure the session cache matches the current model, creating a new cache if needed."""
        if session.model_id != model_info.id:
            session.prompt_cache = mlx_runtime.create_prompt_cache()
            session.model_id = model_info.id

    def chat(
        self,
        session_id: str,
        messages: list[ChatMessage],
        params: GenerationParams,
    ) -> ChatResponse:
        model_info = _require_model()
        session = session_store.get_or_create(session_id)
        self._ensure_session_cache(session, model_info)

        prompt = mlx_runtime.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages]
        )

        try:
            text = mlx_runtime.generate(
                prompt=prompt,
                max_tokens=params.maxTokens,
                temperature=params.temperature,
                top_p=params.topP,
                prompt_cache=session.prompt_cache,
            )
        except Exception as exc:
            raise AppError("GENERATION_FAILED", str(exc)) from exc

        reply = ChatMessage(role="assistant", content=text)
        session.messages.extend(messages)
        session.messages.append(reply)

        return ChatResponse(
            sessionId=session_id,
            model=model_info,
            message=reply,
            usage=UsageInfo(),
        )

    def stream_chat(
        self,
        session_id: str,
        messages: list[ChatMessage],
        params: GenerationParams,
    ) -> Iterator[str]:
        model_info = _require_model()
        session = session_store.get_or_create(session_id)
        self._ensure_session_cache(session, model_info)

        prompt = mlx_runtime.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages]
        )

        response_chunks: list[str] = []
        try:
            for token in mlx_runtime.stream_generate(
                prompt=prompt,
                max_tokens=params.maxTokens,
                temperature=params.temperature,
                top_p=params.topP,
                prompt_cache=session.prompt_cache,
            ):
                response_chunks.append(token)
                yield token
        except Exception as exc:
            raise AppError("GENERATION_FAILED", str(exc)) from exc

        # Save session state once the stream is fully consumed.
        full_response = "".join(response_chunks)
        session.messages.extend(messages)
        session.messages.append(ChatMessage(role="assistant", content=full_response))

    def reset_session(self, session_id: str) -> None:
        session_store.reset(session_id)


chat_service = ChatService()
