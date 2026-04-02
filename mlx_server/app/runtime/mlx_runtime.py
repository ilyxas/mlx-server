from __future__ import annotations

import os
from typing import Iterator, Optional

try:
    import mlx_lm  # type: ignore[import-untyped]
    from mlx_lm import load as mlx_load, generate as mlx_generate  # type: ignore[import-untyped]
    from mlx_lm.utils import stream_generate as mlx_stream_generate  # type: ignore[import-untyped]

    _MLX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MLX_AVAILABLE = False


def _get_mlx_lm_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("mlx-lm")
    except Exception:
        return "unavailable"


class MlxRuntime:
    """Low-level wrapper around mlx_lm load/generate functionality."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._current_repo: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def current_repo(self) -> Optional[str]:
        return self._current_repo

    def load(self, hf_repo: str) -> None:
        """Load a model from a HuggingFace repository path."""
        if not _MLX_AVAILABLE:
            raise RuntimeError(
                "mlx_lm is not installed or not available on this platform."
            )
        self._model, self._tokenizer = mlx_load(hf_repo)
        self._current_repo = hf_repo

    def unload(self) -> None:
        """Unload the currently loaded model."""
        self._model = None
        self._tokenizer = None
        self._current_repo = None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text synchronously."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded.")
        if not _MLX_AVAILABLE:  # pragma: no cover
            raise RuntimeError("mlx_lm is not available.")

        result = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
        )
        return result

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        """Generate text as a token stream."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded.")
        if not _MLX_AVAILABLE:  # pragma: no cover
            raise RuntimeError("mlx_lm is not available.")

        for token, _ in mlx_stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
        ):
            yield token

    def apply_chat_template(self, messages: list[dict]) -> str:
        """Apply the tokenizer's chat template to a list of messages."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded.")
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback: simple concatenation
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)


mlx_runtime = MlxRuntime()
mlx_lm_version: str = _get_mlx_lm_version()
