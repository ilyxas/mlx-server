from __future__ import annotations

import os
from typing import Optional

from mlx_server.app.core.schemas import ModelInfo
from mlx_server.app.runtime.mlx_runtime import mlx_runtime
from mlx_server.app.runtime.model_registry import ModelEntry, model_registry


def _entry_to_info(entry: ModelEntry, loaded: bool = False) -> ModelInfo:
    downloaded = _is_downloaded(entry.hf_repo)
    return ModelInfo(
        id=entry.id,
        hfRepo=entry.hf_repo,
        label=entry.label,
        downloaded=downloaded,
        loaded=loaded,
        recommended=entry.recommended,
    )


def _is_downloaded(hf_repo: str) -> bool:
    """Check whether the model is already cached locally."""
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore[import-untyped]
        result = try_to_load_from_cache(repo_id=hf_repo, filename="config.json")
        return result is not None and result is not False
    except Exception:
        return False


class ModelService:
    def get_models(self) -> list[ModelInfo]:
        current_repo = mlx_runtime.current_repo
        items = []
        for entry in model_registry.all():
            loaded = mlx_runtime.is_loaded and entry.hf_repo == current_repo
            items.append(_entry_to_info(entry, loaded=loaded))
        return items

    def get_current_model(self) -> Optional[ModelInfo]:
        if not mlx_runtime.is_loaded or mlx_runtime.current_repo is None:
            return None
        entry = model_registry.get(mlx_runtime.current_repo)
        if entry is None:
            return None
        return _entry_to_info(entry, loaded=True)

    def load_model(self, model_id: str) -> ModelInfo:
        entry = model_registry.get(model_id)
        if entry is None:
            from mlx_server.app.api.errors import AppError

            raise AppError("MODEL_NOT_FOUND", f"Model '{model_id}' not found in registry.")
        try:
            mlx_runtime.load(entry.hf_repo)
        except Exception as exc:
            from mlx_server.app.api.errors import AppError

            raise AppError("MODEL_LOAD_FAILED", str(exc)) from exc
        return _entry_to_info(entry, loaded=True)


model_service = ModelService()
