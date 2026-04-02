from __future__ import annotations

import platform
import sys

from fastapi import APIRouter

from mlx_server.app.core.schemas import HealthResponse
from mlx_server.app.runtime.mlx_runtime import mlx_lm_version, mlx_runtime
from mlx_server.app.services.model_service import model_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    current = model_service.get_current_model()
    return HealthResponse(
        status="ok",
        service="mlx_server",
        python=sys.version.split()[0],
        mlx_lm=mlx_lm_version,
        modelLoaded=mlx_runtime.is_loaded,
        currentModel=current.hfRepo if current else None,
    )
