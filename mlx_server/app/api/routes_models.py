from __future__ import annotations

from fastapi import APIRouter

from mlx_server.app.core.schemas import (
    CurrentModelResponse,
    LoadModelRequest,
    LoadModelResponse,
    ModelsResponse,
)
from mlx_server.app.services.model_service import model_service

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    return ModelsResponse(items=model_service.get_models())


@router.get("/model/current", response_model=CurrentModelResponse)
def get_current_model() -> CurrentModelResponse:
    model = model_service.get_current_model()
    return CurrentModelResponse(loaded=model is not None, model=model)


@router.post("/model/load", response_model=LoadModelResponse)
def load_model(body: LoadModelRequest) -> LoadModelResponse:
    model = model_service.load_model(body.modelId)
    return LoadModelResponse(ok=True, loaded=True, model=model)
