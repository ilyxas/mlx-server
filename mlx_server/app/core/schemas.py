from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


class ApiModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class ErrorInfo(ApiModel):
    code: str
    message: str


class ErrorResponse(ApiModel):
    ok: bool = False
    error: ErrorInfo


class HealthResponse(ApiModel):
    status: Literal["ok"]
    service: str
    python: str
    mlx_lm: str
    modelLoaded: bool
    currentModel: Optional[str] = None


class ModelInfo(ApiModel):
    id: str
    hfRepo: str
    label: str
    downloaded: bool = False
    loaded: bool = False
    recommended: bool = False


class ModelsResponse(ApiModel):
    items: list[ModelInfo]


class CurrentModelResponse(ApiModel):
    loaded: bool
    model: Optional[ModelInfo] = None


class LoadModelRequest(ApiModel):
    modelId: str


class LoadModelResponse(ApiModel):
    ok: bool = True
    loaded: bool
    model: ModelInfo


Role = Literal["system", "user", "assistant"]


class ChatMessage(ApiModel):
    role: Role
    content: str = Field(min_length=1)


class GenerationParams(ApiModel):
    maxTokens: int = Field(default=300, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    topP: float = Field(default=1.0, gt=0.0, le=1.0)
    stop: list[str] = Field(default_factory=list)


class ChatRequest(ApiModel):
    sessionId: str
    messages: list[ChatMessage] = Field(min_length=1)
    params: GenerationParams = Field(default_factory=GenerationParams)


class UsageInfo(ApiModel):
    promptTokens: Optional[int] = None
    completionTokens: Optional[int] = None
    totalTokens: Optional[int] = None


class ChatResponse(ApiModel):
    sessionId: str
    model: ModelInfo
    message: ChatMessage
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ResetSessionRequest(ApiModel):
    sessionId: str


class ResetSessionResponse(ApiModel):
    ok: bool = True
    sessionId: str
    cleared: bool = True
