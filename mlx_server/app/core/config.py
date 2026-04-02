from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "mlx_server"
    host: str = "0.0.0.0"
    port: int = 8000
    default_model_id: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="MLX_SERVER_", extra="ignore")


settings = Settings()
