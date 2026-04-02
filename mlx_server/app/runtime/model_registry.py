from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelEntry:
    id: str
    hf_repo: str
    label: str
    recommended: bool = False


_REGISTRY: list[ModelEntry] = [
    ModelEntry(
        id="llama32_3b_4bit",
        hf_repo="mlx-community/Llama-3.2-3B-Instruct-4bit",
        label="Llama 3.2 3B Instruct 4bit",
        recommended=True,
    ),
    ModelEntry(
        id="qwen25_7b_4bit",
        hf_repo="mlx-community/Qwen2.5-7B-Instruct-4bit",
        label="Qwen 2.5 7B Instruct 4bit",
        recommended=False,
    ),
]

_ALIAS_MAP: dict[str, str] = {entry.hf_repo: entry.id for entry in _REGISTRY}

DEFAULT_MODEL_ID: str = "llama32_3b_4bit"


class ModelRegistry:
    def __init__(self, entries: list[ModelEntry] | None = None) -> None:
        self._entries: list[ModelEntry] = entries if entries is not None else list(_REGISTRY)

    def all(self) -> list[ModelEntry]:
        return list(self._entries)

    def get(self, model_id: str) -> Optional[ModelEntry]:
        for entry in self._entries:
            if entry.id == model_id or entry.hf_repo == model_id:
                return entry
        return None

    def default(self) -> Optional[ModelEntry]:
        return self.get(DEFAULT_MODEL_ID)

    def resolve_id(self, model_id_or_repo: str) -> Optional[str]:
        entry = self.get(model_id_or_repo)
        return entry.id if entry else None


model_registry = ModelRegistry()
