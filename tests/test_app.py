"""Tests for the mlx-server FastAPI application.

These tests mock out the mlx_runtime and model_registry so the test suite can run
on any platform (not just Apple Silicon).
"""
from __future__ import annotations

from typing import Iterator
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_runtime_and_store():
    """Reset singletons between tests."""
    from mlx_server.app.runtime import mlx_runtime as rt_mod
    from mlx_server.app.runtime import session_store as ss_mod
    from mlx_server.app.services import model_service as ms_mod

    rt_mod.mlx_runtime.unload()
    ss_mod.session_store._sessions.clear()
    yield
    rt_mod.mlx_runtime.unload()
    ss_mod.session_store._sessions.clear()


@pytest.fixture
def client():
    from mlx_server.app.main import app
    return TestClient(app, raise_server_exceptions=False)


def _load_fake_model():
    """Simulate a loaded model without touching mlx."""
    from mlx_server.app.runtime.mlx_runtime import mlx_runtime
    mlx_runtime._model = MagicMock()
    mlx_runtime._tokenizer = MagicMock()
    mlx_runtime._current_repo = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    # make apply_chat_template return a simple string
    mlx_runtime._tokenizer.apply_chat_template = lambda msgs, **kw: " ".join(
        m["content"] for m in msgs
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_no_model(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "mlx_server"
        assert data["modelLoaded"] is False
        assert data["currentModel"] is None

    def test_health_with_model(self, client):
        _load_fake_model()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["modelLoaded"] is True
        assert data["currentModel"] == "mlx-community/Llama-3.2-3B-Instruct-4bit"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModels:
    def test_get_models(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert len(data["items"]) >= 1
        ids = [item["id"] for item in data["items"]]
        assert "llama32_3b_4bit" in ids

    def test_get_current_no_model(self, client):
        resp = client.get("/model/current")
        assert resp.status_code == 200
        data = resp.json()
        assert data["loaded"] is False
        assert data["model"] is None

    def test_get_current_with_model(self, client):
        _load_fake_model()
        resp = client.get("/model/current")
        assert resp.status_code == 200
        data = resp.json()
        assert data["loaded"] is True
        assert data["model"]["id"] == "llama32_3b_4bit"

    def test_load_model_not_found(self, client):
        resp = client.post("/model/load", json={"modelId": "nonexistent"})
        assert resp.status_code == 422
        data = resp.json()
        assert data["ok"] is False
        assert data["error"]["code"] == "MODEL_NOT_FOUND"

    def test_load_model_success(self, client):
        with patch("mlx_server.app.runtime.mlx_runtime.mlx_runtime.load") as mock_load:
            mock_load.side_effect = lambda repo: _load_fake_model()
            resp = client.post("/model/load", json={"modelId": "llama32_3b_4bit"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["loaded"] is True
        assert data["model"]["id"] == "llama32_3b_4bit"

    def test_load_model_failure(self, client):
        with patch("mlx_server.app.runtime.mlx_runtime.mlx_runtime.load") as mock_load:
            mock_load.side_effect = RuntimeError("load error")
            resp = client.post("/model/load", json={"modelId": "llama32_3b_4bit"})
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "MODEL_LOAD_FAILED"


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class TestChat:
    def test_chat_no_model(self, client):
        resp = client.post(
            "/chat",
            json={
                "sessionId": "s1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "NO_MODEL_LOADED"

    def test_chat_success(self, client):
        _load_fake_model()
        with patch("mlx_server.app.runtime.mlx_runtime.mlx_runtime.generate") as mock_gen:
            mock_gen.return_value = "Hi there!"
            resp = client.post(
                "/chat",
                json={
                    "sessionId": "s1",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["sessionId"] == "s1"
        assert data["message"]["role"] == "assistant"
        assert data["message"]["content"] == "Hi there!"
        assert data["model"]["id"] == "llama32_3b_4bit"

    def test_chat_generation_failure(self, client):
        _load_fake_model()
        with patch("mlx_server.app.runtime.mlx_runtime.mlx_runtime.generate") as mock_gen:
            mock_gen.side_effect = RuntimeError("generation error")
            resp = client.post(
                "/chat",
                json={
                    "sessionId": "s1",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "GENERATION_FAILED"

    def test_chat_invalid_request(self, client):
        resp = client.post("/chat", json={"sessionId": "s1", "messages": []})
        assert resp.status_code == 422
        data = resp.json()
        assert data["ok"] is False
        assert data["error"]["code"] == "INVALID_REQUEST"

    def test_chat_stream_no_model(self, client):
        resp = client.post(
            "/chat/stream",
            json={
                "sessionId": "s2",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 422

    def test_chat_stream_success(self, client):
        _load_fake_model()

        def fake_stream(**kwargs):
            yield "Hello"
            yield " world"

        with patch(
            "mlx_server.app.runtime.mlx_runtime.mlx_runtime.stream_generate",
            side_effect=fake_stream,
        ):
            resp = client.post(
                "/chat/stream",
                json={
                    "sessionId": "s2",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
        assert resp.status_code == 200
        assert "Hello" in resp.text

    def test_chat_stream_saves_session(self, client):
        _load_fake_model()

        def fake_stream(**kwargs):
            yield "Hello"
            yield " world"

        with patch(
            "mlx_server.app.runtime.mlx_runtime.mlx_runtime.stream_generate",
            side_effect=fake_stream,
        ):
            resp = client.post(
                "/chat/stream",
                json={
                    "sessionId": "stream-sess",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
        assert resp.status_code == 200

        from mlx_server.app.runtime.session_store import session_store
        session = session_store.get("stream-sess")
        assert session is not None
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "Hi"
        assert session.messages[1].role == "assistant"
        assert session.messages[1].content == "Hello world"

    def test_chat_session_cache_reset_on_model_change(self, client):
        """Session cache is invalidated when a different model is loaded."""
        _load_fake_model()

        with patch("mlx_server.app.runtime.mlx_runtime.mlx_runtime.generate") as mock_gen:
            mock_gen.return_value = "First response"
            client.post(
                "/chat",
                json={"sessionId": "cache-sess", "messages": [{"role": "user", "content": "Hello"}]},
            )

        from mlx_server.app.runtime.session_store import session_store
        session = session_store.get("cache-sess")
        assert session is not None
        assert session.model_id == "llama32_3b_4bit"

        # Simulate switching to a different model
        from mlx_server.app.runtime.mlx_runtime import mlx_runtime as rt
        rt._current_repo = "mlx-community/Qwen2.5-7B-Instruct-4bit"

        with patch("mlx_server.app.runtime.mlx_runtime.mlx_runtime.generate") as mock_gen:
            mock_gen.return_value = "Second response"
            client.post(
                "/chat",
                json={"sessionId": "cache-sess", "messages": [{"role": "user", "content": "Hello again"}]},
            )

        session = session_store.get("cache-sess")
        assert session.model_id == "qwen25_7b_4bit"


# ---------------------------------------------------------------------------
# Session reset
# ---------------------------------------------------------------------------

class TestSession:
    def test_reset_session(self, client):
        resp = client.post("/session/reset", json={"sessionId": "demo-1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["sessionId"] == "demo-1"
        assert data["cleared"] is True

    def test_reset_clears_cache(self, client):
        """Resetting a session wipes its prompt cache and message history."""
        _load_fake_model()

        with patch("mlx_server.app.runtime.mlx_runtime.mlx_runtime.generate") as mock_gen:
            mock_gen.return_value = "reply"
            client.post(
                "/chat",
                json={"sessionId": "r1", "messages": [{"role": "user", "content": "Hello"}]},
            )

        from mlx_server.app.runtime.session_store import session_store
        session = session_store.get("r1")
        assert session is not None
        assert len(session.messages) == 2

        client.post("/session/reset", json={"sessionId": "r1"})

        session = session_store.get("r1")
        assert session is not None
        assert session.messages == []
        assert session.prompt_cache is None
        assert session.model_id is None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_stop_param_not_accepted(self, client):
        """stop is removed from the public API — extra fields are forbidden."""
        resp = client.post(
            "/chat",
            json={
                "sessionId": "s1",
                "messages": [{"role": "user", "content": "Hello"}],
                "params": {"stop": ["\\n"]},
            },
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["ok"] is False
        assert data["error"]["code"] == "INVALID_REQUEST"

    def test_generation_params_defaults(self):
        from mlx_server.app.core.schemas import GenerationParams
        p = GenerationParams()
        assert p.maxTokens == 300
        assert p.temperature == 0.7
        assert p.topP == 1.0

    def test_chat_message_empty_content_invalid(self):
        from mlx_server.app.core.schemas import ChatMessage
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            ChatMessage(role="user", content="")

    def test_model_info_fields(self):
        from mlx_server.app.core.schemas import ModelInfo
        m = ModelInfo(id="x", hfRepo="repo/x", label="X")
        assert m.downloaded is False
        assert m.loaded is False
        assert m.recommended is False
