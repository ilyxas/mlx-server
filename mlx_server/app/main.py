from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mlx_server.app.api.errors import AppError
from mlx_server.app.api.routes_admin import router as admin_router
from mlx_server.app.api.routes_chat import router as chat_router
from mlx_server.app.api.routes_health import router as health_router
from mlx_server.app.api.routes_models import router as models_router

app = FastAPI(title="mlx-server", version="0.1.0")


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"ok": False, "error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"ok": False, "error": {"code": "INVALID_REQUEST", "message": str(exc)}},
    )


app.include_router(health_router)
app.include_router(models_router)
app.include_router(chat_router)
app.include_router(admin_router)
