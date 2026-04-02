from __future__ import annotations

import uvicorn

from mlx_server.app.core.config import settings


def main() -> None:
    uvicorn.run(
        "mlx_server.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
