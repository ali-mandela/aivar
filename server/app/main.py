"""Autonomous Test Orchestration Agent -- HTTP entrypoint.

Give it a URL and it explores the application, plans meaningful test flows,
critiques its own coverage, generates runnable pytest files, executes them,
heals broken locators, and reports what it did and what it could not reach --
with no human intervention between stages.
"""

import logging
from pathlib import Path

from fastapi import FastAPI

from app.api import router
from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=__doc__,
    version="0.1.0",
    debug=settings.debug,
)

app.include_router(router)


@app.get("/health", summary="Liveness plus the two external dependencies")
def health() -> dict:
    """Reports whether the model provider and the database are reachable.

    Neither is fatal: a missing database costs run history, not runs. Knowing
    which one is down before starting a 60-second pipeline is worth the check.
    """
    out: dict = {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
    }

    try:
        from app.llm import LLMConfig

        cfg = LLMConfig.from_env()
        out["llm"] = {"configured": True, **cfg.describe()}
    except Exception as e:
        out["llm"] = {"configured": False, "reason": f"{type(e).__name__}: {e}"}

    try:
        from app.store import health as store_health

        connected, detail = store_health()
        out["database"] = {"connected": connected, "detail": detail}
    except Exception as e:
        out["database"] = {"connected": False, "detail": f"{type(e).__name__}: {e}"}

    return out


def main() -> None:
    """Start the development server. Exposed as `uv run serve`."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        # Watch the source, and nothing this server writes.
        #
        # A run generates pytest files into tests/generated and artifacts into
        # artifacts/. With the whole tree watched, producing the deliverable
        # triggered a reload, which killed the background thread mid-run and
        # emptied the in-memory job registry -- so the client polling /jobs/{id}
        # got a 404 and the run vanished. The pipeline was destroying itself by
        # succeeding.
        reload_dirs=[str(Path(__file__).resolve().parent)],
    )


if __name__ == "__main__":
    main()
