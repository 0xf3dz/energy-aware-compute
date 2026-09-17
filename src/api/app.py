"""HTTP surface for the compute dashboard.

Answers one question: what compute is this machine running, and why? It shows
no electrical detail beyond the scheduler inputs, and it links to VRM for that.
"""

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from api.runtime import Runtime, build_runtime
from api.settings import Settings

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


def create_app(settings: Settings, runtime: Runtime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime or await build_runtime(settings)
        app.state.stop = asyncio.Event()
        app.state.tasks = []
        if runtime is None and settings.serve_scheduler:
            app.state.tasks.append(
                asyncio.create_task(app.state.runtime.run_scheduler(app.state.stop))
            )
        try:
            yield
        finally:
            app.state.stop.set()
            for task in app.state.tasks:
                task.cancel()
            for task in app.state.tasks:
                with contextlib.suppress(BaseException):
                    await task
            if runtime is None:
                await app.state.runtime.close()

    app = FastAPI(title="Energy aware compute", lifespan=lifespan, docs_url="/api/docs")

    def authorize(request: Request, token: str | None = Query(default=None)) -> None:
        configured = settings.dashboard_token
        if configured is None:
            return
        supplied = token or ""
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            supplied = header[7:]
        if supplied != configured.get_secret_value():
            raise HTTPException(status_code=401, detail="Provide the dashboard token")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    async def index(_: None = Depends(authorize)) -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/static/{name}")
    async def static_file(name: str) -> FileResponse:
        path = (STATIC / name).resolve()
        if path.parent != STATIC.resolve() or not path.is_file():
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(path)

    @app.get("/api/snapshot")
    async def snapshot(request: Request, _: None = Depends(authorize)) -> JSONResponse:
        return JSONResponse(await request.app.state.runtime.snapshot())

    @app.get("/api/briefing", response_class=PlainTextResponse)
    async def briefing(request: Request, _: None = Depends(authorize)) -> PlainTextResponse:
        data = await request.app.state.runtime.snapshot()
        latest = data.get("latest_briefing")
        if not latest:
            raise HTTPException(status_code=404, detail="No briefing has been generated yet")
        return PlainTextResponse(latest["text"])

    return app
