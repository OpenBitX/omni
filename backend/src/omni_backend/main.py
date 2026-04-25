"""Unified omni-backend.

Single FastAPI application combining:
  - YOLO: Ultralytics segmentation (WS /ws/yolo)
  - HTTP API: /api/* endpoints (assess, describe, generate-line, group-line,
    converse, teacher-say, gallerize-card, tts/stream, speak, runware/generate)

This is the ONLY backend process — there is no separate Node.js API server.

Run:
    uv run uvicorn omni_backend.main:app --host 0.0.0.0 --port 8000 --reload
Or via npm script:
    pnpm backend
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from omni_backend.config import settings
from omni_backend.routers import api as api_router
from omni_backend.routers import yolo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("omni")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("omni-backend starting (TEST_MODE=%s)", settings.test_mode)
    await yolo.startup()
    yield
    yolo._executor.shutdown(wait=False, cancel_futures=True)
    log.info("omni-backend shutdown complete")


app = FastAPI(title="omni-backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Speak-Meta", "X-Tts-Backend"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    path = request.url.path
    if path == "/health":
        return await call_next(request)
    t0 = time.time()
    log.info("[http] ▶ %s %s", request.method, path)
    try:
        response = await call_next(request)
    except Exception as e:
        log.warning("[http] ✖ %s %s after %dms: %s", request.method, path, int((time.time() - t0) * 1000), e)
        raise
    log.info("[http] ◀ %s %s %d in %dms", request.method, path, response.status_code, int((time.time() - t0) * 1000))
    return response


# ── Include routers ──────────────────────────────────────────────────────────

# YOLO WS → mounted at /ws so the endpoint becomes /ws/yolo
app.include_router(yolo.router, prefix="/ws")

# HTTP /api/* — replaces the old Express server
app.include_router(api_router.router)


# ── Aggregated health ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "ok": True,
        "test_mode": settings.test_mode,
        "yolo": {
            "model_loaded": yolo._model is not None,
            "device": yolo._device,
            "model": settings.yolo_model,
        },
    }


if __name__ == "__main__":
    uvicorn.run(
        "omni_backend.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        reload=False,
    )
