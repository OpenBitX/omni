"""Centralised configuration loaded from environment variables.

All env vars are read here once; the rest of the app imports `settings`.
dotenv loading order (first existing file wins, override=False so real env beats file):
  repo_root/.env.local → backend/.env → repo_root/.env
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent          # src/omni_backend/
_BACKEND = _HERE.parent.parent                   # backend/
_REPO_ROOT = _BACKEND.parent                     # repo root

for _candidate in (
    _REPO_ROOT / ".env.local",
    _BACKEND / ".env",
    _REPO_ROOT / ".env",
):
    if _candidate.exists():
        load_dotenv(_candidate, override=False)
        break


class Settings:
    # ── CORS ──────────────────────────────────────────────────────────
    cors_origins: list[str] = ["*"]

    # ── Server ────────────────────────────────────────────────────────
    host: str = os.environ.get("BACKEND_HOST", "0.0.0.0")
    port: int = int(os.environ.get("BACKEND_PORT", "8000"))

    # ── Model paths ───────────────────────────────────────────────────
    models_dir: Path = Path(os.environ.get("MODELS_DIR", str(_BACKEND / "models")))

    @property
    def yolo_model(self) -> str:
        name = os.environ.get("YOLO_MODEL", "yolo11s-seg.pt")
        if os.path.isabs(name):
            return name
        local = self.models_dir / name
        if local.exists():
            return str(local)
        return name

    yolo_imgsz: int = int(os.environ.get("YOLO_IMGSZ", "704"))

    # ── GLM / Zhipu ───────────────────────────────────────────────────
    @property
    def glm_api_key(self) -> str | None:
        return os.environ.get("ZHIPU_API_KEY")

    @property
    def glm_base_url(self) -> str:
        override = os.environ.get("ZHIPU_BASE_URL")
        if override:
            return override
        key = self.glm_api_key or ""
        if "." in key and not key.startswith("sk-"):
            return "https://open.bigmodel.cn/api/paas/v4/"
        return "https://api.z.ai/api/paas/v4/"

    # ── External provider keys ────────────────────────────────────────
    # Read here for visibility + fast-fail logging; the service modules
    # read them directly from os.environ for parity with the Node impl.
    @property
    def openai_api_key(self) -> str | None:
        return os.environ.get("OPENAI_API_KEY")

    @property
    def cerebras_api_key(self) -> str | None:
        return os.environ.get("CEREBRAS_API_KEY")

    @property
    def cartesia_api_key(self) -> str | None:
        return os.environ.get("CARTESIA_API_KEY")

    @property
    def runware_api_key(self) -> str | None:
        return os.environ.get("RUNWARE_API_KEY")

    # ── Test mode ────────────────────────────────────────────────────
    # When TEST_MODE=1, routers substitute stub models so tests run without
    # real checkpoints downloaded.
    test_mode: bool = os.environ.get("TEST_MODE", "0") == "1"


settings = Settings()
