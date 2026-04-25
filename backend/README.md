# omni-backend

Unified FastAPI backend serving three modules on a single port (:8000):

| Module | Endpoint(s) | Description |
|--------|-------------|-------------|
| **Mirror** | `GET /bases`, `GET /base-image/{n}`, `POST /base/{n}`, `POST /base/{n}/retry`, `POST /upload`, `WS /ws` | dlib 68-point face landmarks + OpenCV seamlessClone composite |
| **YOLO** | `WS /ws/yolo` | Ultralytics YOLO11s-seg server-side inference, matches `lib/yolo-ws.ts` protocol |
| **SAM2** | `WS /sam2/ws` | SAM2-tiny image predictor tracking, matches `components/tracker-v2.tsx` protocol |
| **Health** | `GET /health`, `GET /health/mirror`, `GET /sam2/health/sam2` | Status of all three modules |

## Quick start

```bash
# 1. Install Python deps (requires uv: https://docs.astral.sh/uv/)
cd backend
uv sync

# 2. Download model weights (~270 MB total, idempotent)
uv run python scripts/download_models.py

# 3. Run
uv run uvicorn omni_backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the npm scripts from the repo root:

```bash
pnpm backend:models   # download weights
pnpm backend          # start server
pnpm demo             # start Next.js + backend concurrently
```

## Environment variables

Backend reads from (first found wins, existing env beats file):
`repo_root/.env.local` → `backend/.env` → `repo_root/.env`

See [`backend/.env.example`](.env.example) for all supported variables.

Key variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ZHIPU_API_KEY` | — | GLM coord detection for Mirror base images |
| `YOLO_MODEL` | `yolo11s-seg.pt` | YOLO weight file name (relative to `models/` or absolute) |
| `YOLO_IMGSZ` | `704` | Inference resolution |
| `SAM2_CHECKPOINT` | `models/sam2.1_hiera_tiny.pt` | Override SAM2 checkpoint path |
| `BACKEND_PORT` | `8000` | Listening port |
| `TEST_MODE` | `0` | Set to `1` to skip all model loading (for CI/tests) |

## Running tests

```bash
cd backend
uv run pytest         # runs in TEST_MODE, no models required
```

## Model files (in `models/`, gitignored)

| File | Size | Source |
|------|------|--------|
| `yolo11s-seg.pt` | ~22 MB | Ultralytics releases |
| `sam2.1_hiera_tiny.pt` | ~150 MB | Meta AI / fbaipublicfiles |
| `shape_predictor_68_face_landmarks.dat` | ~100 MB | dlib.net |

Run `pnpm backend:models` to download all three.
