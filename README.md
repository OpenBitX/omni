# omni

A web app where cartoon faces with voices are placed on everyday objects.

## Routes

| Route | App |
|-------|-----|
| `/` | **Tracker** — YOLO WS detects objects, tap to lock a talking face |
| `/v2` | **Tracker V2** — SAM2 tracking backend (experimental) |
| `/gallery` | Card collection + bilingual teacher mode |

## Quick start

### 1. Frontend

```bash
pnpm install
pnpm dev        # https://localhost:3000
```

### 2. Backend (Python)

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
cd backend && uv sync           # install Python deps
pnpm backend:models             # download model weights (~170 MB)
pnpm backend                    # start on https://localhost:8000
```

Or start both together:
```bash
pnpm demo
```

### 3. Environment

Copy `.env.example` to `.env.local` and fill in API keys:

```bash
cp .env.example .env.local
```

Minimum for the main Tracker (`/`):
- `NEXT_PUBLIC_BACKEND_URL=https://localhost:8000`
- `ZHIPU_API_KEY` — GLM vision (assess + describe)
- `OPENAI_API_KEY` — gpt-4o-mini bundled first-tap + Whisper STT fallback
- `CARTESIA_API_KEY` — TTS (caption-only without it)

Recommended:
- `CEREBRAS_API_KEY` — fast ~200ms retap replies

## Tech stack

- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind v4, pnpm
- **Backend**: Python 3.12+, FastAPI, uv
- **ML**: YOLO11s-seg (Ultralytics), SAM2-tiny (Meta)
- **AI**: Zhipu GLM-5V-Turbo, OpenAI gpt-4o-mini, Cerebras Llama, Cartesia Sonic

## Project layout

```
omni-main/
├── app/                    Next.js app router
│   ├── actions.ts          Server actions (GLM / OpenAI / Cerebras)
│   └── api/                API routes (TTS, Runware, converse, etc.)
├── components/
│   ├── tracker.tsx         Main Tracker (~5200 lines)
│   ├── tracker-v2.tsx      SAM2 Tracker
│   ├── mirror.tsx          Mirror component
│   └── face-voice.tsx      Talking face renderer
├── lib/
│   ├── backend-url.ts      Single source for backend HTTP/WS URLs
│   ├── yolo-ws.ts          WS YOLO client
│   ├── iou.ts              Identity tracker + EMA + anchor
│   ├── tts.ts              Cartesia streaming TTS
│   └── ...
├── public/
│   └── facevoice/          Eye video + 9 mouth-shape PNGs
├── backend/                Unified Python backend
│   ├── pyproject.toml      uv project config
│   ├── src/omni_backend/
│   │   ├── main.py         FastAPI entry point
│   │   ├── config.py       Settings from env vars
│   │   ├── routers/        mirror.py / yolo.py / sam2.py
│   │   └── services/       compositor.py / coords.py / sam2_tracker.py
│   ├── assets/base_images/ Mirror base images
│   ├── models/             Downloaded weights (gitignored)
│   ├── scripts/
│   │   └── download_models.py
│   └── tests/              Pytest smoke tests
└── certificates/           mkcert dev HTTPS certs
```

## Deployment

Set `NEXT_PUBLIC_BACKEND_URL` to your deployed backend URL (e.g. `https://api.example.com`). The frontend will use it for all WS and HTTP connections to the backend.

The backend serves CORS `allow_origins=["*"]` by default. Restrict in production via the `CORS_ORIGINS` env var if needed.
