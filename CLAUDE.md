# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Frontend (Next.js):**
- `pnpm dev` — dev server on port 3000 with local HTTPS (certificates/key.pem + cert.pem)
- `pnpm build` / `pnpm start` — prod build / run
- `pnpm typecheck` — `tsc --noEmit` (the only automated gate)

**Backend (Python, under `backend/`):**
- `pnpm backend` — start unified FastAPI server on `0.0.0.0:8000`. Wraps `uv --directory backend run uvicorn omni_backend.main:app --reload`
- `pnpm backend:models` — idempotent download of YOLO model weights to `backend/models/`
- `pnpm backend:test` — run pytest smoke tests (TEST_MODE, no real models required)
- `pnpm demo` — start Next.js + backend concurrently

**Install (Python):**
```bash
cd backend && uv sync   # creates backend/.venv and installs all deps
```

## Architecture

```
┌─────────────────────────────────────────────┐
│  Next.js 15 (app router, React 19)          │
│                                             │
│  /            → components/tracker.tsx      │  ← main app
│  /mirror      → components/mirror.tsx       │
│  /v2          → components/tracker-v2.tsx   │
│  /gallery     → app/gallery/page.tsx        │
│  /hi/*        → app/hi/[lang]/page.tsx      │
│                                             │
│  lib/backend-url.ts                         │
│    NEXT_PUBLIC_BACKEND_URL (env)            │
│    httpUrl(path)  →  https://localhost:8000 │
│    wsUrl(path)    →  wss://localhost:8000   │
└──────────────┬──────────────────────────────┘
               │  WS + HTTP (configurable via NEXT_PUBLIC_BACKEND_URL)
               ▼
┌─────────────────────────────────────────────┐
│  backend/  (omni-backend, port 8000)        │
│  uv + pyproject.toml                        │
│                                             │
│  GET  /health            combined status    │
│  GET  /bases             Mirror base list   │
│  GET  /base-image/{n}    Mirror base image  │
│  POST /base/{n}          Mirror swap base   │
│  POST /base/{n}/retry    re-run GLM coords  │
│  POST /upload            Mirror upload      │
│  WS   /ws                Mirror composite   │
│  WS   /ws/yolo           YOLO11s detection  │
└─────────────────────────────────────────────┘
```

## Front-end/backend decoupling

`lib/backend-url.ts` is the **single source of truth** for all backend URLs:

```typescript
import { httpUrl, wsUrl } from "@/lib/backend-url";
const ws = new WebSocket(wsUrl("/ws/yolo"));
const res = await fetch(httpUrl("/bases"));
```

Set `NEXT_PUBLIC_BACKEND_URL` in `.env.local` to point at a remote server. Default: `https://localhost:8000`.

## What this is

Two live apps + two experimental routes:

1. **Tracker** (route `/`, `components/tracker.tsx`) — the main app. YOLO WS detects objects every frame; tap one and a cartoon `<FaceVoice />` rides on it until it leaves frame. A bundled `gpt-4o-mini` vision call writes the opening line + picks a Cartesia voice + captures a persona card. Retaps + conversation run off that persona card via Cerebras Llama (text-only, ~200ms). Mic button for voice conversation.

2. **Mirror** (route `/mirror`, `components/mirror.tsx`) — streams webcam JPEGs over WebSocket to the Python backend, which runs dlib landmarks + OpenCV `seamlessClone` to paste the user's eyes/mouth onto a chosen base image and streams composites back.

3. **Gallery** (route `/gallery`) — card collection, bilingual teacher mode, comic image generation.

Design language: pink/pastel "bubbly" aesthetic (blob-float, soft-pulse, wiggle-on-hover, pastel radial background in `app/globals.css`). Intentional, not cruft.

## Stack

Next.js 15 App Router, React 19, TypeScript, Tailwind v4, pnpm. Python 3.12+, FastAPI, uv. Path alias `@/*` → repo root.

## Backend modules

### `backend/src/omni_backend/`

| Module | Purpose |
|--------|---------|
| `main.py` | FastAPI entry point + lifespan (startup/shutdown) |
| `config.py` | All settings from env vars (dotenv loading, model paths) |
| `routers/mirror.py` | Mirror HTTP + WS composite pipeline |
| `routers/yolo.py` | YOLO WS inference router |
| `services/compositor.py` | dlib + OpenCV seamlessClone class (Thread-safe) |
| `services/coords.py` | GLM face-coord detection + cache logic |


### TEST_MODE

Set `TEST_MODE=1` to skip all model loading. Stub models are used instead.
All pytest tests run in TEST_MODE automatically (see `backend/tests/conftest.py`).

### Model files (in `backend/models/`, gitignored)

| File | Size | Used by |
|------|------|---------|
| `yolo11s-seg.pt` | ~22 MB | YOLO WS router |
| `shape_predictor_68_face_landmarks.dat` | ~100 MB | Mirror compositor |

Download: `pnpm backend:models`

## Frontend components

### `components/tracker.tsx` (~5200 lines, one default export)

The Tracker UI and control logic. **Do not split this file** — it contains many load-bearing refs and guards that are tightly coupled:

**`generationRef` is load-bearing** — cancels stale async work (assess/speak) when the user taps again mid-flight. **`speakGen` per-track** is the same idea for re-taps on the same track. Do not remove those `if (gen !== …) return;` guards.

Phases: `starting` → `ready` → `locked` → `error`. Up to `MAX_FACES` (=3) tracks can be locked at once.

Per-tap flow:
1. `pickTappedDetection` — picks smallest-area containing detection
2. `assessObject` (GLM `glm-5v-turbo`) — returns suitable/coords in parallel with step 3
3. Lock + anchor — `anchorFromPoint`/`applyAnchor` from `lib/iou.ts`
4. `generateLine` first tap — bundled `gpt-4o-mini` vision call returns `{description, voiceId, line}`
5. Persona pinning — `{voiceId, description}` on `TrackRefs`, used for all retaps

The continuous tracking loop:
- Inference at up to `MAX_INFERENCE_FPS` (=30); mobile lands at 3–8 fps
- Between inferences: face glides at 60 fps using velocity (`EXTRAP_MAX_MS`=220)
- `SUSPECT_SIZE_RATIO` (1.75) rejects big position jumps
- Box smoothing: split-alpha EMA (`BOX_POS_ALPHA`=0.7, `BOX_SIZE_ALPHA`=0.25)
- `LOST_AFTER_MISSES` (4) **must** be the same constant for fade AND snap logic

### `lib/yolo-ws.ts` (~516 lines)

Browser WS client for YOLO. Default URL from `wsUrl("/ws/yolo")` (→ `NEXT_PUBLIC_BACKEND_URL`). Same public API shape as the removed `lib/yolo.ts` (ONNX path) — `initYolo`, `detect`, `resetYolo`, `subscribeYoloStatus`. **`tracker.tsx` imports from here.**

### `lib/iou.ts`

Identity tracker: `iou`, `matchTarget`, `EMAFilter`, `BoxEMA`, `Anchor` with `anchorFromPoint`/`applyAnchor`. The anchor math is the heart of the tracker.

### `components/face-voice.tsx`

Talking face renderer. Looping video (`public/facevoice/eyes.{webm,mp4}`) + 9 mouth PNG shapes (`shape-A` … `shape-X`). `classifyShape(analyser)` consumed by tracker RAF loop.

### `app/actions.ts` (~2100 lines)

Server actions (provider-split by latency):
- `assessObject` → GLM `glm-5v-turbo` (VLM, ~3-5s)
- `describeObject` → GLM (describe crop)
- `generateLine` — dual path: first-tap bundled VLM (OpenAI `gpt-4o-mini`) or retap Cerebras Llama text-only (~200ms)
- `converseWithObject` — voice-in text-out; TTS separately via `/api/tts/stream`
- `groupLine`, `teacherSay`, `gallerizeCard` — gallery/multi-object features

**Provider clients:** GLM via OpenAI-compatible SDK (base URL from key shape or `ZHIPU_BASE_URL`). Cerebras at `api.cerebras.ai/v1`. OpenAI directly. **Cerebras model name is `llama3.1-8b`** (no hyphens inside version).

**Defensive JSON parsing:** `extractJsonObject` strips `<think>` traces, unwraps ```json fences. Don't remove — load-bearing against model drift.

### `app/api/tts/stream/route.ts`

Streaming TTS passthrough. POSTs `{text, voiceId, turnId}` and streams `audio/mpeg` bytes from Cartesia → browser. `latency: "balanced"`, `chunk_length: 100` for low TTFB. Fallback: OpenAI `tts-1/nova`.

### `app/api/runware/generate/route.ts`

Comic image generation via Runware + optional OpenAI prompt enhancement.

## TTS routing

Server action does NOT synthesize audio — returns `{line, voiceId}` and client POSTs to `/api/tts/stream`. Audio routes through Web Audio graph: `source → analyser → gain → destination`. The analyser drives `classifyShape` for lip sync.

## Load-bearing prompts in actions.ts

(Edits here are product changes, not refactors):
- `ASSESS_SYSTEM`: default-to-suitable, only reject people or empty crops
- `DESCRIBE_SYSTEM`: 35-word cap, concrete/visual, no jokes
- `FACE_BUNDLED_SYSTEM(catalog)`: first-tap; emits `{description, voiceId, line}`
- `FACE_WITH_PERSONA_SYSTEM(description)`: retap; text-only
- `RESPOND_SYSTEM(className, description)`: 22-word reply cap

## Environment variables

See `.env.example` for all frontend variables. See `backend/.env.example` for all backend variables.

**Required for `/`**: `ZHIPU_API_KEY` + `OPENAI_API_KEY`. Without OpenAI, Tracker cannot generate lines.  
**Strongly recommended**: `CEREBRAS_API_KEY` (fast ~200ms retap replies; falls back to `gpt-4o-mini` without it).  
**TTS**: `CARTESIA_API_KEY` (falls back to OpenAI `tts-1/nova`; caption-only without either).

## Assets

- `public/facevoice/` — eyes.{webm,mp4} + shape-{A,B,C,D,E,F,G,H,X}.png (9 shapes)
- `backend/assets/base_images/` — Mirror base images (orange.jpg + user uploads)
- `certificates/` — mkcert dev certificates (local HTTPS)

## Invariants

- Esc wipes all tracks (live demo reset)
- Rear camera default (`facingMode: { ideal: "environment" }`)
- `FACE_VOICE_WIDTH` (280) / `FACE_VOICE_HEIGHT` (160) in `face-voice.tsx` must match `FACE_NATIVE_PX` in `tracker.tsx`
- Mirror uses lockstep WS (one frame in flight at a time via `inFlightRef`)
- GLM `glm-4v-flash` was silently deprecated 2026-04-18; both GLM constants default to `glm-5v-turbo`
