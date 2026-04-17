# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Frontend (Next.js):
- `npm run dev` — dev server on port 3000
- `npm run build` / `npm run start` — prod build / run
- `npm run typecheck` — `tsc --noEmit` (the only automated gate — no lint, no tests)

Backend (Python, under `server/`):
- `npm run server` — starts the FastAPI + WebSocket server on `0.0.0.0:8000`. Wraps `server/.venv/bin/python server/server.py`. The server auto-loads `OPENAI_API_KEY` from `.env.local` via python-dotenv; inline env vars still win.

Two terminals: `npm run dev` (3000) and `npm run server` (8000). Typecheck is the only automated gate; there is no lint script and no test framework.

## What this is

Mirror is a realtime face-composite web app with a playful pink/pastel "bubbly" aesthetic. The browser streams webcam frames over a WebSocket to a local Python server. The server runs dlib face detection + OpenCV `seamlessClone` to paste the user's eyes and mouth onto a chosen base image (orange, uploaded photo, etc.), and streams the composited frames back. Both halves live in this repo.

## Stack

Next.js 15 App Router, React 19, TypeScript, Tailwind v4 (via `@tailwindcss/postcss`, configured in `app/globals.css` with `@theme`, no `tailwind.config.*`). Path alias `@/*` → repo root. Fonts: Geist Sans/Mono only. Python 3.12, FastAPI, OpenCV, dlib, OpenAI SDK (for GPT-4o vision coord detection).

## Architecture

Three files do the real work:

1. **`components/mirror.tsx`** (client, ~580 lines, one default export) — the whole frontend. Owns webcam acquisition, WebSocket lifecycle, reconnection, and all UI. The WS flow is **lockstep**: a new frame is only sent after the previous composite comes back. This naturally rate-limits to server throughput without an explicit fps cap. `inFlightRef` is load-bearing — don't convert to a timed `setInterval`. Auto-reconnect with exponential backoff (500ms → 8s) runs as long as `wantConnectedRef` is true; `scheduleReconnect` has an intentional `eslint-disable` for the cycle with `openSocket`. Stall watchdog force-closes the socket after 6s without a frame so the reconnect path takes over. Text frames from the server (`{"event":"face"|"no_face"|"base", ...}`) drive the small pill overlay. Preview is horizontally mirrored client-side before send; the server does not mirror.

2. **`server/server.py`** — FastAPI app exposing:
   - `GET /health` — status + whether an OpenAI key is loaded
   - `GET /bases` — list of allowed images in the server dir
   - `POST /base/{name}` — swap active base (triggers coord detection if uncached)
   - `POST /upload` — multipart image upload; cv2.imdecode verifies bytes **before** writing to disk; 10 MB cap; extension allowlist; path-traversal guarded by `_within_root`
   - `WS /ws` — binary = JPEG in/out, text = control events

   The composite pipeline runs on a `ThreadPoolExecutor` (`run_in_executor`) so dlib + seamlessClone never block the event loop. `_safe_target` clamps paste coordinates to canvas bounds (seamlessClone crashes otherwise). Frame decode / detect / paste are each wrapped — a bad frame drops silently instead of killing the connection.

3. **`app/actions.ts`** — standalone Next server action `generateLine(imageDataUrl)` that sends a data-URL crop to GPT-4o with the `FACE_SYSTEM` prompt (a one-line in-character utterance for whatever the thing is), then synthesizes it with OpenAI TTS (`tts-1`, voice `nova`) and returns `{ line, audioDataUrl }`. **Currently defined but not yet wired into the UI** — it's the server side of an in-progress "make the base speak" feature. The `FACE_SYSTEM` prompt (14-word cap, no meta, vary rhythm, in-character) is **load-bearing for voice/tone** — treat edits as product changes, not refactors. `next.config.mjs` carries `serverActions.bodySizeLimit: "8mb"` specifically because frames travel as data URL strings through this action.

### GPT-4o coord detection

Each new base image gets its eye/mouth paste targets discovered by GPT-4o vision (`ask_gpt_for_coords`, `response_format: json_object`, `detail: "low"`), with two retries, 20s timeout, and a graceful fallback to `default_coords()` (sensibly centered) on any failure. `validate_coords` rejects junk (mouth above eyes, eyes too close) and falls back too. The result is cached next to the image as `<name>.<md5_prefix>.coords.json`; delete that file to re-detect.

## Things to know

- **Landmarks model is gitignored.** `server/shape_predictor_68_face_landmarks.dat` (~100 MB) is not committed. If a fresh checkout is missing it, download from `http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2` and `bunzip2` into `server/`.
- **`server/.venv/` is gitignored** and not portable. Recreate with `python3 -m venv server/.venv && server/.venv/bin/pip install opencv-python dlib imutils numpy openai fastapi "uvicorn[standard]" python-multipart websockets python-dotenv`. dlib builds from source on macOS and needs `cmake` from Homebrew.
- **Env loading order** (in `server.py`): `../.env.local` → `./.env` → `../.env`, first existing file wins. `override=False` so an inline `OPENAI_API_KEY=… npm run server` still beats `.env.local`.
- **The chip strip polls `/bases` every 5s** so it self-heals when the server comes back; no need for manual refresh.
- **Bubbly aesthetic is intentional** — blob-float, soft-pulse, wiggle-on-hover, and sparkle-spin animations in `app/globals.css` plus the pastel radial background are the design language, not cruft. Keep them unless the brief changes.
- **Server actions re-validate at the boundary** even though the client is same-repo — `generateLine` rejects anything that doesn't start with `data:image/`. Keep this check.
