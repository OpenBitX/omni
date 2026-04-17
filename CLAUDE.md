# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Frontend (Next.js):
- `npm run dev` — dev server on port 3000
- `npm run build` / `npm run start` — prod build / run
- `npm run typecheck` — `tsc --noEmit` (the only automated gate — no lint, no tests)

Backend (Python, under `server/`, only needed for `/mirror`):
- `npm run server` — starts the FastAPI + WebSocket server on `0.0.0.0:8000`. Wraps `server/.venv/bin/python server/server.py`. Auto-loads `ZHIPU_API_KEY` (for GLM coord detection) and other env from `.env.local` via python-dotenv; inline env vars still win.

Typecheck is the only automated gate; there is no lint script and no test framework.

## What this is

Two apps live in this repo, sharing `app/actions.ts`:

1. **Tracker** (route `/`, `components/tracker.tsx`) — the main app. Tap anything in the camera feed; GPT-4o assesses it and picks a face-placement point; a local JS Lucas-Kanade tracker then holds a cartoon `<Face />` overlay on the object while the camera moves, and GPT-4o + OpenAI TTS gives it a voice. No server process required — everything runs client-side except the two OpenAI calls, which go through Next server actions.

2. **Mirror** (route `/mirror`, `components/mirror.tsx`) — the original app. Streams webcam JPEGs over WebSocket to the local Python server, which runs dlib landmarks + OpenCV `seamlessClone` to paste the user's eyes/mouth onto a chosen base image (orange, upload, etc.) and streams composites back. Keep running if you're touching `/mirror`; otherwise the Python server is optional.

Design language for both: pink/pastel "bubbly" aesthetic (blob-float, soft-pulse, wiggle-on-hover, pastel radial background in `app/globals.css`). Intentional, not cruft.

## Stack

Next.js 15 App Router, React 19, TypeScript, Tailwind v4 (via `@tailwindcss/postcss`, configured in `app/globals.css` with `@theme`, no `tailwind.config.*`). Path alias `@/*` → repo root. Fonts: Geist Sans/Mono only. Python 3.12, FastAPI, OpenCV, dlib, OpenAI SDK.

## Architecture

Four files carry the real weight:

### `components/tracker.tsx` (~1270 lines, one default export)
The Tracker UI and control logic. Phases: `starting` → `ready` → `locked` → `error`. On tap:

1. **Border detection** (`detectObjectBox`): Shi-Tomasi corners in a wide ROI around the tap, filtered by hard-cap + median-distance gate, produces a rough object bbox. Instant visual feedback (tap frame snaps to this bbox).
2. **AI assessment** (`assessObject` from `app/actions.ts`): sends the bbox crop + tap coords to GPT-4o, which returns `{suitable, cx, cy, bbox, reason}`. Default stance is generous — only rejects people or genuinely empty crops. See the `ASSESS_SYSTEM` prompt.
3. **Seed tracking**: detects fresh corners inside the AI-returned bbox (not the whole scene — this is the single biggest win for "stays in that spot"), anchors the face at the AI's `(cx, cy)` (not the corner centroid), and starts the LK loop.
4. **Speak** (`generateLine`): sends the crop to GPT-4o with the `FACE_SYSTEM` prompt, pipes the line through OpenAI TTS (`tts-1`, voice `nova`), and plays it through a Web Audio analyser that drives the SVG mouth animation.

The tracking loop (`loop`) uses a **two-transform keyframe scheme**: `base` (anchor→keyframe) composed with `transform` (keyframe→current). On every reseed, `transform` folds into `base` and resets to identity, so the displayed pose is exactly continuous across reseeds and drift can't compound inside a single fit. The smoothed display pose (`smoothPose`) decomposes the fit into scale/angle/tx/ty and low-passes each on its own time constant — scale is slowest because sub-pixel point noise makes the overlay "breathe" most visibly on scale.

**Reseed triggers** (`needReseed`): surviving-points drop below threshold, fit RMSE stays above threshold for several frames (drift signal), or a periodic max-age hit. Because reseed is visually free, being aggressive is cheap.

**`generationRef` is load-bearing.** It cancels stale async work (assess/speak) when the user taps again mid-flight. Don't remove those `if (gen !== generationRef.current) return;` guards.

The face scale (`baseFaceScaleRef`) is derived from the AI bbox at lock time — small for a mug, big for a sofa. `<Face />` is 200 CSS px at scale=1, so the multiplier is `target_px / 200`.

### `lib/lk.ts` (~440 lines)
Pure-TS Lucas-Kanade stack, no deps. Exports: `toGray`, `detectCorners` (Shi-Tomasi with ROI), `buildPyramid` / `trackLK` (pyramid LK over Float32 gradients, coarse-to-fine), `estimateSimilarity` / `filterOutliers` / `applyTransform` / `composeTransforms` / `invertTransform`. Processing width is 320px; `PYRAMID_DEPTH=3` handles ~30–40 px inter-frame motion. **`composeTransforms` convention:** `composeTransforms(first, second)(p) = second(first(p))` — reading order matches the math.

### `app/actions.ts`
Two Next server actions, both wired:
- **`assessObject(imageDataUrl, tapX, tapY)`** — GPT-4o vision call that decides suitability and returns a face-placement point + subject bbox. Used by Tracker on tap, before locking.
- **`generateLine(imageDataUrl)`** — GPT-4o in-character line + OpenAI TTS mp3. Used by both Tracker and Mirror.

Both re-validate `data:image/` at the boundary — keep the check. `next.config.mjs` carries `serverActions.bodySizeLimit: "8mb"` specifically because frames travel as data URL strings through these actions.

The `ASSESS_SYSTEM` and `FACE_SYSTEM` prompts are **load-bearing for product behavior**:
- `ASSESS_SYSTEM`: default-to-suitable, only reject people or fully empty crops. Tone of "reason" strings shows in the rejection toast.
- `FACE_SYSTEM`: 14-word cap, in-character voice (no meta, no "as a [thing]"), vary rhythm.

Treat edits to either prompt as product changes, not refactors.

### `server/server.py` (Mirror only, ~700 lines)
FastAPI app:
- `GET /health`, `GET /bases`
- `POST /base/{name}` — swap active base (triggers coord detection if uncached)
- `POST /upload` — multipart image; `cv2.imdecode` verifies bytes **before** writing to disk; 10 MB cap; extension allowlist; path-traversal guarded by `_within_root`
- `WS /ws` — binary = JPEG in/out, text = control events

The composite pipeline runs on a `ThreadPoolExecutor` (`run_in_executor`) so dlib + seamlessClone never block the event loop. `_safe_target` clamps paste coordinates to canvas bounds (seamlessClone crashes otherwise). `Compositor._smooth_shape` EMAs 68-point landmarks across frames with a stale-and-jump fallback to kill dlib jitter without smearing scene cuts.

**GLM base-image coord detection** (`ask_glm_for_coords`, model `glm-5v-turbo` via Zhipu's OpenAI-compatible endpoint): two-phase prompt — (A) identify the main item and return a short `label` (1–3 words, e.g. "Orange", "Sock Monkey"); (B) find the perfect face spot on that item (subject_bbox + eye/mouth points/widths). 60s timeout, retries, fallback to `default_coords()`. `glm-5v-turbo` is a reasoning model — it spends tokens on internal reasoning before emitting `content`, so don't cap `max_tokens` tightly and expect the occasional slow response. Uses the OpenAI Python SDK with a custom `base_url` (`GLM_BASE_URL`, auto-picked from key shape: bigmodel.cn for `<hex>.<secret>` keys, api.z.ai for `sk-` keys; override via `ZHIPU_BASE_URL`). GLM doesn't reliably honor `response_format=json_object`, so `_extract_json_object` strips `<think>` traces and ```json fences before parsing. `validate_coords` rejects junk coords (mouth above eyes, eyes too close, features outside subject_bbox, IOD > 75% of bbox width) and falls back *without caching*; the `label` field is cosmetic and never causes rejection. Results cache as `<name>.v{COORDS_VERSION}.<md5_prefix>.coords.json` next to the image; bump `COORDS_VERSION` (currently 5) if you change the coord schema, prompt, or model — old cache files will be ignored. Label-only prompt tweaks don't need a bump; missing labels fall back to `filename_to_label` (strips Screenshot/IMG/date prefixes, collapses separators).

**Retry behavior** (`POST /base/{name}/retry`): clears all cache files for that base and re-runs detection. If `name` is the currently active base, reloads the compositor in place with new coords. If `name` is NOT active, refreshes the cache silently — no auto-swap — so the user can keep viewing the current base while another one is re-detected in the background. The chip UI stays clickable during retry; only the retrying chip is locked (and the upload button).

The Mirror client (`components/mirror.tsx`) uses a **lockstep** WS flow: a new frame is only sent after the previous composite comes back. `inFlightRef` is load-bearing — don't convert to a timed `setInterval`. Auto-reconnect with exponential backoff (500ms → 8s) runs as long as `wantConnectedRef` is true; `scheduleReconnect` has an intentional `eslint-disable` for the cycle with `openSocket`. Stall watchdog force-closes after 6s without a frame so the reconnect path takes over. Preview is mirrored client-side before send; the server does not mirror.

## Things to know

- **The main route is Tracker.** `app/page.tsx` renders `<Tracker />`. Mirror lives at `/mirror`. Don't swap these without asking.
- **Landmarks model is gitignored.** `server/shape_predictor_68_face_landmarks.dat` (~100 MB) is not committed. If a fresh checkout is missing it, download from `http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2` and `bunzip2` into `server/`. Only needed for `/mirror`.
- **`server/.venv/` is gitignored** and not portable. Recreate with `python3 -m venv server/.venv && server/.venv/bin/pip install opencv-python dlib imutils numpy openai fastapi "uvicorn[standard]" python-multipart websockets python-dotenv`. dlib builds from source on macOS and needs `cmake` from Homebrew.
- **Env loading order** (`server.py`): `../.env.local` → `./.env` → `../.env`, first existing file wins. `override=False` so an inline `ZHIPU_API_KEY=… npm run server` still beats `.env.local`. The Python server needs `ZHIPU_API_KEY` (Mirror's GLM coord detection). On the Next side, server actions read `OPENAI_API_KEY` from the normal Next env chain (`.env.local`, etc.) — Tracker's assess/speak calls still use GPT-4o + OpenAI TTS.
- **The chip strip in Mirror polls `/bases` every 5s** so it self-heals when the server comes back; no manual refresh needed.
- **Tracker uses rear camera by default** (`facingMode: { ideal: "environment" }`). Desktop falls back to the default webcam.
- **Bubbly aesthetic is intentional** — keep blob-float, soft-pulse, wiggle-on-hover, sparkle-spin, and the pastel radial background unless the brief changes.
