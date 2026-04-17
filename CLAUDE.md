# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Frontend (Next.js):
- `npm run dev` — dev server on port 3000
- `npm run build` / `npm run start` — prod build / run
- `npm run typecheck` — `tsc --noEmit` (the only automated gate — no lint, no tests)
- `npm install` — also runs `scripts/setup-ort.mjs` (postinstall) which copies onnxruntime-web's WASM runtime into `public/ort/` so the browser serves it same-origin. If the browser fails to fetch `/ort/*.wasm`, re-run `node scripts/setup-ort.mjs`.

Backend (Python, under `server/`, only needed for `/mirror`):
- `npm run server` — starts the FastAPI + WebSocket server on `0.0.0.0:8000`. Wraps `server/.venv/bin/python server/server.py`. Auto-loads `ZHIPU_API_KEY` and other env from `.env.local` via python-dotenv.

Typecheck is the only automated gate; there is no lint script and no test framework.

## What this is

Two apps live in this repo, sharing `app/actions.ts`:

1. **Tracker** (route `/`, `components/tracker.tsx`) — the main app. On-device YOLO26n-seg detects objects every frame; tap one and a cartoon `<FaceVoice />` rides on it until it leaves frame. GLM-5V writes the line, Fish.audio / OpenAI speaks it, and a mic button lets you talk back (`converseWithObject`). Everything tracking-side runs client-only — the server actions are the only round-trip.

2. **Mirror** (route `/mirror`, `components/mirror.tsx`) — the original app. Streams webcam JPEGs over WebSocket to the local Python server, which runs dlib landmarks + OpenCV `seamlessClone` to paste the user's eyes/mouth onto a chosen base image (orange, upload, etc.) and streams composites back. Keep running only if you're touching `/mirror`.

There is also a `/landing` route (`app/landing/page.tsx`) — a static marketing page, not part of either app flow.

Design language for both: pink/pastel "bubbly" aesthetic (blob-float, soft-pulse, wiggle-on-hover, pastel radial background in `app/globals.css`). Intentional, not cruft.

## Stack

Next.js 15 App Router, React 19, TypeScript, Tailwind v4 (via `@tailwindcss/postcss`, configured in `app/globals.css` with `@theme`, no `tailwind.config.*`). Path alias `@/*` → repo root. Fonts: Geist Sans/Mono only. `onnxruntime-web` runs the detector in the browser (WebGPU preferred, WASM/SIMD fallback, single-threaded). Python 3.12, FastAPI, OpenCV, dlib, OpenAI SDK server-side.

## Architecture

The files that carry the real weight:

### `components/tracker.tsx` (~2400 lines, one default export)
The Tracker UI and control logic. Phases: `starting` → `ready` → `locked` → `error`. Up to `MAX_FACES` (=3) tracks can be locked at once; tapping past that evicts the LRU track. Each locked track gets its own `TrackRefs` (hot-path mutable state written every RAF — box EMA, velocity, analyser, gain, speak generation) paired with a `TrackUI` (React-rendered caption/animation state).

Per-tap flow:

1. **`pickTappedDetection`** — picks the smallest-area detection that contains the tap (nested objects: prefer the cup, not the dining table). Falls back to nearest-center if nothing contains the tap.
2. **`assessObject`** (server action, GLM-5V-Turbo) — returns `{suitable, cx, cy, bbox, reason}` for the AI-chosen face placement on that crop. Runs in parallel with step 3 so we don't pay its latency serially. Default stance is generous — only rejects people or genuinely empty crops. If `suitable=false`, the rejection toast shows `reason` verbatim.
3. **Lock + anchor** — computes `anchor` from `applyAnchor`/`anchorFromPoint` in `lib/iou.ts`: the face's offset relative to the detected box, in box-normalized units. On every subsequent frame the anchor is replayed against the *current* smoothed box so the face stays pinned to the right spot on the object, even as the box shifts/resizes. The YOLO-seg head's mask centroid (if available) is preferred over bbox-center as the anchoring origin — materially more stable on asymmetric objects (the mug-handle problem).
4. **`generateLine`** (server action, GLM-4v-flash + Fish.audio TTS) — sends the crop to GLM for the in-character line and synthesizes audio. Played through a per-track Web Audio graph: `source → analyser → gain → destination`. The analyser drives `classifyShape` (in `face-voice.tsx`) every RAF to pick the mouth-shape letter; the gain is driven from the track's opacity so the voice fades with the face. Concurrent voices are a feature — each track has its own analyser so three objects can talk over each other.

The continuous tracking loop (`loop`):
- Inference runs at up to `MAX_INFERENCE_FPS` (=30) frames/s; on mobile CPU-WASM it naturally lands at 3–8 fps. Between inferences the face glides at 60 fps using per-track velocity (`EXTRAP_MAX_MS`=220, `VELOCITY_EMA`=0.75, `EXTRAP_MISS_LIMIT`=2). Without this extrapolation the face visibly stutters at the inference rate on mobile — this is *not* optional polish.
- Matching per frame: same class + IoU ≥ `IDENTITY_IOU_MIN` (0.3), ties broken by center distance. On `WIDEN_MATCH_AFTER_MISSES` (3) consecutive misses, fall back to same-class + nearest-center to recover through pans/occlusions. `LOST_AFTER_MISSES` (4) both fades the face out AND snaps the smoothed pose on return (the two consequences MUST use the same number — mismatch makes the face visibly glide across the screen during reacquisition).
- `SUSPECT_SIZE_RATIO` (1.75) rejects position updates whose dimensions jumped too much — stops a noisy frame from snapping the face onto a larger same-class neighbor.
- Box smoothing is split-alpha EMA (`BOX_POS_ALPHA`=0.7, `BOX_SIZE_ALPHA`=0.25) on (cx, cy, w, h). Size is slower because bbox edges breathe more than centers.

**`generationRef` is load-bearing** — it cancels stale async work (assess/speak) when the user taps again mid-flight. **`speakGen` per-track** is the same idea for re-taps on the same track. Don't remove those `if (gen !== …) return;` guards.

Face scale (`FACE_BBOX_FRACTION` × min(box.w, box.h)) is derived from the locked box, clamped to [`FACE_SCALE_MIN`, `FACE_SCALE_MAX`]. `<FaceVoice />` is `FACE_VOICE_WIDTH` CSS px at scale=1; the multiplier is `target_px / FACE_VOICE_WIDTH`. `EXCLUDED_CLASS_IDS` bakes in the person-is-excluded policy alongside `ASSESS_SYSTEM`.

Esc wipes all tracks (useful for live demos). Tap-hit uses a short-lived cache (`TAP_CACHE_MAX_AGE_MS`=400) of the continuous loop's detections; otherwise fires a one-shot lower-threshold inference at the tap.

### `lib/yolo.ts` (~820 lines)
Browser-side object detector on top of onnxruntime-web. Default model is `public/models/yolo26n-seg.onnx` (~9.4 MB, served with one-year immutable cache via `next.config.mjs`). Exports `initYolo`, `detect`, `resetYolo`, and `subscribeYoloStatus` (observable for the load-progress HUD — stage/backend/bytes/error). Pipeline: letterbox → CHW float32 [0,1] → ORT session → post-process. Three output heads are supported:
- **`yolo-detr`** (YOLO26n, RF-DETR Nano) — logits [1,N,C] + pred_boxes [1,N,4] normalized cxcywh, per-query sigmoid→argmax, threshold, no NMS.
- **`yolo-seg-detr`** (YOLO26n-seg, **default**) — rows `(x1,y1,x2,y2,score,classId,32 mask coefs)` + prototype masks [1,32,H,W]. Per detection we reconstruct the binary mask, compute `maskCentroid` (in source pixels) and `maskArea` (in 160×160 proto pixels). Centroid → stable anchor origin; sudden area drop → occlusion signal.
- **`yolov8-head`** — legacy anchor-grid [1, 4+C, N] for raw Ultralytics exports.

Presets live in `MODEL_PRESETS` and are auto-picked from the URL basename. **Don't upscale RF-DETR Nano** — it's NAS-tuned to 384×384 and its compute scales with the 4th power of resolution; move up the ladder (small/medium/base) instead. WebGPU is tried first, WASM single-threaded with SIMD is the fallback; multi-threaded WASM needs COOP/COEP which is too brittle across Safari/localhost. `ort.env.logLevel = "error"` and `logSeverityLevel: 3` are set specifically to stop Next.js dev from promoting ORT warnings into a fullscreen error card that covers the camera.

### `lib/iou.ts`
Lightweight identity tracker: `iou`, `centerDistNorm`, `matchTarget`, `EMAFilter`, `BoxEMA` + `newBoxEMA`/`smoothBox`/`seedBoxEMA`, and `Anchor` with `anchorFromPoint`/`applyAnchor`. The anchor math is the heart of the tracker: store a point as box-normalized offsets at lock time, replay each frame against the newest smoothed box.

### `components/face-voice.tsx`
The "talking face" renderer. A looping `<video>` of real eyes (`public/facevoice/eyes.{webm,mp4}`) with a `<img>` mouth (9 PNG shapes `shape-A` … `shape-X` in `public/facevoice/`) swapped per audio frame. `classifyShape(analyser)` is exported and consumed by `tracker.tsx`'s RAF loop. Swap shapes at 30 fps by preloading all 9 PNGs once on mount. `FACE_VOICE_WIDTH` (280) / `FACE_VOICE_HEIGHT` (160) are the native px at scale=1; Tracker's `FACE_NATIVE_PX` must match.

`components/face.tsx` and `lib/lk.ts` are the pre-YOLO generation of the face + a pure-TS Lucas-Kanade tracker. They are **orphaned** (grep shows no imports). Don't wire them back in without asking — the YOLO pipeline supersedes them.

### `app/actions.ts`
Three server actions, all wired. All vision calls go to **GLM (Zhipu)** via the OpenAI-compatible SDK, with a two-model strategy:

- **Deep model** `GLM_MODEL_DEEP` (default `glm-5v-turbo`) — reasoning-style VLM, used only by `assessObject`. Spends tokens on internal reasoning before emitting JSON, so `max_tokens` is set high (1536) and latency is ~3–5s. Quality > latency here; runs once per tap.
- **Fast model** `GLM_MODEL_FAST` (default `glm-4v-flash`) — no reasoning spiral. Used on the hot paths: `generateLine` (~1–2s) and `converseWithObject` reply (~1–2s). Still sees the crop for character continuity.

Both GLM endpoints are auto-picked from the key shape: `<hex>.<secret>` keys → `open.bigmodel.cn`; `sk-…` keys → `api.z.ai`. Override via `ZHIPU_BASE_URL`. GLM doesn't reliably honor `response_format=json_object`, so `extractJsonObject` strips `<think>` traces and ```json fences before parsing.

- **`assessObject(imageDataUrl, tapX, tapY)`** → `{suitable, cx, cy, bbox, reason}`.
- **`generateLine(imageDataUrl)`** → `{line, audioDataUrl, backend}`. Used by both Tracker and Mirror.
- **`converseWithObject(formData)`** → `{transcript, reply, audioDataUrl, backend}`. Voice-in/voice-out: MediaRecorder blob → Whisper → GLM reply (vision-grounded so persona stays consistent) → TTS. Audio blob is bounded [1 KB, 10 MB].

TTS ladder (`synthesizeSpeech`): **Fish.audio** first when `FISH_API_KEY` is set (character-specific voices via `FISH_REFERENCE_ID`, `FISH_MODEL` default `s1`) → **OpenAI `tts-1`/`nova`** fallback → caption-only. Transcription uses OpenAI Whisper; language is left unset so non-English input works.

`assessObject`/`generateLine` both re-validate `data:image/` at the boundary — keep the check. `next.config.mjs` sets `serverActions.bodySizeLimit: "8mb"` because frames travel as data URL strings, and adds immutable 1-year cache headers for `/models/*` and `/ort/*`.

The `ASSESS_SYSTEM`, `FACE_SYSTEM`, and `RESPOND_SYSTEM` prompts are **load-bearing for product behavior**:
- `ASSESS_SYSTEM`: default-to-suitable, only reject people or fully empty crops. Tone of `reason` strings shows verbatim in the rejection toast.
- `FACE_SYSTEM`: 14-word cap, in-character voice (no meta, no "as a [thing]"), vary rhythm.
- `RESPOND_SYSTEM(className)`: 22-word cap, must acknowledge the user's line, same character rules.

Treat edits to any of these prompts as product changes, not refactors.

### `server/server.py` (Mirror only, ~960 lines)
FastAPI app:
- `GET /health`, `GET /bases`
- `POST /base/{name}` — swap active base (triggers coord detection if uncached)
- `POST /base/{name}/retry` — clear cache for that base and re-run detection. Hot-swaps the compositor in place if `name` is active; otherwise refreshes the cache silently so the user can keep viewing the current base.
- `POST /upload` — multipart image; `cv2.imdecode` verifies bytes **before** writing to disk; 10 MB cap; extension allowlist; path-traversal guarded by `_within_root`.
- `WS /ws` — binary = JPEG in/out, text = control events.

The composite pipeline runs on a `ThreadPoolExecutor` (`run_in_executor`) so dlib + seamlessClone never block the event loop. `_safe_target` clamps paste coordinates to canvas bounds (seamlessClone crashes otherwise). `Compositor._smooth_shape` EMAs 68-point landmarks across frames with a stale-and-jump fallback to kill dlib jitter without smearing scene cuts.

**GLM base-image coord detection** (`ask_glm_for_coords`, `glm-5v-turbo`): two-phase prompt — (A) identify main item → short `label` (1–3 words); (B) find the perfect face spot (`subject_bbox` + eye/mouth points/widths). 60s timeout, retries, fallback to `default_coords()`. `validate_coords` rejects junk (mouth above eyes, eyes too close, features outside `subject_bbox`, IOD > 75% of bbox width) and falls back *without caching*. Label is cosmetic — missing labels fall back to `filename_to_label`. Results cache as `<name>.v{COORDS_VERSION}.<md5_prefix>.coords.json`; bump `COORDS_VERSION` (currently 5) if you change the coord schema/prompt/model — old cache files will be ignored. Label-only prompt tweaks don't need a bump.

The Mirror client (`components/mirror.tsx`) uses a **lockstep** WS flow: a new frame is only sent after the previous composite comes back. `inFlightRef` is load-bearing — don't convert to a timed `setInterval`. Auto-reconnect with exponential backoff (500ms → 8s) runs as long as `wantConnectedRef` is true; `scheduleReconnect` has an intentional `eslint-disable` for the cycle with `openSocket`. Stall watchdog force-closes after 6s without a frame so the reconnect path takes over. Preview is mirrored client-side before send; the server does not mirror. The chip strip polls `/bases` every 5s so it self-heals when the server comes back.

## Things to know

- **The main route is Tracker.** `app/page.tsx` renders `<Tracker />`. Mirror lives at `/mirror`. Don't swap these without asking.
- **Tracker uses rear camera by default** (`facingMode: { ideal: "environment" }`). Desktop falls back to the default webcam.
- **ONNX model + ORT runtime are fetched same-origin, not bundled.** `public/models/yolo26n-seg.onnx` (~9.4 MB) ships in git. The ORT WASM files under `public/ort/` are produced by the `postinstall` script (`scripts/setup-ort.mjs`) copying from `node_modules/onnxruntime-web/dist/`. If you upgrade onnxruntime-web, the postinstall picks up the new files automatically.
- **Env loading** — Tracker's server actions read from the normal Next env chain (`.env.local`, etc.). The Python server's loader is `../.env.local` → `./.env` → `../.env`, first existing file wins, `override=False`. Required for `/`: `ZHIPU_API_KEY` (GLM — assess + speak + converse). Optional but recommended: `OPENAI_API_KEY` (Whisper + TTS fallback), `FISH_API_KEY` + `FISH_REFERENCE_ID` (primary TTS). Without `OPENAI_API_KEY` AND without `FISH_API_KEY`, the Tracker still works but runs caption-only. Without `OPENAI_API_KEY`, `converseWithObject` fails (Whisper is required).
- **Landmarks model is gitignored.** `server/shape_predictor_68_face_landmarks.dat` (~100 MB) is not committed. If a fresh checkout is missing it, download from `http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2` and `bunzip2` into `server/`. Only needed for `/mirror`.
- **`server/.venv/` is gitignored** and not portable. Recreate with `python3 -m venv server/.venv && server/.venv/bin/pip install opencv-python dlib imutils numpy openai fastapi "uvicorn[standard]" python-multipart websockets python-dotenv`. dlib builds from source on macOS and needs `cmake` from Homebrew.
- **`scripts/test-*.mjs`/`scripts/inspect-onnx.py`** are ad-hoc probes (Fish/GLM/YOLO/seg/ONNX schema). Not part of any automated test setup.
- **Bubbly aesthetic is intentional** — keep blob-float, soft-pulse, wiggle-on-hover, sparkle-spin, and the pastel radial background unless the brief changes.
