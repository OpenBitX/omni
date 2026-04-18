# CLAUDE.md — v2 (React Native + ExecuTorch)

This is **v2** of Tracker — a ground-up React Native port of the browser app
in the parent repo, built around [react-native-executorch](https://github.com/software-mansion/react-native-executorch)
so the YOLO26n-seg detector runs on-device at 10–20 fps on mid-range phones
instead of 3–8 fps on mobile WASM in Safari.

**Frontend only.** v2 is just the native UI/UX (Expo + Vision Camera +
ExecuTorch). The **backend is the parent browser app's Next.js server** at
`../` — it already implements `assessObject`, `describeObject`, `generateLine`,
`converseWithObject` as server actions, plus `/api/tts/stream`. Four thin
HTTP wrappers were added under `../app/api/{assess,describe,generate-line,
converse}/route.ts` to expose those actions over plain POST JSON for the
phone. They're additive — the browser app itself still uses the server
actions directly and is unaffected.

## Commands

**Native app:**
- `npm install` — once
- `npm run prebuild` — regenerates ios/ + android/ (required: vision-camera + executorch ship native modules, so Expo Go will NOT work — a custom dev client is mandatory)
- `npm run ios` — build + launch on iOS simulator / device
- `npm run android` — build + launch on Android
- `npm run typecheck` — the only automated gate (no lint, no tests)

**Backend (the parent browser app, reused as a LAN server):**
- `cd ..` and `npm run dev -- -H 0.0.0.0 -p 3000` — bound to all interfaces so the phone can reach it
- The parent's existing `.env.local` supplies all keys (`OPENAI_API_KEY`, `ZHIPU_API_KEY`, `CEREBRAS_API_KEY`, `FISH_API_KEY`)
- The four new HTTP wrappers at `../app/api/{assess,describe,generate-line,converse}/route.ts` are additive — browser `/` keeps working unchanged

## Stack deltas vs v1

| v1 (browser)                         | v2 (RN)                                              |
| ------------------------------------ | ---------------------------------------------------- |
| `onnxruntime-web` + `yolo26n-seg.onnx` | `react-native-executorch` `YOLO26N_SEG` (`.pte`, auto-fetched from SWM HuggingFace) |
| `getUserMedia` + `<video>`           | `react-native-vision-camera` v5 + `useFrameOutput`   |
| Web Audio API (AnalyserNode, GainNode) | `react-native-audio-api` (same API, native backend) |
| `MediaSource`-streamed mp3           | Buffered decode via RNAA (TTFB +400–900 ms; streamed append not yet exposed by the lib) |
| Tailwind v4 + `<div>`                | `StyleSheet` + `<View>`                              |
| Next.js server actions               | Parent Next.js at port 3000, four new `/api/*` HTTP wrappers |
| Web Speech API                       | `expo-speech-recognition`                            |
| CSS `mask-image` clip-to-silhouette  | **not ported yet** — would need `@react-native-masked-view` + a dynamic mask Image. Face currently renders unmasked. |

## Architecture

### `components/tracker.tsx`
Single-file RN component, same state machine as v1 (`starting → ready → locked → error`), same constants (MAX_INFERENCE_FPS=30, IDENTITY_IOU_MIN=0.3, BOX_POS_ALPHA=0.7, BOX_SIZE_ALPHA=0.25, LOST_AFTER_MISSES=4, WIDEN_MATCH_AFTER_MISSES=3, SUSPECT_SIZE_RATIO=1.75, EXTRAP_MAX_MS=220, VELOCITY_EMA=0.75, MAX_FACES=3).

The frame processor runs as a worklet on the Vision Camera thread at up to `MAX_INFERENCE_FPS` — each frame is fed to `seg.runOnFrame(frame, false, {...})` which returns raw segmentation instances (bbox, label, score, binary mask). `scheduleOnRN` posts the results back to JS where `normalizeDetection` (in `lib/detector.ts`) computes mask centroids and converts to the v1 `Detection` shape. The render-side RAF loop then matches detections to tracks, runs the same split-alpha EMA, same velocity extrapolation, same "hold-last-good" size-ratio guard — logic ported line-for-line from `../components/tracker.tsx`.

**Features not yet ported** (deliberate v1 omissions, callers may extend later):
- Silhouette mask clip on the face. v1 uses CSS `mask-image: url(<dataUrl>)`; RN equivalent is `@react-native-masked-view/masked-view` wrapping the face, fed a dynamic mask PNG. The `mask` + `maskCentroid` data is already on each Detection — the renderer just doesn't consume it yet.
- Adaptive mic button waveform strip.
- Bubble tail bob animation during `speaking`.
- Tilt rotation of the bubble counter-rotation is approximated — v1 has more careful counter-rotation math.

### `components/face-voice.tsx`
Same 9-shape mouth atlas (`assets/facevoice/shape-*.png`), same looping eyes video (`eyes.mp4` via `expo-video`). `classifyShapeSmooth` + `extractFeatures` + `createLipSyncState` are **ported verbatim** — pure TS, consumes Uint8Array pairs from react-native-audio-api's `AnalyserNode`, emits mouth shapes at 60 fps.

### `lib/detector.ts`
Types + `normalizeDetection`. Converts raw `useInstanceSegmentation` output into the v1 `Detection` shape so every downstream consumer (`lib/iou.ts` anchor math, tracker render loop) runs unchanged. Mask centroid is computed once per detection, in source-pixel space — feeds `applyAnchor` at lock time and the render-frame match loop.

### `lib/iou.ts`
**Copied verbatim** from v1. Pure TS, no DOM — works unchanged in RN.

### `lib/audio-track.ts`
Per-track audio graph via `react-native-audio-api`: `source → analyser → gain → destination`. `speakLine` fetches the TTS stream, decodes to an `AudioBuffer`, starts playback. Cancel returns a handle so a retap can kill the current line before the new one starts.

### `lib/api.ts`
HTTP client against `EXPO_PUBLIC_API_BASE`. One function per endpoint — `assessObject`, `describeObject`, `generateLine`, `converseWithObject`. Base URL defaults to `http://localhost:3001` (simulator-only); set in `.env` for physical devices.

### Backend — the parent browser app
v2 does **not** run its own backend. The parent's Next.js dev server at
`../` is the server. Four thin wrappers at `../app/api/{assess,describe,
generate-line,converse}/route.ts` just call the existing server actions over
HTTP so the phone can reach them. The TTS stream endpoint at
`../app/api/tts/stream/route.ts` was already there and is reused.

**Endpoints** (all POST, all on port 3000):
- `/api/assess` → `{suitable, cx, cy, bbox, reason}` (GLM `glm-5v-turbo`)
- `/api/describe` → `{description}` (OpenAI `gpt-4o-mini` vision)
- `/api/generate-line` → `{line, voiceId, description}` — bundled first-tap vision call, Cerebras text-only on retap. Accepts null `imageDataUrl` (wrapper substitutes a 1×1 transparent PNG) because retap never touches the image.
- `/api/converse` → `{transcript, reply, voiceId}` — FormData passthrough
- `/api/tts/stream` → streams `audio/mpeg` bytes from Fish or OpenAI fallback

Uses the keys already in `../.env.local`.

## Things to know

- **New React Native architecture is REQUIRED.** Set via `"newArchEnabled": true` in `app.json`. react-native-executorch doesn't work on the old arch.
- **iOS 17 min, Android SDK 33 min.** Hardcoded in `app.json`. Older OSes crash at runtime.
- **Expo Go will not work.** vision-camera, executorch, and audio-api are all native modules. You need `expo run:ios` or `expo run:android` (i.e. a custom dev client).
- **Physical device requires LAN base URL.** `localhost:3001` only resolves from the simulator. On a real phone, set `EXPO_PUBLIC_API_BASE=http://<your-machine-IP>:3001` and make sure `backend/` is bound to `0.0.0.0` (Next.js dev server is by default).
- **Model downloads on first launch.** `YOLO26N_SEG` is fetched from SWM's HuggingFace mirror (~9 MB) on first run — `initExecutorch` in `app/_layout.tsx` registers the resource fetcher; downloads cache to app storage after.
- **Frame processor must return quickly.** `dropFramesWhileBusy: true` is set so the camera drops frames rather than queuing when inference is slow — this is what gives the smooth preview on mid-range Androids where inference dips to 50–80 ms.
- **`generationRef` guards cancel stale tap work** when a new tap lands mid-flight — same load-bearing pattern as v1. Don't remove the `if (gen !== generationRef.current) return` guards.
- **`speakGen` per-track** cancels a prior line on the same track when a retap replaces it — same contract as v1.
- **Mask silhouette clip is deliberately skipped** in the current render. The `Detection.mask` + `maskCentroid` + `maskArea` are still populated by `normalizeDetection`, so reintroducing the silhouette via `@react-native-masked-view/masked-view` + a dynamic PNG is localized to `TrackOverlay` — no data-plane changes needed.
