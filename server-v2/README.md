# server-v2 — SAM2-tiny streaming backend

Isolated FastAPI server that hosts SAM2-tiny segmentation + tracking for the `/v2` route. **Does not share a process or venv with `server/`** (the Mirror backend) — you can run either, both, or neither without interference.

- Port: `8001` (Mirror is on `8000`)
- Venv: `server-v2/.venv/` (gitignored)
- Model checkpoint: `server-v2/checkpoints/sam2.1_hiera_tiny.pt` (gitignored, ~150 MB)

## Setup

```bash
./scripts/setup-sam2.sh
```

That script:
1. Creates `server-v2/.venv/` with Python 3.10+.
2. Installs `torch` — default CPU wheel, with a note on MPS (Apple Silicon picks it up automatically) / CUDA (use `--cuda` flag).
3. Installs `fastapi`, `uvicorn`, `opencv-python`, etc. from `requirements.txt`.
4. Installs SAM2 from `git+https://github.com/facebookresearch/sam2.git`.
5. Downloads the `sam2.1_hiera_tiny.pt` checkpoint to `server-v2/checkpoints/` (skips if present).

Re-run it any time; it's idempotent.

## Run

```bash
pnpm run server:v2
```

Equivalent to `server-v2/.venv/bin/python server-v2/server.py`. Listens on `0.0.0.0:8001`. Loads `.env.local` from the repo root (same pattern as `server/`).

## Protocol

**`GET /health`** → `{"ok": true, "device": "mps|cuda|cpu", "model_loaded": bool}`

**`WS /sam2/ws`** — per-connection state machine.

Client → Server:
- Text `{"type": "init", "x": 0.47, "y": 0.62}` — tap point in normalized `[0,1]` frame coords. Sets session to `PENDING_INIT`.
- Binary (JPEG bytes) — processed as the frame for whichever state the session is in.
- Text `{"type": "reset"}` — drop current track, return to `IDLE`.

Server → Client (all text JSON):
- `{"type": "initialized", "trackId": "...", "bbox": [cx, cy, w, h], "score": 0.93}` — bbox in normalized coords.
- `{"type": "track", "bbox": [...], "score": 0.82}` — per-frame update.
- `{"type": "lost", "reason": "low_score"}` — tracker gave up; client should treat this like the YOLO `LOST_AFTER_MISSES` state (fade face, wait for re-tap).
- `{"type": "error", "message": "..."}` — malformed input or backend exception.

Lockstep: server replies exactly once per client binary frame, so the client can use an `inFlightRef` flag just like Mirror does.

## Implementation notes

- One global SAM2 image predictor, serialized via `asyncio.Lock`. For a laptop demo with one user this is the right cost/complexity tradeoff; add per-session predictors if we ever want concurrent tracks.
- Tracking uses `SAM2ImagePredictor` per frame with `mask_input` chained from the previous frame's low-res logits. This is simpler than `SAM2VideoPredictor` streaming and works well for short horizons. Swap in the video predictor later if occlusion robustness needs to improve — the WS protocol is predictor-agnostic.
- `score < SCORE_LOST_THRESHOLD` (default `0.35`) emits `lost` and clears session state.
