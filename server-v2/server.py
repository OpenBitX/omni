"""Server V2 — SAM2-tiny streaming segmentation backend.

Isolated from server/ (the Mirror dlib pipeline). Listens on port 8001.

Protocol (WS /sam2/ws):
  Client → Server:
    text  {"type":"init","x":0..1,"y":0..1}   -- tap point, normalized frame coords
    bin   JPEG bytes                           -- current frame
    text  {"type":"reset"}                     -- drop current track

  Server → Client (text JSON):
    {"type":"initialized","trackId":"...","bbox":[cx,cy,w,h],"score":float}
    {"type":"track","bbox":[...],"score":float}
    {"type":"lost","reason":"low_score"|"exception"}
    {"type":"error","message":"..."}

bbox is in normalized [0,1] coords. Lockstep: one server reply per client JPEG.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sam2-v2")

ROOT = Path(__file__).parent.resolve()

# Match server/server.py env loading: look up from server-v2/ to repo root.
for candidate in (ROOT.parent / ".env.local", ROOT / ".env", ROOT.parent / ".env"):
    if candidate.exists():
        load_dotenv(candidate, override=False)
        log.info("loaded env from %s", candidate)
        break

PORT = int(os.environ.get("SAM2_PORT", "8001"))
SCORE_LOST_THRESHOLD = float(os.environ.get("SAM2_LOST_SCORE", "0.35"))
MAX_FRAME_BYTES = 4 * 1024 * 1024  # match server/ limit
MAX_LONG_EDGE = int(os.environ.get("SAM2_MAX_LONG_EDGE", "640"))
CHECKPOINT_ENV = os.environ.get("SAM2_CHECKPOINT", "")
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
DEFAULT_CONFIG_NAME = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")


# =====================================================================
# Model loading (lazy, device autodetect)
# =====================================================================

_model_lock = asyncio.Lock()
_predictor: Optional[Any] = None
_device: str = "cpu"
_dtype: Any = None


def _pick_device() -> str:
    import torch  # imported lazily so --help works without torch

    if torch.cuda.is_available():
        return "cuda"
    # MPS is great on M-series but has fp16 quirks — we clamp to fp32 below.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_predictor() -> Any:
    """Build the SAM2 image predictor once. Returns the predictor instance."""
    global _predictor, _device, _dtype

    if _predictor is not None:
        return _predictor

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _device = _pick_device()
    # fp16 on CUDA for speed; fp32 everywhere else (MPS fp16 has known SAM2 issues).
    _dtype = torch.float16 if _device == "cuda" else torch.float32

    checkpoint = Path(CHECKPOINT_ENV) if CHECKPOINT_ENV else DEFAULT_CHECKPOINT
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint missing at {checkpoint}. Run scripts/setup-sam2.sh."
        )

    log.info(
        "loading SAM2 checkpoint=%s config=%s device=%s dtype=%s",
        checkpoint, DEFAULT_CONFIG_NAME, _device, _dtype,
    )
    t0 = time.time()
    model = build_sam2(DEFAULT_CONFIG_NAME, str(checkpoint), device=_device)
    _predictor = SAM2ImagePredictor(model)
    log.info("SAM2 ready in %.2fs", time.time() - t0)
    return _predictor


# =====================================================================
# Frame + mask helpers
# =====================================================================


def _decode_jpeg(buf: bytes) -> np.ndarray:
    """Decode JPEG → RGB uint8 ndarray. Downscale so long edge ≤ MAX_LONG_EDGE
    to keep SAM2 latency bounded regardless of source resolution."""
    im = Image.open(io.BytesIO(buf)).convert("RGB")
    w, h = im.size
    long_edge = max(w, h)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        im = im.resize(new_size, Image.BILINEAR)
    return np.asarray(im)


def _bbox_from_mask(mask: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    """Return (cx, cy, w, h) in normalized [0,1] coords, or None if empty."""
    if mask.size == 0:
        return None
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    H, W = mask.shape
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    cx = (x1 + x2) * 0.5 / W
    cy = (y1 + y2) * 0.5 / H
    w = (x2 - x1 + 1) / W
    h = (y2 - y1 + 1) / H
    return float(cx), float(cy), float(w), float(h)


def _centroid_from_mask(mask: np.ndarray) -> Optional[tuple[float, float]]:
    """Mask centroid in pixel coords on the mask's own grid. Used to re-prompt
    the next frame — the centroid is more stable than the bbox center for
    elongated/asymmetric objects (same reason YOLO-seg prefers it)."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


# =====================================================================
# Per-session state
# =====================================================================


class Session:
    """One WS connection → one tracking session.

    State machine:
        IDLE          -- no track. Waiting for {"type":"init", x, y}.
        PENDING_INIT  -- got init text, waiting for the JPEG binary that goes with it.
        TRACKING      -- have a mask. Each incoming JPEG extends the track.
    """

    def __init__(self) -> None:
        self.state: str = "IDLE"
        self.track_id: Optional[str] = None
        self.pending_point: Optional[tuple[float, float]] = None  # normalized
        self.prev_low_res: Optional[np.ndarray] = None  # (1, 256, 256) logits
        self.prev_centroid_px: Optional[tuple[float, float]] = None  # in resized-frame px
        self.frames_since_ok: int = 0

    def reset(self) -> None:
        self.state = "IDLE"
        self.track_id = None
        self.pending_point = None
        self.prev_low_res = None
        self.prev_centroid_px = None
        self.frames_since_ok = 0


# =====================================================================
# SAM2 call (under global lock)
# =====================================================================


async def _predict(
    frame_rgb: np.ndarray,
    point_xy_px: tuple[float, float],
    prev_low_res: Optional[np.ndarray],
) -> tuple[np.ndarray, float, np.ndarray]:
    """Run SAM2 on a single frame with a point prompt (+ optional prev mask).

    Returns (mask_bool, score, low_res_logits). Serialized on `_model_lock`
    so concurrent sessions don't thrash the predictor's internal `set_image`
    state. Actual torch work runs in a thread so the asyncio loop stays free.
    """
    import torch

    predictor = _load_predictor()

    def _run():
        predictor.set_image(frame_rgb)
        # SAM2 expects shape (N,2) float32 for point_coords, (N,) int for labels.
        pts = np.array([[point_xy_px[0], point_xy_px[1]]], dtype=np.float32)
        lbl = np.array([1], dtype=np.int32)  # 1 = foreground
        kwargs: dict[str, Any] = dict(
            point_coords=pts,
            point_labels=lbl,
            multimask_output=False,
        )
        if prev_low_res is not None:
            kwargs["mask_input"] = prev_low_res

        with torch.inference_mode():
            # autocast only on CUDA; MPS autocast is flaky, CPU doesn't need it.
            if _device == "cuda":
                with torch.autocast(device_type="cuda", dtype=_dtype):
                    masks, scores, low_res = predictor.predict(**kwargs)
            else:
                masks, scores, low_res = predictor.predict(**kwargs)

        # masks: (1, H, W) bool-ish; scores: (1,) float; low_res: (1, 256, 256)
        return masks[0].astype(bool), float(scores[0]), low_res

    async with _model_lock:
        return await asyncio.to_thread(_run)


# =====================================================================
# FastAPI app
# =====================================================================

app = FastAPI(title="sam2-v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "device": _device if _predictor is not None else "unloaded",
        "model_loaded": _predictor is not None,
        "max_long_edge": MAX_LONG_EDGE,
        "score_lost_threshold": SCORE_LOST_THRESHOLD,
    }


@app.on_event("startup")
async def _startup() -> None:
    # Load eagerly so the first tap doesn't pay ~3–6s of cold model load on
    # top of SAM2 inference. If the checkpoint is missing, surface it loudly
    # at boot rather than on first WS frame.
    try:
        await asyncio.to_thread(_load_predictor)
    except Exception as e:  # pragma: no cover — surfaced at dev time
        log.error("SAM2 failed to load at startup: %s", e)


async def _send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_text(json.dumps(payload, separators=(",", ":")))


@app.websocket("/sam2/ws")
async def sam2_ws(ws: WebSocket) -> None:
    await ws.accept()
    sess = Session()
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
    log.info("[ws %s] connected", client)

    try:
        while True:
            msg = await ws.receive()
            # FastAPI exposes either {"text": str} or {"bytes": bytes}.
            if "text" in msg and msg["text"] is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await _send_json(ws, {"type": "error", "message": "invalid json"})
                    continue
                kind = data.get("type")
                if kind == "init":
                    try:
                        x = float(data["x"])
                        y = float(data["y"])
                    except (KeyError, TypeError, ValueError):
                        await _send_json(ws, {"type": "error", "message": "init needs numeric x,y"})
                        continue
                    x = min(1.0, max(0.0, x))
                    y = min(1.0, max(0.0, y))
                    sess.pending_point = (x, y)
                    sess.state = "PENDING_INIT"
                    sess.prev_low_res = None
                    sess.prev_centroid_px = None
                    sess.frames_since_ok = 0
                    sess.track_id = secrets.token_hex(4)
                    log.info("[ws %s] init point=(%.3f,%.3f) track=%s", client, x, y, sess.track_id)
                elif kind == "reset":
                    sess.reset()
                    log.info("[ws %s] reset", client)
                else:
                    await _send_json(ws, {"type": "error", "message": f"unknown type {kind!r}"})
                continue

            if "bytes" in msg and msg["bytes"] is not None:
                frame_bytes: bytes = msg["bytes"]
                if len(frame_bytes) > MAX_FRAME_BYTES:
                    await _send_json(
                        ws,
                        {"type": "error", "message": f"frame too large ({len(frame_bytes)} B)"},
                    )
                    continue

                try:
                    frame = _decode_jpeg(frame_bytes)
                except Exception as e:
                    await _send_json(ws, {"type": "error", "message": f"decode: {e}"})
                    continue

                H, W = frame.shape[:2]

                if sess.state == "PENDING_INIT" and sess.pending_point is not None:
                    px = sess.pending_point[0] * W
                    py = sess.pending_point[1] * H
                    try:
                        mask, score, low_res = await _predict(frame, (px, py), None)
                    except Exception as e:
                        log.exception("[ws %s] predict init failed", client)
                        await _send_json(ws, {"type": "error", "message": f"predict: {e}"})
                        continue

                    bbox = _bbox_from_mask(mask)
                    cen = _centroid_from_mask(mask)
                    if bbox is None or score < SCORE_LOST_THRESHOLD:
                        sess.reset()
                        await _send_json(ws, {"type": "lost", "reason": "low_score_init"})
                        continue

                    sess.state = "TRACKING"
                    sess.prev_low_res = low_res
                    sess.prev_centroid_px = cen
                    sess.frames_since_ok = 0
                    await _send_json(ws, {
                        "type": "initialized",
                        "trackId": sess.track_id,
                        "bbox": list(bbox),
                        "score": round(score, 4),
                    })
                    continue

                if sess.state == "TRACKING":
                    # Re-prompt using previous mask centroid (in pixels, scaled
                    # to this frame's dims since both resize to MAX_LONG_EDGE).
                    if sess.prev_centroid_px is None:
                        sess.reset()
                        await _send_json(ws, {"type": "lost", "reason": "no_centroid"})
                        continue
                    cx_px, cy_px = sess.prev_centroid_px
                    cx_px = min(max(cx_px, 0.0), W - 1.0)
                    cy_px = min(max(cy_px, 0.0), H - 1.0)
                    try:
                        mask, score, low_res = await _predict(
                            frame, (cx_px, cy_px), sess.prev_low_res
                        )
                    except Exception as e:
                        log.exception("[ws %s] predict track failed", client)
                        await _send_json(ws, {"type": "error", "message": f"predict: {e}"})
                        continue

                    bbox = _bbox_from_mask(mask)
                    cen = _centroid_from_mask(mask)
                    if bbox is None or score < SCORE_LOST_THRESHOLD:
                        sess.frames_since_ok += 1
                        # One bad frame isn't fatal — give the object two strikes
                        # before declaring a lost track (matches YOLO's
                        # LOST_AFTER_MISSES philosophy, just smaller because each
                        # SAM2 frame is already doing temporal work).
                        if sess.frames_since_ok >= 3:
                            sess.reset()
                            await _send_json(ws, {"type": "lost", "reason": "low_score"})
                        else:
                            await _send_json(ws, {
                                "type": "track",
                                "bbox": None,
                                "score": round(score, 4),
                                "stale": True,
                            })
                        continue

                    sess.prev_low_res = low_res
                    sess.prev_centroid_px = cen
                    sess.frames_since_ok = 0
                    await _send_json(ws, {
                        "type": "track",
                        "bbox": list(bbox),
                        "score": round(score, 4),
                    })
                    continue

                # IDLE or unexpected: nothing to do, tell client to init.
                await _send_json(ws, {"type": "error", "message": "no active track — send init first"})
                continue

            # Neither text nor bytes — connection closing.
            break

    except WebSocketDisconnect:
        log.info("[ws %s] disconnected", client)
    except Exception:
        log.exception("[ws %s] handler crashed", client)
        try:
            await _send_json(ws, {"type": "error", "message": "server exception"})
        except Exception:
            pass
    finally:
        sess.reset()


def main() -> None:
    log.info("starting sam2-v2 on :%d", PORT)
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
