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

# === Tracking stickiness tunables ====================================
#
# These shape how aggressively we trust and re-prompt from the previous
# frame. Set with conservative defaults — if tracks are still drifting,
# the knobs to turn first are BBOX_PAD_FRAC (more context for SAM2) and
# DRIFT_AREA_RATIO (stricter = stickier, at the cost of slower handling
# of genuinely fast size changes like zooming in).

# Amount to pad the previous frame's mask bbox before feeding it as the
# box prompt. SAM2 refines the boundary *within* the prompted box, so a
# tight bbox can amputate parts of a moving object. 0.15 = 15% padding
# on each side — enough slack for motion without dragging in background.
BBOX_PAD_FRAC = float(os.environ.get("SAM2_BBOX_PAD", "0.15"))
# Max acceptable frame-to-frame area change. New mask area > N× previous
# (or < 1/N×) is assumed to be bleed into the background or dropout onto
# a tiny neighbor. Rejected updates don't advance `prev_low_res`, so the
# next frame re-prompts from the last good state instead of compounding
# the drift. `3.0` is a generous cap — a hand entering the frame grows
# 2–3× over a couple frames; beyond that it's almost always bleed.
DRIFT_AREA_RATIO = float(os.environ.get("SAM2_DRIFT_RATIO", "3.0"))
# Max consecutive bad frames (low score OR drift-rejected) before we
# give up and emit "lost". Higher = stickier through brief occlusion.
LOST_AFTER_BAD_FRAMES = int(os.environ.get("SAM2_LOST_AFTER", "5"))


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


def _bbox_xyxy_from_mask(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) in *pixel* coords on the mask's own grid, or
    None if empty. This is the form SAM2's `box=` prompt expects. Kept
    alongside `_bbox_from_mask` which returns normalized center-width for
    client serialization."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _pad_box_xyxy(
    box: tuple[int, int, int, int], W: int, H: int, pad_frac: float
) -> tuple[float, float, float, float]:
    """Grow a pixel bbox by `pad_frac` of its longer edge on each side,
    clamped to frame bounds. Purpose: give SAM2 slack to refine the mask
    for objects that moved/rotated between frames — a tight bbox prompt
    clips the mask at the previous silhouette and the object visibly
    "chips off" along motion direction. Padding by ~15% is a sweet spot
    for 640px frames."""
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pad = int(round(max(w, h) * pad_frac))
    return (
        float(max(0, x1 - pad)),
        float(max(0, y1 - pad)),
        float(min(W - 1, x2 + pad)),
        float(min(H - 1, y2 + pad)),
    )


def _centroid_from_mask(mask: np.ndarray) -> Optional[tuple[float, float]]:
    """Mask centroid in pixel coords on the mask's own grid. Used to re-prompt
    the next frame — the centroid is more stable than the bbox center for
    elongated/asymmetric objects (same reason YOLO-seg prefers it)."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _mask_area_px(mask: np.ndarray) -> int:
    """Foreground pixel count. Used by the drift-rejection heuristic:
    frame-to-frame area ratios > DRIFT_AREA_RATIO mean the mask has
    bled into the background or collapsed to a tiny neighbor region."""
    return int(mask.sum())


# =====================================================================
# Per-session state
# =====================================================================


class Session:
    """One WS connection → one tracking session.

    State machine:
        IDLE          -- no track. Waiting for {"type":"init", x, y}.
        PENDING_INIT  -- got init text, waiting for the JPEG binary that goes with it.
        TRACKING      -- have a mask. Each incoming JPEG extends the track.

    Tracking state accumulates per frame:
        prev_low_res        -- last good (1, 256, 256) logits for mask_input chaining
        prev_centroid_px    -- last good mask centroid (stable re-prompt point)
        prev_bbox_xyxy      -- last good mask bbox in pixels (box prompt for next frame)
        prev_area_px        -- last good mask foreground pixel count (drift detection)
        initial_tap_norm    -- the user's original tap, stashed for life of the track
        bad_frame_streak    -- consecutive low-score / drift-rejected frames; triggers
                               "lost" after LOST_AFTER_BAD_FRAMES so one blurry frame
                               or brief occlusion doesn't kill the track.
        frames_tracked      -- monotonically increases. First few frames skip drift
                               checks because the mask naturally grows as SAM2 locks
                               on (point-only prompt → triple prompt produces a fuller
                               mask, which would otherwise trip the area ratio).
    """

    def __init__(self) -> None:
        self.state: str = "IDLE"
        self.track_id: Optional[str] = None
        self.pending_point: Optional[tuple[float, float]] = None  # normalized
        self.initial_tap_norm: Optional[tuple[float, float]] = None
        self.prev_low_res: Optional[np.ndarray] = None  # (1, 256, 256) logits
        self.prev_centroid_px: Optional[tuple[float, float]] = None
        self.prev_bbox_xyxy: Optional[tuple[int, int, int, int]] = None
        self.prev_area_px: int = 0
        self.bad_frame_streak: int = 0
        self.frames_tracked: int = 0

    def reset(self) -> None:
        self.state = "IDLE"
        self.track_id = None
        self.pending_point = None
        self.initial_tap_norm = None
        self.prev_low_res = None
        self.prev_centroid_px = None
        self.prev_bbox_xyxy = None
        self.prev_area_px = 0
        self.bad_frame_streak = 0
        self.frames_tracked = 0


# =====================================================================
# SAM2 call (under global lock)
# =====================================================================


async def _predict(
    frame_rgb: np.ndarray,
    points_xy_px: list[tuple[float, float]],
    point_labels: list[int],
    box_xyxy: Optional[tuple[float, float, float, float]],
    prev_low_res: Optional[np.ndarray],
) -> tuple[np.ndarray, float, np.ndarray]:
    """Run SAM2 on a single frame with any combination of point / box / mask
    prompts. Returns (mask_bool, score, low_res_logits).

    The "sticky tracking" recipe is the triple-prompt: a point at the last
    centroid (says *this is the object*), a box from the last mask padded
    15% (says *roughly where to look*), and the previous frame's low-res
    logits via mask_input (says *here's what the silhouette looked like*).
    Any one of these alone leaks; together they pin the segmentation.

    Serialized on `_model_lock` because SAM2's image predictor stores the
    encoded image on `self`, so concurrent `set_image` calls race. Heavy
    torch work goes through `asyncio.to_thread` so the event loop stays
    responsive for other WS messages.
    """
    import torch

    predictor = _load_predictor()

    def _run():
        predictor.set_image(frame_rgb)
        kwargs: dict[str, Any] = dict(multimask_output=False)
        if points_xy_px:
            kwargs["point_coords"] = np.array(points_xy_px, dtype=np.float32)
            kwargs["point_labels"] = np.array(point_labels, dtype=np.int32)
        if box_xyxy is not None:
            kwargs["box"] = np.array(box_xyxy, dtype=np.float32)
        if prev_low_res is not None:
            kwargs["mask_input"] = prev_low_res

        with torch.inference_mode():
            if _device == "cuda":
                with torch.autocast(device_type="cuda", dtype=_dtype):
                    masks, scores, low_res = predictor.predict(**kwargs)
            else:
                masks, scores, low_res = predictor.predict(**kwargs)

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
        "bbox_pad_frac": BBOX_PAD_FRAC,
        "drift_area_ratio": DRIFT_AREA_RATIO,
        "lost_after_bad_frames": LOST_AFTER_BAD_FRAMES,
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
                    sess.initial_tap_norm = (x, y)
                    sess.state = "PENDING_INIT"
                    sess.prev_low_res = None
                    sess.prev_centroid_px = None
                    sess.prev_bbox_xyxy = None
                    sess.prev_area_px = 0
                    sess.bad_frame_streak = 0
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
                    # Init uses only the point prompt — we have no previous
                    # mask or bbox yet, and the tap is the strongest signal
                    # about what the user wants.
                    px = sess.pending_point[0] * W
                    py = sess.pending_point[1] * H
                    try:
                        mask, score, low_res = await _predict(
                            frame,
                            points_xy_px=[(px, py)],
                            point_labels=[1],
                            box_xyxy=None,
                            prev_low_res=None,
                        )
                    except Exception as e:
                        log.exception("[ws %s] predict init failed", client)
                        await _send_json(ws, {"type": "error", "message": f"predict: {e}"})
                        continue

                    bbox_cxcywh = _bbox_from_mask(mask)
                    bbox_xyxy = _bbox_xyxy_from_mask(mask)
                    cen = _centroid_from_mask(mask)
                    area = _mask_area_px(mask)
                    if bbox_cxcywh is None or bbox_xyxy is None or score < SCORE_LOST_THRESHOLD:
                        sess.reset()
                        await _send_json(ws, {"type": "lost", "reason": "low_score_init"})
                        continue

                    sess.state = "TRACKING"
                    sess.prev_low_res = low_res
                    sess.prev_centroid_px = cen
                    sess.prev_bbox_xyxy = bbox_xyxy
                    sess.prev_area_px = area
                    sess.bad_frame_streak = 0
                    await _send_json(ws, {
                        "type": "initialized",
                        "trackId": sess.track_id,
                        "bbox": list(bbox_cxcywh),
                        "score": round(score, 4),
                    })
                    continue

                if sess.state == "TRACKING":
                    # Re-prompt using previous mask centroid + bbox + mask.
                    # The triple prompt is what makes SAM2 "stick": the point
                    # says *this pixel is foreground*, the box limits search
                    # to a plausible region, and the mask_input gives a shape
                    # prior. Missing any one and SAM2 tends to drift into
                    # adjacent same-texture regions (shadows, cloth, skin).
                    if sess.prev_centroid_px is None or sess.prev_bbox_xyxy is None:
                        sess.reset()
                        await _send_json(ws, {"type": "lost", "reason": "no_prev_state"})
                        continue

                    cx_px, cy_px = sess.prev_centroid_px
                    cx_px = min(max(cx_px, 0.0), W - 1.0)
                    cy_px = min(max(cy_px, 0.0), H - 1.0)

                    # Pad the bbox so SAM2 can refine into areas the object
                    # moved into between frames. Without padding, fast
                    # translation makes the mask "chip off" its leading edge.
                    box_prompt = _pad_box_xyxy(
                        sess.prev_bbox_xyxy, W, H, BBOX_PAD_FRAC
                    )

                    # Build the point prompt set. Base case: just the
                    # centroid. Recovery case (after a bad frame): ALSO
                    # include the user's original tap, but ONLY if that
                    # tap still lies inside the padded previous bbox —
                    # otherwise the object has moved away from where it
                    # was picked and the original tap now points at
                    # background, which would actively mislead SAM2.
                    #
                    # This gives us a "re-lock" pass whenever tracking
                    # wobbles: two positive points constrain the mask to
                    # the intersection, snapping the segmentation back
                    # to the intended object.
                    pts: list[tuple[float, float]] = [(cx_px, cy_px)]
                    labels: list[int] = [1]
                    if (
                        sess.bad_frame_streak > 0
                        and sess.initial_tap_norm is not None
                    ):
                        tap_px_x = sess.initial_tap_norm[0] * W
                        tap_px_y = sess.initial_tap_norm[1] * H
                        bx1, by1, bx2, by2 = box_prompt
                        if bx1 <= tap_px_x <= bx2 and by1 <= tap_px_y <= by2:
                            pts.append((tap_px_x, tap_px_y))
                            labels.append(1)

                    try:
                        mask, score, low_res = await _predict(
                            frame,
                            points_xy_px=pts,
                            point_labels=labels,
                            box_xyxy=box_prompt,
                            prev_low_res=sess.prev_low_res,
                        )
                    except Exception as e:
                        log.exception("[ws %s] predict track failed", client)
                        await _send_json(ws, {"type": "error", "message": f"predict: {e}"})
                        continue

                    bbox_cxcywh = _bbox_from_mask(mask)
                    bbox_xyxy = _bbox_xyxy_from_mask(mask)
                    cen = _centroid_from_mask(mask)
                    area = _mask_area_px(mask)

                    # === Drift detection ================================
                    # Compare against previous frame. Two failure modes:
                    #  a) score dipped (SAM2 itself is uncertain)
                    #  b) area exploded or collapsed (mask bled into the
                    #     background or snapped onto a tiny neighbor)
                    # Both count as "bad frames" — we don't advance state
                    # on bad frames so the next frame re-prompts from the
                    # last KNOWN-GOOD centroid/bbox/mask, not the drifted
                    # one. This is what stops runaway bleed.
                    low_score = score < SCORE_LOST_THRESHOLD
                    area_ratio = (
                        float(area) / max(1, sess.prev_area_px)
                        if sess.prev_area_px > 0
                        else 1.0
                    )
                    # Grace the first couple of frames: SAM2 with just the
                    # init point tends to produce a tight mask, and the
                    # triple-prompt on frame 2+ produces a fuller mask —
                    # that legitimate ~2-3× growth would falsely trip drift.
                    # After frames_tracked > 2 we expect the mask area to
                    # be stable frame-to-frame, so drift rejection kicks in.
                    grace_phase = sess.frames_tracked < 2
                    drifted = (not grace_phase) and (
                        area_ratio > DRIFT_AREA_RATIO
                        or area_ratio < (1.0 / DRIFT_AREA_RATIO)
                    )
                    bad = (
                        bbox_cxcywh is None
                        or bbox_xyxy is None
                        or low_score
                        or drifted
                    )

                    if bad:
                        sess.bad_frame_streak += 1
                        if sess.bad_frame_streak >= LOST_AFTER_BAD_FRAMES:
                            reason = (
                                "drift" if drifted else "low_score"
                                if low_score else "empty_mask"
                            )
                            sess.reset()
                            await _send_json(ws, {"type": "lost", "reason": reason})
                        else:
                            # Hold the face where it was — tell the client
                            # this frame is stale so it fades slightly but
                            # doesn't move. Importantly we DO NOT write to
                            # prev_low_res / prev_bbox — the next frame
                            # re-prompts from the last good state.
                            await _send_json(ws, {
                                "type": "track",
                                "bbox": None,
                                "score": round(score, 4),
                                "stale": True,
                            })
                        continue

                    # Good frame — commit to state.
                    sess.prev_low_res = low_res
                    sess.prev_centroid_px = cen
                    sess.prev_bbox_xyxy = bbox_xyxy
                    sess.prev_area_px = area
                    sess.bad_frame_streak = 0
                    sess.frames_tracked += 1
                    await _send_json(ws, {
                        "type": "track",
                        "bbox": list(bbox_cxcywh),
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
