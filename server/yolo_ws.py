"""YOLO segmentation over WebSocket.

Mirrors the Detection shape of `lib/yolo.ts` so the browser client is a
drop-in replacement. The server does the heavy lifting (seg model, mask
decode, centroid + pole + PCA) on MPS — the browser just renders.

Protocol:
  - Client connects to /ws/yolo and sends binary JPEG frames.
  - Client sends a small JSON text preamble per frame with the options
    the current detect() call uses (conf, max, exclude class ids). The
    preamble is optional — defaults apply when it's missing.
  - Server responds with JSON (text) containing a detection list, keyed
    by a monotonically-increasing `seq` so the client can correlate
    responses with in-flight requests. Mask bytes are base64'd.

Model: yolo11s-seg (~22 MB), a solid step up from the browser's yolo26n.
Device: MPS on Apple Silicon, CUDA if present, CPU fallback.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

log = logging.getLogger("yolo_ws")

# Single background thread — seg inference is GIL-bound-ish through torch
# but single-stream is what we want for a single browser tab. Parallelism
# inside the model (MPS) is already there.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")

# Lazily-constructed model + device. First /ws/yolo connection triggers load.
_model: Any = None
_device: str | None = None
# Model ladder on MPS (M2/M3-class Apple Silicon). All stay real-time; each
# rung up is meaningfully cleaner masks + less box jitter but costs latency:
#   yolo11s-seg.pt  (~22 MB, ~20 ms/frame)  — default, fast. Some jitter on borderline angles.
#   yolo11m-seg.pt  (~50 MB, ~35 ms/frame)  — noticeably cleaner, needs one-time download
#   yolo11l-seg.pt  (~90 MB, ~60-80 ms)    — very clean, diminishing returns
#   yolo11x-seg.pt  (~140 MB, ~100-150 ms) — crispest, felt in click-to-sound latency
# Override with YOLO_MODEL. We resolve relative filenames against the repo
# root and prefer the repo-local file when it exists, so ultralytics doesn't
# re-download into whatever cwd the server started from (which has
# previously left corrupt partials on slow links).
def _resolve_model_path() -> str:
    name = os.environ.get("YOLO_MODEL", "yolo11s-seg.pt")
    if os.path.isabs(name):
        return name
    local = _REPO_ROOT / name
    if local.exists():
        return str(local)
    return name


_model_name: str = _resolve_model_path()
# Inference resolution. 704 gives a sharper mask (→ more stable pole/
# centroid per frame) than 640 at only ~15% more compute. Bump via
# YOLO_IMGSZ if you want to trade further.
_imgsz: int = int(os.environ.get("YOLO_IMGSZ", "704"))
_load_error: str | None = None
_load_lock = asyncio.Lock()


def _select_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model_blocking() -> tuple[Any, str]:
    """Load ultralytics model on the best device. Called inside executor."""
    from ultralytics import YOLO

    dev = _select_device()
    t0 = time.time()
    model = YOLO(_model_name)
    # Move to device and warm with a dummy frame so the first real inference
    # doesn't pay kernel-compile latency.
    dummy = np.zeros((_imgsz, _imgsz, 3), dtype=np.uint8)
    _ = model.predict(dummy, device=dev, verbose=False, imgsz=_imgsz, conf=0.4)
    log.info(
        "[yolo] model=%s device=%s imgsz=%d warm=%.0fms",
        _model_name, dev, _imgsz, (time.time() - t0) * 1000,
    )
    return model, dev


async def ensure_model() -> tuple[Any, str]:
    global _model, _device, _load_error
    if _model is not None and _device is not None:
        return _model, _device
    async with _load_lock:
        if _model is not None and _device is not None:
            return _model, _device
        loop = asyncio.get_running_loop()
        try:
            _model, _device = await loop.run_in_executor(_executor, _load_model_blocking)
            _load_error = None
        except Exception as e:  # noqa: BLE001
            _load_error = f"{type(e).__name__}: {e}"
            log.exception("[yolo] model load failed")
            raise
        return _model, _device


@dataclass
class DetectOpts:
    conf: float = 0.35
    iou: float = 0.45
    max_det: int = 10
    # Ids to exclude (applied after inference). None = no filter.
    exclude: set[int] | None = None


def _parse_opts(raw: dict | None) -> DetectOpts:
    if not raw:
        return DetectOpts()
    conf = float(raw.get("conf", 0.35))
    iou = float(raw.get("iou", 0.45))
    max_det = int(raw.get("maxDet", 10))
    exclude_raw = raw.get("excludeClassIds")
    exclude: set[int] | None = None
    if isinstance(exclude_raw, list):
        exclude = {int(x) for x in exclude_raw if isinstance(x, (int, float))}
    return DetectOpts(conf=conf, iou=iou, max_det=max_det, exclude=exclude)


def _pole_of_inaccessibility(mask: np.ndarray) -> tuple[int, int, float]:
    """2-pass chamfer distance transform on a binary mask, returns the pixel
    farthest from any edge plus its distance. Input: uint8 mask where inside
    pixels are non-zero. Coordinates are in the mask's local frame."""
    if mask.size == 0:
        return 0, 0, 0.0
    # OpenCV's distanceTransform needs uint8 0/1 input and handles L2
    # approximation efficiently. Way faster than a Python double-loop.
    m8 = (mask > 0).astype(np.uint8)
    if m8.sum() == 0:
        return 0, 0, 0.0
    dt = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
    idx = int(np.argmax(dt))
    h, w = m8.shape
    y = idx // w
    x = idx % w
    return x, y, float(dt[y, x])


def _pca_orientation(mask_bool: np.ndarray) -> tuple[float | None, float | None]:
    """Principal axis angle (radians, screen space) and major/minor ratio."""
    ys, xs = np.nonzero(mask_bool)
    if xs.size < 20:
        return None, None
    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)
    mx = xs_f.mean()
    my = ys_f.mean()
    dx = xs_f - mx
    dy = ys_f - my
    cxx = float((dx * dx).mean())
    cyy = float((dy * dy).mean())
    cxy = float((dx * dy).mean())
    angle = 0.5 * math.atan2(2 * cxy, cxx - cyy)
    trace = cxx + cyy
    disc = max(0.0, trace * trace * 0.25 - (cxx * cyy - cxy * cxy))
    l1 = trace * 0.5 + math.sqrt(disc)
    l2 = trace * 0.5 - math.sqrt(disc)
    ratio = math.sqrt(max(1.0, l1) / max(1.0, l2))
    return angle, ratio


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _run_inference(model: Any, device: str, frame_bgr: np.ndarray, opts: DetectOpts) -> list[dict]:
    """Run the seg model and pack detections matching the browser's shape."""
    # ultralytics accepts BGR np arrays directly. imgsz=640 matches the COCO
    # pretrain; bumping higher hurts latency more than accuracy on webcam.
    results = model.predict(
        frame_bgr,
        device=device,
        imgsz=_imgsz,
        conf=opts.conf,
        iou=opts.iou,
        max_det=max(opts.max_det, 10),
        verbose=False,
        # retina_masks=True renders masks at the input frame's native
        # resolution (not the 160×160 proto grid), which is the other
        # meaningful "cleaner masks" win. Adds ~1-2 ms on MPS; worth it —
        # sharper silhouettes = more stable pole/centroid → less visible
        # micro-jitter in face placement.
        retina_masks=True,
    )
    if not results:
        return []
    res = results[0]
    H, W = frame_bgr.shape[:2]

    if res.boxes is None or len(res.boxes) == 0:
        return []

    boxes_xyxy = res.boxes.xyxy.cpu().numpy()  # (N,4) in source pixels
    scores = res.boxes.conf.cpu().numpy()      # (N,)
    classes = res.boxes.cls.cpu().numpy().astype(int)  # (N,)

    # Masks are at model input resolution by default (640x640 letterbox).
    # ultralytics already uncrops them to original frame size when we read
    # res.masks.data — cpu() gives (N, H, W) float/bool-ish. Use .data for
    # the canonical representation.
    masks_np: np.ndarray | None = None
    if res.masks is not None:
        # .data is a torch tensor at model-input resolution. Use .xy polygons?
        # For speed we prefer the rasterized form at *frame* resolution which
        # is available via res.masks.data after the built-in upsample. To be
        # safe and avoid per-version differences, resize ourselves from the
        # model's native mask tensor.
        mtensor = res.masks.data.cpu().numpy()  # (N, mh, mw)
        if mtensor.shape[1] != H or mtensor.shape[2] != W:
            resized = np.empty((mtensor.shape[0], H, W), dtype=np.uint8)
            for i in range(mtensor.shape[0]):
                # Binary threshold then resize — cheaper than resize-then-threshold
                # for our use case and produces the same silhouette.
                m = (mtensor[i] > 0.5).astype(np.uint8) * 255
                resized[i] = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            masks_np = resized
        else:
            masks_np = ((mtensor > 0.5).astype(np.uint8) * 255)

    out: list[dict] = []
    # Sort by score descending, apply exclude filter, cap at max_det.
    order = np.argsort(-scores)
    for idx in order:
        cls = int(classes[idx])
        if opts.exclude and cls in opts.exclude:
            continue
        if cls < 0 or cls >= len(COCO_CLASSES):
            continue
        x1f, y1f, x2f, y2f = [float(v) for v in boxes_xyxy[idx]]
        x1 = max(0.0, min(float(W), x1f))
        y1 = max(0.0, min(float(H), y1f))
        x2 = max(0.0, min(float(W), x2f))
        y2 = max(0.0, min(float(H), y2f))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        det: dict[str, Any] = {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": cx, "cy": cy, "w": w, "h": h,
            "score": float(scores[idx]),
            "classId": cls,
            "className": COCO_CLASSES[cls],
        }

        if masks_np is not None:
            full = masks_np[idx]
            # Crop to bbox in source pixels.
            ix1, iy1 = int(math.floor(x1)), int(math.floor(y1))
            ix2, iy2 = int(math.ceil(x2)), int(math.ceil(y2))
            ix1 = max(0, ix1)
            iy1 = max(0, iy1)
            ix2 = min(W, ix2)
            iy2 = min(H, iy2)
            crop = full[iy1:iy2, ix1:ix2]
            if crop.size > 0 and crop.any():
                above = int(np.count_nonzero(crop))
                # Full-resolution mask bytes get sent base64; at 160x160-ish
                # crops that's small but webcam-sized crops can be much bigger.
                # Downsample for transit to 96 px on the longer side — the
                # browser uses this only for silhouette clipping of a ~200 px
                # face, so this resolution is plenty.
                mh, mw = crop.shape
                max_side = 96
                longer = max(mh, mw)
                if longer > max_side:
                    scale_m = max_side / longer
                    new_w = max(1, int(round(mw * scale_m)))
                    new_h = max(1, int(round(mh * scale_m)))
                    small = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                else:
                    small = crop

                # Centroid in source pixels (use full-res crop for accuracy).
                ys, xs = np.nonzero(crop)
                if xs.size > 0:
                    cxm = float(xs.mean()) + ix1
                    cym = float(ys.mean()) + iy1
                    det["maskCentroid"] = {"x": cxm, "y": cym}

                # Pole in source pixels.
                px, py, _pd = _pole_of_inaccessibility(crop)
                det["maskPole"] = {"x": float(px + ix1), "y": float(py + iy1)}

                # PCA orientation on the crop (screen space).
                angle, ratio = _pca_orientation(crop > 0)
                if angle is not None and ratio is not None:
                    det["principalAngle"] = angle
                    det["axisRatio"] = ratio

                det["maskArea"] = above
                det["mask"] = {
                    "w": int(small.shape[1]),
                    "h": int(small.shape[0]),
                    "data": base64.b64encode(bytes(small)).decode("ascii"),
                    # Source-pixel rect the mask covers, so the client can
                    # place it without knowing any letterbox math.
                    "ox": int(ix1),
                    "oy": int(iy1),
                    "ow": int(ix2 - ix1),
                    "oh": int(iy2 - iy1),
                }

        out.append(det)
        if len(out) >= opts.max_det:
            break
    return out


async def ws_handler(websocket: WebSocket) -> None:
    """Main WS loop. Each frame is a pair:
       1. (optional) text JSON with options + seq id for this frame
       2. binary JPEG bytes

    To keep the protocol simple and forgiving, we actually accept either:
       - a bare binary frame (uses previous options, no seq)
       - a text preamble followed immediately by a binary frame
    The text always carries seq; the response echoes it.
    """
    await websocket.accept()
    log.info("[yolo] ws connected")

    # Load the model eagerly on the first connection so the client can
    # differentiate "connecting" from "connected-but-model-loading".
    try:
        await ensure_model()
    except Exception as e:  # noqa: BLE001
        try:
            await websocket.send_text(json.dumps({"type": "error", "error": f"model load failed: {e}"}))
        except Exception:
            pass
        await websocket.close()
        return

    try:
        await websocket.send_text(json.dumps({
            "type": "ready",
            "model": _model_name,
            "device": _device,
            "imgsz": _imgsz,
        }))
    except Exception:
        return

    # Per-connection state
    pending_opts = DetectOpts()
    pending_seq: int | None = None
    loop = asyncio.get_running_loop()

    # Rolling FPS / infer-time window so we can see cadence without logging
    # every single frame. 60-frame window = roughly every 6–15 seconds.
    frame_count = 0
    window_start = time.time()
    window_infer_ms = 0.0

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "opts":
                    pending_opts = _parse_opts(payload)
                    seq = payload.get("seq")
                    pending_seq = int(seq) if isinstance(seq, (int, float)) else None
                continue

            raw = msg.get("bytes")
            if not raw:
                continue

            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            t0 = time.time()
            try:
                model, device = await ensure_model()
                dets = await loop.run_in_executor(
                    _executor, _run_inference, model, device, frame, pending_opts
                )
            except Exception as e:  # noqa: BLE001
                log.exception("[yolo] inference failed")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "detections",
                        "seq": pending_seq,
                        "error": f"{type(e).__name__}: {e}",
                        "detections": [],
                        "inferMs": int((time.time() - t0) * 1000),
                    }))
                except Exception:
                    break
                pending_seq = None
                continue

            infer_ms = int((time.time() - t0) * 1000)
            out = {
                "type": "detections",
                "seq": pending_seq,
                "detections": dets,
                "inferMs": infer_ms,
                "w": int(frame.shape[1]),
                "h": int(frame.shape[0]),
            }
            pending_seq = None
            try:
                await websocket.send_text(json.dumps(out))
            except Exception:
                break
            frame_count += 1
            window_infer_ms += infer_ms
            if frame_count % 60 == 0:
                elapsed = time.time() - window_start
                fps = 60 / elapsed if elapsed > 0 else 0
                avg_ms = window_infer_ms / 60
                log.info(
                    "[yolo] ◦ frames=%d fps=%.1f avg-infer=%.1fms dets=last=%d",
                    frame_count,
                    fps,
                    avg_ms,
                    len(dets),
                )
                window_start = time.time()
                window_infer_ms = 0.0
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("[yolo] ws crashed")
    finally:
        log.info("[yolo] ws closed frames=%d", frame_count)
