"""YOLO WebSocket router: /ws/yolo

Migrated from server/yolo_ws.py. Protocol contract is unchanged.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from omni_backend.config import settings

log = logging.getLogger("yolo_ws")

router = APIRouter()

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")
_model: Any = None
_device: str | None = None
_load_error: str | None = None
_load_lock = asyncio.Lock()

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


class _StubModel:
    """Minimal stub for TEST_MODE — returns a single fake detection."""

    def predict(self, frame, device, imgsz, conf, iou, max_det, verbose, retina_masks):
        class FakeBoxes:
            def __init__(self):
                self.xyxy = np.array([[10.0, 10.0, 60.0, 60.0]], dtype=np.float32)
                self.conf = np.array([0.9], dtype=np.float32)
                self.cls = np.array([41], dtype=np.float32)  # cup

            def __len__(self):
                return int(self.xyxy.shape[0])

        class FakeResult:
            def __init__(self):
                self.boxes = FakeBoxes()
                self.masks = None

        return [FakeResult()]


def _select_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model_blocking() -> tuple[Any, str]:
    from ultralytics import YOLO

    model_name = settings.yolo_model
    imgsz = settings.yolo_imgsz
    dev = _select_device()
    t0 = time.time()
    model = YOLO(model_name)
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    _ = model.predict(dummy, device=dev, verbose=False, imgsz=imgsz, conf=0.4)
    log.info("[yolo] model=%s device=%s imgsz=%d warm=%.0fms", model_name, dev, imgsz, (time.time() - t0) * 1000)
    return model, dev


async def startup() -> None:
    """Eagerly load on startup (optional — also loads on first connection)."""
    global _executor
    # TestClient can start/stop the app multiple times within one process,
    # which may leave the module-level executor shut down.
    if getattr(_executor, "_shutdown", False):
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")
    if settings.test_mode:
        global _model, _device
        _model = _StubModel()
        _device = "cpu"
        log.info("[yolo] TEST_MODE — stub model loaded")
        return
    try:
        await ensure_model()
    except Exception as e:
        log.error("[yolo] startup model load failed: %s", e)


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
        except Exception as e:
            _load_error = f"{type(e).__name__}: {e}"
            log.exception("[yolo] model load failed")
            raise
        return _model, _device


@dataclass
class DetectOpts:
    conf: float = 0.35
    iou: float = 0.45
    max_det: int = 10
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
    if mask.size == 0:
        return 0, 0, 0.0
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
    ys, xs = np.nonzero(mask_bool)
    if xs.size < 20:
        return None, None
    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)
    mx, my = xs_f.mean(), ys_f.mean()
    dx, dy = xs_f - mx, ys_f - my
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


def _run_inference(model: Any, device: str, frame_bgr: np.ndarray, opts: DetectOpts) -> list[dict]:
    imgsz = settings.yolo_imgsz
    results = model.predict(
        frame_bgr,
        device=device,
        imgsz=imgsz,
        conf=opts.conf,
        iou=opts.iou,
        max_det=max(opts.max_det, 10),
        verbose=False,
        retina_masks=True,
    )
    if not results:
        return []
    res = results[0]
    H, W = frame_bgr.shape[:2]

    if res.boxes is None or len(res.boxes) == 0:
        return []

    boxes_xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else res.boxes.xyxy
    scores = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, "cpu") else res.boxes.conf
    classes = (
        res.boxes.cls.cpu().numpy().astype(int)
        if hasattr(res.boxes.cls, "cpu")
        else res.boxes.cls.astype(int)
    )

    masks_np: np.ndarray | None = None
    if res.masks is not None:
        mtensor = res.masks.data.cpu().numpy() if hasattr(res.masks.data, "cpu") else res.masks.data
        if mtensor.shape[1] != H or mtensor.shape[2] != W:
            resized = np.empty((mtensor.shape[0], H, W), dtype=np.uint8)
            for i in range(mtensor.shape[0]):
                m = (mtensor[i] > 0.5).astype(np.uint8) * 255
                resized[i] = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            masks_np = resized
        else:
            masks_np = ((mtensor > 0.5).astype(np.uint8) * 255)

    out: list[dict] = []
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
        det: dict[str, Any] = {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) * 0.5, "cy": (y1 + y2) * 0.5,
            "w": w, "h": h,
            "score": float(scores[idx]),
            "classId": cls,
            "className": COCO_CLASSES[cls],
        }

        if masks_np is not None:
            full = masks_np[idx]
            ix1, iy1 = int(math.floor(x1)), int(math.floor(y1))
            ix2, iy2 = int(math.ceil(x2)), int(math.ceil(y2))
            ix1, iy1 = max(0, ix1), max(0, iy1)
            ix2, iy2 = min(W, ix2), min(H, iy2)
            crop = full[iy1:iy2, ix1:ix2]
            if crop.size > 0 and crop.any():
                above = int(np.count_nonzero(crop))
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

                ys, xs = np.nonzero(crop)
                if xs.size > 0:
                    det["maskCentroid"] = {"x": float(xs.mean()) + ix1, "y": float(ys.mean()) + iy1}

                px, py, _pd = _pole_of_inaccessibility(crop)
                det["maskPole"] = {"x": float(px + ix1), "y": float(py + iy1)}

                angle, ratio = _pca_orientation(crop > 0)
                if angle is not None and ratio is not None:
                    det["principalAngle"] = angle
                    det["axisRatio"] = ratio

                det["maskArea"] = above
                det["mask"] = {
                    "w": int(small.shape[1]),
                    "h": int(small.shape[0]),
                    "data": base64.b64encode(bytes(small)).decode("ascii"),
                    "ox": int(ix1), "oy": int(iy1),
                    "ow": int(ix2 - ix1), "oh": int(iy2 - iy1),
                }

        out.append(det)
        if len(out) >= opts.max_det:
            break
    return out


@router.websocket("/yolo")
async def ws_yolo(websocket: WebSocket):
    await websocket.accept()
    log.info("[yolo] ws connected")

    try:
        model, device = await ensure_model()
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "error": f"model load failed: {e}"}))
        except Exception:
            pass
        await websocket.close()
        return

    try:
        await websocket.send_text(json.dumps({
            "type": "ready",
            "model": settings.yolo_model,
            "device": device,
            "imgsz": settings.yolo_imgsz,
        }))
    except Exception:
        return

    pending_opts = DetectOpts()
    pending_seq: int | None = None
    loop = asyncio.get_running_loop()
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
                m, dev = await ensure_model()
                dets = await loop.run_in_executor(_executor, _run_inference, m, dev, frame, pending_opts)
            except Exception as e:
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
            try:
                await websocket.send_text(json.dumps({
                    "type": "detections",
                    "seq": pending_seq,
                    "detections": dets,
                    "inferMs": infer_ms,
                    "w": int(frame.shape[1]),
                    "h": int(frame.shape[0]),
                }))
            except Exception:
                break
            pending_seq = None
            frame_count += 1
            window_infer_ms += infer_ms
            if frame_count % 60 == 0:
                elapsed = time.time() - window_start
                fps = 60 / elapsed if elapsed > 0 else 0
                log.info("[yolo] ◦ frames=%d fps=%.1f avg=%.1fms", frame_count, fps, window_infer_ms / 60)
                window_start = time.time()
                window_infer_ms = 0.0
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("[yolo] ws crashed")
    finally:
        log.info("[yolo] ws closed frames=%d", frame_count)
