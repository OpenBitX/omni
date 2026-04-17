"""Realtime face-composite WebSocket server.

Protocol:
  - Client opens WS at /ws.
  - Client sends binary frames: JPEG bytes of the webcam frame.
  - Server responds with binary frames: JPEG bytes of the composite, and
    occasional text status frames: {"event":"no_face"|"face"|"base", ...}.
  - Client text message '{"baseImage": "orange.jpg"}' swaps the base.

Run:
  OPENAI_API_KEY=... .venv/bin/python server.py
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import dlib
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from imutils import face_utils, resize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mirror")

ROOT = Path(__file__).parent.resolve()

# Auto-load OPENAI_API_KEY from the repo's .env.local (parent of server/).
# Existing env vars win, so `OPENAI_API_KEY=... python server.py` still works.
for candidate in (ROOT.parent / ".env.local", ROOT / ".env", ROOT.parent / ".env"):
    if candidate.exists():
        load_dotenv(candidate, override=False)
        log.info("loaded env from %s", candidate)
        break
CANVAS = 512
DEFAULT_BASE = "orange.jpg"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_FRAME_BYTES = 4 * 1024 * 1024  # 4 MB
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
GPT_TIMEOUT_S = 30.0
GPT_RETRIES = 3
# Bump when the coords schema or prompt changes — invalidates old cache files.
COORDS_VERSION = 2
EYE_WIDTH_MIN, EYE_WIDTH_MAX = 60, 210
MOUTH_WIDTH_MIN, MOUTH_WIDTH_MAX = 100, 380


def default_coords() -> dict:
    """Sane centered coords used when GPT-4o is unavailable or returns junk."""
    return {
        "left_eye": [int(CANVAS * 0.36), int(CANVAS * 0.42)],
        "right_eye": [int(CANVAS * 0.64), int(CANVAS * 0.42)],
        "mouth": [int(CANVAS * 0.50), int(CANVAS * 0.68)],
        "eye_width": int(CANVAS * 0.18),
        "mouth_width": int(CANVAS * 0.30),
    }


def _clamp_point(p, margin: int = 40) -> list[int]:
    try:
        x, y = int(p[0]), int(p[1])
    except (TypeError, ValueError, IndexError):
        return [CANVAS // 2, CANVAS // 2]
    return [max(margin, min(CANVAS - margin, x)), max(margin, min(CANVAS - margin, y))]


def validate_coords(raw: dict) -> dict | None:
    """Coerce whatever GPT-4o returned into a valid coords dict. Returns None
    if the response is unusable so the caller can fall back to defaults
    WITHOUT caching — we never want bad data frozen on disk."""
    if not isinstance(raw, dict):
        return None
    try:
        out = {
            "left_eye": _clamp_point(raw["left_eye"]),
            "right_eye": _clamp_point(raw["right_eye"]),
            "mouth": _clamp_point(raw["mouth"]),
        }
    except (KeyError, TypeError):
        return None

    # left_eye must actually be on the left (lower x)
    if out["left_eye"][0] > out["right_eye"][0]:
        out["left_eye"], out["right_eye"] = out["right_eye"], out["left_eye"]

    iod = abs(out["right_eye"][0] - out["left_eye"][0])
    if iod < 40:
        log.warning("coords rejected: eyes too close (iod=%d)", iod)
        return None

    # mouth should be below eyes
    eyes_y = (out["left_eye"][1] + out["right_eye"][1]) / 2
    if out["mouth"][1] <= eyes_y + 10:
        log.warning("coords rejected: mouth not below eyes")
        return None

    # eyes should be roughly level
    if abs(out["left_eye"][1] - out["right_eye"][1]) > CANVAS * 0.12:
        log.warning("coords rejected: eyes not level")
        return None

    # Scale fields — derive from IOD when missing, clamp when given.
    eye_w = raw.get("eye_width")
    eye_w = int(eye_w) if isinstance(eye_w, (int, float)) else int(iod * 0.55)
    # eye patches must not overlap horizontally — cap at 95% of IOD
    eye_w = min(eye_w, int(iod * 0.95))
    out["eye_width"] = max(EYE_WIDTH_MIN, min(EYE_WIDTH_MAX, eye_w))

    mouth_w = raw.get("mouth_width")
    mouth_w = int(mouth_w) if isinstance(mouth_w, (int, float)) else int(iod * 1.4)
    out["mouth_width"] = max(MOUTH_WIDTH_MIN, min(MOUTH_WIDTH_MAX, mouth_w))

    return out


def ask_gpt_for_coords(img_path: Path) -> dict | None:
    """Call GPT-4o for coords with retries + timeout. Returns None on any
    failure (bad key, network, validation) so the caller can fall back to
    defaults without caching them."""
    if "OPENAI_API_KEY" not in os.environ:
        log.warning("OPENAI_API_KEY unset — using default coords for %s", img_path.name)
        return None

    from openai import OpenAI

    try:
        img_bytes = img_path.read_bytes()
    except OSError as e:
        log.warning("read failed for %s (%s)", img_path.name, e)
        return None

    b64 = base64.b64encode(img_bytes).decode()
    ext = img_path.suffix.lstrip(".").lower() or "jpeg"
    if ext == "jpg":
        ext = "jpeg"

    prompt = (
        f"You are placing a composite of a live person's eyes and mouth onto this "
        f"base image. The base will be scaled to exactly {CANVAS}x{CANVAS} pixels "
        "before compositing. Use the top-left pixel as the origin (0,0), with x "
        "increasing rightward and y increasing downward.\n\n"
        "TASK: Find the MAIN SUBJECT that should receive a face — the dominant "
        "face, head, or face-like object (orange, pumpkin, mug, toy, animal, "
        "etc.). If multiple candidates exist, pick the largest / most central. "
        "Return the CENTER pixel where each feature should be pasted, plus the "
        "horizontal WIDTH in pixels that each feature should occupy on the base.\n\n"
        "Return ONLY this JSON (no prose, no markdown fence):\n"
        '{"left_eye":[x,y],"right_eye":[x,y],"mouth":[x,y],'
        '"eye_width":N,"mouth_width":N}\n\n'
        "Rules:\n"
        "- 'left_eye' is the eye on the viewer's LEFT (lower x) — not anatomical left.\n"
        "- Eyes sit on roughly the same y (within a few percent of canvas height).\n"
        "- Mouth y is strictly below both eye y values.\n"
        f"- eye_width = horizontal size of ONE eye on the base (normally "
        f"{int(CANVAS*0.10)}–{int(CANVAS*0.24)}; small for distant subjects, "
        "larger for close-ups).\n"
        f"- mouth_width = horizontal size of the mouth on the base (normally "
        f"{int(CANVAS*0.18)}–{int(CANVAS*0.45)}).\n"
        "- Keep all points at least 40px from the image border.\n"
        "- Eyes must not overlap: distance between eye centers > eye_width.\n"
        "- Think about face proportions: vertical gap eyes→mouth ≈ 1.2–1.8× the "
        "distance between the eye centers.\n\n"
        "Study the image carefully. Locate the feature positions precisely on the "
        "actual subject — do not default to center. Return only JSON."
    )

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=GPT_TIMEOUT_S)
    last_err: Exception | None = None
    for attempt in range(1, GPT_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{ext};base64,{b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
            )
            raw = json.loads(resp.choices[0].message.content or "{}")
            validated = validate_coords(raw)
            if validated is not None:
                log.info("GPT coords for %s: %s", img_path.name, validated)
                return validated
            log.warning(
                "GPT attempt %d returned unusable coords for %s: %s",
                attempt, img_path.name, raw,
            )
        except Exception as e:  # openai errors, json errors, network, etc.
            last_err = e
            log.warning("GPT-4o attempt %d failed: %s", attempt, e)
        if attempt < GPT_RETRIES:
            time.sleep(0.6 * attempt)

    log.warning(
        "GPT-4o gave up after %d attempts (last: %s) for %s",
        GPT_RETRIES, last_err, img_path.name,
    )
    return None


def load_or_detect_coords(img_path: Path) -> dict:
    """Try the cache first; otherwise ask GPT-4o. Never raises. Defaults are
    returned (not cached) when GPT fails — a future server restart will retry."""
    try:
        img_hash = hashlib.md5(img_path.read_bytes()).hexdigest()[:8]
    except OSError:
        return default_coords()
    cache = img_path.with_suffix(f".v{COORDS_VERSION}.{img_hash}.coords.json")
    if cache.exists():
        try:
            cached = validate_coords(json.loads(cache.read_text()))
            if cached is not None:
                return cached
            log.warning("cached coords invalid, re-detecting: %s", cache.name)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("cache read failed (%s), re-detecting: %s", e, cache.name)
    coords = ask_gpt_for_coords(img_path)
    if coords is not None:
        try:
            cache.write_text(json.dumps(coords, indent=2))
        except OSError as e:
            log.warning("cache write failed: %s", e)
        return coords
    log.warning("falling back to default coords (uncached) for %s", img_path.name)
    return default_coords()


def _within_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


class Compositor:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            str(ROOT / "shape_predictor_68_face_landmarks.dat")
        )
        self.lock = threading.Lock()
        self.base_img: np.ndarray | None = None
        self.coords: dict | None = None
        self.base_name: str | None = None
        self.load_base(DEFAULT_BASE)

    def load_base(self, name: str):
        safe = Path(name).name  # strip any path segments
        path = (ROOT / safe).resolve()
        if not _within_root(path) or not path.exists():
            raise FileNotFoundError(str(safe))
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"cv2 could not read {safe}")
        img = cv2.resize(img, (CANVAS, CANVAS))
        coords = load_or_detect_coords(path)
        with self.lock:
            self.base_img = img
            self.coords = coords
            self.base_name = safe
        log.info("base loaded: %s coords=%s", safe, coords)

    def snapshot(self):
        with self.lock:
            if self.base_img is None or self.coords is None:
                return None, None, None
            return self.base_img.copy(), dict(self.coords), self.base_name

    def composite(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        """Returns (image, face_detected). Never raises."""
        base, coords, _ = self.snapshot()
        if base is None or coords is None:
            return frame_bgr, False

        try:
            faces = self.detector(frame_bgr)
        except Exception as e:
            log.warning("detector failed: %s", e)
            return base, False

        if not faces:
            return base, False

        try:
            # Pick the largest face — robust against background faces/noise.
            face = max(
                faces,
                key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()),
            )
            shape = face_utils.shape_to_np(self.predictor(frame_bgr, face))
            base = self._paste_eyes(frame_bgr, shape, base, coords)
            base = self._paste_mouth(frame_bgr, shape, base, coords)
        except Exception as e:
            log.warning("composite failed: %s", e)
        return base, True

    @staticmethod
    def _paste_eyes(src, shape, base, coords):
        def crop(i1, i2, iy1, iy2, margin_scale):
            x1, x2 = int(shape[i1, 0]), int(shape[i2, 0])
            y1, y2 = int(shape[iy1, 1]), int(shape[iy2, 1])
            margin = int(max(0, (x2 - x1) * margin_scale))
            y1c, y2c = max(0, y1 - margin), min(src.shape[0], y2 + margin)
            x1c, x2c = max(0, x1 - margin), min(src.shape[1], x2 + margin)
            return src[y1c:y2c, x1c:x2c].copy()

        try:
            left = crop(36, 39, 37, 41, 0.18)
            right = crop(42, 45, 43, 47, 0.18)
        except Exception:
            return base

        if left.size == 0 or right.size == 0:
            return base
        if left.shape[0] < 4 or left.shape[1] < 4 or right.shape[0] < 4 or right.shape[1] < 4:
            return base

        eye_w = int(coords.get("eye_width", 160))
        eye_w = max(EYE_WIDTH_MIN, min(EYE_WIDTH_MAX, eye_w))
        try:
            left = resize(left, width=eye_w)
            right = resize(right, width=eye_w)
        except Exception:
            return base

        for patch, target_key in ((left, "left_eye"), (right, "right_eye")):
            try:
                target = Compositor._safe_target(patch, tuple(coords[target_key]))
                base = cv2.seamlessClone(
                    patch, base,
                    np.full(patch.shape[:2], 255, patch.dtype),
                    target, cv2.NORMAL_CLONE,
                )
            except cv2.error:
                pass
        return base

    @staticmethod
    def _paste_mouth(src, shape, base, coords):
        try:
            mx1, mx2 = int(shape[48, 0]), int(shape[54, 0])
            my1, my2 = int(shape[50, 1]), int(shape[57, 1])
            margin = int(max(0, (mx2 - mx1) * 0.1))
            y1c, y2c = max(0, my1 - margin), min(src.shape[0], my2 + margin)
            x1c, x2c = max(0, mx1 - margin), min(src.shape[1], mx2 + margin)
            patch = src[y1c:y2c, x1c:x2c].copy()
        except Exception:
            return base

        if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
            return base

        mouth_w = int(coords.get("mouth_width", 320))
        mouth_w = max(MOUTH_WIDTH_MIN, min(MOUTH_WIDTH_MAX, mouth_w))
        try:
            patch = resize(patch, width=mouth_w)
        except Exception:
            return base

        try:
            target = Compositor._safe_target(patch, tuple(coords["mouth"]))
            base = cv2.seamlessClone(
                patch, base,
                np.full(patch.shape[:2], 255, patch.dtype),
                target, cv2.NORMAL_CLONE,
            )
        except cv2.error:
            pass
        return base

    @staticmethod
    def _safe_target(patch: np.ndarray, target: tuple[int, int]) -> tuple[int, int]:
        """Clamp target so the patch rect stays fully inside the CANVAS×CANVAS base."""
        h, w = patch.shape[:2]
        half_w = (w // 2) + 1
        half_h = (h // 2) + 1
        x = int(target[0])
        y = int(target[1])
        x = max(half_w, min(CANVAS - half_w, x))
        y = max(half_h, min(CANVAS - half_h, y))
        return (x, y)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

compositor = Compositor()
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="composite")


@app.on_event("shutdown")
def _on_shutdown():
    executor.shutdown(wait=False, cancel_futures=True)


@app.get("/health")
def health():
    return {
        "ok": True,
        "base": compositor.base_name,
        "coords": compositor.coords,
        "openai_key_present": "OPENAI_API_KEY" in os.environ,
    }


@app.get("/bases")
def bases():
    allowed = sorted(
        p.name for p in ROOT.iterdir()
        if p.suffix.lower() in ALLOWED_EXT
    )
    return {"bases": allowed, "current": compositor.base_name}


@app.post("/base/{name}")
def set_base(name: str):
    try:
        compositor.load_base(name)
    except FileNotFoundError:
        raise HTTPException(404, f"base not found: {name}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "base": compositor.base_name, "coords": compositor.coords}


def _safe_name(original: str) -> str:
    stem = Path(original).stem
    ext = Path(original).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"unsupported extension {ext!r}")
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:40]
    if not cleaned:
        cleaned = "upload"
    return f"{cleaned}{ext}"


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file too large (>{MAX_UPLOAD_BYTES // 1024 // 1024} MB)")

    # Verify bytes are actually a decodable image before touching disk
    probe = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if probe is None:
        raise HTTPException(400, "not a decodable image")

    name = _safe_name(file.filename or "upload.jpg")
    dest = ROOT / name
    if not _within_root(dest):
        raise HTTPException(400, "bad path")
    if dest.exists() and dest.read_bytes() != data:
        i = 2
        while True:
            candidate = ROOT / f"{dest.stem}-{i}{dest.suffix}"
            if not candidate.exists():
                dest = candidate
                name = candidate.name
                break
            i += 1
    dest.write_bytes(data)
    try:
        compositor.load_base(name)
    except Exception as e:
        log.exception("load_base failed after upload")
        raise HTTPException(500, f"loaded but couldn't activate: {e}")
    return {"ok": True, "base": compositor.base_name, "coords": compositor.coords}


async def _composite_async(frame_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, compositor.composite, frame_bgr)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    last_face_state: bool | None = None
    log.info("ws connected")
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("bytes") is not None:
                payload: bytes = msg["bytes"]
                if not payload or len(payload) > MAX_FRAME_BYTES:
                    continue
                buf = np.frombuffer(payload, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                try:
                    out, has_face = await _composite_async(frame)
                except Exception:
                    log.exception("composite pipeline failed")
                    continue

                ok_enc, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if not ok_enc:
                    continue
                try:
                    await websocket.send_bytes(enc.tobytes())
                except Exception:
                    break

                if last_face_state != has_face:
                    last_face_state = has_face
                    try:
                        await websocket.send_text(
                            json.dumps({"event": "face" if has_face else "no_face"})
                        )
                    except Exception:
                        break
                continue

            if msg.get("text") is not None:
                try:
                    text_payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if "baseImage" in text_payload:
                    try:
                        compositor.load_base(text_payload["baseImage"])
                        await websocket.send_text(
                            json.dumps({
                                "event": "base",
                                "ok": True,
                                "base": compositor.base_name,
                            })
                        )
                    except Exception as e:
                        await websocket.send_text(
                            json.dumps({"event": "base", "ok": False, "error": str(e)})
                        )
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("ws loop crashed")
    finally:
        log.info("ws closed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
