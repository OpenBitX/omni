"""Realtime face-composite WebSocket server.

Protocol:
  - Client opens WS at /ws.
  - Client sends binary frames: JPEG bytes of the webcam frame.
  - Server responds with binary frames: JPEG bytes of the composite, and
    occasional text status frames: {"event":"no_face"|"face"|"base", ...}.
  - Client text message '{"baseImage": "orange.jpg"}' swaps the base.

Run:
  ZHIPU_API_KEY=... .venv/bin/python server.py
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
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
from fastapi.responses import FileResponse
from imutils import face_utils, resize

from yolo_ws import ws_handler as yolo_ws_handler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mirror")

ROOT = Path(__file__).parent.resolve()

# Auto-load API keys from the repo's .env.local (parent of server/).
# Existing env vars win, so `ZHIPU_API_KEY=... python server.py` still works.
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
GLM_TIMEOUT_S = 60.0  # glm-5v-turbo is a reasoning model; needs headroom for reasoning tokens
GLM_RETRIES = 3
GLM_MODEL = "glm-5v-turbo"
# bigmodel.cn keys look like `<hex>.<secret>`; api.z.ai keys look like `sk-...`.
# Both endpoints are OpenAI-compatible; pick by key shape unless overridden.
GLM_BASE_URL = os.environ.get(
    "ZHIPU_BASE_URL",
    "https://open.bigmodel.cn/api/paas/v4/"
    if "." in (os.environ.get("ZHIPU_API_KEY") or "") and not (os.environ.get("ZHIPU_API_KEY") or "").startswith("sk-")
    else "https://api.z.ai/api/paas/v4/",
)
# Bump when the coords schema, prompt, or model changes — invalidates old cache files.
COORDS_VERSION = 5
EYE_WIDTH_MIN, EYE_WIDTH_MAX = 60, 210
MOUTH_WIDTH_MIN, MOUTH_WIDTH_MAX = 100, 380


def default_coords() -> dict:
    """Sane centered coords used when GLM is unavailable or returns junk."""
    return {
        "left_eye": [int(CANVAS * 0.36), int(CANVAS * 0.42)],
        "right_eye": [int(CANVAS * 0.64), int(CANVAS * 0.42)],
        "mouth": [int(CANVAS * 0.50), int(CANVAS * 0.68)],
        "eye_width": int(CANVAS * 0.18),
        "mouth_width": int(CANVAS * 0.30),
    }


_FILENAME_PREFIX_RE = re.compile(
    r"^(screenshot|img|image|pxl|photo|dsc|capture)[\s_-]*", re.IGNORECASE,
)
_FILENAME_DATE_RE = re.compile(
    r"\d{4}[-_]\d{2}[-_]\d{2}"
    r"(?:[\s_-]+at[\s_-]+\d{1,2}[-_:]\d{2}(?:[-_:]\d{2})?(?:[\s_-]*[ap]m)?)?",
    re.IGNORECASE,
)


def filename_to_label(name: str) -> str:
    """Best-effort human label from a filename when no AI label is cached yet.
    Strips screenshot/photo prefixes and date-stamps, collapses separators,
    title-cases a lowercase slug. Fallback when GLM hasn't run yet."""
    original = Path(name).stem
    stem = _FILENAME_PREFIX_RE.sub("", original)
    stem = _FILENAME_DATE_RE.sub("", stem)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" -_")
    if len(stem) < 2:
        # Whole name was prefix+date; use the prefix word as the label.
        m = _FILENAME_PREFIX_RE.match(original)
        if m:
            return m.group(1).title()
        return (original[:20] or "Image").strip() or "Image"
    if len(stem) > 24:
        stem = stem[:22].rstrip() + "…"
    if stem.islower() or stem.isupper():
        stem = stem.title()
    return stem


def _cached_label(img_path: Path) -> str | None:
    """Cheap read of the label field from this image's current-version cache,
    without re-hashing the image. Returns None if no cache or no label there."""
    pattern = f"{img_path.stem}.v{COORDS_VERSION}.*.coords.json"
    caches = sorted(img_path.parent.glob(pattern))
    for cache in reversed(caches):
        try:
            data = json.loads(cache.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        label = data.get("label")
        if isinstance(label, str) and label.strip():
            return label.strip()
    return None


def _clamp_point(p, margin: int = 40) -> list[int]:
    try:
        x, y = int(p[0]), int(p[1])
    except (TypeError, ValueError, IndexError):
        return [CANVAS // 2, CANVAS // 2]
    return [max(margin, min(CANVAS - margin, x)), max(margin, min(CANVAS - margin, y))]


def _parse_bbox(raw) -> tuple[int, int, int, int] | None:
    """Parse a [x1,y1,x2,y2] bbox and clamp to canvas. Returns None if unusable."""
    try:
        x1, y1, x2, y2 = (int(v) for v in raw)
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0, min(CANVAS, x1)), max(0, min(CANVAS, x2))))
    y1, y2 = sorted((max(0, min(CANVAS, y1)), max(0, min(CANVAS, y2))))
    w, h = x2 - x1, y2 - y1
    # Reject degenerate, tiny, or near-full-frame boxes — both hurt paste quality.
    if w < CANVAS * 0.15 or h < CANVAS * 0.15:
        return None
    if w > CANVAS * 0.98 and h > CANVAS * 0.98:
        return None
    return (x1, y1, x2, y2)


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

    # mouth x should sit between the eyes — catches "eyes on different objects"
    # where the model picks a mouth on a third location. Allow 10% canvas slack.
    eyes_cx = (out["left_eye"][0] + out["right_eye"][0]) / 2
    if abs(out["mouth"][0] - eyes_cx) > max(iod * 0.5, CANVAS * 0.10):
        log.warning(
            "coords rejected: mouth x (%d) not between eyes (cx=%.0f iod=%d)",
            out["mouth"][0], eyes_cx, iod,
        )
        return None

    # eyes should be roughly level
    if abs(out["left_eye"][1] - out["right_eye"][1]) > CANVAS * 0.12:
        log.warning("coords rejected: eyes not level")
        return None

    # Subject bbox gates everything: all 3 features must sit inside it, and
    # IOD must be a sane fraction of bbox width. This is the main defense
    # against "one eye on the cap, one eye on the rope handle" failures.
    bbox = _parse_bbox(raw.get("subject_bbox"))
    if bbox is None:
        log.warning("coords rejected: missing/invalid subject_bbox")
        return None
    x1, y1, x2, y2 = bbox
    pad = 6  # tolerate tiny off-by-a-pixel errors at the bbox edge
    for key in ("left_eye", "right_eye", "mouth"):
        px, py = out[key]
        if not (x1 - pad <= px <= x2 + pad and y1 - pad <= py <= y2 + pad):
            log.warning(
                "coords rejected: %s (%d,%d) outside bbox %s", key, px, py, bbox,
            )
            return None
    bbox_w = x2 - x1
    if iod > bbox_w * 0.75:
        log.warning(
            "coords rejected: IOD=%d too wide for bbox width %d (eyes likely on different objects)",
            iod, bbox_w,
        )
        return None
    out["subject_bbox"] = list(bbox)

    # Scale fields — derive from IOD when missing, clamp when given.
    eye_w = raw.get("eye_width")
    eye_w = int(eye_w) if isinstance(eye_w, (int, float)) else int(iod * 0.55)
    # eye patches must not overlap horizontally — cap at 95% of IOD
    eye_w = min(eye_w, int(iod * 0.95))
    out["eye_width"] = max(EYE_WIDTH_MIN, min(EYE_WIDTH_MAX, eye_w))

    mouth_w = raw.get("mouth_width")
    mouth_w = int(mouth_w) if isinstance(mouth_w, (int, float)) else int(iod * 1.4)
    out["mouth_width"] = max(MOUTH_WIDTH_MIN, min(MOUTH_WIDTH_MAX, mouth_w))

    # Optional, cosmetic — short human label for the UI chip.
    label = raw.get("label")
    if isinstance(label, str):
        cleaned = label.strip().strip('"\'.,;:!?()[]{}').strip()
        if len(cleaned) > 28:
            cleaned = cleaned[:26].rstrip() + "…"
        if cleaned:
            out["label"] = cleaned

    return out


def _extract_json_object(text: str) -> dict | None:
    """Pull the first JSON object out of a model response. GLM sometimes wraps
    output in ```json fences or prefixes it with <think>…</think> reasoning
    traces; the OpenAI `response_format=json_object` flag isn't reliably honored
    across Zhipu models, so we parse defensively."""
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip fenced blocks like ```json ... ``` or ``` ... ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    # Find the outermost {...} span as a fallback.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def ask_glm_for_coords(img_path: Path) -> dict | None:
    """Call GLM (Zhipu) for coords with retries + timeout. Returns None on any
    failure (bad key, network, validation) so the caller can fall back to
    defaults without caching them."""
    if "ZHIPU_API_KEY" not in os.environ:
        log.warning("ZHIPU_API_KEY unset — using default coords for %s", img_path.name)
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
        f"You are placing a composite of a live person's eyes and mouth onto "
        f"this base image. The base will be scaled to exactly {CANVAS}x{CANVAS} "
        "pixels before compositing. Origin is the top-left pixel (0,0); x "
        "increases rightward, y increases downward.\n\n"
        "HOW THE PASTE WORKS — critical context for the rules below.\n"
        "Each feature is composited as an elliptical patch centered on the "
        "point you return, feathered into the surrounding pixels:\n"
        "  • Each eye patch occupies a rectangle ~eye_width wide × "
        "    0.7·eye_width tall, centered on the eye point.\n"
        "  • The mouth patch occupies a rectangle ~mouth_width wide × "
        "    0.55·mouth_width tall, centered on the mouth point.\n"
        "  • The blend pulls color from WHATEVER pixels sit under those "
        "    rectangles. Pixels on rope, background, hair, shadow, stem, "
        "    label, handle, or a different surface will pollute the blend "
        "    and ruin the result.\n"
        "So 'inside the subject' means the whole patch rectangle — not just "
        "the center — must sit on one clean face-bearing surface.\n\n"
        "YOUR JOB, IN TWO PHASES:\n\n"
        "PHASE A — FIND THE MAIN ITEM.\n"
        "Scan the image and lock onto the single dominant subject that "
        "should receive a face: a face, head, or face-like object (orange, "
        "pumpkin, mug, bottle cap, rock, toy, doll, fruit, animal, etc.). "
        "If several candidates exist, pick the largest and most central. "
        "Label this subject in 1–3 words (title case, no punctuation, e.g. "
        "\"Orange\", \"Sock Monkey\", \"Ceramic Mug\", \"Garfield\").\n\n"
        "PHASE B — FIND THE PERFECT FACE SPOT ON THAT ITEM.\n"
        "Now decide exactly where the face goes on the chosen subject.\n"
        "  1. Output subject_bbox = a tight rectangle [x1,y1,x2,y2] hugging "
        "     ONLY the face-bearing surface of that item — not the whole "
        "     silhouette. Exclude accessories, rims, shadows, handles, "
        "     stems, ropes, labels, hair, cords, cap threads. For tall "
        "     objects box just the face region (e.g. water bottle → cap).\n"
        "  2. Place all three features SNUGLY and SAFELY inside that bbox:\n"
        "     - Eyes on roughly the same y, both on the same continuous "
        "       surface, naturally spaced for a face (not stretched).\n"
        "     - Mouth directly below and between the eyes, on the same "
        "       surface.\n"
        "     - The WHOLE patch rectangle of every feature (center ± half "
        "       width, center ± half height) must lie inside subject_bbox "
        "       with breathing room on every side. No patch edge may touch "
        "       or cross the bbox boundary.\n"
        "     - If they don't fit with breathing room, SHRINK eye_width "
        "       and mouth_width until they do. A small well-placed face "
        "       reads far better than a big face that crops into rope or "
        "       background. Do not spread eyes wider to fill the bbox.\n\n"
        "Return ONLY this JSON (no prose, no markdown fence):\n"
        '{"label":"ShortName",'
        '"subject_bbox":[x1,y1,x2,y2],'
        '"left_eye":[x,y],"right_eye":[x,y],"mouth":[x,y],'
        '"eye_width":N,"mouth_width":N}\n\n'
        "Rules (checked automatically — violations cause fallback):\n"
        "- subject_bbox MUST fully contain every patch rectangle, not just "
        "  the center points.\n"
        "- 'left_eye' is the eye on the viewer's LEFT (lower x).\n"
        "- Eyes sit on roughly the same y (within a few percent of canvas).\n"
        "- IOD (|right_eye.x − left_eye.x|) ≤ 0.6 × bbox_width. If you're "
        "  tempted to exceed that, the bbox is too wide — shrink it to the "
        "  real face surface. Wide-spread eyes reaching onto background is "
        "  the #1 failure mode.\n"
        "- Mouth y strictly below both eye y values.\n"
        "- Mouth x between the two eye x values.\n"
        f"- eye_width = width of ONE eye on the base (normally "
        f"{int(CANVAS*0.09)}–{int(CANVAS*0.20)}; small for distant subjects, "
        "larger for close-ups). When in doubt, pick the smaller end.\n"
        f"- mouth_width = width of the mouth on the base (normally "
        f"{int(CANVAS*0.16)}–{int(CANVAS*0.38)}). Usually ≈ IOD + eye_width; "
        "never wider than bbox_width minus a safety gap.\n"
        "- Keep all points at least 40px from the image border.\n"
        "- SAFETY MARGINS inside the bbox (clean surface for the blend):\n"
        "    left_eye.x  ≥ bbox.x1 + 0.6 × eye_width\n"
        "    right_eye.x ≤ bbox.x2 − 0.6 × eye_width\n"
        "    eye.y       ≥ bbox.y1 + 0.4 × eye_width   (both eyes)\n"
        "    mouth.x ± 0.55 × mouth_width ⊆ [bbox.x1, bbox.x2]\n"
        "    mouth.y + 0.35 × mouth_width ≤ bbox.y2\n"
        "- Eyes must not overlap: IOD > eye_width.\n"
        "- Face proportions: vertical gap eyes→mouth ≈ 1.2–1.8 × IOD.\n\n"
        "Study the image carefully. Before returning, mentally draw each "
        "patch rectangle on the image and confirm every one sits cleanly "
        "inside the chosen surface. If any rectangle clips an edge or "
        "crosses onto a different surface, shrink the widths or tighten "
        "the bbox. Return only JSON."
    )

    client = OpenAI(
        api_key=os.environ["ZHIPU_API_KEY"],
        base_url=GLM_BASE_URL,
        timeout=GLM_TIMEOUT_S,
    )
    last_err: Exception | None = None
    for attempt in range(1, GLM_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=GLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{ext};base64,{b64}",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            content = resp.choices[0].message.content or ""
            raw = _extract_json_object(content)
            if raw is None:
                log.warning(
                    "GLM attempt %d returned unparseable content for %s: %r",
                    attempt, img_path.name, content[:300],
                )
            else:
                validated = validate_coords(raw)
                if validated is not None:
                    log.info("GLM coords for %s: %s", img_path.name, validated)
                    return validated
                log.warning(
                    "GLM attempt %d returned unusable coords for %s: %s",
                    attempt, img_path.name, raw,
                )
        except Exception as e:  # openai errors, json errors, network, etc.
            last_err = e
            log.warning("GLM attempt %d failed: %s", attempt, e)
        if attempt < GLM_RETRIES:
            time.sleep(0.6 * attempt)

    log.warning(
        "GLM gave up after %d attempts (last: %s) for %s",
        GLM_RETRIES, last_err, img_path.name,
    )
    return None


def load_or_detect_coords(img_path: Path) -> dict:
    """Try the cache first; otherwise ask GLM. Never raises. Defaults are
    returned (not cached) when GLM fails — a future server restart will retry.

    When a v{COORDS_VERSION} cache is missing but an older v{N<COORDS_VERSION}
    cache exists and still validates under the current rules, promote it to
    the current version rather than pay for a fresh GLM detection. This is the
    common case right after a COORDS_VERSION bump."""
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
    # Migrate older-version caches when the schema is still compatible. Walk
    # newest → oldest so a v4 wins over a v3 if both exist. `None` is the
    # pre-versioning format `{stem}.{hash}.coords.json`.
    candidates: list[Path] = [
        img_path.with_suffix(f".v{older}.{img_hash}.coords.json")
        for older in range(COORDS_VERSION - 1, 0, -1)
    ] + [img_path.with_suffix(f".{img_hash}.coords.json")]
    for old_cache in candidates:
        if not old_cache.exists():
            continue
        try:
            migrated = validate_coords(json.loads(old_cache.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
        if migrated is None:
            continue
        try:
            cache.write_text(json.dumps(migrated, indent=2))
            log.info(
                "migrated %s → v%d cache for %s",
                old_cache.name, COORDS_VERSION, img_path.name,
            )
        except OSError as e:
            log.warning("cache promote failed (%s) for %s", e, img_path.name)
        return migrated
    coords = ask_glm_for_coords(img_path)
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
    # EMA weight on the current frame's landmarks. Lower = smoother but laggier.
    SMOOTH_ALPHA = 0.55
    # Skip smoothing if the previous sample is older than this (face moved/lost).
    SMOOTH_STALE_S = 0.25
    # Skip smoothing if mean landmark jump is too large (scene cut, new face).
    SMOOTH_JUMP_PX = 60.0

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            str(ROOT / "shape_predictor_68_face_landmarks.dat")
        )
        self.lock = threading.Lock()
        self.base_img: np.ndarray | None = None
        self.coords: dict | None = None
        self.base_name: str | None = None
        self.base_label: str | None = None
        # Temporal landmark smoothing state (see _smooth_shape).
        self.shape_lock = threading.Lock()
        self.last_shape: np.ndarray | None = None
        self.last_shape_at: float = 0.0
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
        label = coords.get("label") if isinstance(coords, dict) else None
        if not isinstance(label, str) or not label.strip():
            label = filename_to_label(safe)
        with self.lock:
            self.base_img = img
            self.coords = coords
            self.base_name = safe
            self.base_label = label
        log.info("base loaded: %s label=%r coords=%s", safe, label, coords)

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
            raw_shape = face_utils.shape_to_np(self.predictor(frame_bgr, face))
            shape = self._smooth_shape(raw_shape)
            base = self._paste_eyes(frame_bgr, shape, base, coords)
            base = self._paste_mouth(frame_bgr, shape, base, coords)
        except Exception as e:
            log.warning("composite failed: %s", e)
        return base, True

    def _smooth_shape(self, shape: np.ndarray) -> np.ndarray:
        """EMA-smooth 68-point landmarks across frames to suppress dlib jitter.
        Falls back to the raw shape when the previous sample is stale or when
        landmarks jumped too far (new face / scene cut)."""
        now = time.monotonic()
        with self.shape_lock:
            prev = self.last_shape
            prev_at = self.last_shape_at
            can_smooth = (
                prev is not None
                and prev.shape == shape.shape
                and now - prev_at < self.SMOOTH_STALE_S
            )
            if can_smooth:
                # mean pixel distance between corresponding landmarks
                diff = float(
                    np.abs(
                        prev.astype(np.float32) - shape.astype(np.float32)
                    ).mean()
                )
                if diff < self.SMOOTH_JUMP_PX:
                    a = self.SMOOTH_ALPHA
                    smoothed = (
                        a * shape.astype(np.float32)
                        + (1 - a) * prev.astype(np.float32)
                    ).astype(shape.dtype)
                else:
                    smoothed = shape
            else:
                smoothed = shape
            self.last_shape = smoothed
            self.last_shape_at = now
        return smoothed

    @staticmethod
    def _blend_mask(h: int, w: int, rx_frac: float = 0.47, ry_frac: float = 0.45) -> np.ndarray:
        """Elliptical mask for seamlessClone. Feathers the patch edges so the
        blend doesn't show a rectangular seam. rx/ry fractions leave a small
        black border so the cloning gradient has skin/base to blend into."""
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (w // 2, h // 2),
            (max(1, int(w * rx_frac)), max(1, int(h * ry_frac))),
            0, 0, 360, 255, -1,
        )
        return mask

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
            left = crop(36, 39, 37, 41, 0.28)
            right = crop(42, 45, 43, 47, 0.28)
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
                mask = Compositor._blend_mask(patch.shape[0], patch.shape[1])
                base = cv2.seamlessClone(
                    patch, base, mask, target, cv2.NORMAL_CLONE,
                )
            except cv2.error:
                pass
        return base

    @staticmethod
    def _paste_mouth(src, shape, base, coords):
        try:
            mx1, mx2 = int(shape[48, 0]), int(shape[54, 0])
            my1, my2 = int(shape[50, 1]), int(shape[57, 1])
            # Extra margin — the ellipse mask needs skin to blend into so the
            # paste doesn't look like a cut-out rectangle.
            margin_x = int(max(0, (mx2 - mx1) * 0.20))
            margin_y = int(max(0, (mx2 - mx1) * 0.28))
            y1c, y2c = max(0, my1 - margin_y), min(src.shape[0], my2 + margin_y)
            x1c, x2c = max(0, mx1 - margin_x), min(src.shape[1], mx2 + margin_x)
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
            # Slightly wider ellipse for the mouth — mouths are more horizontal
            # than square so the blend reads more naturally with a wider ry.
            mask = Compositor._blend_mask(
                patch.shape[0], patch.shape[1], rx_frac=0.48, ry_frac=0.46
            )
            base = cv2.seamlessClone(
                patch, base, mask, target, cv2.NORMAL_CLONE,
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


@app.middleware("http")
async def log_requests(request, call_next):
    """Per-request access log with wall-clock timing. Skips /health to
    keep the log readable under liveness-probe traffic."""
    path = request.url.path
    if path == "/health":
        return await call_next(request)
    t0 = time.time()
    log.info("[http] ▶ %s %s", request.method, path)
    try:
        response = await call_next(request)
    except Exception as e:
        log.warning(
            "[http] ✖ %s %s after %dms: %s",
            request.method,
            path,
            int((time.time() - t0) * 1000),
            e,
        )
        raise
    log.info(
        "[http] ◀ %s %s %d in %dms",
        request.method,
        path,
        response.status_code,
        int((time.time() - t0) * 1000),
    )
    return response

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
        "zhipu_key_present": "ZHIPU_API_KEY" in os.environ,
        "glm_model": GLM_MODEL,
        "glm_base_url": GLM_BASE_URL,
    }


@app.get("/bases")
def bases():
    items = []
    for p in sorted(ROOT.iterdir()):
        if p.suffix.lower() not in ALLOWED_EXT:
            continue
        label = _cached_label(p) or filename_to_label(p.name)
        items.append({"name": p.name, "label": label})
    return {
        "bases": items,
        "current": compositor.base_name,
        "current_label": compositor.base_label,
    }


@app.get("/base-image/{name}")
def base_image(name: str):
    """Serve the raw base file so the client can optimistically preview a
    chip-swap before the WS delivers the first composite on the new base."""
    safe = Path(name).name
    path = (ROOT / safe).resolve()
    if not _within_root(path) or not path.exists():
        raise HTTPException(404, f"base not found: {safe}")
    if path.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(400, "bad extension")
    return FileResponse(path)


@app.post("/base/{name}")
def set_base(name: str):
    try:
        compositor.load_base(name)
    except FileNotFoundError:
        raise HTTPException(404, f"base not found: {name}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "ok": True,
        "base": compositor.base_name,
        "label": compositor.base_label,
        "coords": compositor.coords,
    }


@app.post("/base/{name}/retry")
def retry_base(name: str):
    """Re-run GLM coord detection for `name`. If it's the currently active
    base, applies the new coords in place. Otherwise just refreshes the cache
    file so a future swap picks up the new spot — no surprise auto-swap."""
    safe = Path(name).name
    path = (ROOT / safe).resolve()
    if not _within_root(path) or not path.exists():
        raise HTTPException(404, f"base not found: {safe}")
    cleared: list[str] = []
    for cache in ROOT.glob(f"{path.stem}.v*.*.coords.json"):
        try:
            cache.unlink()
            cleared.append(cache.name)
        except OSError as e:
            log.warning("could not delete %s: %s", cache.name, e)

    is_active = compositor.base_name == safe
    if is_active:
        try:
            compositor.load_base(safe)
        except ValueError as e:
            raise HTTPException(400, str(e))
        coords = compositor.coords
        label = compositor.base_label
    else:
        # Refresh cache silently — don't steal the user's active view.
        coords = load_or_detect_coords(path)
        raw_label = coords.get("label") if isinstance(coords, dict) else None
        label = raw_label if isinstance(raw_label, str) and raw_label.strip() else filename_to_label(safe)

    return {
        "ok": True,
        "base": compositor.base_name,
        "label": label,
        "name": safe,
        "activated": is_active,
        "coords": coords,
        "cleared": cleared,
    }


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
    return {
        "ok": True,
        "base": compositor.base_name,
        "label": compositor.base_label,
        "coords": compositor.coords,
    }


async def _composite_async(frame_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, compositor.composite, frame_bgr)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    last_face_state: bool | None = None
    # Rolling FPS / composite-time window so the log shows cadence without
    # spamming per-frame lines. 100-frame window ~= every 3-5 seconds.
    frame_count = 0
    window_start = time.time()
    window_composite_ms = 0.0
    window_face_frames = 0
    log.info("[ws/mirror] ▶ connected")
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
                frame_t0 = time.time()
                try:
                    out, has_face = await _composite_async(frame)
                except Exception:
                    log.exception("composite pipeline failed")
                    continue
                window_composite_ms += (time.time() - frame_t0) * 1000

                ok_enc, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if not ok_enc:
                    continue
                try:
                    await websocket.send_bytes(enc.tobytes())
                except Exception:
                    break
                frame_count += 1
                if has_face:
                    window_face_frames += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - window_start
                    fps = 100 / elapsed if elapsed > 0 else 0
                    avg_ms = window_composite_ms / 100
                    log.info(
                        "[ws/mirror] ◦ frames=%d fps=%.1f avg-composite=%.1fms face-ratio=%d/100",
                        frame_count,
                        fps,
                        avg_ms,
                        window_face_frames,
                    )
                    window_start = time.time()
                    window_composite_ms = 0.0
                    window_face_frames = 0

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
        log.info("[ws/mirror] ◀ closed frames=%d", frame_count)


@app.websocket("/ws/yolo")
async def ws_yolo(websocket: WebSocket):
    await yolo_ws_handler(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
