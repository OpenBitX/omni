"""GLM-based face coordinate detection for Mirror base images.

Extracted from the original server/server.py. All public functions are
pure (no global state) so they can be tested without mocking the compositor.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

log = logging.getLogger("coords")

CANVAS = 512
COORDS_VERSION = 5
EYE_WIDTH_MIN, EYE_WIDTH_MAX = 60, 210
MOUTH_WIDTH_MIN, MOUTH_WIDTH_MAX = 100, 380
GLM_TIMEOUT_S = 60.0
GLM_RETRIES = 3
GLM_MODEL = "glm-5v-turbo"

_FILENAME_PREFIX_RE = re.compile(
    r"^(screenshot|img|image|pxl|photo|dsc|capture)[\s_-]*", re.IGNORECASE,
)
_FILENAME_DATE_RE = re.compile(
    r"\d{4}[-_]\d{2}[-_]\d{2}"
    r"(?:[\s_-]+at[\s_-]+\d{1,2}[-_:]\d{2}(?:[-_:]\d{2})?(?:[\s_-]*[ap]m)?)?",
    re.IGNORECASE,
)


def default_coords() -> dict:
    return {
        "left_eye": [int(CANVAS * 0.36), int(CANVAS * 0.42)],
        "right_eye": [int(CANVAS * 0.64), int(CANVAS * 0.42)],
        "mouth": [int(CANVAS * 0.50), int(CANVAS * 0.68)],
        "eye_width": int(CANVAS * 0.18),
        "mouth_width": int(CANVAS * 0.30),
    }


def filename_to_label(name: str) -> str:
    original = Path(name).stem
    stem = _FILENAME_PREFIX_RE.sub("", original)
    stem = _FILENAME_DATE_RE.sub("", stem)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" -_")
    if len(stem) < 2:
        m = _FILENAME_PREFIX_RE.match(original)
        if m:
            return m.group(1).title()
        return (original[:20] or "Image").strip() or "Image"
    if len(stem) > 24:
        stem = stem[:22].rstrip() + "…"
    if stem.islower() or stem.isupper():
        stem = stem.title()
    return stem


def _clamp_point(p, margin: int = 40) -> list[int]:
    try:
        x, y = int(p[0]), int(p[1])
    except (TypeError, ValueError, IndexError):
        return [CANVAS // 2, CANVAS // 2]
    return [max(margin, min(CANVAS - margin, x)), max(margin, min(CANVAS - margin, y))]


def _parse_bbox(raw) -> tuple[int, int, int, int] | None:
    try:
        x1, y1, x2, y2 = (int(v) for v in raw)
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0, min(CANVAS, x1)), max(0, min(CANVAS, x2))))
    y1, y2 = sorted((max(0, min(CANVAS, y1)), max(0, min(CANVAS, y2))))
    w, h = x2 - x1, y2 - y1
    if w < CANVAS * 0.15 or h < CANVAS * 0.15:
        return None
    if w > CANVAS * 0.98 and h > CANVAS * 0.98:
        return None
    return (x1, y1, x2, y2)


def validate_coords(raw: dict) -> dict | None:
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

    if out["left_eye"][0] > out["right_eye"][0]:
        out["left_eye"], out["right_eye"] = out["right_eye"], out["left_eye"]

    iod = abs(out["right_eye"][0] - out["left_eye"][0])
    if iod < 40:
        log.warning("coords rejected: eyes too close (iod=%d)", iod)
        return None

    eyes_y = (out["left_eye"][1] + out["right_eye"][1]) / 2
    if out["mouth"][1] <= eyes_y + 10:
        log.warning("coords rejected: mouth not below eyes")
        return None

    eyes_cx = (out["left_eye"][0] + out["right_eye"][0]) / 2
    if abs(out["mouth"][0] - eyes_cx) > max(iod * 0.5, CANVAS * 0.10):
        log.warning(
            "coords rejected: mouth x (%d) not between eyes (cx=%.0f iod=%d)",
            out["mouth"][0], eyes_cx, iod,
        )
        return None

    if abs(out["left_eye"][1] - out["right_eye"][1]) > CANVAS * 0.12:
        log.warning("coords rejected: eyes not level")
        return None

    bbox = _parse_bbox(raw.get("subject_bbox"))
    if bbox is None:
        log.warning("coords rejected: missing/invalid subject_bbox")
        return None
    x1, y1, x2, y2 = bbox
    pad = 6
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
            "coords rejected: IOD=%d too wide for bbox width %d",
            iod, bbox_w,
        )
        return None
    out["subject_bbox"] = list(bbox)

    eye_w = raw.get("eye_width")
    eye_w = int(eye_w) if isinstance(eye_w, (int, float)) else int(iod * 0.55)
    eye_w = min(eye_w, int(iod * 0.95))
    out["eye_width"] = max(EYE_WIDTH_MIN, min(EYE_WIDTH_MAX, eye_w))

    mouth_w = raw.get("mouth_width")
    mouth_w = int(mouth_w) if isinstance(mouth_w, (int, float)) else int(iod * 1.4)
    out["mouth_width"] = max(MOUTH_WIDTH_MIN, min(MOUTH_WIDTH_MAX, mouth_w))

    label = raw.get("label")
    if isinstance(label, str):
        cleaned = label.strip().strip('"\'.,;:!?()[]{}').strip()
        if len(cleaned) > 28:
            cleaned = cleaned[:26].rstrip() + "…"
        if cleaned:
            out["label"] = cleaned

    return out


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def _build_coord_prompt() -> str:
    return (
        f"You are placing a composite of a live person's eyes and mouth onto "
        f"this base image. The base will be scaled to exactly {CANVAS}x{CANVAS} "
        "pixels before compositing. Origin is the top-left pixel (0,0); x "
        "increases rightward, y increases downward.\n\n"
        "Return ONLY this JSON (no prose, no markdown fence):\n"
        '{"label":"ShortName",'
        '"subject_bbox":[x1,y1,x2,y2],'
        '"left_eye":[x,y],"right_eye":[x,y],"mouth":[x,y],'
        '"eye_width":N,"mouth_width":N}\n\n'
        "Rules: left_eye has lower x; mouth strictly below eyes; all features "
        "inside subject_bbox; IOD ≤ 0.6 × bbox_width; keep points ≥ 40px from border."
    )


def ask_glm_for_coords(img_path: Path, glm_api_key: str, glm_base_url: str) -> dict | None:
    """Call GLM for face placement coords. Returns None on any failure."""
    try:
        img_bytes = img_path.read_bytes()
    except OSError as e:
        log.warning("read failed for %s (%s)", img_path.name, e)
        return None

    b64 = base64.b64encode(img_bytes).decode()
    ext = img_path.suffix.lstrip(".").lower() or "jpeg"
    if ext == "jpg":
        ext = "jpeg"

    from openai import OpenAI

    client = OpenAI(api_key=glm_api_key, base_url=glm_base_url, timeout=GLM_TIMEOUT_S)
    last_err: Exception | None = None
    for attempt in range(1, GLM_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=GLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
                        {"type": "text", "text": _build_coord_prompt()},
                    ],
                }],
            )
            content = resp.choices[0].message.content or ""
            raw = _extract_json_object(content)
            if raw is None:
                log.warning("GLM attempt %d returned unparseable content: %r", attempt, content[:300])
            else:
                validated = validate_coords(raw)
                if validated is not None:
                    log.info("GLM coords for %s: %s", img_path.name, validated)
                    return validated
                log.warning("GLM attempt %d returned unusable coords: %s", attempt, raw)
        except Exception as e:
            last_err = e
            log.warning("GLM attempt %d failed: %s", attempt, e)
        if attempt < GLM_RETRIES:
            time.sleep(0.6 * attempt)

    log.warning("GLM gave up after %d attempts (last: %s)", GLM_RETRIES, last_err)
    return None


def cached_label(img_path: Path) -> str | None:
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


def load_or_detect_coords(
    img_path: Path,
    glm_api_key: str | None,
    glm_base_url: str,
) -> dict:
    """Try disk cache first, then GLM, then defaults. Never raises."""
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
        except (OSError, json.JSONDecodeError) as e:
            log.warning("cache read failed (%s), re-detecting: %s", e, cache.name)

    # Try migrating older-version caches
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
            log.info("migrated %s → v%d cache", old_cache.name, COORDS_VERSION)
        except OSError as e:
            log.warning("cache promote failed (%s)", e)
        return migrated

    if glm_api_key:
        coords = ask_glm_for_coords(img_path, glm_api_key, glm_base_url)
        if coords is not None:
            try:
                cache.write_text(json.dumps(coords, indent=2))
            except OSError as e:
                log.warning("cache write failed: %s", e)
            return coords

    log.warning("falling back to default coords (uncached) for %s", img_path.name)
    return default_coords()
