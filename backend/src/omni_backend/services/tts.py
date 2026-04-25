"""Cartesia-only TTS passthrough used by ``/api/tts/stream`` and the
combined ``/api/speak`` route.

No fallback providers by design — the old Cartesia → Fish → OpenAI
cascade was what caused the "long dead air / no audio at all"
behaviour: an aggressive Cartesia timeout would abort a request that
was about to succeed, bounce through Fish (usually unconfigured), and
finally land on OpenAI tts-1/nova for another ~1–2s of silence.
Timeout is generous (120s) so a slow-but-real response wins instead
of being aborted.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Literal, Optional, Union

import httpx

logger = logging.getLogger(__name__)

CARTESIA_TTS_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_VERSION = "2024-11-13"
CARTESIA_TTFB_TIMEOUT_MS = 120_000

TtsBackend = Literal["cartesia"]

VALID_EMOTIONS = {"anger", "positivity", "surprise", "sadness", "curiosity"}
VALID_INTENSITIES = {"lowest", "low", "high", "highest"}
VALID_SPEEDS = {"slowest", "slow", "normal", "fast", "fastest"}

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


@dataclass
class StreamTtsResult:
    stream: AsyncIterator[bytes]
    backend: TtsBackend


def sanitize_emotion(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        trimmed = item.strip().lower()
        if not trimmed:
            continue
        parts = trimmed.split(":", 1)
        name = parts[0]
        intensity = parts[1] if len(parts) > 1 else ""
        if name not in VALID_EMOTIONS:
            continue
        if intensity and intensity not in VALID_INTENSITIES:
            continue
        out.append(f"{name}:{intensity}" if intensity else name)
        if len(out) >= 3:
            break
    return out


def sanitize_speed(raw: Any) -> Union[str, float, None]:
    if isinstance(raw, bool):
        # JS `typeof true === "boolean"`, not "number" — exclude explicitly.
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return max(-1.0, min(1.0, value))
    if isinstance(raw, str):
        trimmed = raw.strip().lower()
        if trimmed in VALID_SPEEDS:
            return trimmed
    return None


def fix_mojibake(s: str) -> str:
    """Recover UTF-8 text that was double-decoded as latin-1.

    Detects every char sitting in the 0x80–0xFF range (no codepoints
    above) and, if the re-byte-then-decode-as-UTF-8 round trip yields
    real CJK, returns the recovered text. On any ambiguity we leave
    the original alone.
    """
    if not s:
        return s
    suspect = 0
    for ch in s:
        c = ord(ch)
        if c > 0xFF:
            return s
        if c >= 0x80:
            suspect += 1
    if suspect < 3 or suspect < len(s) * 0.3:
        return s
    try:
        raw_bytes = bytes(ord(ch) for ch in s)
        decoded = raw_bytes.decode("utf-8")
        if _CJK_RE.search(decoded):
            logger.info(
                "[tts] \u26a0 mojibake recovered: %s\u2026 \u2192 %s\u2026",
                s[:40],
                decoded[:40],
            )
            return decoded
    except (UnicodeDecodeError, ValueError):
        # Not valid UTF-8 — leave it.
        pass
    return s


async def stream_tts(
    text: str,
    voice_id: Optional[str] = None,
    lang: Optional[str] = None,
    turn_id: Optional[str] = None,
    emotion: Optional[Iterable[str]] = None,
    speed: Union[str, float, None] = None,
) -> Optional[StreamTtsResult]:
    """Return the Cartesia audio stream, or ``None`` when the key is
    missing or Cartesia returned a non-audio response. The caller 503s
    in that case — there is no secondary provider anymore.
    """
    text = fix_mojibake(text or "").strip()[:600]
    if not text:
        return None
    voice_id_s = (voice_id or "").strip()
    lang_s = "zh" if lang == "zh" else "en"
    turn = (turn_id or "?")[:16] or "?"
    tag = f" #{turn}"
    emotion_list = list(emotion) if emotion else []

    cartesia_key_raw = os.environ.get("CARTESIA_API_KEY")
    cartesia_key = cartesia_key_raw.strip() if cartesia_key_raw else ""
    if not cartesia_key:
        logger.info("[tts cartesia%s] \u2716 CARTESIA_API_KEY not set", tag)
        return None

    model_id_raw = os.environ.get("CARTESIA_MODEL_ID")
    model_id = (model_id_raw.strip() if model_id_raw else "") or "sonic-3"

    if lang_s == "zh":
        voice_env = os.environ.get("CARTESIA_VOICE_ID_ZH")
        default_voice = (
            voice_env.strip() if voice_env else ""
        ) or "0cd0cde2-3b93-42b5-bcb9-f214a591aa29"
    else:
        voice_env = os.environ.get("CARTESIA_VOICE_ID_EN")
        default_voice = (
            voice_env.strip() if voice_env else ""
        ) or "a0e99841-438c-4a64-b679-ae501e7d6091"
    chosen_voice = voice_id_s if _UUID_RE.match(voice_id_s) else default_voice

    voice_payload: dict[str, Any] = {"mode": "id", "id": chosen_voice}
    if emotion_list or speed is not None:
        controls: dict[str, Any] = {}
        if emotion_list:
            controls["emotion"] = emotion_list
        if speed is not None:
            controls["speed"] = speed
        voice_payload["experimental_controls"] = controls

    t0 = time.monotonic()
    logger.info(
        "[tts cartesia%s] \u2192 model=%s voice=%s lang=%s text=%dch emotion=[%s] speed=%s",
        tag,
        model_id,
        chosen_voice,
        lang_s,
        len(text),
        ",".join(emotion_list) if emotion_list else "-",
        speed if speed is not None else "-",
    )

    body = {
        "model_id": model_id,
        "transcript": text,
        "voice": voice_payload,
        "output_format": {
            "container": "mp3",
            "bit_rate": 128000,
            "sample_rate": 44100,
        },
        "language": "zh" if lang_s == "zh" else "en",
    }
    headers = {
        "X-API-Key": cartesia_key,
        "Cartesia-Version": CARTESIA_VERSION,
        "Content-Type": "application/json",
    }
    ttfb_s = CARTESIA_TTFB_TIMEOUT_MS / 1000.0
    timeout = httpx.Timeout(ttfb_s, connect=ttfb_s, read=ttfb_s, write=ttfb_s)

    client = httpx.AsyncClient(timeout=timeout)
    try:
        req = client.build_request(
            "POST", CARTESIA_TTS_URL, headers=headers, json=body
        )
        response = await client.send(req, stream=True)
    except Exception as err:  # noqa: BLE001
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "[tts cartesia%s] \u2716 exception after %dms: %s",
            tag,
            elapsed_ms,
            str(err),
        )
        await client.aclose()
        return None

    ct = response.headers.get("content-type", "") or ""
    if response.status_code < 400 and "application/json" not in ct:
        ttfb_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "[tts cartesia%s] \u2713 ttfb=%dms streaming audio/mpeg", tag, ttfb_ms
        )

        async def _iter() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return StreamTtsResult(stream=_iter(), backend="cartesia")

    try:
        err_bytes = await response.aread()
        err_text = err_bytes.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        err_text = ""
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "[tts cartesia%s] \u2716 %d in %dms: %s",
        tag,
        response.status_code,
        elapsed_ms,
        err_text[:200],
    )
    await response.aclose()
    await client.aclose()
    return None
