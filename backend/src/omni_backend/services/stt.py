"""Server-side speech-to-text.

All STT runs on the backend now — there is no longer any browser-side
Whisper / transformers.js model load. The client simply uploads the
recorded audio blob (webm/opus on Chrome & Firefox, mp4/aac on Safari)
and gets back a transcript.

Default backend is OpenAI's ``gpt-4o-mini-transcribe`` — Whisper-family
accuracy with ~300-600ms latency on short mic turns. Override via
``OPENAI_STT_MODEL`` (e.g. ``whisper-1`` for cost/latency trade-offs, or
``gpt-4o-transcribe`` for higher accuracy at higher cost).

Public API:
    transcribe_audio(audio_bytes, *, lang=None, mime_type=None,
                     prompt=None, turn_tag="") -> TranscribeResult

Returns an empty string + ``backend="none"`` when no STT provider is
configured (missing ``OPENAI_API_KEY``); callers decide whether that's
a hard error or a fallback-to-"huh?" path.
"""
from __future__ import annotations

import io
import logging
import os
import time
from typing import Optional, TypedDict

from openai import AsyncOpenAI

logger = logging.getLogger("stt")


class TranscribeResult(TypedDict):
    text: str
    backend: str
    ms: int


# Model + timeout knobs. OpenAI's transcribe endpoints accept webm/opus,
# mp4/aac, wav, flac, etc. directly — no client-side decode needed.
OPENAI_STT_MODEL = (
    (os.environ.get("OPENAI_STT_MODEL") or "").strip() or "gpt-4o-mini-transcribe"
)
OPENAI_STT_TIMEOUT_MS = int(os.environ.get("OPENAI_STT_TIMEOUT_MS", "30000"))

# Client is lazily created so importing this module doesn't fail when
# OPENAI_API_KEY is missing at import time (tests, TEST_MODE).
_openai_client: Optional[AsyncOpenAI] = None


def _get_openai_client() -> Optional[AsyncOpenAI]:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    _openai_client = AsyncOpenAI(
        api_key=key, timeout=OPENAI_STT_TIMEOUT_MS / 1000.0
    )
    return _openai_client


# Map a MediaRecorder Blob.type (e.g. ``audio/webm;codecs=opus``) to a
# filename extension OpenAI's multipart endpoint likes. The extension is
# the ONLY thing OpenAI looks at to pick a decoder, so we err on the side
# of giving it something unambiguous.
def _filename_for_mime(mime_type: Optional[str]) -> str:
    mt = (mime_type or "").lower()
    if "webm" in mt:
        return "audio.webm"
    if "ogg" in mt:
        return "audio.ogg"
    if "mp4" in mt or "m4a" in mt or "aac" in mt:
        return "audio.m4a"
    if "wav" in mt:
        return "audio.wav"
    if "mpeg" in mt or "mp3" in mt:
        return "audio.mp3"
    if "flac" in mt:
        return "audio.flac"
    # Safe default — OpenAI's server sniffs the container anyway.
    return "audio.webm"


async def transcribe_audio(
    audio_bytes: bytes,
    *,
    lang: Optional[str] = None,
    mime_type: Optional[str] = None,
    prompt: Optional[str] = None,
    turn_tag: str = "",
) -> TranscribeResult:
    """Transcribe a single mic-recording blob.

    ``lang`` is an ISO-639-1 hint (``"en"`` / ``"zh"``) that improves
    accuracy + latency on very short utterances. ``mime_type`` is the
    browser's ``Blob.type`` so we can pick a correct file extension.
    Empty audio or missing ``OPENAI_API_KEY`` return ``text=""``.
    """
    tag = turn_tag or ""
    size = len(audio_bytes) if audio_bytes else 0
    if size == 0:
        return {"text": "", "backend": "none", "ms": 0}

    client = _get_openai_client()
    if client is None:
        logger.info("[stt%s] ∅ OPENAI_API_KEY not set — skipping STT", tag)
        return {"text": "", "backend": "none", "ms": 0}

    # The OpenAI SDK accepts a (filename, bytes, content_type) tuple as
    # ``file``. We pass the raw bytes in an in-memory buffer so we don't
    # spill the recording to disk.
    buf = io.BytesIO(audio_bytes)
    buf.name = _filename_for_mime(mime_type)

    kwargs: dict = {
        "file": (buf.name, buf.getvalue(), mime_type or "application/octet-stream"),
        "model": OPENAI_STT_MODEL,
        "response_format": "json",
    }
    if lang in ("en", "zh"):
        kwargs["language"] = lang
    if prompt:
        # OpenAI caps prompts at ~224 tokens — cap here defensively.
        kwargs["prompt"] = prompt[:800]

    t0 = time.time()
    try:
        resp = await client.audio.transcriptions.create(**kwargs)
    except Exception as e:  # noqa: BLE001
        ms = int((time.time() - t0) * 1000)
        logger.warning(
            "[stt%s] ✖ openai/%s failed in %dms: %s",
            tag,
            OPENAI_STT_MODEL,
            ms,
            e,
        )
        return {"text": "", "backend": f"openai/{OPENAI_STT_MODEL}", "ms": ms}

    ms = int((time.time() - t0) * 1000)
    text = (getattr(resp, "text", None) or "").strip()
    trunc = text[:120] + ("…" if len(text) > 120 else "")
    logger.info(
        '[stt%s] ✓ openai/%s %dms (%dB, %s) → "%s"',
        tag,
        OPENAI_STT_MODEL,
        ms,
        size,
        lang or "auto",
        trunc,
    )
    return {"text": text, "backend": f"openai/{OPENAI_STT_MODEL}", "ms": ms}
