"""Unified HTTP API router — replaces the old Express server.

Endpoints (all mounted at /api/*):
  POST /api/assess            → llm.assess_object
  POST /api/describe          → llm.describe_object
  POST /api/generate-line     → llm.generate_line
  POST /api/group-line        → llm.group_line
  POST /api/converse          → llm.converse_with_object           (multipart)
  POST /api/transcribe        → stt.transcribe_audio               (multipart)
  POST /api/teacher-say       → llm.teacher_say
  POST /api/gallerize-card    → llm.gallerize_card
  POST /api/tts/stream        → tts.stream_tts                     (audio stream)
  POST /api/speak             → generate_line + stream_tts         (audio stream)
  POST /api/runware/generate  → runware.generate_comic_image

Frontend contract lives in lib/api-client.ts — camelCase JSON fields are
the product contract; do NOT snake_case them on the wire.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from omni_backend.services import llm, runware, stt, tts

logger = logging.getLogger("api")

router = APIRouter(prefix="/api")


# 1×1 transparent PNG, used when the client forgets imageDataUrl on generate-line.
_BLANK_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _parse_lang(value: Any) -> Optional[str]:
    if value == "zh" or value == "en":
        return value
    return None


def _tag_from_body(body: dict, key: str = "tag", limit: int = 32) -> Optional[str]:
    v = body.get(key)
    if isinstance(v, str):
        t = v.strip()[:limit]
        return t or None
    return None


def _error_response(exc: BaseException, status: int = 500) -> JSONResponse:
    msg = str(exc) or exc.__class__.__name__
    return JSONResponse({"error": msg}, status_code=status)


# ── /api/assess ───────────────────────────────────────────────────────

@router.post("/assess")
async def assess(request: Request):
    body = await request.json()
    if not isinstance(body, dict) or not isinstance(body.get("imageDataUrl"), str):
        return JSONResponse({"error": "imageDataUrl required"}, status_code=400)
    try:
        result = await llm.assess_object(
            image_data_url=body["imageDataUrl"],
            tap_x=float(body.get("tapX") or 0) or 0.0,
            tap_y=float(body.get("tapY") or 0) or 0.0,
            tag=_tag_from_body(body),
            learn_lang=_parse_lang(body.get("learnLang")),
        )
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning("[api /assess] %s", e)
        return _error_response(e)


# ── /api/describe ─────────────────────────────────────────────────────

@router.post("/describe")
async def describe(request: Request):
    body = await request.json()
    if not isinstance(body, dict) or not isinstance(body.get("imageDataUrl"), str):
        return JSONResponse({"error": "imageDataUrl required"}, status_code=400)
    try:
        result = await llm.describe_object(
            image_data_url=body["imageDataUrl"],
            class_name=str(body.get("className") or ""),
            lang="zh" if body.get("lang") == "zh" else "en",
            tag=_tag_from_body(body),
        )
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning("[api /describe] %s", e)
        return _error_response(e)


# ── /api/generate-line ────────────────────────────────────────────────

def _resolve_langs(body: dict) -> tuple[str, str, str]:
    """Return (legacy_lang, spoken_lang, learn_lang) with the same
    fallback semantics as the old Express route."""
    lang = "zh" if body.get("lang") == "zh" else "en"
    spoken_provided = body.get("spokenLang") in ("zh", "en")
    learn_provided = body.get("learnLang") in ("zh", "en")
    spoken = body["spokenLang"] if spoken_provided else lang
    if learn_provided:
        learn = body["learnLang"]
    elif spoken_provided:
        learn = "en" if spoken == "zh" else "zh"
    else:
        learn = lang
    return lang, spoken, learn


@router.post("/generate-line")
async def generate_line_endpoint(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse({"error": "bad body"}, status_code=400)

    raw_image = body.get("imageDataUrl")
    image_data_url = (
        raw_image if isinstance(raw_image, str) and raw_image.startswith("data:image/")
        else _BLANK_PNG
    )
    voice_id = body.get("voiceId") if isinstance(body.get("voiceId"), str) else None
    description = body.get("description") if isinstance(body.get("description"), str) else None
    history = body.get("history") if isinstance(body.get("history"), list) else []
    lang, spoken, learn = _resolve_langs(body)
    tag = _tag_from_body(body)

    try:
        result = await llm.generate_line(
            image_data_url=image_data_url,
            voice_id=voice_id,
            description=description,
            history=history,
            lang=lang,
            tag=tag,
            spoken_lang=spoken,
            learn_lang=learn,
        )
        return {
            "line": result["line"],
            "voiceId": result["voiceId"],
            "description": result["description"],
            "name": result["name"],
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("[api /generate-line] %s", e)
        return _error_response(e)


# ── /api/group-line ───────────────────────────────────────────────────

@router.post("/group-line")
async def group_line_endpoint(request: Request):
    try:
        body = await request.json()
        result = await llm.group_line(body or {})
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning("[api /group-line] %s", e)
        return _error_response(e)


# ── /api/converse ─────────────────────────────────────────────────────
# multipart/form-data: audio (blob) + text fields.

@router.post("/converse")
async def converse(
    audio: UploadFile = File(...),
    className: str = Form(""),
    voiceId: Optional[str] = Form(None),
    history: Optional[str] = Form(None),
    lang: Optional[str] = Form(None),
    spokenLang: Optional[str] = Form(None),
    learnLang: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    transcript: Optional[str] = Form(None),
    turnId: Optional[str] = Form(None),
):
    audio_bytes = await audio.read()

    history_list: list[dict] = []
    if history:
        try:
            parsed = json.loads(history)
            if isinstance(parsed, list):
                history_list = parsed
        except (json.JSONDecodeError, ValueError):
            history_list = []

    try:
        result = await llm.converse_with_object(
            audio_bytes=audio_bytes,
            class_name=className or "",
            voice_id=voiceId,
            history=history_list,
            lang=_parse_lang(lang),
            spoken_lang=_parse_lang(spokenLang),
            learn_lang=_parse_lang(learnLang),
            description=description,
            transcript=transcript or "",
            turn_id=turnId,
            audio_mime_type=audio.content_type,
        )
        return result
    except ValueError as e:
        return _error_response(e, status=400)
    except Exception as e:  # noqa: BLE001
        logger.warning("[api /converse] %s", e)
        return _error_response(e)


# ── /api/transcribe ───────────────────────────────────────────────────
# Standalone STT: upload an audio blob, get back { text, backend, ms }.
# Useful when the client wants to display what the user said before the
# /api/converse round-trip finishes.

@router.post("/transcribe")
async def transcribe_endpoint(
    audio: UploadFile = File(...),
    lang: Optional[str] = Form(None),
):
    audio_bytes = await audio.read()
    if not audio_bytes:
        return JSONResponse({"error": "missing audio"}, status_code=400)
    try:
        result = await stt.transcribe_audio(
            audio_bytes,
            lang=_parse_lang(lang),
            mime_type=audio.content_type,
        )
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning("[api /transcribe] %s", e)
        return _error_response(e)


# ── /api/teacher-say ──────────────────────────────────────────────────

@router.post("/teacher-say")
async def teacher_say_endpoint(request: Request):
    try:
        body = await request.json()
        result = await llm.teacher_say(body or {})
        return result
    except ValueError as e:
        return _error_response(e, status=400)
    except Exception as e:  # noqa: BLE001
        logger.warning("[api /teacher-say] %s", e)
        return _error_response(e)


# ── /api/gallerize-card ───────────────────────────────────────────────

@router.post("/gallerize-card")
async def gallerize_card_endpoint(request: Request):
    try:
        body = await request.json()
        result = await llm.gallerize_card(body or {})
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning("[api /gallerize-card] %s", e)
        return _error_response(e)


# ── /api/tts/stream ───────────────────────────────────────────────────

@router.post("/tts/stream")
async def tts_stream(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse({"error": "bad body"}, status_code=400)

    raw_text = body.get("text")
    text = raw_text.strip()[:600] if isinstance(raw_text, str) else ""
    voice_id = body.get("voiceId").strip() if isinstance(body.get("voiceId"), str) else ""
    lang = "zh" if (isinstance(body.get("lang"), str) and body["lang"].strip() == "zh") else "en"
    turn_id = body.get("turnId").strip()[:16] if isinstance(body.get("turnId"), str) else ""
    turn_id = turn_id or "?"
    if not text:
        return JSONResponse({"error": "missing text"}, status_code=400)

    emotion = tts.sanitize_emotion(body.get("emotion"))
    speed = tts.sanitize_speed(body.get("speed"))

    try:
        result = await tts.stream_tts(
            text=text, voice_id=voice_id, lang=lang, turn_id=turn_id,
            emotion=emotion, speed=speed,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[api /tts/stream] %s", e)
        return _error_response(e)

    if result is None:
        return JSONResponse({"error": "no TTS backend configured"}, status_code=503)

    return StreamingResponse(
        result.stream,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Tts-Backend": result.backend,
        },
    )


# ── /api/speak ────────────────────────────────────────────────────────

@router.post("/speak")
async def speak(request: Request):
    payload = await request.json()
    if not isinstance(payload, dict):
        return JSONResponse({"error": "bad body"}, status_code=400)

    image_data_url = payload.get("imageDataUrl")
    if not (isinstance(image_data_url, str) and image_data_url.startswith("data:image/")):
        return JSONResponse({"error": "expected data:image/ URL"}, status_code=400)

    voice_id = payload.get("voiceId")
    voice_id = voice_id.strip() if isinstance(voice_id, str) and voice_id.strip() else None

    description = payload.get("description")
    description = description.strip() if isinstance(description, str) and description.strip() else None

    history_raw = payload.get("history") if isinstance(payload.get("history"), list) else []
    history: list[dict] = []
    for item in history_raw:
        if not isinstance(item, dict):
            continue
        r = item.get("role")
        c = item.get("content")
        if (r == "user" or r == "assistant") and isinstance(c, str):
            history.append({"role": r, "content": c})
    history = history[-32:]

    lang, spoken, learn = _resolve_langs(payload)
    turn_id = payload.get("turnId")
    turn_id = turn_id.strip()[:16] if isinstance(turn_id, str) and turn_id.strip() else "?"
    turn_id = turn_id or "?"

    try:
        gen = await llm.generate_line(
            image_data_url=image_data_url,
            voice_id=voice_id,
            description=description,
            history=history,
            lang=lang,
            tag=turn_id,
            spoken_lang=spoken,
            learn_lang=learn,
        )
    except Exception as e:  # noqa: BLE001
        return _error_response(e, status=502)

    line = gen["line"]
    chosen_voice_id = gen["voiceId"]
    chosen_description = gen["description"]
    chosen_name = gen["name"]

    try:
        tts_result = await tts.stream_tts(
            text=line,
            voice_id=chosen_voice_id or "",
            lang=learn,
            turn_id=turn_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[api /speak tts] %s", e)
        tts_result = None

    meta_json = json.dumps({
        "line": line,
        "voiceId": chosen_voice_id,
        "description": chosen_description,
        "name": chosen_name,
        "spokenLang": spoken,
        "learnLang": learn,
        "teachMode": False,
    }, ensure_ascii=False)
    meta_b64 = base64.b64encode(meta_json.encode("utf-8")).decode("ascii")

    if tts_result is None:
        return Response(
            status_code=204,
            headers={
                "X-Speak-Meta": meta_b64,
                "Cache-Control": "no-store",
            },
        )

    return StreamingResponse(
        tts_result.stream,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Speak-Meta": meta_b64,
            "X-Tts-Backend": tts_result.backend,
        },
    )


# ── /api/runware/generate ─────────────────────────────────────────────

@router.post("/runware/generate")
async def runware_generate(request: Request):
    payload = await request.json()
    if not isinstance(payload, dict):
        return JSONResponse({"error": "bad body"}, status_code=400)

    card_id = payload.get("cardId")
    card_id = card_id.strip() if isinstance(card_id, str) else ""
    class_name = payload.get("className")
    class_name = class_name.strip() if isinstance(class_name, str) else ""
    description = payload.get("description")
    description = description.strip() if isinstance(description, str) else ""

    if not card_id:
        return JSONResponse({"error": "cardId required"}, status_code=400)
    if not class_name:
        return JSONResponse({"error": "className required"}, status_code=400)
    if not description:
        return JSONResponse({"error": "description required"}, status_code=400)

    history_raw = payload.get("history") if isinstance(payload.get("history"), list) else []
    history: list[dict] = []
    for v in history_raw:
        if not isinstance(v, dict):
            continue
        r = v.get("role")
        c = v.get("content")
        if (r == "user" or r == "assistant") and isinstance(c, str):
            history.append({"role": r, "content": c})
    history = history[-8:]

    img = payload.get("imageDataUrl")
    image_data_url: Optional[str] = None
    if isinstance(img, str) and img.startswith("data:image/") and len(img) <= 4_000_000:
        image_data_url = img

    inp = runware.RunwareGenerateInput(
        class_name=class_name,
        description=description,
        history=history,
        spoken_lang=_parse_lang(payload.get("spokenLang")),
        learn_lang=_parse_lang(payload.get("learnLang")),
        image_data_url=image_data_url,
    )

    cancel = asyncio.Event()

    async def run_with_budget() -> runware.RunwareGenerateResult:
        try:
            return await asyncio.wait_for(
                runware.generate_comic_image(inp, cancel=cancel),
                timeout=35.0,
            )
        except asyncio.TimeoutError:
            cancel.set()
            return runware.RunwareGenerateResult(ok=False, error="timeout")

    try:
        result = await run_with_budget()
    except Exception as e:  # noqa: BLE001
        logger.warning("[api /runware/generate] %s", e)
        return _error_response(e)

    if not result.ok:
        err = result.error or "unknown"
        status = (
            503 if err == "RUNWARE_API_KEY missing"
            else 499 if err == "cancelled"
            else 504 if err == "timeout"
            else 500
        )
        return JSONResponse({"error": err}, status_code=status)

    return {
        "imageUrl": result.image_url,
        "prompt": result.prompt,
        "promptSource": result.prompt_source,
    }
