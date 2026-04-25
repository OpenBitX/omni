"""Runware image generation client + prompt builder.

The /gallery page displays comic-book style illustrations of the physical
objects the user tapped in the tracker. We feed Runware a prompt built
from the card's className + persona description + a tiny summary of the
conversation topic, and return one 512x512 JPEG URL.

Keep this file server-only -- ``generate_comic_image`` reads the API key
from the environment and is invoked from the FastAPI Runware route.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import httpx
from openai import AsyncOpenAI

log = logging.getLogger("runware")


class ChatTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass
class RunwareGenerateInput:
    class_name: str
    description: str
    # Recent conversation turns, newest last, already truncated by caller.
    history: list[ChatTurn] = field(default_factory=list)
    spoken_lang: Literal["en", "zh"] | None = None
    learn_lang: Literal["en", "zh"] | None = None
    # Optional data URL of the cropped object. When provided + OPENAI_API_KEY
    # is set, we ask gpt-4o-mini to craft the comic prompt directly from the
    # image (it has seen the actual object, including all the specific
    # details a heuristic prompt misses -- chewed straw, dust on the screen,
    # a peeling sticker). Falls back to the heuristic if anything goes wrong.
    image_data_url: str | None = None


@dataclass
class RunwareGenerateResult:
    ok: bool
    image_url: str | None = None
    prompt: str | None = None
    prompt_source: Literal["vlm", "heuristic"] | None = None
    error: str | None = None


RUNWARE_ENDPOINT = "https://api.runware.ai/v1"
RUNWARE_MODEL = "runware:100@1"
REQUEST_TIMEOUT_MS = 20_000

# Max allowed image data URL size (base64 inflation ~1.37x). 3 MB covers
# every crop the tracker produces (typical: 30-180 KB). Anything bigger is
# almost certainly a mistake -- fail fast rather than ship it to OpenAI.
MAX_IMAGE_DATA_URL_BYTES = 3_000_000

# Art-style opener we expect every prompt to start with. If the VLM
# forgets it (it sometimes does), we prepend it so Runware stays on-brand.
# Gallery page composites these onto a wooden bookshelf via mix-blend-mode:
# multiply, so solid pure-white backgrounds disappear into the wood tone.
STYLE_OPENER = (
    "bold comic-book illustration, thick black outlines, flat cel-shading, "
    "retro manga × kawaii energy, vivid pastel palette."
)

# Tail appended to every prompt -- forces the die-cut silhouette so the
# composited thumbnail reads as a sticker sitting on the shelf, not a
# framed painting.
DIE_CUT_TAIL = (
    "full character portrait, die-cut sticker, centered, isolated on solid "
    "pure white (#ffffff) background, no ground, no shadow, no scenery, "
    "clean product-shot lighting. No text, no speech bubbles, no watermarks."
)


# --- Prompt construction --------------------------------------------------

def sanitize_snippet(s: str) -> str:
    """Strip/normalize a snippet for inclusion in the prompt.

    We don't want stray quotes or newlines confusing the image model.
    """
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'`]", "", s)
    return s.strip()


def summarize_history(history: list[ChatTurn]) -> str:
    """Deterministic one-sentence summary of the conversation topic.

    If there are no user turns yet, fall back to the generic intro line.
    Otherwise compress the last user + last assistant turn into ~120 chars.
    """
    has_user = any(t.get("role") == "user" for t in history)
    if not has_user:
        return "introducing itself, looking playful."

    # Walk from the tail to find the most recent user and assistant turns.
    last_user: str | None = None
    last_assistant: str | None = None
    for t in reversed(history):
        if not t or not t.get("content"):
            continue
        if last_user is None and t.get("role") == "user":
            last_user = t["content"]
        elif last_assistant is None and t.get("role") == "assistant":
            last_assistant = t["content"]
        if last_user and last_assistant:
            break

    parts: list[str] = []
    if last_user:
        parts.append(f"user says {sanitize_snippet(last_user)}")
    if last_assistant:
        parts.append(f"it replies {sanitize_snippet(last_assistant)}")
    joined = "; ".join(parts)
    if not joined:
        return "introducing itself, looking playful."

    if len(joined) > 120:
        capped = joined[:117].rstrip() + "..."
    else:
        capped = joined
    # Ensure the scene context reads as a sentence fragment we can slot in.
    return capped if capped.endswith(".") else f"{capped}."


def build_comic_prompt(inp: RunwareGenerateInput) -> str:
    class_name = sanitize_snippet(inp.class_name) or "object"
    description = sanitize_snippet(inp.description) or "expressive character"
    scene = summarize_history(inp.history or [])

    return " ".join([
        STYLE_OPENER,
        f"Center subject: {class_name}, {description}.",
        f"Scene context: {scene}",
        "Anthropomorphized — give it a face, eyes, mouth, expressive posture.",
        DIE_CUT_TAIL,
    ])


# --- VLM-crafted prompt (lets the model that saw the object write the
# prompt). We feed it the actual crop + the persona card + the latest
# conversation turn, and ask for a specific, fun, comic-style Runware
# prompt. Falls back to the heuristic when the VLM errors or the JSON
# comes back malformed.
# --------------------------------------------------------------------------

VLM_PROMPT_MODEL = (os.environ.get("RUNWARE_PROMPT_MODEL") or "").strip() or "gpt-4o-mini"
VLM_PROMPT_TIMEOUT_MS = 12_000

VLM_PROMPT_SYSTEM = """You are a comic-book art director. You will see a cropped photo of a real physical object and a short persona card about it. You write ONE image prompt for a text-to-image model that will produce a fun, specific cartoon portrait of THIS object — not a generic one. The result will be composited onto a wooden bookshelf as a die-cut sticker, so the background MUST be solid pure white.

Rules:
- 55–90 words. One paragraph. No line breaks.
- Start with the art style: "bold comic-book illustration, thick black outlines, flat cel-shading, retro manga × kawaii energy, vivid pastel palette."
- Then describe the subject. Lock in 2–3 hyper-specific visual details you can actually see in the photo (a chip, a smudge, a sticker, a worn edge, a color, a reflection). These details are what make the image feel like a portrait of THIS object instead of a stock cartoon.
- Give the object a face + posture + mood that matches the persona card and the conversation topic. Anthropomorphize: give it eyes, mouth, little arms if it fits.
- Centered composition, full figure, subject fills the frame, expressive, alive.
- End with EXACTLY: "full character portrait, die-cut sticker, centered, isolated on solid pure white (#ffffff) background, no ground, no shadow, no scenery, clean product-shot lighting. No text, no speech bubbles, no watermarks."
- Do NOT include real brand names or logos. Paraphrase ("sleek black phone" not "iPhone 15 Pro").
- Return JSON: {"prompt": "..."}."""


def extract_prompt_field(raw: str | None) -> str | None:
    if not raw:
        return None
    trimmed = raw.strip()
    # Strip code fences if present.
    unfenced = re.sub(r"^```(?:json)?\s*", "", trimmed, flags=re.IGNORECASE)
    unfenced = re.sub(r"```\s*$", "", unfenced, flags=re.IGNORECASE)
    # Find the first {...} block.
    start = unfenced.find("{")
    end = unfenced.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(unfenced[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    prompt = obj.get("prompt") if isinstance(obj, dict) else None
    if isinstance(prompt, str) and len(prompt.strip()) > 20:
        return prompt.strip()
    return None


# Module-level OpenAI client; created once per process on first use so we
# don't pay connection churn per request.
_openai: AsyncOpenAI | None = None


def _get_openai() -> AsyncOpenAI | None:
    global _openai
    if _openai is not None:
        return _openai
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    _openai = AsyncOpenAI(api_key=key, timeout=VLM_PROMPT_TIMEOUT_MS / 1000.0)
    return _openai


def normalize_crafted_prompt(raw: str) -> str:
    """Ensure the crafted prompt still starts with the brand art-style opener
    and ends with the die-cut / safety clause. VLMs sometimes drop one or
    the other when they get creative.
    """
    compact = re.sub(r"\s+", " ", raw).strip()
    with_opener = (
        compact
        if re.match(r"bold comic-book", compact, flags=re.IGNORECASE)
        else f"{STYLE_OPENER} {compact}"
    )
    with_die_cut = (
        with_opener
        if re.search(
            r"die-cut|isolated on .*white|pure white.*background",
            with_opener,
            flags=re.IGNORECASE,
        )
        else f"{with_opener} {DIE_CUT_TAIL}"
    )
    with_safety = (
        with_die_cut
        if re.search(
            r"no text|no speech bubbles|no watermarks",
            with_die_cut,
            flags=re.IGNORECASE,
        )
        else f"{with_die_cut} No text, no speech bubbles, no watermarks."
    )
    return with_safety[:1400] if len(with_safety) > 1400 else with_safety


async def _run_with_cancel(coro: Any, cancel: asyncio.Event | None) -> Any:
    """Race ``coro`` against ``cancel``; raise CancelledError if cancel wins."""
    if cancel is None:
        return await coro
    task = asyncio.ensure_future(coro)
    cancel_task = asyncio.ensure_future(cancel.wait())
    try:
        done, _pending = await asyncio.wait(
            {task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if cancel_task in done and task not in done:
            task.cancel()
            try:
                await task
            except BaseException:
                pass
            raise asyncio.CancelledError()
        cancel_task.cancel()
        return task.result()
    finally:
        if not cancel_task.done():
            cancel_task.cancel()


async def craft_prompt_from_image(
    inp: RunwareGenerateInput,
    cancel: asyncio.Event | None = None,
) -> str | None:
    client = _get_openai()
    if client is None:
        return None
    if not inp.image_data_url or not inp.image_data_url.startswith("data:image/"):
        return None
    if len(inp.image_data_url) > MAX_IMAGE_DATA_URL_BYTES:
        # Too large -- the crop is absurd, don't pay the OpenAI vision bill.
        return None

    description = sanitize_snippet(inp.description) or "expressive character"
    class_name = sanitize_snippet(inp.class_name) or "object"
    topic = summarize_history(inp.history or [])

    user_text = "\n".join([
        f"Object class (YOLO guess): {class_name}",
        f"Persona card: {description}",
        f"Conversation so far: {topic}",
        "",
        "Look at the image. Pick 2–3 specific visual details you see. Then write the comic prompt per the system rules. Return JSON.",
    ])

    try:
        resp = await _run_with_cancel(
            client.chat.completions.create(
                model=VLM_PROMPT_MODEL,
                response_format={"type": "json_object"},
                max_tokens=420,
                temperature=0.85,
                messages=[
                    {"role": "system", "content": VLM_PROMPT_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": inp.image_data_url,
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
            ),
            cancel,
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 -- match TS best-effort behaviour
        log.debug("VLM prompt craft failed: %s", e)
        return None

    try:
        raw = resp.choices[0].message.content if resp.choices else None
    except (AttributeError, IndexError):
        raw = None
    prompt = extract_prompt_field(raw if isinstance(raw, str) else None)
    if not prompt:
        return None
    return normalize_crafted_prompt(prompt)


# --- Runware HTTP call ----------------------------------------------------

async def generate_comic_image(
    inp: RunwareGenerateInput,
    cancel: asyncio.Event | None = None,
) -> RunwareGenerateResult:
    api_key = os.environ.get("RUNWARE_API_KEY")
    if not api_key:
        return RunwareGenerateResult(ok=False, error="RUNWARE_API_KEY missing")

    # Prefer the VLM-crafted prompt when we have the image crop; that model
    # can lock in specific visible details ("chipped yellow lid", "peeling
    # sticker on the back") that the heuristic prompt can't reach.
    try:
        crafted = await craft_prompt_from_image(inp, cancel)
    except asyncio.CancelledError:
        return RunwareGenerateResult(ok=False, error="cancelled")

    prompt = crafted if crafted is not None else build_comic_prompt(inp)
    prompt_source: Literal["vlm", "heuristic"] = "vlm" if crafted else "heuristic"
    task_uuid = str(uuid.uuid4())

    body = [
        {
            "taskType": "imageInference",
            "taskUUID": task_uuid,
            "positivePrompt": prompt,
            "model": RUNWARE_MODEL,
            "width": 384,
            "height": 384,
            "numberResults": 1,
            "outputType": "URL",
            # WEBP at moderate quality keeps the file well under ~40 KB so
            # the gallery tile renders immediately instead of showing the
            # alt-text filename while a ~1 MB PNG streams in. The
            # whiteToAlpha SVG filter in globals.css thresholds near-white
            # pixels, so mild compression artefacts near the silhouette
            # still get clipped.
            "outputFormat": "WEBP",
            "outputQuality": 72,
            "checkNSFW": False,
        }
    ]

    timeout_s = REQUEST_TIMEOUT_MS / 1000.0
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            res = await _run_with_cancel(
                client.post(RUNWARE_ENDPOINT, headers=headers, json=body),
                cancel,
            )
    except asyncio.CancelledError:
        return RunwareGenerateResult(
            ok=False, error="cancelled", prompt_source=prompt_source
        )
    except httpx.TimeoutException:
        return RunwareGenerateResult(
            ok=False, error="timeout", prompt_source=prompt_source
        )
    except httpx.HTTPError as err:
        message = str(err) or "network error"
        return RunwareGenerateResult(
            ok=False, error=message, prompt_source=prompt_source
        )

    if res.status_code >= 400:
        # Try to surface the provider's error message but don't crash on bad JSON.
        detail = f"runware http {res.status_code}"
        try:
            j = res.json()
            errs = j.get("errors") if isinstance(j, dict) else None
            if isinstance(errs, list) and errs:
                msg = errs[0].get("message") if isinstance(errs[0], dict) else None
                if msg:
                    detail = msg
        except (ValueError, json.JSONDecodeError):
            try:
                t = res.text
                if t:
                    detail = t[:200]
            except Exception:  # noqa: BLE001
                pass
        return RunwareGenerateResult(
            ok=False, error=detail, prompt_source=prompt_source
        )

    try:
        payload = res.json()
    except (ValueError, json.JSONDecodeError):
        return RunwareGenerateResult(
            ok=False, error="invalid runware response", prompt_source=prompt_source
        )

    errors = payload.get("errors") if isinstance(payload, dict) else None
    if isinstance(errors, list) and len(errors) > 0:
        first = errors[0] if isinstance(errors[0], dict) else {}
        msg = first.get("message") or first.get("code") or "runware error"
        return RunwareGenerateResult(
            ok=False, error=msg, prompt_source=prompt_source
        )

    image_url: str | None = None
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        for d in data:
            if isinstance(d, dict) and d.get("imageURL"):
                image_url = d["imageURL"]
                break
    if not image_url:
        return RunwareGenerateResult(
            ok=False, error="no image url in response", prompt_source=prompt_source
        )

    return RunwareGenerateResult(
        ok=True,
        image_url=image_url,
        prompt=prompt,
        prompt_source=prompt_source,
    )
