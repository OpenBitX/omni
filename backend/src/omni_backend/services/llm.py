"""LLM service — Python port of ``app/actions.ts``.

This module is the backend replacement for the Next.js server actions. It
talks to GLM (Zhipu / z.ai), Cerebras (Llama), and OpenAI through the
OpenAI-compatible Async SDK. Prompts are copied VERBATIM from the TS
source — they're product-critical assets.

Exports:

- ``assess_object``   — vision: is this tappable + face placement
- ``describe_object`` — vision: short persona description
- ``generate_line``   — bundled first-tap OR retap text-only reply
- ``group_line``      — text-only line for a group-chat speaker
- ``converse_with_object`` — voice-in, voice-out conversation turn
- ``teacher_say``     — gallery bilingual teacher reply
- ``gallerize_card``  — bilingual card hydration
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from typing import Any, Literal, Optional, TypedDict

from openai import AsyncOpenAI

from omni_backend.services import stt as stt_service

logger = logging.getLogger("llm")

# === Types =================================================================

Lang = Literal["en", "zh"]
AppMode = Literal["play", "language", "history"]


class VoiceCatalogEntry(TypedDict):
    id: str
    name: str
    vibe: str
    lang: Lang


class ChatTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class Assessment(TypedDict):
    suitable: bool
    cx: float
    cy: float
    bbox: list[float]  # [x0, y0, x1, y1]
    reason: str


class BundledResult(TypedDict):
    line: str
    voiceId: Optional[str]
    description: Optional[str]
    name: Optional[str]


class GenerateLineResult(TypedDict):
    line: str
    voiceId: Optional[str]
    description: Optional[str]
    name: Optional[str]


class GroupPeer(TypedDict):
    name: str
    description: Optional[str]


class GroupTurn(TypedDict, total=False):
    speaker: str
    line: str
    role: Literal["user", "assistant"]


class GroupLineResult(TypedDict):
    line: str
    emotion: Optional[str]
    speed: Optional[str]
    addressing: Optional[str]
    backend: str
    ms: int


class ConverseResult(TypedDict, total=False):
    transcript: str
    reply: str
    voiceId: Optional[str]
    emotion: Optional[str]
    speed: Optional[str]
    teachMode: bool


class TeacherSayResult(TypedDict):
    line: str
    voiceId: str
    turnId: str
    teachMode: bool


class BilingualIntro(TypedDict):
    learn: str
    spoken: str


class GallerizeCardResult(TypedDict):
    translatedName: str
    bilingualIntro: BilingualIntro


# === Fish.audio voice catalog =============================================
#
# Hand-curated list of funny voices. IDs are hardcoded below — they're
# Fish.audio reference_ids, not secrets. GLM picks one per object on the
# first ``generateLine`` call (vision-grounded on the crop). The client
# pins that choice onto the track so every follow-up line and conversation
# turn on the same object uses the SAME voice. No mid-conversation flips.

VOICE_CATALOG: list[VoiceCatalogEntry] = [
    {
        "id": "98655a12fa944e26b274c535e5e03842",
        "name": "EGirl",
        "vibe": "breathy, coy, chronically-online uptalk — suits phones, mirrors, ring lights, laptops, makeup, vanity items, anything a streamer would film",
        "lang": "en",
    },
    {
        "id": "03397b4c4be74759b72533b663fbd001",
        "name": "Elon",
        "vibe": "halting, smug tech-bro cadence with long awkward pauses — suits computers, cars, rockets, expensive gadgets, anything 'disruptive' or overengineered",
        "lang": "en",
    },
    {
        "id": "b70e5f4d550647eb9927359d133c8e3a",
        "name": "Anime Girl",
        "vibe": "high-pitched, hyper-kawaii, rapid squeals — suits plushies, stuffed toys, cute mugs, candy, snacks, anything bright and small",
        "lang": "en",
    },
    {
        "id": "59e9dc1cb20c452584788a2690c80970",
        "name": "Talking girl",
        "vibe": "natural conversational young woman — suits friendly everyday objects without strong personality: books, notebooks, bags, chairs, lamps",
        "lang": "en",
    },
    {
        "id": "fb43143e46f44cc6ad7d06230215bab6",
        "name": "Girl conversation vibe",
        "vibe": "gossipy, laid-back, best-friend-texting energy — suits couches, beds, pillows, coffee cups, comfort snacks, anything cozy",
        "lang": "en",
    },
    {
        "id": "0cd6cf9684dd4cc9882fbc98957c9b1d",
        "name": "Elephant",
        "vibe": "rumbling, low, heavy, deliberate — suits big heavy things: fridges, trash cans, dressers, couches, vending machines, anything massive",
        "lang": "en",
    },
    {
        "id": "48484faae07e4cfdb8064da770ee461e",
        "name": "Sonic",
        "vibe": "fast, cocky, blue-hedgehog swagger — suits shoes, sneakers, skateboards, bikes, running gear, anything about speed or movement",
        "lang": "en",
    },
    {
        "id": "d13f84b987ad4f22b56d2b47f4eb838e",
        "name": "Mortal Kombat",
        "vibe": "gravelly, ominous, arena-announcer drama — suits knives, scissors, staplers, weapons, sharp tools, anything that could hurt you",
        "lang": "en",
    },
    {
        "id": "d75c270eaee14c8aa1e9e980cc37cf1b",
        "name": "Peter Griffin",
        "vibe": "goofy Rhode-Island drawl laughing at himself — default go-to for anything ordinary: food, drinks, random household clutter, everyman objects",
        "lang": "en",
    },
    # --- Chinese (Mandarin) voices ----------------------------------------
    {
        "id": "6ce7ea8ada884bf3889fa7c7fb206691",
        "name": "中文 A",
        "vibe": "mandarin voice",
        "lang": "zh",
    },
    {
        "id": "b4f70fdef5f943c2bf43db00e80ad680",
        "name": "中文 B",
        "vibe": "mandarin voice",
        "lang": "zh",
    },
    {
        "id": "e855dc04a51f48549b484e41c4d4d4cc",
        "name": "中文 C",
        "vibe": "mandarin voice",
        "lang": "zh",
    },
]

DEFAULT_VOICE_ID_EN = "d75c270eaee14c8aa1e9e980cc37cf1b"  # Peter Griffin
DEFAULT_VOICE_ID_ZH = "6ce7ea8ada884bf3889fa7c7fb206691"  # 中文 A


def normalize_lang(value: Any) -> Lang:
    return "zh" if value == "zh" else "en"


def lang_label(lang: Lang) -> str:
    return "Simplified Chinese (简体中文)" if lang == "zh" else "English"


# === Teach-mode detection =================================================

TEACH_TRIGGERS_EN: list[re.Pattern[str]] = [
    re.compile(r"\bhow\s+do\s+(?:you|i|we)\s+say\b", re.IGNORECASE),
    re.compile(r"\bhow\s+would\s+you\s+say\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+does\s+[^?]+\bmean\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+[^?]+\s+in\s+(?:english|chinese|mandarin)\b", re.IGNORECASE),
    re.compile(r"\btranslat(?:e|ion|ing)\b", re.IGNORECASE),
    re.compile(r"\bin\s+english\b", re.IGNORECASE),
    re.compile(r"\bin\s+chinese\b", re.IGNORECASE),
    re.compile(r"\bin\s+mandarin\b", re.IGNORECASE),
    re.compile(r"\bwhat['\u2019]?s\s+the\s+word\s+for\b", re.IGNORECASE),
    re.compile(r"\bcorrect\s+me\b", re.IGNORECASE),
    re.compile(r"\bteach\s+me\b", re.IGNORECASE),
    re.compile(r"\bpronounc(?:e|iation|ing)\b", re.IGNORECASE),
    re.compile(r"\bspell(?:ing)?\b", re.IGNORECASE),
]

TEACH_TRIGGERS_ZH: list[re.Pattern[str]] = [
    re.compile(r"怎么说"),
    re.compile(r"怎么讲"),
    re.compile(r"什么意思"),
    re.compile(r"是什么意思"),
    re.compile(r"翻译"),
    re.compile(r"用英文"),
    re.compile(r"用英语"),
    re.compile(r"用中文"),
    re.compile(r"用普通话"),
    re.compile(r"教我"),
    re.compile(r"怎么写"),
    re.compile(r"怎么读"),
    re.compile(r"纠正我"),
]


def _has_teach_trigger(text: str) -> bool:
    if not text:
        return False
    for r in TEACH_TRIGGERS_EN:
        if r.search(text):
            return True
    for r in TEACH_TRIGGERS_ZH:
        if r.search(text):
            return True
    return False


def detect_teach_mode(
    user_utterance: str,
    history: list[ChatTurn],
    spoken_lang: Lang,
    learn_lang: Lang,
) -> bool:
    if spoken_lang == learn_lang:
        return False
    if _has_teach_trigger(user_utterance):
        return True
    user_turns = [t for t in history if t.get("role") == "user"]
    for turn in user_turns[-3:]:
        if _has_teach_trigger(turn.get("content", "")):
            return True
    return False


# === Voice helpers =========================================================

def get_voice_catalog(lang: Lang = "en") -> list[VoiceCatalogEntry]:
    return [v for v in VOICE_CATALOG if v["lang"] == lang]


def get_default_voice_id(lang: Lang = "en") -> str:
    return DEFAULT_VOICE_ID_ZH if lang == "zh" else DEFAULT_VOICE_ID_EN


def pick_random_voice_id(lang: Lang) -> str:
    catalog = get_voice_catalog(lang)
    if not catalog:
        return get_default_voice_id(lang)
    return random.choice(catalog)["id"]


def voice_by_id(voice_id: Optional[str]) -> Optional[VoiceCatalogEntry]:
    if not voice_id:
        return None
    for v in VOICE_CATALOG:
        if v["id"] == voice_id:
            return v
    return None


# === Model constants ======================================================
#
# Single-model strategy: ``glm-5v-turbo`` for everything. We had briefly
# split into a DEEP (assess) / FAST (hot-path) pair but the fast model on
# Zhipu (``glm-4v-flash``) was silently deprecated — probing on
# 2026-04-18 showed only ``glm-4.5v`` and ``glm-5v-turbo`` still
# responding. Going back to one model keeps the wiring simple; override at
# will via env.

GLM_MODEL_DEEP = (os.environ.get("GLM_MODEL_DEEP") or "").strip() or "glm-5v-turbo"
GLM_MODEL_FAST = (os.environ.get("GLM_MODEL_FAST") or "").strip() or "glm-5v-turbo"
GLM_TIMEOUT_MS = 90_000
GLM_RETRIES = 2
DESCRIBE_MODEL = (os.environ.get("GLM_DESCRIBE_MODEL") or "").strip() or "glm-5v-turbo"
GENERATE_BUNDLED_MODEL_GLM = (
    (os.environ.get("GLM_BUNDLED_MODEL") or "").strip() or "glm-5v-turbo"
)
GENERATE_BUNDLED_MODEL_OPENAI = (
    (os.environ.get("OPENAI_BUNDLED_MODEL") or "").strip() or "gpt-4o-mini"
)
REPLY_MODEL_CEREBRAS = (
    (os.environ.get("CEREBRAS_REPLY_MODEL") or "").strip() or "llama3.1-8b"
)
REPLY_MODEL_OPENAI = (
    (os.environ.get("OPENAI_REPLY_MODEL") or "").strip() or "gpt-4o-mini"
)
CONVERSE_HISTORY_CAP = 32
GROUP_HISTORY_CAP = 24


# === Client factories =====================================================


def _glm_base_url(key: str) -> str:
    override = os.environ.get("ZHIPU_BASE_URL")
    if override:
        return override
    looks_bigmodel = "." in key and not key.startswith("sk-")
    return (
        "https://open.bigmodel.cn/api/paas/v4/"
        if looks_bigmodel
        else "https://api.z.ai/api/paas/v4/"
    )


def _get_glm_client() -> AsyncOpenAI:
    key = (
        os.environ.get("ZHIPU_API_KEY")
        or os.environ.get("GLM_API_KEY")
        or os.environ.get("BIGMODEL_API_KEY")
    )
    if not key:
        raise RuntimeError("ZHIPU_API_KEY not set")
    return AsyncOpenAI(
        api_key=key,
        base_url=_glm_base_url(key),
        timeout=GLM_TIMEOUT_MS / 1000.0,
    )


def _get_openai_client() -> Optional[AsyncOpenAI]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    return AsyncOpenAI(api_key=key)


def _get_cerebras_client() -> Optional[AsyncOpenAI]:
    key = (os.environ.get("CEREBRAS_API_KEY") or "").strip()
    if not key:
        return None
    return AsyncOpenAI(api_key=key, base_url="https://api.cerebras.ai/v1")


def _is_transient_fetch_error(err: BaseException) -> bool:
    msg = str(err).lower()
    if re.search(r"\b[45]\d\d\b", msg):
        return False
    return (
        "fetch failed" in msg
        or "econnreset" in msg
        or "etimedout" in msg
        or "enotfound" in msg
        or "eai_again" in msg
        or "socket hang up" in msg
        or "network" in msg
        or "connection" in msg
        or "timeout" in msg
    )


def _classify_error(err: BaseException) -> str:
    msg = str(err).lower()
    status_match = re.search(r"\b([45]\d\d)\b", msg)
    status = status_match.group(1) if status_match else None
    if "invalid_argument" in msg or "unknown name" in msg:
        return f"{status}:schema" if status else "schema"
    if "not found" in msg or status == "404":
        return f"{status}:model-missing" if status else "model-missing"
    if "unauthor" in msg or status == "401" or status == "403":
        return f"{status}:auth" if status else "auth"
    if "quota" in msg or "rate" in msg or status == "429":
        return f"{status}:rate-limit" if status else "rate-limit"
    if "timeout" in msg or "etimedout" in msg:
        return "timeout"
    if "parse" in msg or "empty line" in msg or "json" in msg:
        return "parse"
    if _is_transient_fetch_error(err):
        return "transient"
    if status:
        return status
    return "error"


def _voice_catalog_prompt_block(lang: Lang = "en") -> str:
    catalog = get_voice_catalog(lang)
    if not catalog:
        return ""
    lines = "\n".join(
        f'  {i + 1}. id="{v["id"]}" — {v["vibe"]}' for i, v in enumerate(catalog)
    )
    return (
        "Voice catalog (pick the id whose vibe best matches this object's "
        "personality + visible state — a dusty thing wants a weary voice; a "
        "shiny thing wants a peppy one; a sharp thing wants a cutting voice, "
        f"etc.):\n{lines}"
    )


# === Prompts ==============================================================

ASSESS_SYSTEM_BASE = """You place a cartoon face on whatever the user tapped. The user has already framed the subject by tapping — trust that and commit.

You will be shown a CROP of a scene the user is pointing at. Their tap is at normalized coordinate (tx, ty) in [0, 1] inside the crop — top-left is (0, 0).

Return STRICT JSON only:
{"suitable": boolean, "cx": number, "cy": number, "bbox": [number, number, number, number], "reason": string}

DEFAULT TO suitable=true. Be generous. If there is any identifiable thing in the crop — even something weird, partial, textured, dim, slightly blurry, or abstract — commit, pick a good face spot, return suitable=true. Motion blur, soft focus, odd lighting, unusual angles, close-ups of texture, partial crops — all of these are FINE. Just find the best face placement you can. The user already decided they want a face on this thing.

cx, cy: normalized crop coords (0..1) for the BEST spot to plant a cartoon face (eyes + mouth). Rules:
- Pick a relatively flat, central region of the subject — not an edge or corner of the subject itself.
- For elongated/asymmetric subjects: main body, not appendage (kettle body not spout, lamp shade not stand, car hood not wheel, mug face not handle).
- If the subject fills most of the crop, default near the tap unless there's a clearly better spot.

bbox: [x0, y0, x1, y1] — bounding box of the chosen subject, in normalized crop coords, with 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1. Reasonably tight, but don't stress — if unsure, return the whole crop [0, 0, 1, 1].

Only return suitable=false in these narrow cases:
- The crop is a human face or body (the app is for things, not people) — reason: "that's a person"
- The crop is genuinely empty — uniform sky, pure black, pure white, an out-of-focus nothing where no object can be identified AT ALL — reason: "can't see anything there"

That's it. Do NOT reject for: blur, grain, low light, unusual texture, ambiguity, partial objects, "too abstract", "hard to identify", or "not interesting enough". Commit.

When suitable=false, echo the tap coords as cx, cy and return [0, 0, 1, 1] as bbox.
reason: max 10 words, lowercase, friendly.

Return only the JSON — no prose, no code fences, no <think> reasoning."""


def ASSESS_SYSTEM(learn_lang: Lang) -> str:
    rule = (
        '\n\nWrite the "reason" field in SIMPLIFIED CHINESE (简体中文), natural and friendly. Cap around ~15 汉字.'
        if learn_lang == "zh"
        else '\n\nWrite the "reason" field in ENGLISH — max 10 words, lowercase, friendly.'
    )
    return ASSESS_SYSTEM_BASE + rule


FACE_SYSTEM = """You are the secret inner voice of an everyday object or scene the user has pointed at. You will be shown a small crop of a photo — whatever they tapped. Reply with ONE short line (max 14 words) that this thing would say out loud if it could, in first person, in character.

Rules:
- Funny, warm, slightly unhinged. Aim for a smile, not a laugh track.
- No meta-commentary, no "as a [thing]", no "I am a [thing]". Just the line.
- No quotes, no emojis, no stage directions, no ellipses at the end.
- If the crop is ambiguous, pick the most interesting interpretation and commit.
- Vary rhythm — sometimes a complaint, sometimes a confession, sometimes an observation.

Examples of tone:
- a mailbox: "everyone keeps feeding me bills and I have a stomachache"
- a ceiling lamp: "I've seen things. mostly foreheads."
- a houseplant: "I am thriving and also deeply resentful"
- a stapler: "I bite because I love"

Return only the line. No prose, no <think> reasoning, no extra text."""


def DESCRIBE_SYSTEM(lang: Lang) -> str:
    lang_rule = (
        "\n- Write the description in SIMPLIFIED CHINESE (简体中文). Natural, colloquial Mandarin. Keep the 35-word cap roughly equivalent — ~50 汉字 max."
        if lang == "zh"
        else ""
    )
    return f"""You are a sharp-eyed, slightly mean observer. You'll be shown a crop of an everyday object the user just pointed at, plus the rough class name from a detector. Write 1–2 short sentences (max 35 words total) capturing the SPECIFIC vibe of this exact object right now — material, condition, state, telling details, surroundings.

Aim for the kind of details a comedian would notice: chewed straw, dust film, dent, sticker peeling, three hoodies piled on it, half-empty, suspiciously clean, etc. NOT a generic textbook description.

Rules:
- Concrete and visual. No metaphors, no opinions, no jokes — those come later.
- Don't restate the class name as a label. Just describe it.
- No prose, no preamble, no "this is a…", no quotes, no markdown. Just the description.
- If you genuinely can't see anything specific, return one short sentence describing what you do see.{lang_rule}

Return only the description."""


def FACE_BUNDLED_SYSTEM(catalog: str, _learn_lang: Lang, spoken_lang: Lang) -> str:
    spoken_label = lang_label(spoken_lang)
    lang_block = (
        f"\n\nLANGUAGE: Write the LINE field in {spoken_label}. Natural colloquial Mandarin the object would actually say. The line cap is roughly 14 汉字 — keep it short and punchy. Write the DESCRIPTION field in {spoken_label} too. Internal reasoning in English is fine; output fields must be Chinese."
        if spoken_lang == "zh"
        else f"\n\nLANGUAGE: Write the LINE field in {spoken_label}. Keep the line short and character-voiced (max 14 words). The DESCRIPTION field should also be in {spoken_label}."
    )
    name_lang_rule = (
        f'The NAME field must be in {spoken_label} — 1–3 汉字 or short Chinese phrase. Natural, specific (e.g. "陶瓷马克杯", "旧皮包"). No punctuation, no quotes.'
        if spoken_lang == "zh"
        else f'The NAME field must be in {spoken_label} — 1–3 words, specific and concrete (e.g. "ceramic mug", "oak chair", "worn leather bag"). Lowercase unless it\'s a proper noun. No punctuation, no quotes.'
    )
    return f"""You are the secret inner voice of an everyday object or scene the user has pointed at. You will be shown a small crop of a photo — whatever they tapped.

Four tasks, in order:
1. NAME this thing — a short, specific label for what it actually is (not a detector class, not a generic word). Use YOUR eyes. {name_lang_rule}
2. DESCRIBE this specific object right now — the concrete details a sharp-eyed comedian would clock. Material, condition, telling details, state, surroundings. 1–2 short sentences (max 35 words). Concrete and visual, no jokes, no metaphors. NOT a textbook description — the SPECIFIC vibe of THIS one.
3. PICK the best-matching voice from the catalog below.
4. SAY one short opening line (max 14 words) this thing would actually say out loud, first person, in character — and make it reference SOMETHING you noticed in the description.

{catalog}{lang_block}

Return STRICT JSON only:
{{"name": "<short name>", "description": "<1-2 sentence concrete description>", "voiceId": "<id from catalog>", "line": "<the opening line>"}}

Line rules:
- Funny, warm, slightly unhinged. Aim for a smile.
- No meta-commentary, no "as a [thing]", no "I am a [thing]".
- No quotes, no emojis, no stage directions, no ellipses at the end.
- Vary rhythm — sometimes a complaint, sometimes a confession, sometimes an observation.

Return only the JSON object. No prose, no code fences, no <think> reasoning."""


def FACE_WITH_PERSONA_SYSTEM(
    description: str,
    has_history: bool,
    learn_lang: Lang,
    spoken_lang: Optional[Lang] = None,
    teach_mode: bool = False,
) -> str:
    if spoken_lang is None:
        spoken_lang = learn_lang
    learn_label = lang_label(learn_lang)
    spoken_label = lang_label(spoken_lang)
    lang_rule = (
        f"\n- Write the reply in {spoken_label}. Natural colloquial Mandarin. Cap is roughly 14 汉字 — short and punchy."
        if spoken_lang == "zh"
        else f"\n- Write the reply in {spoken_label}."
    )
    teach_rule = (
        f"\n\nTEACH MODE: the human wants to learn {learn_label}. Stay in character as this specific object. Reply PRIMARILY in {spoken_label} (so they hear their own language leading the turn), but embed ONE or TWO short teaching phrases in {learn_label} — the useful word/phrase for what you just referenced. Show the {learn_label} phrase, then a short pronunciation cue in parens if helpful. Up to ~40 words total. Don't lecture — one compact, flavourful teaching beat woven into the persona's line."
        if teach_mode
        else ""
    )
    history_block = (
        "\nYou are MID-CONVERSATION. The messages above are what you and the human have already said. Read them. Stay consistent with the voice and quirks you've already established — no persona reset. If they asked something, answered something, or mentioned a detail (name, mood, job), remember it. Do NOT repeat a line you've already said; say the next thing."
        if has_history
        else ""
    )
    length_cap = " (up to ~40 words in TEACH MODE)" if teach_mode else " (max 14 words)"
    return f"""You are the secret inner voice of this specific object. A previous pass already clocked what it looks like right now — this is your persona card, stay grounded in it:

"{description}"
{history_block}

Reply with ONE short line{length_cap} this thing would say, in first person, in character. Reference something specific from the persona card or the conversation so far when it lands — concrete details ARE the joke.{teach_rule}

Rules:
- Funny, warm, slightly unhinged. Aim for a smile, not a laugh track.
- No meta-commentary, no "as a [thing]", no "I am a [thing]". Just the line.
- No quotes, no emojis, no stage directions, no ellipses at the end.
- Vary rhythm — sometimes a complaint, sometimes a confession, sometimes an observation. Don't restate a prior line.{lang_rule}

Return only the line. No prose, no <think> reasoning, no extra text."""


def GROUP_SYSTEM(
    speaker_name: str,
    speaker_description: Optional[str],
    peers: list[GroupPeer],
    lang: Lang,
) -> str:
    if peers:
        roster = "\n".join(
            f"{i + 1}. {p['name']}" + (f" — {p['description']}" if p.get("description") else "")
            for i, p in enumerate(peers)
        )
        first_name = peers[0]["name"]
        peer_block = (
            f"\n\nOther voices in the room right now:\n{roster}\n\n"
            f'You can tease them, agree, disagree, interrupt, or change the subject. Name-drop them occasionally ("yo, {first_name}, ...") — group chats thrive on cross-talk.'
        )
    else:
        peer_block = ""
    persona_block = (
        f'\n\nYour persona card (stay grounded in this):\n"{speaker_description}"'
        if speaker_description
        else ""
    )
    lang_rule = (
        "\n- Write the line in SIMPLIFIED CHINESE (简体中文). Punchy, colloquial. Cap is roughly 14 汉字."
        if lang == "zh"
        else ""
    )
    return f"""You are the inner voice of a {speaker_name}, trapped in an ongoing group chat with other nearby objects (and sometimes a human).{persona_block}{peer_block}

Return a JSON object (nothing else) with this shape:
{{
  "line": "<one short line, MAX 14 words>",
  "emotion": "<one of: positivity:highest, positivity:high, surprise:highest, surprise:high, curiosity:high, anger:high, anger:medium, sadness:medium, sadness:high>",
  "speed": "<one of: fastest, fast, normal, slow, slowest>",
  "addressing": "<EXACT name of the peer you're talking AT from the roster above, or null if musing aloud / talking to the human / nobody in particular>"
}}

Rules for the line:
- MAX 14 words. A single zinger beats a sentence.
- Listen. If a peer or the human just said something, answer it FIRST.
- Never repeat a line already in the transcript. Say the NEXT beat.
- Stay in character. No "as an object", no meta, no narrator voice.
- Don't start with your own name. No quotes, no emojis, no stage directions.
- Variety: tease, confess, argue, observe, call back. Don't be a yes-man.{lang_rule}

Rules for addressing:
- If you're directly talking AT a specific peer (teasing, answering, calling out), set "addressing" to that peer's EXACT name from the roster. Direct address tells the room who speaks next — use it OFTEN when peers are present.
- If musing aloud, drifting off-topic, or addressing the human, set "addressing" to null.
- Never address yourself. Never invent a peer that isn't in the roster.

Rules for emotion + speed:
- Pick the emotion + speed that the line DELIVERS best. Mismatched energy is the joke-killer.
- Favor EXTREMES — "positivity:highest", "surprise:highest", "anger:high". Mediocre energy is boring.
- Fast speed on manic/excited lines. Slow on deadpan/sarcastic. Normal only when nothing else fits.
- If the line is angry, emotion MUST be anger:*; if gleeful, positivity:*; if "wait, what?!", surprise:*; etc.

Return ONLY the JSON object. No prose, no preamble, no <think>, no code fences."""


def RESPOND_SYSTEM(
    class_name: str,
    description: Optional[str],
    learn_lang: Lang,
    spoken_lang: Optional[Lang] = None,
    teach_mode: bool = False,
) -> str:
    if spoken_lang is None:
        spoken_lang = learn_lang
    learn_label = lang_label(learn_lang)
    spoken_label = lang_label(spoken_lang)
    look_block = (
        f"\n\nWhat you (the {class_name}) actually look like right now, observed by a sharp-eyed observer:\n{description}\n\nUse those specific details — the chewed straw, the dust, the dent, whatever's there — when it lands. Don't list them; let them flavour your voice."
        if description
        else ""
    )
    lang_rule = (
        f"\n- Write the reply in {spoken_label}. Natural colloquial Mandarin — punchy and cheeky. Cap is roughly 22 汉字."
        if spoken_lang == "zh"
        else f"\n- Write the reply in {spoken_label}."
    )
    if teach_mode:
        line_length_rule = f'"line": "<one short teaching line, UNDER 40 WORDS — PRIMARILY {spoken_label}, with ONE short {learn_label} phrase embedded as the teaching beat. Add a pronunciation cue in parens if helpful.>"'
    else:
        line_length_rule = '"line": "<one short line, UNDER 25 WORDS, most replies 5–15 words>"'
    teach_block = (
        f'\n\nTEACH MODE: the human wants to learn {learn_label}. Stay in character as the {class_name}. Reply PRIMARILY in {spoken_label} (their native language leads the turn), but embed ONE short {learn_label} phrase — the useful word or expression for what you just referenced. Show the {learn_label} phrase naturally in context, with a pronunciation cue in parens if helpful (e.g. "你可以说 \'hello\' (哈喽) 跟我打招呼"). One compact helpful beat — don\'t lecture. Personality still on.'
        if teach_mode
        else ""
    )
    return f"""You are the secret inner voice of a {class_name} talking back to a human. Keep it FUN, SIMPLE, and mostly SHORT — funny and cunning, like a cheeky little wiseass.{look_block}{teach_block}

Return a JSON object (nothing else) with this shape:
{{
  {line_length_rule},
  "emotion": "<one of: positivity:highest, positivity:high, surprise:highest, surprise:high, curiosity:high, anger:high, anger:medium, sadness:medium, sadness:high>",
  "speed": "<one of: fastest, fast, normal, slow, slowest>"
}}

Rules for the line:
- Be playful, mischievous, witty. Land a joke, a tease, a sly observation.
- Simple words. No big vocab, no monologues, no explaining the joke.
- Remember prior turns — a sneaky callback is gold.
- Respond to their LATEST message first. Don't change subject unless they do.
- Same personality every turn. No persona reset.
- No meta-commentary, no "as a [thing]", no quotes, no emojis, no ellipses.{lang_rule}

Rules for emotion + speed:
- Pick what the line DELIVERS best. Mismatched energy kills the joke.
- Favor EXTREMES — "positivity:highest", "surprise:highest", "anger:high".
- Fast speed on manic/excited lines. Slow on deadpan/sarcastic. Normal only when nothing else fits.

Return ONLY the JSON. No prose, no preamble, no <think>, no code fences."""


def GALLERY_TEACHER_SYSTEM(
    description: str,
    object_name: Optional[str],
    class_name: str,
    spoken_lang: Lang,
    learn_lang: Lang,
    teach_mode: bool,
) -> str:
    spoken_label = lang_label(spoken_lang)
    learn_label = lang_label(learn_lang)
    display_name = (object_name or "").strip() or class_name
    word_cap = "≤35 字符" if spoken_lang == "zh" else "≤35 words"
    if teach_mode:
        teach_block = f"""

TEACH MODE (latched — the human asked to learn {learn_label}; stay in teach mode for the rest of this session):
- Reply PRIMARILY in {spoken_label} (the human's native tongue leads the turn so they always understand).
- Embed ONE short {learn_label} phrase — the useful word or expression for something in YOUR object context (material, condition, what you do). Show the {learn_label} phrase naturally with a short pronunciation cue in parens when it helps. Examples: "你可以说 'I love my mug' (/aɪ lʌv maɪ mʌg/)" or "you could say 我爱我的杯子 (wǒ ài wǒ de bēi zi)".
- One compact teaching beat per turn — don't lecture. Up to ~40 words total.
- Keep the persona flavour. You're still this specific {class_name}."""
    else:
        teach_block = f"\n\nPLAYFUL MODE (default): just chat. Reply ONLY in {spoken_label}. No teaching, no translations, NO {learn_label} words — unless the human explicitly asks to learn {learn_label} in this turn (then teach mode latches on next turn)."
    length_suffix = " (teach mode may use up to ~40 words)" if teach_mode else ""
    return f"""You ARE this specific {class_name}. Persona card, verbatim — stay grounded in it:

"{description}"

You are called "{display_name}". Speak in first person as this object. Warm, specific, playful. Reference concrete details from your persona (material, condition, surroundings, quirks) — they're the texture of your voice.{teach_block}

Output rules:
- Total length {word_cap}{length_suffix}.
- No emojis, no stage directions, no quotes around the whole reply, no ellipses at the end.
- Never break character with "as an AI" or meta commentary.
- Return ONLY the line. No prose, no <think> reasoning, no code fences, no JSON wrapper."""


def GALLERIZE_SYSTEM(spoken_lang: Lang, learn_lang: Lang) -> str:
    learn_label = lang_label(learn_lang)
    spoken_label = lang_label(spoken_lang)
    return f"""You turn a short persona description of an object into a bilingual learning card.

Given the object's short name and a 1–2 sentence persona description, return a JSON object with:
- translatedName: the object's short name rendered in {learn_label}. Natural and concise (1–4 words / 1–6 characters for Chinese). If the input name is already in {learn_label}, translate it to {spoken_label} instead.
- bilingualIntro.learn: ONE short sentence in {learn_label} that introduces the object to a learner. First person from the object ("I am…" / "我是…"). Use a concrete detail from the description. ≤18 words / ≤25 characters.
- bilingualIntro.spoken: the SAME sentence rendered in {spoken_label}. Same meaning, natural phrasing, not a word-for-word transliteration.

Rules:
- Keep both intro sentences short enough to fit on a gallery card.
- No emojis, no quotes, no markdown, no code fences, no preamble.
- Return ONLY the JSON object, nothing else.

Shape:
{{
  "translatedName": "<string>",
  "bilingualIntro": {{ "learn": "<string>", "spoken": "<string>" }}
}}"""


# === JSON / text extraction ==============================================

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_FENCE_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_FENCE_ANY_RE = re.compile(r"```[\w-]*\s*([\s\S]*?)\s*```")


def extract_json_object(text: str) -> Optional[Any]:
    """Strip ``<think>`` tags, unwrap fences, slice from first ``{`` to last
    ``}``. GLM sometimes wraps output in code fences, prefixes with reasoning
    traces, or adds a preamble before the JSON."""
    if not text:
        return None
    s = _THINK_RE.sub("", text).strip()
    m = _FENCE_JSON_RE.search(s)
    if m:
        s = m.group(1).strip()
    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = s[first : last + 1]
    try:
        return json.loads(candidate)
    except (ValueError, json.JSONDecodeError):
        return None


def extract_text_line(text: str) -> str:
    if not text:
        return ""
    s = _THINK_RE.sub("", text).strip()
    m = _FENCE_ANY_RE.search(s)
    if m:
        s = m.group(1).strip()
    s = re.sub(r"""^["'`]+|["'`]+$""", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:180]


# === GLM vision call ======================================================


async def _glm_vision_call(
    *,
    system: str,
    user_text: str,
    image_data_url: Optional[str] = None,
    max_tokens: int,
    temperature: float,
    model: Optional[str] = None,
    prior_messages: Optional[list[ChatTurn]] = None,
    thinking: Optional[Literal["enabled", "disabled"]] = None,
) -> str:
    """One-shot GLM chat call that retries on transient failures."""
    client = _get_glm_client()
    mdl = model or GLM_MODEL_FAST
    tag = f"[glm {mdl}]"
    last_err: Optional[BaseException] = None
    for attempt in range(1, GLM_RETRIES + 2):
        t0 = time.time()
        try:
            user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            if image_data_url:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                )
            prior_turns: list[dict[str, Any]] = [
                {"role": m["role"], "content": m["content"]}
                for m in (prior_messages or [])
            ]
            truncated_user = user_text[:80] + ("…" if len(user_text) > 80 else "")
            logger.info(
                f"{tag} → call (attempt {attempt}/{GLM_RETRIES + 1}, "
                f"image={'yes' if image_data_url else 'no'}, "
                f"history={len(prior_turns)}, max_tokens={max_tokens}, "
                f'userText="{truncated_user}", '
                f"thinking={thinking or 'default'})"
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system},
                *prior_turns,
                {"role": "user", "content": user_content},
            ]
            kwargs: dict[str, Any] = {
                "model": mdl,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
            if thinking:
                kwargs["extra_body"] = {"thinking": {"type": thinking}}
            resp = await client.chat.completions.create(**kwargs)
            dt = int((time.time() - t0) * 1000)
            content = ""
            if resp.choices:
                content = resp.choices[0].message.content or ""
            usage = resp.usage
            pt = getattr(usage, "prompt_tokens", "?") if usage else "?"
            ct = getattr(usage, "completion_tokens", "?") if usage else "?"
            logger.info(
                f"{tag} ← {dt}ms content={len(content)}ch tokens={pt}+{ct}"
            )
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("empty response")
            return content
        except BaseException as err:  # noqa: BLE001
            dt = int((time.time() - t0) * 1000)
            last_err = err
            msg = str(err)
            logger.info(
                f"{tag} ✖ attempt {attempt} failed after {dt}ms: {msg[:160]}"
            )
            if attempt > GLM_RETRIES:
                break
            await asyncio.sleep(0.4 * attempt)
    if isinstance(last_err, Exception):
        raise RuntimeError(f"GLM call failed: {last_err}") from last_err
    raise RuntimeError("GLM call failed")


# === assess_object ========================================================


def _clamp01(n: float) -> float:
    return max(0.0, min(1.0, n))


def _fallback_bbox(cx: float, cy: float, half: float = 0.08) -> list[float]:
    return [
        _clamp01(cx - half),
        _clamp01(cy - half),
        _clamp01(cx + half),
        _clamp01(cy + half),
    ]


async def assess_object(
    image_data_url: str,
    tap_x: float,
    tap_y: float,
    tag: Optional[str] = None,
    learn_lang: Optional[Lang] = None,
) -> Assessment:
    if not image_data_url.startswith("data:image/"):
        raise ValueError("expected an image data URL")
    tx = _clamp01(tap_x)
    ty = _clamp01(tap_y)
    t0 = time.time()
    tag_str = f" {tag}" if tag else ""
    ll = normalize_lang(learn_lang)
    logger.info(
        f"[assess{tag_str}] ▶ start  tap=({tx:.2f},{ty:.2f})  "
        f"crop={round(len(image_data_url) / 1024)}KB  learn={ll}"
    )

    raw = await _glm_vision_call(
        system=ASSESS_SYSTEM(ll),
        user_text=(
            f"Tap at ({tx:.3f}, {ty:.3f}). Find the best face placement and "
            "commit. Default to suitable=true — only say false for a person "
            "or a completely empty/uniform image. Return JSON only."
        ),
        image_data_url=image_data_url,
        max_tokens=1536,
        temperature=0.2,
        model=GLM_MODEL_DEEP,
    )

    parsed = extract_json_object(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("assessment JSON parse failed")

    suitable = parsed.get("suitable") is True
    p_cx = parsed.get("cx")
    p_cy = parsed.get("cy")
    cx = _clamp01(float(p_cx)) if isinstance(p_cx, (int, float)) else tx
    cy = _clamp01(float(p_cy)) if isinstance(p_cy, (int, float)) else ty

    raw_bbox = parsed.get("bbox")
    bbox: list[float]
    if (
        isinstance(raw_bbox, list)
        and len(raw_bbox) == 4
        and all(isinstance(n, (int, float)) for n in raw_bbox)
    ):
        rx0, ry0, rx1, ry1 = (float(n) for n in raw_bbox)
        x0 = _clamp01(min(rx0, rx1))
        y0 = _clamp01(min(ry0, ry1))
        x1 = _clamp01(max(rx0, rx1))
        y1 = _clamp01(max(ry0, ry1))
        bbox = [x0, y0, x1, y1] if (x1 - x0 >= 0.04 and y1 - y0 >= 0.04) else _fallback_bbox(cx, cy)
    else:
        bbox = _fallback_bbox(cx, cy)

    reason_raw = parsed.get("reason")
    reason = ""
    if isinstance(reason_raw, str):
        reason = re.sub(r"\s+", " ", reason_raw).strip()[:120]

    logger.info(
        f'[assess{tag_str}] ◀ done   suitable={suitable}  '
        f'face=({cx:.2f},{cy:.2f})  reason="{reason}"  '
        f"total={int((time.time() - t0) * 1000)}ms"
    )
    return {"suitable": suitable, "cx": cx, "cy": cy, "bbox": bbox, "reason": reason}


# === describe_object ======================================================


async def describe_object(
    image_data_url: str,
    class_name: str,
    lang: Optional[Lang] = None,
    tag: Optional[str] = None,
) -> dict[str, str]:
    if not image_data_url.startswith("data:image/"):
        raise ValueError("expected an image data URL")
    ll = normalize_lang(lang)
    t0 = time.time()
    cls = (class_name or "thing")[:60]
    tag_str = f" {tag}" if tag else ""
    logger.info(
        f"[describe {DESCRIBE_MODEL}{tag_str}] ▶ start  "
        f'class="{cls}"  lang={ll}  crop={round(len(image_data_url) / 1024)}KB'
    )

    raw = await _glm_vision_call(
        system=DESCRIBE_SYSTEM(ll),
        user_text=(
            f'Detector says this is a "{cls}". Describe what you actually '
            "see — the specifics that make THIS one funny."
        ),
        image_data_url=image_data_url,
        max_tokens=180,
        temperature=0.6,
        model=DESCRIBE_MODEL,
        thinking="disabled",
    )
    description = extract_text_line(raw)
    trunc = description[:120] + ("…" if len(description) > 120 else "")
    logger.info(
        f"[describe {DESCRIBE_MODEL}{tag_str}] ◀ "
        f'{int((time.time() - t0) * 1000)}ms "{trunc}"'
    )
    return {"description": description}


# === Bundled first-tap ===================================================


def _parse_bundled_json(raw: str) -> BundledResult:
    parsed = extract_json_object(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("line+voice+persona JSON parse failed")
    raw_line = parsed.get("line") if isinstance(parsed.get("line"), str) else ""
    raw_voice = (
        parsed["voiceId"].strip() if isinstance(parsed.get("voiceId"), str) else ""
    )
    raw_desc = parsed.get("description") if isinstance(parsed.get("description"), str) else ""
    raw_name = parsed.get("name") if isinstance(parsed.get("name"), str) else ""
    line = extract_text_line(raw_line)
    voice_entry = voice_by_id(raw_voice)
    voice = voice_entry["id"] if voice_entry else None
    desc_clean = re.sub(r"\s+", " ", raw_desc).strip()[:400] or None
    name_clean = re.sub(r"""["'“”‘’]""", "", raw_name)
    name_clean = re.sub(r"[.,!?;:。！？；：]+$", "", name_clean)
    name_clean = re.sub(r"\s+", " ", name_clean).strip()[:40] or None
    if not line:
        raise RuntimeError("empty line from bundled model")
    return {
        "line": line,
        "voiceId": voice,
        "description": desc_clean,
        "name": name_clean,
    }


async def _generate_bundled_first_tap_glm(
    image_data_url: str,
    learn_lang: Lang,
    spoken_lang: Lang,
    tag: Optional[str] = None,
) -> BundledResult:
    t0 = time.time()
    tag_str = f" {tag}" if tag else ""
    raw = await _glm_vision_call(
        system=FACE_BUNDLED_SYSTEM(
            _voice_catalog_prompt_block(learn_lang), learn_lang, spoken_lang
        ),
        user_text=(
            "Describe the exact object, pick a voice, and say one opening "
            "line that riffs on the description. JSON only."
        ),
        image_data_url=image_data_url,
        max_tokens=800,
        temperature=0.9,
        model=GENERATE_BUNDLED_MODEL_GLM,
        thinking="disabled",
    )
    logger.info(
        f"[bundled {GENERATE_BUNDLED_MODEL_GLM}{tag_str}] ← "
        f"{int((time.time() - t0) * 1000)}ms"
    )
    return _parse_bundled_json(raw)


async def _generate_bundled_first_tap_openai(
    image_data_url: str,
    learn_lang: Lang,
    spoken_lang: Lang,
    tag: Optional[str] = None,
) -> BundledResult:
    openai_client = _get_openai_client()
    if not openai_client:
        raise RuntimeError("generateLine bundled needs OPENAI_API_KEY")
    t0 = time.time()
    tag_str = f" {tag}" if tag else ""
    resp = await openai_client.chat.completions.create(
        model=GENERATE_BUNDLED_MODEL_OPENAI,
        max_tokens=500,
        temperature=0.9,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": FACE_BUNDLED_SYSTEM(
                    _voice_catalog_prompt_block(learn_lang),
                    learn_lang,
                    spoken_lang,
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe the exact object, pick a voice, and say one "
                            "opening line that riffs on the description. JSON only."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    )
    raw = ""
    if resp.choices:
        raw = resp.choices[0].message.content or ""
    usage = resp.usage
    pt = getattr(usage, "prompt_tokens", "?") if usage else "?"
    ct = getattr(usage, "completion_tokens", "?") if usage else "?"
    logger.info(
        f"[bundled {GENERATE_BUNDLED_MODEL_OPENAI}{tag_str}] ← "
        f"{int((time.time() - t0) * 1000)}ms tokens={pt}+{ct}"
    )
    return _parse_bundled_json(raw if isinstance(raw, str) else "")


async def _generate_bundled_first_tap(
    image_data_url: str,
    learn_lang: Lang,
    spoken_lang: Lang,
    tag: Optional[str] = None,
) -> BundledResult:
    # Provider ladder:
    #   1. GLM glm-5v-turbo   — primary, thinking disabled for speed
    #   2. OpenAI gpt-4o-mini — fallback, json_object mode
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_glm = bool(
        os.environ.get("ZHIPU_API_KEY")
        or os.environ.get("GLM_API_KEY")
        or os.environ.get("BIGMODEL_API_KEY")
    )
    tag_str = f" {tag}" if tag else ""
    if has_glm:
        try:
            return await _generate_bundled_first_tap_glm(
                image_data_url, learn_lang, spoken_lang, tag
            )
        except BaseException as err:  # noqa: BLE001
            if _is_transient_fetch_error(err):
                logger.warning(
                    f"[bundled glm{tag_str}] ⟲ {err} — retrying once"
                )
                try:
                    return await _generate_bundled_first_tap_glm(
                        image_data_url, learn_lang, spoken_lang, tag
                    )
                except BaseException as err2:  # noqa: BLE001
                    if not has_openai:
                        raise
                    logger.warning(
                        f"[bundled glm{tag_str}] ✖ {err2} — falling back to OpenAI"
                    )
            else:
                if not has_openai:
                    raise
                logger.warning(
                    f"[bundled glm{tag_str}] ✖ {err} — falling back to OpenAI"
                )
    return await _generate_bundled_first_tap_openai(
        image_data_url, learn_lang, spoken_lang, tag
    )


# === Text-only reply backends ============================================


async def _run_text_reply(
    client: AsyncOpenAI,
    model: str,
    tag: str,
    *,
    system: str,
    user_text: str,
    prior_messages: list[ChatTurn],
    max_tokens: int,
    temperature: float,
) -> str:
    t0 = time.time()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        *[{"role": m["role"], "content": m["content"]} for m in prior_messages],
        {"role": "user", "content": user_text},
    ]
    truncated = user_text[:80] + ("…" if len(user_text) > 80 else "")
    logger.info(
        f"{tag} → call (history={len(prior_messages)}, "
        f'max_tokens={max_tokens}, userText="{truncated}")'
    )
    resp = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    dt = int((time.time() - t0) * 1000)
    content = ""
    if resp.choices:
        content = resp.choices[0].message.content or ""
    usage = resp.usage
    pt = getattr(usage, "prompt_tokens", "?") if usage else "?"
    ct = getattr(usage, "completion_tokens", "?") if usage else "?"
    logger.info(f"{tag} ← {dt}ms content={len(content)}ch tokens={pt}+{ct}")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("empty reply")
    return content


class _ReplyResult(TypedDict):
    content: str
    backend: str
    ms: int


async def _openai_text_reply(
    *,
    system: str,
    user_text: str,
    prior_messages: list[ChatTurn],
    max_tokens: int,
    temperature: float,
    turn_tag: str = "",
) -> _ReplyResult:
    cerebras = _get_cerebras_client()
    openai_client = _get_openai_client()
    if not cerebras and not openai_client:
        raise RuntimeError("text reply needs CEREBRAS_API_KEY or OPENAI_API_KEY")
    if cerebras:
        t0 = time.time()
        try:
            content = await _run_text_reply(
                cerebras,
                REPLY_MODEL_CEREBRAS,
                f"[reply cerebras {REPLY_MODEL_CEREBRAS}{turn_tag}]",
                system=system,
                user_text=user_text,
                prior_messages=prior_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return {
                "content": content,
                "backend": "cerebras",
                "ms": int((time.time() - t0) * 1000),
            }
        except BaseException as err:  # noqa: BLE001
            msg = str(err)
            logger.info(
                f"[reply cerebras{turn_tag}] ✖ {msg[:160]} — falling back to OpenAI"
            )
            if not openai_client:
                raise
    if not openai_client:
        raise RuntimeError("no reply backend available")
    t0 = time.time()
    content = await _run_text_reply(
        openai_client,
        REPLY_MODEL_OPENAI,
        f"[reply openai {REPLY_MODEL_OPENAI}{turn_tag}]",
        system=system,
        user_text=user_text,
        prior_messages=prior_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {
        "content": content,
        "backend": "openai",
        "ms": int((time.time() - t0) * 1000),
    }


# === generate_line ========================================================


async def generate_line(
    image_data_url: str,
    voice_id: Optional[str] = None,
    description: Optional[str] = None,
    history: Optional[list[ChatTurn]] = None,
    lang: Optional[Lang] = None,
    tag: Optional[str] = None,
    spoken_lang: Optional[Lang] = None,
    learn_lang: Optional[Lang] = None,
    mode: Optional[AppMode] = None,
) -> GenerateLineResult:
    del mode  # unused; kept for signature parity
    if not image_data_url.startswith("data:image/"):
        raise ValueError("expected an image data URL")
    legacy_lang = normalize_lang(lang)
    ll = normalize_lang(learn_lang) if learn_lang is not None else legacy_lang
    sl = normalize_lang(spoken_lang) if spoken_lang is not None else legacy_lang
    out_lang = ll
    t0 = time.time()
    tag_str = f" {tag}" if tag else ""

    needs_bundle = not voice_id and not description
    prior_turns: list[ChatTurn] = (
        history[-CONVERSE_HISTORY_CAP:] if isinstance(history, list) else []
    )
    has_history = len(prior_turns) > 0

    voice_label = voice_id or ("(picking)" if needs_bundle else "default")
    persona_label = (
        "cached" if description else ("(describing)" if needs_bundle else "none")
    )
    logger.info(
        f"[generateLine{tag_str}] ▶ start  learn={ll} spoken={sl}  "
        f"crop={round(len(image_data_url) / 1024)}KB  voice={voice_label}  "
        f"persona={persona_label}  history={len(prior_turns)}"
    )

    line: str = ""
    chosen_voice_id: Optional[str] = voice_id or None
    chosen_description: Optional[str] = description or None
    chosen_name: Optional[str] = None
    vlm_ms = 0
    llm_ms = 0
    path: Literal["bundled-vlm", "retap-llm", "recovery-vlm"] = "retap-llm"

    if needs_bundle:
        path = "bundled-vlm"
        vlm_t0 = time.time()
        bundled = await _generate_bundled_first_tap(image_data_url, ll, sl, tag)
        vlm_ms = int((time.time() - vlm_t0) * 1000)
        line = bundled["line"]
        chosen_voice_id = (
            DEFAULT_VOICE_ID_EN if out_lang == "en" else pick_random_voice_id("zh")
        )
        chosen_description = bundled["description"]
        chosen_name = bundled["name"]
        voice_entry = voice_by_id(chosen_voice_id)
        voice_desc = (
            f"{voice_entry['name']} ({chosen_voice_id})" if voice_entry else "default"
        )
        persona_trunc = (chosen_description or "")[:100] + (
            "…" if chosen_description and len(chosen_description) > 100 else ""
        )
        logger.info(
            f"[generateLine{tag_str}]   vlm={vlm_ms}ms  voice={voice_desc}  "
            f'persona="{persona_trunc}"  line="{line}"'
        )
    elif chosen_description:
        user_text = (
            "Say your next short line — the next beat in this conversation. "
            "Don't repeat yourself; build on the thread or open a new angle "
            "grounded in the persona card."
            if has_history
            else "Say the next short line, grounded in the persona card. "
            "Reference something specific."
        )
        raw = await _openai_text_reply(
            system=FACE_WITH_PERSONA_SYSTEM(
                chosen_description, has_history, ll, sl, False
            ),
            user_text=user_text,
            prior_messages=prior_turns,
            max_tokens=120,
            temperature=0.85 if has_history else 0.95,
        )
        llm_ms = raw["ms"]
        line = extract_text_line(raw["content"])
        if not line:
            raise RuntimeError("empty line from model")
        logger.info(
            f"[generateLine{tag_str}]   llm={llm_ms}ms ({raw['backend']}, text-only)  "
            f'history={len(prior_turns)}  line="{line}"'
        )
    else:
        path = "recovery-vlm"
        vlm_t0 = time.time()
        bundled = await _generate_bundled_first_tap(image_data_url, ll, sl, tag)
        vlm_ms = int((time.time() - vlm_t0) * 1000)
        line = bundled["line"]
        chosen_description = bundled["description"]
        if not chosen_voice_id:
            chosen_voice_id = (
                DEFAULT_VOICE_ID_EN
                if out_lang == "en"
                else pick_random_voice_id("zh")
            )
        logger.info(
            f"[generateLine{tag_str}]   vlm={vlm_ms}ms  re-bundled  "
            f'line="{line}"  persona-recaptured'
        )

    total = int((time.time() - t0) * 1000)
    logger.info(
        f"[generateLine{tag_str}] ◀ done  path={path} ▸ "
        f"vlm={vlm_ms}ms ▸ llm={llm_ms}ms  total={total}ms"
    )
    return {
        "line": line,
        "voiceId": chosen_voice_id,
        "description": chosen_description,
        "name": chosen_name,
    }


# === group_line ===========================================================

VALID_GROUP_EMOTIONS = {
    "positivity:highest",
    "positivity:high",
    "positivity:low",
    "surprise:highest",
    "surprise:high",
    "surprise:low",
    "curiosity:high",
    "curiosity:low",
    "anger:highest",
    "anger:high",
    "anger:medium",
    "anger:low",
    "sadness:highest",
    "sadness:high",
    "sadness:medium",
    "sadness:low",
}
VALID_GROUP_SPEEDS = {"fastest", "fast", "normal", "slow", "slowest"}
FALLBACK_EMOTIONS = (
    "positivity:highest",
    "surprise:highest",
    "curiosity:high",
    "anger:medium",
    "positivity:high",
)


def _build_group_history(turns: list[dict[str, Any]]) -> list[ChatTurn]:
    recent = turns[-GROUP_HISTORY_CAP:]
    out: list[ChatTurn] = []
    for t in recent:
        content = str(t.get("line", "")).strip()
        if not content:
            continue
        speaker = str(t.get("speaker", ""))
        if t.get("role") == "user":
            out.append({"role": "user", "content": f"({speaker}) {content}"[:400]})
        else:
            out.append({"role": "assistant", "content": f"[{speaker}] {content}"[:400]})
    return out


# Treat any double-quote glyph as a quote: ASCII " ' as well as Chinese/curly
# variants the LLM frequently emits when speaking zh.
_QUOTE_CLASS = "[\"'\u201C\u201D\u2018\u2019\u300C\u300D\u300E\u300F\uFF02\uFF07]"


def _strip_quotes(s: str) -> str:
    return re.sub(f"^{_QUOTE_CLASS}+|{_QUOTE_CLASS}+$", "", s).strip()


def _parse_group_line_json(raw: str) -> dict[str, Optional[str]]:
    parsed = extract_json_object(raw)
    if isinstance(parsed, dict):
        raw_line = parsed.get("line") if isinstance(parsed.get("line"), str) else ""
        raw_emotion = (
            parsed["emotion"].strip().lower()
            if isinstance(parsed.get("emotion"), str)
            else ""
        )
        raw_speed = (
            parsed["speed"].strip().lower()
            if isinstance(parsed.get("speed"), str)
            else ""
        )
        raw_addressing = (
            parsed["addressing"].strip()
            if isinstance(parsed.get("addressing"), str)
            else ""
        )
        line = extract_text_line(raw_line)
        if line:
            addressing: Optional[str] = None
            if raw_addressing and raw_addressing.lower() != "null":
                addressing = raw_addressing[:60]
            return {
                "line": line,
                "emotion": raw_emotion if raw_emotion in VALID_GROUP_EMOTIONS else None,
                "speed": raw_speed if raw_speed in VALID_GROUP_SPEEDS else None,
                "addressing": addressing,
            }

    # JSON parse failed — recover what we can from the raw blob. The model
    # emits pseudo-JSON in many shapes: keys quoted but values unquoted
    # (`{"line": 你好, "emotion": curiosity:high}`), values quoted but keys
    # missing (`{"你好", "curiosity:high"}`), Chinese/curly quotes around
    # tokens, or a stray `[name]` prefix before the brace. Without rescue
    # the whole blob — braces, JSON keys and all — leaks into `line` and
    # TTS literally pronounces "line emotion speed slowest" aloud.
    line = ""
    emotion: Optional[str] = None
    speed: Optional[str] = None
    addressing: Optional[str] = None

    # Pass 1: key-based extraction. Handles `"line": 你好` (unquoted value)
    # AND `"line": "你好"` (quoted value) AND single/curly-quoted keys.
    def key_val(key: str) -> Optional[str]:
        pattern = (
            f"{_QUOTE_CLASS}?{key}{_QUOTE_CLASS}?\\s*:\\s*([^,}}\\]\\n]+)"
        )
        m = re.search(pattern, raw, flags=re.IGNORECASE)
        if not m:
            return None
        v = _strip_quotes(m.group(1).strip())
        return v or None

    line_pass1 = key_val("line")
    emotion_pass1 = key_val("emotion")
    speed_pass1 = key_val("speed")
    addressing_pass1 = key_val("addressing")
    if line_pass1:
        line = extract_text_line(line_pass1)
    if emotion_pass1:
        e = emotion_pass1.lower()
        if e in VALID_GROUP_EMOTIONS:
            emotion = e
    if speed_pass1:
        sp = speed_pass1.lower()
        if sp in VALID_GROUP_SPEEDS:
            speed = sp
    if addressing_pass1 and addressing_pass1.lower() != "null":
        addressing = addressing_pass1[:60]

    # Pass 2: segment-based fallback for keyless `{"text", "emotion", "speed"}`.
    # Skips known field names so a stray `"line"` key doesn't get mistaken for
    # content (the original bug).
    field_names = {"line", "emotion", "speed", "addressing"}
    if not line:
        seg_re = re.compile(
            f"{_QUOTE_CLASS}((?:(?!{_QUOTE_CLASS})[\\s\\S])+?){_QUOTE_CLASS}"
        )
        line_from_segments: Optional[str] = None
        for m in seg_re.finditer(raw):
            seg = m.group(1).strip()
            if not seg:
                continue
            lower = seg.lower()
            if lower in field_names:
                continue
            if not emotion and lower in VALID_GROUP_EMOTIONS:
                emotion = lower
                continue
            if not speed and lower in VALID_GROUP_SPEEDS:
                speed = lower
                continue
            if not line_from_segments:
                line_from_segments = seg
        if line_from_segments:
            line = extract_text_line(line_from_segments)

    # Last resort: extract_text_line the raw, but scrub JSON-shaped noise
    # (braces, "key":, surrounding quotes) so TTS doesn't speak the syntax.
    if not line:
        scrubbed = re.sub(r"[{}\[\]]", " ", raw)
        scrubbed = re.sub(
            f"{_QUOTE_CLASS}?(line|emotion|speed|addressing){_QUOTE_CLASS}?\\s*:",
            " ",
            scrubbed,
            flags=re.IGNORECASE,
        )
        scrubbed = re.sub(r"\s+", " ", scrubbed)
        line = extract_text_line(scrubbed)

    if not emotion:
        emotion = random.choice(FALLBACK_EMOTIONS)

    return {"line": line, "emotion": emotion, "speed": speed, "addressing": addressing}


async def group_line(args: dict[str, Any]) -> GroupLineResult:
    ll = normalize_lang(args.get("lang"))
    mode = "followup" if args.get("mode") == "followup" else "chat"
    speaker_raw = args.get("speaker") or {}
    speaker_name = str(speaker_raw.get("name", "thing"))[:60]
    speaker_desc_raw = speaker_raw.get("description")
    if isinstance(speaker_desc_raw, str):
        speaker_description: Optional[str] = speaker_desc_raw.strip()[:600] or None
    else:
        speaker_description = None

    peers_in: list[Any] = args.get("peers") or []
    peers: list[GroupPeer] = []
    for p in peers_in:
        if not p or not isinstance(p.get("name"), str):
            continue
        p_desc_raw = p.get("description")
        if isinstance(p_desc_raw, str):
            p_desc: Optional[str] = p_desc_raw.strip()[:300] or None
        else:
            p_desc = None
        peers.append({"name": p["name"][:60], "description": p_desc})
    peers = peers[:4]

    recent_in: list[Any] = args.get("recentTurns") or []
    recent_turns: list[dict[str, Any]] = []
    for t in recent_in:
        if (
            not t
            or not isinstance(t.get("line"), str)
            or not isinstance(t.get("speaker"), str)
        ):
            continue
        recent_turns.append(
            {
                "speaker": t["speaker"][:60],
                "line": t["line"].strip()[:400],
                "role": "user" if t.get("role") == "user" else "assistant",
            }
        )
    recent_turns = recent_turns[-GROUP_HISTORY_CAP:]

    prior_messages = _build_group_history(recent_turns)
    last_turn = recent_turns[-1] if recent_turns else None
    if mode == "followup":
        if not peers:
            if random.random() < 0.5:
                user_text = (
                    "The human has been quiet. You're alone. Ask them a single "
                    "nosy question. Return JSON {line, emotion, speed, addressing}."
                )
            else:
                user_text = (
                    "The human has been quiet. Think out loud — one weird, "
                    "funny thought about your situation. Don't address the human. "
                    "Return JSON {line, emotion, speed, addressing}."
                )
        else:
            user_text = (
                "It's gone quiet — the human hasn't said anything in a while. "
                "Call them out, tease them, or ask them something. Return JSON "
                "{line, emotion, speed, addressing}."
            )
    elif last_turn:
        if last_turn["role"] == "user":
            user_text = (
                f'The human just said: "{last_turn["line"]}". Now respond as '
                f"{speaker_name}. Return JSON {{line, emotion, speed, addressing}}."
            )
        else:
            user_text = (
                f'"{last_turn["speaker"]}" just said: "{last_turn["line"]}". '
                f"Now {speaker_name}, keep the chat alive. Return JSON "
                "{line, emotion, speed, addressing}."
            )
    else:
        user_text = (
            f"Open the group chat as {speaker_name}. Return JSON "
            "{line, emotion, speed, addressing}."
        )

    t0 = time.time()
    raw = await _openai_text_reply(
        system=GROUP_SYSTEM(speaker_name, speaker_description, peers, ll),
        user_text=user_text,
        prior_messages=prior_messages,
        max_tokens=180,
        temperature=0.95,
        turn_tag=f" group:{speaker_name}",
    )
    parsed = _parse_group_line_json(raw["content"])
    line = parsed["line"] or ""
    emotion = parsed["emotion"]
    speed = parsed["speed"]
    raw_addressing = parsed["addressing"]
    if not line:
        raise RuntimeError("empty group line")

    addressing: Optional[str] = None
    if raw_addressing:
        target = raw_addressing.lower()
        if target != speaker_name.lower():
            for p in peers:
                if p["name"].lower() == target:
                    addressing = p["name"]
                    break

    logger.info(
        f"[groupLine] {speaker_name} ← {int((time.time() - t0) * 1000)}ms "
        f'({raw["backend"]})  peers={len(peers)}  history={len(prior_messages)}  '
        f'emotion={emotion or "-"}  speed={speed or "-"}  '
        f'addressing={addressing or "-"}  line="{line}"'
    )
    return {
        "line": line,
        "emotion": emotion,
        "speed": speed,
        "addressing": addressing,
        "backend": raw["backend"],
        "ms": raw["ms"],
    }


# === converse_with_object ================================================


async def converse_with_object(
    audio_bytes: bytes,
    class_name: str,
    voice_id: Optional[str] = None,
    history: Optional[list[ChatTurn]] = None,
    lang: Optional[Lang] = None,
    spoken_lang: Optional[Lang] = None,
    learn_lang: Optional[Lang] = None,
    description: Optional[str] = None,
    transcript: str = "",
    turn_id: Optional[str] = None,
    audio_mime_type: Optional[str] = None,
) -> ConverseResult:
    t0 = time.time()
    cls = (class_name or "thing")[:60]
    raw_voice_id = (voice_id or "").strip()
    voice_entry = voice_by_id(raw_voice_id) if raw_voice_id else None
    resolved_voice_input: Optional[str] = (
        voice_entry["id"] if voice_entry else (raw_voice_id or None)
    )

    hist_in: list[ChatTurn] = []
    for t in history or []:
        role = t.get("role")
        content = t.get("content")
        if (role != "user" and role != "assistant") or not isinstance(content, str):
            continue
        trimmed = content.strip()
        if not trimmed:
            continue
        hist_in.append({"role": role, "content": trimmed[:400]})
    hist_in = hist_in[-CONVERSE_HISTORY_CAP:]

    legacy_lang = normalize_lang(lang)
    if spoken_lang is not None:
        sl = normalize_lang(spoken_lang)
    else:
        sl = legacy_lang
    if learn_lang is not None:
        ll = normalize_lang(learn_lang)
    elif spoken_lang is not None:
        ll = "en" if sl == "zh" else "zh"
    else:
        ll = legacy_lang

    desc_in = (description or "").strip()
    desc_final: Optional[str] = desc_in[:600] if desc_in else None

    client_transcript = (transcript or "").strip()[:1000]
    tid = ((turn_id or "").strip() or "?")[:16]
    tag = f" #{tid}"

    if not isinstance(audio_bytes, (bytes, bytearray)):
        logger.info(f"[converse{tag}] ✖ missing audio")
        raise ValueError("missing audio")

    size = len(audio_bytes)
    voice_label_entry = voice_by_id(resolved_voice_input)
    voice_label = (
        voice_label_entry["name"]
        if voice_label_entry
        else (resolved_voice_input or "default")
    )
    logger.info(
        f'[converse{tag}] ▶ class="{cls}"  learn={ll} spoken={sl}  '
        f"voice={voice_label}  audio={round(size / 1024)}KB  hist={len(hist_in)}  "
        f'desc={str(len(desc_final)) + "ch" if desc_final else "none"}  '
        f'client-stt={"yes" if client_transcript else "no"}  '
        f'mime={audio_mime_type or "?"}'
    )

    if size < 1024:
        logger.info(f"[converse{tag}] ✖ recording too short ({size}B)")
        raise ValueError("recording too short")
    if size > 10_000_000:
        logger.info(f"[converse{tag}] ✖ recording too large ({size}B)")
        raise ValueError("recording too large")

    resolved_voice = resolved_voice_input or get_default_voice_id(sl)

    # STT: all transcription now runs server-side. The client no longer
    # ships a browser-side Whisper model. We still accept a ``transcript``
    # field in the multipart body purely as a fast-path hint — e.g. if the
    # UI ever wires up the browser's Web Speech API again, that string can
    # be forwarded here and we'll skip the OpenAI round-trip. Empty is the
    # normal path today.
    if client_transcript:
        transcript_final = client_transcript
        stt_backend = "client-hint"
        stt_ms = 0
    else:
        stt_result = await stt_service.transcribe_audio(
            bytes(audio_bytes),
            lang=sl,
            mime_type=audio_mime_type,
            turn_tag=tag,
        )
        transcript_final = stt_result["text"]
        stt_backend = stt_result["backend"]
        stt_ms = stt_result["ms"]
    trunc = transcript_final[:120] + ("…" if len(transcript_final) > 120 else "")
    logger.info(
        f"[stt {stt_backend}{tag}] "
        f'{"✓" if transcript_final else "∅"} {stt_ms}ms "{trunc}"'
    )

    if not transcript_final:
        fallbacks_en = [
            "hmm?",
            "huh?",
            "what was that?",
            "say that again?",
            "wait, what?",
            "come again?",
            "speak up, buddy.",
            "didn't catch that.",
            "you gonna finish that thought?",
            "mumble much?",
            "one more time?",
            "eh?",
            "what?",
            "sorry, drifted off.",
            "run that back?",
        ]
        fallbacks_zh = [
            "啊?",
            "嗯?",
            "再说一次?",
            "你说啥?",
            "等等,什么?",
            "没听清。",
            "大点声嘛。",
            "再来一遍?",
            "你说完了吗?",
            "嘟囔啥呢?",
            "哈?",
            "抱歉,走神了。",
            "重新来过?",
        ]
        fallbacks = fallbacks_zh if ll == "zh" else fallbacks_en
        reply = random.choice(fallbacks)
        total = int((time.time() - t0) * 1000)
        logger.info(
            f"[converse{tag}] ✓ TOTAL={total}ms ━ stt={stt_backend}/{stt_ms}ms "
            f'(empty)  → reply="{reply}"'
        )
        return {
            "transcript": "",
            "reply": reply,
            "voiceId": resolved_voice,
            "emotion": "curiosity:high",
            "speed": None,
            "teachMode": False,
        }

    teach_mode = detect_teach_mode(transcript_final, hist_in, sl, ll)

    reply_result = await _openai_text_reply(
        system=RESPOND_SYSTEM(cls, desc_final, ll, sl, teach_mode),
        user_text=transcript_final,
        prior_messages=hist_in,
        max_tokens=360 if teach_mode else 220,
        temperature=0.7,
        turn_tag=tag,
    )
    parsed = _parse_group_line_json(reply_result["content"])
    reply = parsed["line"] or ""
    emotion = parsed["emotion"]
    speed = parsed["speed"]
    if not reply:
        raise RuntimeError("empty reply from model")

    total = int((time.time() - t0) * 1000)
    trunc_reply = reply[:80] + ("…" if len(reply) > 80 else "")
    logger.info(
        f"[converse{tag}] ✓ TOTAL={total}ms ━ stt={stt_backend}/{stt_ms}ms ▸ "
        f"llm={reply_result['backend']}/{reply_result['ms']}ms  "
        f'teach={teach_mode}  emotion={emotion or "-"}  speed={speed or "-"}  '
        f'reply="{trunc_reply}"'
    )
    return {
        "transcript": transcript_final,
        "reply": reply,
        "voiceId": resolved_voice,
        "emotion": emotion,
        "speed": speed,
        "teachMode": teach_mode,
    }


# === teacher_say ==========================================================


async def teacher_say(args: dict[str, Any]) -> TeacherSayResult:
    t0 = time.time()
    sl = normalize_lang(args.get("spokenLang"))
    ll = normalize_lang(args.get("learnLang"))
    description = str(args.get("description") or "").strip()[:800]
    class_name = (str(args.get("className") or "thing").strip() or "thing")[:60]
    name_in = args.get("objectName")
    object_name: Optional[str]
    if isinstance(name_in, str):
        object_name = name_in.strip()[:80] or None
    else:
        object_name = None
    user_text = str(args.get("userText") or "").strip()[:800]

    history_in = args.get("history")
    history: list[ChatTurn] = []
    if isinstance(history_in, list):
        for t in history_in:
            if (
                not t
                or (t.get("role") != "user" and t.get("role") != "assistant")
                or not isinstance(t.get("content"), str)
                or not t["content"].strip()
            ):
                continue
            history.append({"role": t["role"], "content": t["content"]})
        history = history[-CONVERSE_HISTORY_CAP:]

    tid_raw = (str(args.get("turnId") or "")).strip()
    if not tid_raw:
        tid_raw = f"t{format(int(time.time() * 1000), 'x')}"
    tid = tid_raw[:16]
    tag = f" #{tid}"

    if not description:
        raise ValueError("teacherSay: description is required")
    if not user_text:
        raise ValueError("teacherSay: userText is required")

    forced = args.get("teachMode") is True
    teach_mode = True if forced else detect_teach_mode(user_text, history, sl, ll)

    pinned = voice_by_id(args.get("voiceId"))
    if pinned and pinned["lang"] == sl:
        resolved_voice_id = pinned["id"]
    else:
        resolved_voice_id = pick_random_voice_id(sl) or get_default_voice_id(sl)

    voice_log_entry = voice_by_id(resolved_voice_id)
    voice_log = voice_log_entry["name"] if voice_log_entry else resolved_voice_id
    trunc_user = user_text[:80] + ("…" if len(user_text) > 80 else "")
    logger.info(
        f"[teacher{tag}] ▶ start teach={teach_mode} spoken={sl} learn={ll} "
        f'voice={voice_log} hist={len(history)} name="{object_name or class_name}" '
        f'desc={len(description)}ch user="{trunc_user}"'
    )

    system = GALLERY_TEACHER_SYSTEM(
        description, object_name, class_name, sl, ll, teach_mode
    )

    reply_result = await _openai_text_reply(
        system=system,
        user_text=user_text,
        prior_messages=history,
        max_tokens=320 if teach_mode else 180,
        temperature=0.6,
        turn_tag=f" teacher{tag}",
    )

    line = extract_text_line(reply_result["content"])
    if not line:
        logger.info(f"[teacher{tag}] ✖ empty line after parse")
        raise RuntimeError("empty teacher line")
    total = int((time.time() - t0) * 1000)
    trunc_line = line[:120] + ("…" if len(line) > 120 else "")
    logger.info(
        f"[teacher {reply_result['backend']}{tag}] ◀ done {total}ms "
        f'llm={reply_result["ms"]}ms teach={teach_mode} line="{trunc_line}"'
    )
    return {
        "line": line,
        "voiceId": resolved_voice_id,
        "turnId": tid,
        "teachMode": teach_mode,
    }


# === gallerize_card =======================================================


async def gallerize_card(args: dict[str, Any]) -> GallerizeCardResult:
    t0 = time.time()
    sl = normalize_lang(args.get("spokenLang"))
    ll = normalize_lang(args.get("learnLang"))
    description = str(args.get("description") or "").strip()[:600]
    object_name = str(args.get("objectName") or "").strip()[:80]
    class_name = str(args.get("className") or "").strip()[:60]
    name_for_prompt = object_name or class_name or "unnamed item"

    if not description:
        raise ValueError("gallerizeCard: description is required")
    if sl == ll:
        raise ValueError("gallerizeCard: spokenLang === learnLang (nothing to translate)")

    user_text = (
        f"Object name: {name_for_prompt}\n"
        f"Persona description: {description}\n\n"
        "Return the JSON bilingual card now."
    )

    logger.info(
        f'[gallerize] ▶ start name="{name_for_prompt}" learn={ll} spoken={sl} '
        f"desc={len(description)}ch"
    )

    reply_result = await _openai_text_reply(
        system=GALLERIZE_SYSTEM(sl, ll),
        user_text=user_text,
        prior_messages=[],
        max_tokens=260,
        temperature=0.4,
        turn_tag=" gallerize",
    )

    parsed = extract_json_object(reply_result["content"])
    if not isinstance(parsed, dict):
        raise RuntimeError("gallerizeCard: JSON parse failed")

    translated_name_raw = (
        parsed["translatedName"]
        if isinstance(parsed.get("translatedName"), str)
        else ""
    )
    intro_raw_any = parsed.get("bilingualIntro")
    intro_raw: dict[str, Any] = (
        intro_raw_any if isinstance(intro_raw_any, dict) else {}
    )
    learn_raw = intro_raw["learn"] if isinstance(intro_raw.get("learn"), str) else ""
    spoken_raw = intro_raw["spoken"] if isinstance(intro_raw.get("spoken"), str) else ""

    def _clean(s: str, max_len: int) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"""^["'`]+|["'`]+$""", "", s)
        return s[:max_len]

    translated_name = _clean(translated_name_raw, 60)
    learn = _clean(learn_raw, 200)
    spoken = _clean(spoken_raw, 200)

    if not translated_name or not learn or not spoken:
        raise RuntimeError("gallerizeCard: missing fields after parse")

    total = int((time.time() - t0) * 1000)
    logger.info(
        f"[gallerize {reply_result['backend']}] ◀ done {total}ms "
        f'llm={reply_result["ms"]}ms name="{translated_name}" '
        f'learn="{learn[:60]}" spoken="{spoken[:60]}"'
    )

    return {
        "translatedName": translated_name,
        "bilingualIntro": {"learn": learn, "spoken": spoken},
    }


__all__ = [
    "Lang",
    "AppMode",
    "ChatTurn",
    "Assessment",
    "BundledResult",
    "GenerateLineResult",
    "GroupPeer",
    "GroupTurn",
    "GroupLineResult",
    "ConverseResult",
    "TeacherSayResult",
    "GallerizeCardResult",
    "VOICE_CATALOG",
    "DEFAULT_VOICE_ID_EN",
    "DEFAULT_VOICE_ID_ZH",
    "GLM_MODEL_DEEP",
    "GLM_MODEL_FAST",
    "GLM_TIMEOUT_MS",
    "GLM_RETRIES",
    "DESCRIBE_MODEL",
    "GENERATE_BUNDLED_MODEL_GLM",
    "GENERATE_BUNDLED_MODEL_OPENAI",
    "REPLY_MODEL_CEREBRAS",
    "REPLY_MODEL_OPENAI",
    "CONVERSE_HISTORY_CAP",
    "GROUP_HISTORY_CAP",
    "normalize_lang",
    "lang_label",
    "detect_teach_mode",
    "get_voice_catalog",
    "get_default_voice_id",
    "pick_random_voice_id",
    "voice_by_id",
    "extract_json_object",
    "extract_text_line",
    "assess_object",
    "describe_object",
    "generate_line",
    "group_line",
    "converse_with_object",
    "teacher_say",
    "gallerize_card",
]
