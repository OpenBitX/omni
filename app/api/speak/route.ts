// Combined VLM + TTS endpoint for first-tap.
//
// Why this exists: previously first-tap took two client↔server round
// trips — one server action for the bundled VLM (returning {line,
// voiceId, description}), then a POST to /api/tts/stream for the audio.
// Between them we measured ~600ms of dead air (server-action response
// serialization + network back to browser + client firing a second
// fetch) on top of Cartesia's own ~800ms TTFB, so the caption appeared
// ~1.4s before the first audio sample.
//
// This route folds both steps into one response. The server runs the VLM,
// and the instant the line is parsed it fires Cartesia in parallel. The
// response body is audio/mpeg (identical to /api/tts/stream). The line +
// voiceId + description are base64-encoded into an `X-Speak-Meta` header
// so the client sees them the moment response headers arrive (before the
// first audio byte). Net effect: caption and audio surface together,
// saving the full client-side RTT.

import { NextResponse } from "next/server";
import { generateLine, type AppMode, type ChatTurn, type Lang } from "@/app/actions";
import { streamTts } from "@/lib/tts";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type SpeakBody = {
  imageDataUrl?: unknown;
  voiceId?: unknown;
  description?: unknown;
  history?: unknown;
  lang?: unknown;
  // New: language pair for the spoken/learn split. Either/both optional —
  // when absent, fall back to `lang`.
  spokenLang?: unknown;
  learnLang?: unknown;
  // Lens (play / language / history). Defaults to "play" when absent.
  mode?: unknown;
  turnId?: unknown;
};

function normalizeModeField(raw: unknown): AppMode {
  return raw === "language" || raw === "history" ? raw : "play";
}

function normalizeLangField(raw: unknown, fallback: Lang): Lang {
  if (raw === "zh") return "zh";
  if (raw === "en") return "en";
  return fallback;
}

function coerceHistory(raw: unknown): ChatTurn[] {
  if (!Array.isArray(raw)) return [];
  const out: ChatTurn[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const r = (item as { role?: unknown }).role;
    const c = (item as { content?: unknown }).content;
    if ((r === "user" || r === "assistant") && typeof c === "string") {
      out.push({ role: r, content: c });
    }
  }
  return out.slice(-32);
}

export async function POST(req: Request) {
  let payload: SpeakBody;
  try {
    payload = (await req.json()) as SpeakBody;
  } catch {
    return NextResponse.json({ error: "bad json" }, { status: 400 });
  }

  const imageDataUrl =
    typeof payload.imageDataUrl === "string" ? payload.imageDataUrl : "";
  if (!imageDataUrl.startsWith("data:image/")) {
    return NextResponse.json(
      { error: "expected data:image/ URL" },
      { status: 400 }
    );
  }
  const voiceId =
    typeof payload.voiceId === "string" && payload.voiceId.trim()
      ? payload.voiceId.trim()
      : null;
  const description =
    typeof payload.description === "string" && payload.description.trim()
      ? payload.description.trim()
      : null;
  const history = coerceHistory(payload.history);
  const lang: Lang = payload.lang === "zh" ? "zh" : "en";
  // Language pair. If only one side is present we infer the other as the
  // opposite — matches the server-action fallback in `converseWithObject`.
  const spokenLangProvided = payload.spokenLang === "zh" || payload.spokenLang === "en";
  const learnLangProvided = payload.learnLang === "zh" || payload.learnLang === "en";
  const spokenLang: Lang = spokenLangProvided
    ? normalizeLangField(payload.spokenLang, lang)
    : lang;
  const learnLang: Lang = learnLangProvided
    ? normalizeLangField(payload.learnLang, lang)
    : spokenLangProvided
      ? (spokenLang === "zh" ? "en" : "zh")
      : lang;
  const mode = normalizeModeField(payload.mode);
  const turnId =
    typeof payload.turnId === "string"
      ? payload.turnId.trim().slice(0, 16) || "?"
      : "?";
  const tag = ` #${turnId}`;

  // Step 1 — run the VLM (or text-only retap path). This is the single
  // slowest step on the critical path. Nothing can begin until `line`
  // exists, so we don't try to parallelize around it.
  const tVlm = Date.now();
  let line: string;
  let chosenVoiceId: string | null = voiceId;
  let chosenDescription: string | null = description;
  let chosenName: string | null = null;
  try {
    const result = await generateLine(
      imageDataUrl,
      voiceId,
      description,
      history,
      lang,
      turnId,
      spokenLang,
      learnLang,
      mode
    );
    line = result.line;
    chosenVoiceId = result.voiceId;
    chosenDescription = result.description;
    chosenName = result.name;
  } catch (err) {
    // eslint-disable-next-line no-console
    console.log(
      `[speak${tag}] ✖ generateLine failed: ${err instanceof Error ? err.message : String(err)}`
    );
    return NextResponse.json(
      {
        error: "generate failed",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 502 }
    );
  }
  const vlmMs = Date.now() - tVlm;
  // eslint-disable-next-line no-console
  console.log(
    `[speak${tag}] ◀ line ready in ${vlmMs}ms → firing TTS in same response  line="${line}"`
  );

  // Step 2 — fire Cartesia/Fish/OpenAI the moment `line` is ready. Runs
  // on the same server so the only wait is upstream TTFB.
  const tts = await streamTts({
    text: line,
    voiceId: chosenVoiceId ?? "",
    // TTS pronunciation is driven by the LEARN language (what the object
    // speaks), not by what the user speaks.
    lang: learnLang,
    turnId,
  });

  // `/api/speak` is the first-tap opening line — teach mode can't yet be
  // triggered from a user utterance (there isn't one), so we report it as
  // false. Prompt still picks the simple-language framing when spoken !=
  // learn, which is the only behavior that mattered here.
  const teachMode = false;
  const meta = {
    line,
    voiceId: chosenVoiceId,
    description: chosenDescription,
    // VLM-emitted short label. The client uses this as the user-facing
    // object name everywhere — never the YOLO class.
    name: chosenName,
    spokenLang,
    learnLang,
    teachMode,
  };
  // Base64 so we can carry the (potentially non-ASCII) metadata safely in
  // an HTTP header — headers can't hold raw newlines or binary.
  const metaB64 = Buffer.from(JSON.stringify(meta), "utf-8").toString("base64");

  if (!tts) {
    // No TTS provider available — return metadata only with 204 body.
    // Client will show caption without audio (same degraded mode as the
    // existing /api/tts/stream 503 path).
    // eslint-disable-next-line no-console
    console.log(
      `[speak${tag}] ⚠ no TTS backend — returning caption-only (meta in header, empty body)`
    );
    return new Response(null, {
      status: 204,
      headers: {
        "X-Speak-Meta": metaB64,
        "Cache-Control": "no-store",
      },
    });
  }

  return new Response(tts.stream, {
    status: 200,
    headers: {
      "Content-Type": "audio/mpeg",
      "Cache-Control": "no-store",
      "X-Speak-Meta": metaB64,
      "X-Tts-Backend": tts.backend,
      Connection: "keep-alive",
    },
  });
}
