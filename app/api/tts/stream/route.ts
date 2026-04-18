// Streaming TTS passthrough.
//
// Why this exists: the server-action path serializes the whole mp3 as a
// base64 data URL inside JSON before the client even knows audio is ready,
// which burned 600–1500ms of dead air after the LLM finished. This route
// POSTs {text, voiceId, lang} and streams audio/mpeg bytes straight from
// the TTS provider to the browser — the client starts playing as soon as
// the first chunk lands.
//
// The provider ladder (Cartesia → Fish → OpenAI) lives in `lib/tts.ts`
// and is shared with `/api/speak`, which runs the VLM + TTS in one round
// trip for first-tap.

import { NextResponse } from "next/server";
import {
  sanitizeEmotion,
  sanitizeSpeed,
  streamTts,
} from "@/lib/tts";

export const runtime = "nodejs";
// Tell Next this is a long-lived streaming response.
export const dynamic = "force-dynamic";

type TtsBody = {
  text?: unknown;
  voiceId?: unknown;
  turnId?: unknown;
  lang?: unknown;
  emotion?: unknown;
  speed?: unknown;
};

export async function POST(req: Request) {
  let payload: TtsBody;
  try {
    payload = (await req.json()) as TtsBody;
  } catch {
    return NextResponse.json({ error: "bad json" }, { status: 400 });
  }
  const text =
    typeof payload.text === "string" ? payload.text.trim().slice(0, 600) : "";
  const voiceId =
    typeof payload.voiceId === "string" ? payload.voiceId.trim() : "";
  const lang =
    typeof payload.lang === "string" && payload.lang.trim() === "zh"
      ? "zh"
      : "en";
  const turnId =
    typeof payload.turnId === "string"
      ? payload.turnId.trim().slice(0, 16) || "?"
      : "?";
  if (!text) {
    return NextResponse.json({ error: "missing text" }, { status: 400 });
  }
  const emotion = sanitizeEmotion(payload.emotion);
  const speed = sanitizeSpeed(payload.speed);

  const result = await streamTts({
    text,
    voiceId,
    lang,
    turnId,
    emotion,
    speed,
  });
  if (!result) {
    return NextResponse.json(
      { error: "no TTS backend configured" },
      { status: 503 }
    );
  }
  return new Response(result.stream, {
    status: 200,
    headers: {
      "Content-Type": "audio/mpeg",
      "Cache-Control": "no-store",
      "X-Tts-Backend": result.backend,
      Connection: "keep-alive",
    },
  });
}
