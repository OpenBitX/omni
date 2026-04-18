// Streaming TTS passthrough.
//
// Why this exists: the server-action path serializes the whole mp3 as a
// base64 data URL inside JSON before the client even knows audio is ready,
// which burned 600–1500ms of dead air after the LLM finished. This route
// POSTs {text, voiceId} and streams audio/mpeg bytes straight from Fish
// (or OpenAI) to the browser — the client starts playing as soon as the
// first chunk lands.
//
// Fallback ladder: Fish (when FISH_API_KEY set) → OpenAI tts-1/nova →
// 503. OpenAI's audio endpoint doesn't truly stream, but piping its
// response body through still saves the base64 roundtrip and lets the
// browser's MediaSource decode as bytes arrive.

import { NextResponse } from "next/server";

export const runtime = "nodejs";
// Tell Next this is a long-lived streaming response.
export const dynamic = "force-dynamic";

const FISH_TTS_URL = "https://api.fish.audio/v1/tts";
const OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech";

type TtsBody = {
  text?: unknown;
  voiceId?: unknown;
  turnId?: unknown;
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
  // Turn id from the client — same correlation id used in [converse #N]
  // logs so you can grep one full press-to-sound flow out of the log.
  const turnId =
    typeof payload.turnId === "string"
      ? payload.turnId.trim().slice(0, 16) || "?"
      : "?";
  const tag = ` #${turnId}`;
  if (!text) {
    return NextResponse.json({ error: "missing text" }, { status: 400 });
  }

  const fishKey = process.env.FISH_API_KEY?.trim();
  if (fishKey) {
    const body: Record<string, unknown> = {
      text,
      format: "mp3",
      mp3_bitrate: 128,
      normalize: true,
      // `balanced` gets chunks out faster than `normal`; for 22-word
      // replies the quality cost is imperceptible and the latency win is
      // the entire point of this route.
      latency: "balanced",
      chunk_length: 100,
    };
    if (voiceId) body.reference_id = voiceId;

    const t0 = Date.now();
    // eslint-disable-next-line no-console
    console.log(
      `[tts fish${tag}] → text=${text.length}ch voice=${voiceId || "default"}`
    );
    const fishRes = await fetch(FISH_TTS_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${fishKey}`,
        "Content-Type": "application/json",
        model: process.env.FISH_MODEL?.trim() || "s1",
      },
      body: JSON.stringify(body),
    });
    const ct = fishRes.headers.get("content-type") ?? "";
    if (fishRes.ok && fishRes.body && !ct.includes("application/json")) {
      // eslint-disable-next-line no-console
      console.log(
        `[tts fish${tag}] ✓ ttfb=${Date.now() - t0}ms streaming audio/mpeg`
      );
      return new Response(fishRes.body, {
        status: 200,
        headers: {
          "Content-Type": "audio/mpeg",
          "Cache-Control": "no-store",
          "X-Tts-Backend": "fish",
          // Keep the connection hot for the whole stream.
          Connection: "keep-alive",
        },
      });
    }
    const err = await fishRes.text().catch(() => "");
    // eslint-disable-next-line no-console
    console.log(
      `[tts fish${tag}] ✖ ${fishRes.status} in ${Date.now() - t0}ms: ${err.slice(0, 160)} — falling through`
    );
  }

  const openaiKey = process.env.OPENAI_API_KEY?.trim();
  if (openaiKey) {
    const t0 = Date.now();
    // eslint-disable-next-line no-console
    console.log(`[tts openai${tag}] → text=${text.length}ch`);
    const oaRes = await fetch(OPENAI_TTS_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${openaiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "tts-1",
        voice: "nova",
        input: text,
        response_format: "mp3",
      }),
    });
    if (oaRes.ok && oaRes.body) {
      // eslint-disable-next-line no-console
      console.log(
        `[tts openai${tag}] ✓ ttfb=${Date.now() - t0}ms streaming audio/mpeg`
      );
      return new Response(oaRes.body, {
        status: 200,
        headers: {
          "Content-Type": "audio/mpeg",
          "Cache-Control": "no-store",
          "X-Tts-Backend": "openai",
          Connection: "keep-alive",
        },
      });
    }
    const err = await oaRes.text().catch(() => "");
    // eslint-disable-next-line no-console
    console.log(
      `[tts openai${tag}] ✖ ${oaRes.status} in ${Date.now() - t0}ms: ${err.slice(0, 160)}`
    );
  }

  return NextResponse.json(
    { error: "no TTS backend configured" },
    { status: 503 }
  );
}
