// Streaming TTS passthrough.
//
// Why this exists: the server-action path serializes the whole mp3 as a
// base64 data URL inside JSON before the client even knows audio is ready,
// which burned 600–1500ms of dead air after the LLM finished. This route
// POSTs {text, voiceId, lang} and streams audio/mpeg bytes straight from
// the TTS provider to the browser — the client starts playing as soon as
// the first chunk lands.
//
// Fallback ladder: Cartesia Sonic (when CARTESIA_API_KEY set; ~90ms TTFB)
// → Fish (when FISH_API_KEY set) → OpenAI tts-1/nova → 503. OpenAI's
// audio endpoint doesn't truly stream, but piping its response body
// through still saves the base64 roundtrip and lets the browser's
// MediaSource decode as bytes arrive.

import { NextResponse } from "next/server";

export const runtime = "nodejs";
// Tell Next this is a long-lived streaming response.
export const dynamic = "force-dynamic";

const CARTESIA_TTS_URL = "https://api.cartesia.ai/tts/bytes";
const CARTESIA_VERSION = "2024-11-13";
const FISH_TTS_URL = "https://api.fish.audio/v1/tts";
const OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech";

type TtsBody = {
  text?: unknown;
  voiceId?: unknown;
  turnId?: unknown;
  lang?: unknown;
  // Cartesia Sonic experimental_controls — one "emotion:intensity" string
  // per entry. Valid emotions: anger, positivity, surprise, sadness,
  // curiosity. Intensities: lowest, low, (omit for medium), high, highest.
  // Speed: "slowest" | "slow" | "normal" | "fast" | "fastest" OR a number
  // in [-1, 1]. These flags are ignored on non-Cartesia fallbacks.
  emotion?: unknown;
  speed?: unknown;
};

const VALID_EMOTIONS = new Set([
  "anger",
  "positivity",
  "surprise",
  "sadness",
  "curiosity",
]);
const VALID_INTENSITIES = new Set(["lowest", "low", "high", "highest"]);
const VALID_SPEEDS = new Set([
  "slowest",
  "slow",
  "normal",
  "fast",
  "fastest",
]);

function sanitizeEmotion(raw: unknown): string[] {
  if (!Array.isArray(raw)) return [];
  const out: string[] = [];
  for (const item of raw) {
    if (typeof item !== "string") continue;
    const trimmed = item.trim().toLowerCase();
    if (!trimmed) continue;
    const [name, intensity] = trimmed.split(":");
    if (!VALID_EMOTIONS.has(name)) continue;
    if (intensity && !VALID_INTENSITIES.has(intensity)) continue;
    out.push(intensity ? `${name}:${intensity}` : name);
    if (out.length >= 3) break;
  }
  return out;
}

function sanitizeSpeed(raw: unknown): string | number | null {
  if (typeof raw === "number" && Number.isFinite(raw)) {
    return Math.max(-1, Math.min(1, raw));
  }
  if (typeof raw === "string") {
    const trimmed = raw.trim().toLowerCase();
    if (VALID_SPEEDS.has(trimmed)) return trimmed;
  }
  return null;
}

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
  const emotion = sanitizeEmotion(payload.emotion);
  const speed = sanitizeSpeed(payload.speed);

  // --- Cartesia Sonic (primary) ------------------------------------------
  // Sonic's /tts/bytes endpoint streams mp3 chunks over a single HTTP
  // response. TTFB is consistently ~90-150ms vs. Fish's 500-1200ms — this
  // is the single biggest win for perceived "instant" replies.
  const cartesiaKey = process.env.CARTESIA_API_KEY?.trim();
  if (cartesiaKey) {
    const modelId = process.env.CARTESIA_MODEL_ID?.trim() || "sonic-2";
    const defaultVoice =
      lang === "zh"
        ? process.env.CARTESIA_VOICE_ID_ZH?.trim() ||
          "0cd0cde2-3b93-42b5-bcb9-f214a591aa29"
        : process.env.CARTESIA_VOICE_ID_EN?.trim() ||
          "a0e99841-438c-4a64-b679-ae501e7d6091";
    // The client still pins a per-track Fish reference_id for character
    // variety, but Cartesia voice IDs live in a different namespace. Until
    // we map Fish→Cartesia, fall back to the lang-appropriate default when
    // the passed voiceId isn't a UUID (Cartesia voice IDs are UUIDs).
    const uuidRe = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    const chosenVoice = uuidRe.test(voiceId) ? voiceId : defaultVoice;
    const t0 = Date.now();
    // Build voice payload with optional experimental_controls. sonic-2 may
    // ignore or 400 on these fields depending on rollout; on failure we
    // fall through to Fish, so sending them is safe.
    type CartesiaVoice = {
      mode: "id";
      id: string;
      experimental_controls?: {
        emotion?: string[];
        speed?: string | number;
      };
    };
    const voicePayload: CartesiaVoice = { mode: "id", id: chosenVoice };
    if (emotion.length > 0 || speed !== null) {
      voicePayload.experimental_controls = {
        ...(emotion.length > 0 ? { emotion } : {}),
        ...(speed !== null ? { speed } : {}),
      };
    }
    // eslint-disable-next-line no-console
    console.log(
      `[tts cartesia${tag}] → model=${modelId} voice=${chosenVoice} lang=${lang} text=${text.length}ch emotion=[${emotion.join(",") || "-"}] speed=${speed ?? "-"}`
    );
    // TTFB watchdog. Normal Cartesia TTFB is 300-900ms; we've seen 14s+
    // hangs that block the route forever. Abort on no response headers
    // within 4s and fall through to Fish/OpenAI. The signal is cleared the
    // moment headers arrive so the body stream is never interrupted.
    const CARTESIA_TTFB_TIMEOUT_MS = 4000;
    const cartCtrl = new AbortController();
    const cartTimeout = setTimeout(
      () => cartCtrl.abort(new Error(`cartesia ttfb > ${CARTESIA_TTFB_TIMEOUT_MS}ms`)),
      CARTESIA_TTFB_TIMEOUT_MS
    );
    try {
      const cartRes = await fetch(CARTESIA_TTS_URL, {
        method: "POST",
        headers: {
          "X-API-Key": cartesiaKey,
          "Cartesia-Version": CARTESIA_VERSION,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model_id: modelId,
          transcript: text,
          voice: voicePayload,
          output_format: {
            container: "mp3",
            bit_rate: 128000,
            sample_rate: 44100,
          },
          language: lang === "zh" ? "zh" : "en",
        }),
        signal: cartCtrl.signal,
      });
      clearTimeout(cartTimeout);
      const ct = cartRes.headers.get("content-type") ?? "";
      if (cartRes.ok && cartRes.body && !ct.includes("application/json")) {
        // eslint-disable-next-line no-console
        console.log(
          `[tts cartesia${tag}] ✓ ttfb=${Date.now() - t0}ms streaming audio/mpeg`
        );
        return new Response(cartRes.body, {
          status: 200,
          headers: {
            "Content-Type": "audio/mpeg",
            "Cache-Control": "no-store",
            "X-Tts-Backend": "cartesia",
            Connection: "keep-alive",
          },
        });
      }
      const err = await cartRes.text().catch(() => "");
      // eslint-disable-next-line no-console
      console.log(
        `[tts cartesia${tag}] ✖ ${cartRes.status} in ${Date.now() - t0}ms: ${err.slice(0, 200)} — falling through`
      );
    } catch (err) {
      clearTimeout(cartTimeout);
      // eslint-disable-next-line no-console
      console.log(
        `[tts cartesia${tag}] ✖ exception: ${err instanceof Error ? err.message : String(err)} — falling through`
      );
    }
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
