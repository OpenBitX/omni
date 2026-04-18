// Shared TTS provider ladder used by both `/api/tts/stream` and the
// combined `/api/speak` route. Lifted out of the route handler so the
// speak endpoint can fire Cartesia the instant the VLM returns instead
// of paying a second client→server round-trip.
//
// Ladder: Cartesia Sonic → Fish → OpenAI tts-1/nova → null (caller 503s).
// Whoever responds first wins; behavior is identical to the previous
// inlined version.

const CARTESIA_TTS_URL = "https://api.cartesia.ai/tts/bytes";
const CARTESIA_VERSION = "2024-11-13";
const FISH_TTS_URL = "https://api.fish.audio/v1/tts";
const OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech";

export type TtsBackend = "cartesia" | "fish" | "openai";

export type StreamTtsOptions = {
  text: string;
  voiceId?: string;
  lang?: "en" | "zh";
  turnId?: string;
  emotion?: string[];
  speed?: string | number | null;
};

export type StreamTtsResult = {
  stream: ReadableStream<Uint8Array>;
  backend: TtsBackend;
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

export function sanitizeEmotion(raw: unknown): string[] {
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

export function sanitizeSpeed(raw: unknown): string | number | null {
  if (typeof raw === "number" && Number.isFinite(raw)) {
    return Math.max(-1, Math.min(1, raw));
  }
  if (typeof raw === "string") {
    const trimmed = raw.trim().toLowerCase();
    if (VALID_SPEEDS.has(trimmed)) return trimmed;
  }
  return null;
}

// Some Cerebras Llama replies come back as UTF-8 bytes *interpreted* as
// latin-1 — e.g. `è¾²ä¹°è¿ç­é¼ ææ­¥` instead of the actual Chinese it
// meant to produce. When that string is piped to Cartesia, the voice
// pronounces the garbled latin characters verbatim (gibberish). This
// detects the pattern (every char sitting in the 0x80–0xFF range, no
// codepoints above) and, if the re-byte-then-decode-as-UTF-8 round trip
// yields real CJK, returns the recovered text. On any ambiguity we
// leave the original alone.
function fixMojibake(s: string): string {
  if (!s) return s;
  let suspect = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c > 0xff) return s;
    if (c >= 0x80) suspect++;
  }
  // Need meaningful signal — a single `é` in an English sentence is
  // not mojibake. Require at least 30% extended-latin density AND a
  // minimum absolute count so one-stray-char strings are left alone.
  if (suspect < 3 || suspect < s.length * 0.3) return s;
  try {
    const bytes = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
    const decoded = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    // Only accept the recovery if it yields CJK / kana / hangul — that's
    // the only universe where mojibake is plausibly happening for us.
    // Otherwise the source text was probably legitimate accented latin.
    if (/[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]/.test(decoded)) {
      // eslint-disable-next-line no-console
      console.log(
        `[tts] ⚠ mojibake recovered: ${s.slice(0, 40)}… → ${decoded.slice(0, 40)}…`
      );
      return decoded;
    }
  } catch {
    // Not valid UTF-8 — leave it.
  }
  return s;
}

// Returns the first provider that yields a streamable audio body. `null`
// means every configured provider failed — caller should 503.
export async function streamTts(
  opts: StreamTtsOptions
): Promise<StreamTtsResult | null> {
  const text = fixMojibake(opts.text).trim().slice(0, 600);
  if (!text) return null;
  const voiceId = (opts.voiceId ?? "").trim();
  const lang = opts.lang === "zh" ? "zh" : "en";
  const turnId = (opts.turnId ?? "?").slice(0, 16) || "?";
  const tag = ` #${turnId}`;
  const emotion = opts.emotion ?? [];
  const speed = opts.speed ?? null;

  // --- Cartesia Sonic (primary) ------------------------------------------
  const cartesiaKey = process.env.CARTESIA_API_KEY?.trim();
  if (cartesiaKey) {
    const modelId = process.env.CARTESIA_MODEL_ID?.trim() || "sonic-3";
    const defaultVoice =
      lang === "zh"
        ? process.env.CARTESIA_VOICE_ID_ZH?.trim() ||
          "0cd0cde2-3b93-42b5-bcb9-f214a591aa29"
        : process.env.CARTESIA_VOICE_ID_EN?.trim() ||
          "a0e99841-438c-4a64-b679-ae501e7d6091";
    const uuidRe =
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    const chosenVoice = uuidRe.test(voiceId) ? voiceId : defaultVoice;
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
    const t0 = Date.now();
    // eslint-disable-next-line no-console
    console.log(
      `[tts cartesia${tag}] → model=${modelId} voice=${chosenVoice} lang=${lang} text=${text.length}ch emotion=[${emotion.join(",") || "-"}] speed=${speed ?? "-"}`
    );
    const CARTESIA_TTFB_TIMEOUT_MS = 4000;
    const cartCtrl = new AbortController();
    const cartTimeout = setTimeout(
      () =>
        cartCtrl.abort(
          new Error(`cartesia ttfb > ${CARTESIA_TTFB_TIMEOUT_MS}ms`)
        ),
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
        return { stream: cartRes.body, backend: "cartesia" };
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

  // --- Fish (secondary) --------------------------------------------------
  const fishKey = process.env.FISH_API_KEY?.trim();
  if (fishKey) {
    const body: Record<string, unknown> = {
      text,
      format: "mp3",
      mp3_bitrate: 128,
      normalize: true,
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
      return { stream: fishRes.body, backend: "fish" };
    }
    const err = await fishRes.text().catch(() => "");
    // eslint-disable-next-line no-console
    console.log(
      `[tts fish${tag}] ✖ ${fishRes.status} in ${Date.now() - t0}ms: ${err.slice(0, 160)} — falling through`
    );
  }

  // --- OpenAI (tertiary) -------------------------------------------------
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
      return { stream: oaRes.body, backend: "openai" };
    }
    const err = await oaRes.text().catch(() => "");
    // eslint-disable-next-line no-console
    console.log(
      `[tts openai${tag}] ✖ ${oaRes.status} in ${Date.now() - t0}ms: ${err.slice(0, 160)}`
    );
  }

  return null;
}
