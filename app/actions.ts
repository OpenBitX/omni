"use server";

import OpenAI from "openai";

// Vision + line model is Zhipu's GLM-5V-Turbo (same as the server-side
// Mirror pipeline in server/server.py). TTS stays on OpenAI because GLM
// doesn't ship a drop-in `tts-1 / nova` equivalent. The app degrades to
// caption-only (no audio) when `OPENAI_API_KEY` isn't set.
//
// NOTE: glm-5v-turbo is a reasoning model. It spends a chunk of its
// completion budget on `reasoning_content` before emitting `content`, so
// don't cap max_tokens tightly — leave enough headroom that reasoning
// doesn't crowd out the real answer.
const GLM_MODEL = "glm-5v-turbo";
const GLM_TIMEOUT_MS = 90_000; // reasoning-style model; give it headroom
const GLM_RETRIES = 2;

// bigmodel.cn keys look like `<hex>.<secret>`; api.z.ai keys look like `sk-…`.
// Both endpoints are OpenAI-compatible; pick by key shape unless overridden.
function glmBaseUrl(key: string): string {
  const override = process.env.ZHIPU_BASE_URL;
  if (override) return override;
  const looksBigModel = key.includes(".") && !key.startsWith("sk-");
  return looksBigModel
    ? "https://open.bigmodel.cn/api/paas/v4/"
    : "https://api.z.ai/api/paas/v4/";
}

function getGlmClient(): OpenAI {
  const key =
    process.env.ZHIPU_API_KEY ??
    process.env.GLM_API_KEY ??
    process.env.BIGMODEL_API_KEY;
  if (!key) throw new Error("ZHIPU_API_KEY not set");
  return new OpenAI({
    apiKey: key,
    baseURL: glmBaseUrl(key),
    timeout: GLM_TIMEOUT_MS,
  });
}

function getOpenAIClient(): OpenAI | null {
  const key = process.env.OPENAI_API_KEY;
  if (!key) return null;
  return new OpenAI({ apiKey: key });
}

const ASSESS_SYSTEM = `You place a cartoon face on whatever the user tapped. The user has already framed the subject by tapping — trust that and commit.

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

Return only the JSON — no prose, no code fences, no <think> reasoning.`;

const FACE_SYSTEM = `You are the secret inner voice of an everyday object or scene the user has pointed at. You will be shown a small crop of a photo — whatever they tapped. Reply with ONE short line (max 14 words) that this thing would say out loud if it could, in first person, in character.

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

Return only the line. No prose, no <think> reasoning, no extra text.`;

export type Assessment = {
  suitable: boolean;
  cx: number;
  cy: number;
  bbox: [number, number, number, number];
  reason: string;
};

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));

const fallbackBbox = (
  cx: number,
  cy: number,
  half = 0.08
): [number, number, number, number] => [
  clamp01(cx - half),
  clamp01(cy - half),
  clamp01(cx + half),
  clamp01(cy + half),
];

// GLM sometimes wraps output in ```json fences, prefixes with <think>…</think>
// reasoning traces, or adds a preamble before the JSON. The OpenAI
// response_format=json_object flag isn't reliably honored across Zhipu models,
// so we parse defensively: strip think tags, unwrap fences, slice from first
// `{` to last `}`.
function extractJsonObject(text: string): unknown | null {
  if (!text) return null;
  let s = text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  const fence = s.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fence) s = fence[1].trim();
  const first = s.indexOf("{");
  const last = s.lastIndexOf("}");
  if (first === -1 || last === -1 || last <= first) return null;
  const candidate = s.slice(first, last + 1);
  try {
    return JSON.parse(candidate);
  } catch {
    return null;
  }
}

// Strip GLM's occasional wrapping around a plain-text line response.
function extractTextLine(text: string): string {
  if (!text) return "";
  let s = text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  const fence = s.match(/```[\w-]*\s*([\s\S]*?)\s*```/);
  if (fence) s = fence[1].trim();
  return s
    .replace(/^["'`]+|["'`]+$/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 180);
}

// One-shot GLM chat call that retries on transient failures. `prompt` is the
// system-scoped instruction; `userText` + optional image form the user turn.
async function glmVisionCall(args: {
  system: string;
  userText: string;
  imageDataUrl?: string;
  maxTokens: number;
  temperature: number;
}): Promise<string> {
  const client = getGlmClient();
  let lastErr: unknown = null;
  for (let attempt = 1; attempt <= GLM_RETRIES + 1; attempt++) {
    try {
      const userContent: OpenAI.Chat.ChatCompletionContentPart[] = [
        { type: "text", text: args.userText },
      ];
      if (args.imageDataUrl) {
        userContent.push({
          type: "image_url",
          image_url: { url: args.imageDataUrl },
        });
      }
      const resp = await client.chat.completions.create({
        model: GLM_MODEL,
        max_tokens: args.maxTokens,
        temperature: args.temperature,
        messages: [
          { role: "system", content: args.system },
          { role: "user", content: userContent },
        ],
      });
      const content = resp.choices[0]?.message?.content ?? "";
      if (typeof content !== "string" || !content.trim()) {
        throw new Error("empty response");
      }
      return content;
    } catch (err) {
      lastErr = err;
      if (attempt > GLM_RETRIES) break;
      // Mild backoff — Zhipu rate-limits on bursts.
      await new Promise((r) => setTimeout(r, 400 * attempt));
    }
  }
  throw lastErr instanceof Error
    ? new Error(`GLM call failed: ${lastErr.message}`)
    : new Error("GLM call failed");
}

export async function assessObject(
  imageDataUrl: string,
  tapX: number,
  tapY: number
): Promise<Assessment> {
  if (!imageDataUrl.startsWith("data:image/")) {
    throw new Error("expected an image data URL");
  }
  const tx = clamp01(tapX);
  const ty = clamp01(tapY);

  const raw = await glmVisionCall({
    system: ASSESS_SYSTEM,
    userText: `Tap at (${tx.toFixed(3)}, ${ty.toFixed(3)}). Find the best face placement and commit. Default to suitable=true — only say false for a person or a completely empty/uniform image. Return JSON only.`,
    imageDataUrl,
    // Reasoning headroom — GLM spends most of its budget on inner thought
    // before emitting the actual JSON answer.
    maxTokens: 1536,
    temperature: 0.2,
  });

  const parsed = extractJsonObject(raw);
  if (!parsed || typeof parsed !== "object") {
    throw new Error("assessment JSON parse failed");
  }
  const p = parsed as Partial<Record<string, unknown>>;
  const suitable = p.suitable === true;
  const cx =
    typeof p.cx === "number" && Number.isFinite(p.cx) ? clamp01(p.cx) : tx;
  const cy =
    typeof p.cy === "number" && Number.isFinite(p.cy) ? clamp01(p.cy) : ty;

  let bbox: [number, number, number, number];
  const rawBbox = p.bbox;
  if (
    Array.isArray(rawBbox) &&
    rawBbox.length === 4 &&
    rawBbox.every((n) => typeof n === "number" && Number.isFinite(n))
  ) {
    const [rx0, ry0, rx1, ry1] = rawBbox as number[];
    const x0 = clamp01(Math.min(rx0, rx1));
    const y0 = clamp01(Math.min(ry0, ry1));
    const x1 = clamp01(Math.max(rx0, rx1));
    const y1 = clamp01(Math.max(ry0, ry1));
    bbox = x1 - x0 >= 0.04 && y1 - y0 >= 0.04 ? [x0, y0, x1, y1] : fallbackBbox(cx, cy);
  } else {
    bbox = fallbackBbox(cx, cy);
  }

  const reason =
    typeof p.reason === "string"
      ? p.reason.replace(/\s+/g, " ").trim().slice(0, 120)
      : "";
  return { suitable, cx, cy, bbox, reason };
}

export async function generateLine(
  imageDataUrl: string
): Promise<{ line: string; audioDataUrl: string | null }> {
  if (!imageDataUrl.startsWith("data:image/")) {
    throw new Error("expected an image data URL");
  }

  const raw = await glmVisionCall({
    system: FACE_SYSTEM,
    userText: "What does this thing say?",
    imageDataUrl,
    // Reasoning headroom — the actual line is ~14 words but the model
    // can spend 400+ tokens reasoning about tone, count, and vibe first.
    maxTokens: 1024,
    temperature: 0.95,
  });
  const line = extractTextLine(raw);
  if (!line) throw new Error("empty line from model");

  // TTS stays on OpenAI — GLM has no tts-1/nova equivalent. Skip audio if the
  // key is missing so a GLM-only deploy still shows captions.
  const openai = getOpenAIClient();
  if (!openai) {
    return { line, audioDataUrl: null };
  }

  try {
    const speech = await openai.audio.speech.create({
      model: "tts-1",
      voice: "nova",
      input: line,
      response_format: "mp3",
    });
    const buf = Buffer.from(await speech.arrayBuffer());
    const audioDataUrl = `data:audio/mpeg;base64,${buf.toString("base64")}`;
    return { line, audioDataUrl };
  } catch {
    // TTS failed — still return the line so the caption can render.
    return { line, audioDataUrl: null };
  }
}
