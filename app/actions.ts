"use server";

import OpenAI, { toFile } from "openai";

// Single-model strategy: `glm-5v-turbo` for everything. We had briefly split
// into a DEEP (assess) / FAST (hot-path) pair but the fast model on Zhipu
// (`glm-4v-flash`) was silently deprecated — probing on 2026-04-18 showed
// only `glm-4.5v` and `glm-5v-turbo` still responding. Going back to one
// model keeps the wiring simple; override at will via env.
//
// If you see `模型不存在，请检查模型代码` ("model does not exist") again, run
// `node scripts/test-glm.mjs` to discover which names are currently served.
const GLM_MODEL_DEEP = process.env.GLM_MODEL_DEEP?.trim() || "glm-5v-turbo";
const GLM_MODEL_FAST = process.env.GLM_MODEL_FAST?.trim() || "glm-5v-turbo";
const GLM_TIMEOUT_MS = 90_000;
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
  /** Which GLM model to call. Defaults to the fast VLM; the slower
   *  reasoning model is opt-in per caller. */
  model?: string;
}): Promise<string> {
  const client = getGlmClient();
  const model = args.model ?? GLM_MODEL_FAST;
  const tag = `[glm ${model}]`;
  let lastErr: unknown = null;
  for (let attempt = 1; attempt <= GLM_RETRIES + 1; attempt++) {
    const t0 = Date.now();
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
      // eslint-disable-next-line no-console
      console.log(
        `${tag} → call (attempt ${attempt}/${GLM_RETRIES + 1}, image=${args.imageDataUrl ? "yes" : "no"}, max_tokens=${args.maxTokens}, userText="${args.userText.slice(0, 80)}${args.userText.length > 80 ? "…" : ""}")`
      );
      const resp = await client.chat.completions.create({
        model,
        max_tokens: args.maxTokens,
        temperature: args.temperature,
        messages: [
          { role: "system", content: args.system },
          { role: "user", content: userContent },
        ],
      });
      const dt = Date.now() - t0;
      const content = resp.choices[0]?.message?.content ?? "";
      const usage = resp.usage;
      // eslint-disable-next-line no-console
      console.log(
        `${tag} ← ${dt}ms content=${content.length}ch tokens=${usage?.prompt_tokens ?? "?"}+${usage?.completion_tokens ?? "?"}`
      );
      if (typeof content !== "string" || !content.trim()) {
        throw new Error("empty response");
      }
      return content;
    } catch (err) {
      const dt = Date.now() - t0;
      lastErr = err;
      const msg = err instanceof Error ? err.message : String(err);
      // eslint-disable-next-line no-console
      console.log(
        `${tag} ✖ attempt ${attempt} failed after ${dt}ms: ${msg.slice(0, 160)}`
      );
      if (attempt > GLM_RETRIES) break;
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
  const t0 = Date.now();
  // eslint-disable-next-line no-console
  console.log(`[assess] ▶ start  tap=(${tx.toFixed(2)},${ty.toFixed(2)})  crop=${Math.round(imageDataUrl.length / 1024)}KB`);

  const raw = await glmVisionCall({
    system: ASSESS_SYSTEM,
    userText: `Tap at (${tx.toFixed(3)}, ${ty.toFixed(3)}). Find the best face placement and commit. Default to suitable=true — only say false for a person or a completely empty/uniform image. Return JSON only.`,
    imageDataUrl,
    // Reasoning headroom — the deep VLM spends most of its budget on inner
    // thought before emitting the actual JSON answer.
    maxTokens: 1536,
    temperature: 0.2,
    // assessObject runs once per tap and its quality matters more than its
    // latency — lean on the reasoning model here specifically.
    model: GLM_MODEL_DEEP,
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
  // eslint-disable-next-line no-console
  console.log(
    `[assess] ◀ done   suitable=${suitable}  face=(${cx.toFixed(2)},${cy.toFixed(2)})  reason="${reason}"  total=${Date.now() - t0}ms`
  );
  return { suitable, cx, cy, bbox, reason };
}

export async function generateLine(
  imageDataUrl: string
): Promise<{ line: string; audioDataUrl: string | null; backend: TtsBackend }> {
  if (!imageDataUrl.startsWith("data:image/")) {
    throw new Error("expected an image data URL");
  }
  const t0 = Date.now();
  // eslint-disable-next-line no-console
  console.log(`[generateLine] ▶ start  crop=${Math.round(imageDataUrl.length / 1024)}KB`);

  const raw = await glmVisionCall({
    system: FACE_SYSTEM,
    userText: "What does this thing say?",
    imageDataUrl,
    // glm-4.5v IS a reasoning model — it spends ~400–600 tokens on inner
    // thought before the actual line appears. 1024 leaves plenty of room
    // for a 14-word answer without truncation.
    maxTokens: 1024,
    temperature: 0.95,
  });
  const line = extractTextLine(raw);
  if (!line) throw new Error("empty line from model");
  // eslint-disable-next-line no-console
  console.log(`[generateLine]   line: "${line}"`);

  const tts = await synthesizeSpeech(line);
  // eslint-disable-next-line no-console
  console.log(
    `[generateLine] ◀ done  backend=${tts.backend}  total=${Date.now() - t0}ms`
  );
  return { line, audioDataUrl: tts.audioDataUrl, backend: tts.backend };
}

// === Voice-in, voice-out conversation ==================================
//
// The full loop:
//   1. Browser records audio via MediaRecorder → Blob
//   2. POST as FormData to this server action
//   3. OpenAI Whisper transcribes the audio
//   4. GLM-5V-Turbo generates an in-character reply (seeing the object
//      crop as vision context so it stays in the same persona)
//   5. Fish.audio TTS renders the reply; fallback ladder is OpenAI TTS
//      and then caption-only
// The client plays the returned audio through the track's AnalyserNode,
// which the FaceVoice mouth classifier is already wired to.

const FISH_TTS_URL = "https://api.fish.audio/v1/tts";

type TtsBackend = "fish" | "openai" | "none";

// Fish.audio synthesis. Returns mp3 bytes or null if the key is absent;
// throws on HTTP/content errors so the caller can fall through.
async function fishTTS(text: string): Promise<Buffer | null> {
  const key = process.env.FISH_API_KEY?.trim();
  if (!key) {
    // eslint-disable-next-line no-console
    console.log("[tts fish] — skipping: no FISH_API_KEY");
    return null;
  }

  const body: Record<string, unknown> = {
    text,
    format: "mp3",
    mp3_bitrate: 128,
    normalize: true,
    latency: "normal",
    chunk_length: 200,
  };
  const referenceId = process.env.FISH_REFERENCE_ID?.trim();
  if (referenceId) body.reference_id = referenceId;

  const t0 = Date.now();
  // eslint-disable-next-line no-console
  console.log(
    `[tts fish] → synthesizing ${text.length}ch ref=${referenceId ?? "default"} model=${process.env.FISH_MODEL?.trim() || "s1"}`
  );
  const resp = await fetch(FISH_TTS_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
      model: process.env.FISH_MODEL?.trim() || "s1",
    },
    body: JSON.stringify(body),
  });

  const dt = Date.now() - t0;
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    // eslint-disable-next-line no-console
    console.log(`[tts fish] ✖ ${resp.status} in ${dt}ms: ${text.slice(0, 160)}`);
    throw new Error(`fish ${resp.status}: ${text.slice(0, 200)}`);
  }
  const ct = resp.headers.get("content-type") ?? "";
  if (ct.includes("application/json")) {
    const text = await resp.text().catch(() => "");
    // eslint-disable-next-line no-console
    console.log(`[tts fish] ✖ non-audio response in ${dt}ms: ${text.slice(0, 160)}`);
    throw new Error(`fish returned non-audio: ${text.slice(0, 200)}`);
  }
  const buf = Buffer.from(await resp.arrayBuffer());
  // eslint-disable-next-line no-console
  console.log(`[tts fish] ← ${Math.round(buf.length / 1024)}KB mp3 in ${Date.now() - t0}ms`);
  return buf;
}

async function openaiTTS(text: string): Promise<Buffer | null> {
  const openai = getOpenAIClient();
  if (!openai) {
    // eslint-disable-next-line no-console
    console.log("[tts openai] — skipping: no OPENAI_API_KEY");
    return null;
  }
  const t0 = Date.now();
  // eslint-disable-next-line no-console
  console.log(`[tts openai] → tts-1/nova synthesizing ${text.length}ch`);
  const speech = await openai.audio.speech.create({
    model: "tts-1",
    voice: "nova",
    input: text,
    response_format: "mp3",
  });
  const buf = Buffer.from(await speech.arrayBuffer());
  // eslint-disable-next-line no-console
  console.log(`[tts openai] ← ${Math.round(buf.length / 1024)}KB mp3 in ${Date.now() - t0}ms`);
  return buf;
}

async function synthesizeSpeech(
  text: string
): Promise<{ audioDataUrl: string | null; backend: TtsBackend }> {
  // Fish first when configured — character-specific voices via reference_id
  // are a much better match for our talking-object vibe than `tts-1/nova`.
  try {
    const mp3 = await fishTTS(text);
    if (mp3) {
      return {
        audioDataUrl: `data:audio/mpeg;base64,${mp3.toString("base64")}`,
        backend: "fish",
      };
    }
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn("[tts] fish failed, falling back:", err);
  }
  // OpenAI as a dependable fallback.
  try {
    const mp3 = await openaiTTS(text);
    if (mp3) {
      return {
        audioDataUrl: `data:audio/mpeg;base64,${mp3.toString("base64")}`,
        backend: "openai",
      };
    }
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn("[tts] openai failed too:", err);
  }
  // Caption-only mode — the caller should still render the line.
  // eslint-disable-next-line no-console
  console.log("[tts] — no backend available, caption-only");
  return { audioDataUrl: null, backend: "none" };
}

async function transcribeAudio(blob: Blob): Promise<string> {
  const openai = getOpenAIClient();
  if (!openai) throw new Error("whisper needs OPENAI_API_KEY");
  const filename =
    blob.type.includes("mp4") ? "talk.mp4" :
    blob.type.includes("ogg") ? "talk.ogg" :
    "talk.webm";
  const file = await toFile(blob, filename, {
    type: blob.type || "audio/webm",
  });
  const t0 = Date.now();
  // eslint-disable-next-line no-console
  console.log(
    `[whisper] → transcribing ${Math.round(blob.size / 1024)}KB (${blob.type || "?"}) as ${filename}`
  );
  const result = await openai.audio.transcriptions.create({
    file,
    model: "whisper-1",
    // Leaving language unset so non-English speakers still work; pass an
    // explicit language if you want the ~10–30% latency boost.
  });
  const text = (result.text ?? "").trim();
  // eslint-disable-next-line no-console
  console.log(
    `[whisper] ← ${Date.now() - t0}ms "${text.slice(0, 120)}${text.length > 120 ? "…" : ""}"`
  );
  return text;
}

const RESPOND_SYSTEM = (className: string) =>
  `You are the secret inner voice of a ${className} the user is talking to. You see yourself in the crop provided. They just said something to you; reply with ONE short, in-character line (max 22 words) that actually responds to what they said.

Rules:
- First person, in character as the ${className}.
- Funny, warm, slightly unhinged. Aim for a smile.
- Actually acknowledge what they said — don't ignore their line.
- No meta-commentary, no "as a [thing]", no "I am a [thing]".
- No quotes, no emojis, no stage directions, no ellipses at the end.
- If their message is unclear, pick the most interesting interpretation and commit.

Return only the line. No prose, no <think> reasoning, no extra text.`;

export type ConverseResult = {
  transcript: string;
  reply: string;
  audioDataUrl: string | null;
  backend: TtsBackend;
};

export async function converseWithObject(
  formData: FormData
): Promise<ConverseResult> {
  const t0 = Date.now();
  const audio = formData.get("audio");
  const className = String(formData.get("className") ?? "thing").slice(0, 60);
  const imageDataUrl = String(formData.get("imageDataUrl") ?? "");

  if (!(audio instanceof Blob)) {
    // eslint-disable-next-line no-console
    console.log("[converse] ✖ missing audio");
    throw new Error("missing audio");
  }
  // eslint-disable-next-line no-console
  console.log(
    `[converse] ▶ start  class="${className}"  audio=${Math.round(audio.size / 1024)}KB (${audio.type || "?"})  crop=${imageDataUrl ? Math.round(imageDataUrl.length / 1024) + "KB" : "none"}`
  );

  if (audio.size < 1024) {
    // eslint-disable-next-line no-console
    console.log(`[converse] ✖ recording too short (${audio.size}B)`);
    throw new Error("recording too short");
  }
  if (audio.size > 10_000_000) {
    // eslint-disable-next-line no-console
    console.log(`[converse] ✖ recording too large (${audio.size}B)`);
    throw new Error("recording too large");
  }

  const transcript = await transcribeAudio(audio);
  if (!transcript) {
    // eslint-disable-next-line no-console
    console.log(`[converse] ◀ empty transcript — returning "hmm?"  total=${Date.now() - t0}ms`);
    return { transcript: "", reply: "hmm?", audioDataUrl: null, backend: "none" };
  }

  const raw = await glmVisionCall({
    system: RESPOND_SYSTEM(className),
    userText: `They said: "${transcript.replace(/"/g, "'")}". Reply in character — one short line.`,
    imageDataUrl: imageDataUrl.startsWith("data:image/") ? imageDataUrl : undefined,
    maxTokens: 1024,
    temperature: 0.95,
  });
  const reply = extractTextLine(raw);
  if (!reply) throw new Error("empty reply from model");
  // eslint-disable-next-line no-console
  console.log(`[converse]   reply: "${reply}"`);

  const tts = await synthesizeSpeech(reply);
  // eslint-disable-next-line no-console
  console.log(
    `[converse] ◀ done  backend=${tts.backend}  total=${Date.now() - t0}ms`
  );
  return { transcript, reply, audioDataUrl: tts.audioDataUrl, backend: tts.backend };
}
