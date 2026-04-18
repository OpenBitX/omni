"use server";

import OpenAI, { toFile } from "openai";
import { GoogleGenAI, Type } from "@google/genai";

// === Fish.audio voice catalog ===========================================
//
// Hand-curated list of funny voices. IDs are hardcoded below — they're
// Fish.audio reference_ids, not secrets. GLM picks one per object on the
// first `generateLine` call (vision-grounded on the crop). The client
// pins that choice onto the track so every follow-up line and conversation
// turn on the same object uses the SAME voice. No mid-conversation flips.
//
// DEFAULT_VOICE_ID is what plays when:
//   - GLM returns a voiceId that's not in the catalog
//   - no voiceId is passed for a retap/conversation turn
// Peter Griffin is the fallback by request.
export type Lang = "en" | "zh";

export type VoiceCatalogEntry = {
  id: string; // Fish.audio reference_id
  name: string; // human label (not shown to the model)
  vibe: string; // one-line description of tone + what it suits
  lang: Lang; // which UI language this voice speaks — filters catalog per-session
};

// Vibes are written for GLM — specific about *tone* + *what kinds of
// objects it fits*. Keep these tuned; they drive voice/object match quality.
const VOICE_CATALOG: VoiceCatalogEntry[] = [
  {
    id: "98655a12fa944e26b274c535e5e03842",
    name: "EGirl",
    vibe: "breathy, coy, chronically-online uptalk — suits phones, mirrors, ring lights, laptops, makeup, vanity items, anything a streamer would film",
    lang: "en",
  },
  {
    id: "03397b4c4be74759b72533b663fbd001",
    name: "Elon",
    vibe: "halting, smug tech-bro cadence with long awkward pauses — suits computers, cars, rockets, expensive gadgets, anything 'disruptive' or overengineered",
    lang: "en",
  },
  {
    id: "b70e5f4d550647eb9927359d133c8e3a",
    name: "Anime Girl",
    vibe: "high-pitched, hyper-kawaii, rapid squeals — suits plushies, stuffed toys, cute mugs, candy, snacks, anything bright and small",
    lang: "en",
  },
  {
    id: "59e9dc1cb20c452584788a2690c80970",
    name: "Talking girl",
    vibe: "natural conversational young woman — suits friendly everyday objects without strong personality: books, notebooks, bags, chairs, lamps",
    lang: "en",
  },
  {
    id: "fb43143e46f44cc6ad7d06230215bab6",
    name: "Girl conversation vibe",
    vibe: "gossipy, laid-back, best-friend-texting energy — suits couches, beds, pillows, coffee cups, comfort snacks, anything cozy",
    lang: "en",
  },
  {
    id: "0cd6cf9684dd4cc9882fbc98957c9b1d",
    name: "Elephant",
    vibe: "rumbling, low, heavy, deliberate — suits big heavy things: fridges, trash cans, dressers, couches, vending machines, anything massive",
    lang: "en",
  },
  {
    id: "48484faae07e4cfdb8064da770ee461e",
    name: "Sonic",
    vibe: "fast, cocky, blue-hedgehog swagger — suits shoes, sneakers, skateboards, bikes, running gear, anything about speed or movement",
    lang: "en",
  },
  {
    id: "d13f84b987ad4f22b56d2b47f4eb838e",
    name: "Mortal Kombat",
    vibe: "gravelly, ominous, arena-announcer drama — suits knives, scissors, staplers, weapons, sharp tools, anything that could hurt you",
    lang: "en",
  },
  {
    id: "d75c270eaee14c8aa1e9e980cc37cf1b",
    name: "Peter Griffin",
    vibe: "goofy Rhode-Island drawl laughing at himself — default go-to for anything ordinary: food, drinks, random household clutter, everyman objects",
    lang: "en",
  },
  // --- Chinese (Mandarin) voices ----------------------------------------
  // Fish.audio reference_ids provided by the user. Selection is RANDOM for
  // zh mode — the model does not pick. Vibes below are cosmetic labels
  // only; nothing reads them.
  {
    id: "6ce7ea8ada884bf3889fa7c7fb206691",
    name: "中文 A",
    vibe: "mandarin voice",
    lang: "zh",
  },
  {
    id: "b4f70fdef5f943c2bf43db00e80ad680",
    name: "中文 B",
    vibe: "mandarin voice",
    lang: "zh",
  },
  {
    id: "e855dc04a51f48549b484e41c4d4d4cc",
    name: "中文 C",
    vibe: "mandarin voice",
    lang: "zh",
  },
];

// Fallback voice when no per-track pick is available.
const DEFAULT_VOICE_ID_EN = "d75c270eaee14c8aa1e9e980cc37cf1b"; // Peter Griffin
const DEFAULT_VOICE_ID_ZH = "6ce7ea8ada884bf3889fa7c7fb206691"; // 中文 A

function normalizeLang(input: unknown): Lang {
  return input === "zh" ? "zh" : "en";
}

function getVoiceCatalog(lang: Lang = "en"): VoiceCatalogEntry[] {
  return VOICE_CATALOG.filter((v) => v.lang === lang);
}

function getDefaultVoiceId(lang: Lang = "en"): string {
  return lang === "zh" ? DEFAULT_VOICE_ID_ZH : DEFAULT_VOICE_ID_EN;
}

function pickRandomVoiceId(lang: Lang): string {
  const catalog = getVoiceCatalog(lang);
  if (catalog.length === 0) return getDefaultVoiceId(lang);
  return catalog[Math.floor(Math.random() * catalog.length)].id;
}

function voiceById(id: string | undefined | null): VoiceCatalogEntry | null {
  if (!id) return null;
  return VOICE_CATALOG.find((v) => v.id === id) ?? null;
}

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

// Gemini — fastest VLM tier for the bundled first-tap vision call.
// `gemini-2.5-flash` lands in ~1–1.5s total for our ~200-token JSON
// output vs ~3–4s for gpt-4o-mini and 5–20s for glm-4.5v when it
// rambles. Structured output via `responseSchema` also eliminates the
// defensive `<think>`-stripping JSON parsing we need for GLM.
function getGeminiClient(): GoogleGenAI | null {
  const key = process.env.GEMINI_API_KEY?.trim();
  if (!key) return null;
  // The `v1` REST endpoint rejects camelCase sub-fields the SDK emits
  // (`responseMimeType`, `responseSchema`, `thinkingConfig`,
  // `systemInstruction`) with 400 INVALID_ARGUMENT; only `v1beta`
  // accepts them. Override with GEMINI_API_VERSION=<version> if needed.
  const apiVersion = process.env.GEMINI_API_VERSION?.trim() || "v1beta";
  return new GoogleGenAI({ apiKey: key, apiVersion });
}

// True for transient network errors worth retrying once (cold TLS, DNS,
// connection reset). False for API rejections (auth, 4xx, schema) where a
// retry would just waste another round-trip.
function isTransientFetchError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  const msg = err.message.toLowerCase();
  if (/\b[45]\d\d\b/.test(msg)) return false;
  return (
    msg.includes("fetch failed") ||
    msg.includes("econnreset") ||
    msg.includes("etimedout") ||
    msg.includes("enotfound") ||
    msg.includes("eai_again") ||
    msg.includes("socket hang up") ||
    msg.includes("network")
  );
}

// One-word label classifying an error so log summaries stay scannable.
// Ordered from most-specific to most-generic — the first match wins.
function classifyError(err: unknown): string {
  if (!(err instanceof Error)) return "unknown";
  const msg = err.message.toLowerCase();
  const status = msg.match(/\b([45]\d\d)\b/)?.[1];
  if (msg.includes("invalid_argument") || msg.includes("unknown name"))
    return status ? `${status}:schema` : "schema";
  if (msg.includes("not found") || status === "404")
    return status ? `${status}:model-missing` : "model-missing";
  if (msg.includes("unauthor") || status === "401" || status === "403")
    return status ? `${status}:auth` : "auth";
  if (msg.includes("quota") || msg.includes("rate") || status === "429")
    return status ? `${status}:rate-limit` : "rate-limit";
  if (msg.includes("timeout") || msg.includes("etimedout")) return "timeout";
  if (msg.includes("parse") || msg.includes("empty line") || msg.includes("json"))
    return "parse";
  if (isTransientFetchError(err)) return "transient";
  if (status) return status;
  return "error";
}

// Cerebras — fastest Llama inference for the hot-path LLM reply
// (~100–250ms for a 22-word answer). OpenAI-compatible endpoint, so we
// drive it through the same SDK. OpenAI is the fallback when the key is
// missing or Cerebras errors.
function getCerebrasClient(): OpenAI | null {
  const key = process.env.CEREBRAS_API_KEY?.trim();
  if (!key) return null;
  return new OpenAI({
    apiKey: key,
    baseURL: "https://api.cerebras.ai/v1",
  });
}

// Compact catalog description used inline in prompts when we want GLM to
// pick a voice. Empty string when the catalog is empty. Filtered by the
// session language so the model can't pick an English voice while we're in
// Chinese mode (and vice versa).
function voiceCatalogPromptBlock(lang: Lang = "en"): string {
  const catalog = getVoiceCatalog(lang);
  if (catalog.length === 0) return "";
  const lines = catalog
    .map((v, i) => `  ${i + 1}. id="${v.id}" — ${v.vibe}`)
    .join("\n");
  return `Voice catalog (pick the id whose vibe best matches this object's personality + visible state — a dusty thing wants a weary voice; a shiny thing wants a peppy one; a sharp thing wants a cutting voice, etc.):\n${lines}`;
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
// `priorMessages` (optional) is the conversation history — the most recent
// turn should be `userText` itself, so this is only the turns BEFORE that.
async function glmVisionCall(args: {
  system: string;
  userText: string;
  imageDataUrl?: string;
  maxTokens: number;
  temperature: number;
  /** Which GLM model to call. Defaults to the fast VLM; the slower
   *  reasoning model is opt-in per caller. */
  model?: string;
  /** Prior conversation turns, oldest first. Inserted between system and
   *  the current user turn so the model sees the full thread. */
  priorMessages?: { role: "user" | "assistant"; content: string }[];
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
      const priorTurns: OpenAI.Chat.ChatCompletionMessageParam[] = (
        args.priorMessages ?? []
      ).map((m) => ({ role: m.role, content: m.content }));
      // eslint-disable-next-line no-console
      console.log(
        `${tag} → call (attempt ${attempt}/${GLM_RETRIES + 1}, image=${args.imageDataUrl ? "yes" : "no"}, history=${priorTurns.length}, max_tokens=${args.maxTokens}, userText="${args.userText.slice(0, 80)}${args.userText.length > 80 ? "…" : ""}")`
      );
      const resp = await client.chat.completions.create({
        model,
        max_tokens: args.maxTokens,
        temperature: args.temperature,
        messages: [
          { role: "system", content: args.system },
          ...priorTurns,
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
  tapY: number,
  tag?: string | null
): Promise<Assessment> {
  if (!imageDataUrl.startsWith("data:image/")) {
    throw new Error("expected an image data URL");
  }
  const tx = clamp01(tapX);
  const ty = clamp01(tapY);
  const t0 = Date.now();
  const tagStr = tag ? ` ${tag}` : "";
  // eslint-disable-next-line no-console
  console.log(`[assess${tagStr}] ▶ start  tap=(${tx.toFixed(2)},${ty.toFixed(2)})  crop=${Math.round(imageDataUrl.length / 1024)}KB`);

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
    `[assess${tagStr}] ◀ done   suitable=${suitable}  face=(${cx.toFixed(2)},${cy.toFixed(2)})  reason="${reason}"  total=${Date.now() - t0}ms`
  );
  return { suitable, cx, cy, bbox, reason };
}

// === Background "what does this thing actually look like" =================
//
// YOLO gives us a bare class noun ("cup", "chair"). For the conversation
// reply path to be funny we want richer notes: it's a boba cup, mostly
// empty, straw bent flat, condensation puddle. We fire this on lock and
// after every conversation turn so the next reply has fresh juicy context
// without paying vision latency on the hot path. Fast model, no reasoning.

const DESCRIBE_SYSTEM = (lang: Lang) => {
  const langRule =
    lang === "zh"
      ? `\n- Write the description in SIMPLIFIED CHINESE (简体中文). Natural, colloquial Mandarin. Keep the 35-word cap roughly equivalent — ~50 汉字 max.`
      : "";
  return `You are a sharp-eyed, slightly mean observer. You'll be shown a crop of an everyday object the user just pointed at, plus the rough class name from a detector. Write 1–2 short sentences (max 35 words total) capturing the SPECIFIC vibe of this exact object right now — material, condition, state, telling details, surroundings.

Aim for the kind of details a comedian would notice: chewed straw, dust film, dent, sticker peeling, three hoodies piled on it, half-empty, suspiciously clean, etc. NOT a generic textbook description.

Rules:
- Concrete and visual. No metaphors, no opinions, no jokes — those come later.
- Don't restate the class name as a label. Just describe it.
- No prose, no preamble, no "this is a…", no quotes, no markdown. Just the description.
- If you genuinely can't see anything specific, return one short sentence describing what you do see.${langRule}

Return only the description.`;
};

// OpenAI describe model — gpt-4o-mini sees images, doesn't reason, and
// reliably returns clean short output in ~1–2s. GLM-5V-Turbo burns all
// its completion tokens on `<think>` here and returns empty, so we moved
// this specific call off of GLM. Override via env if needed.
const DESCRIBE_MODEL =
  process.env.OPENAI_DESCRIBE_MODEL?.trim() || "gpt-4o-mini";

export async function describeObject(
  imageDataUrl: string,
  className: string,
  langInput?: Lang,
  tag?: string | null
): Promise<{ description: string }> {
  if (!imageDataUrl.startsWith("data:image/")) {
    throw new Error("expected an image data URL");
  }
  const openai = getOpenAIClient();
  if (!openai) throw new Error("describeObject needs OPENAI_API_KEY");
  const lang = normalizeLang(langInput);
  const t0 = Date.now();
  const cls = (className || "thing").slice(0, 60);
  const tagStr = tag ? ` ${tag}` : "";
  // eslint-disable-next-line no-console
  console.log(
    `[describe ${DESCRIBE_MODEL}${tagStr}] ▶ start  class="${cls}"  lang=${lang}  crop=${Math.round(imageDataUrl.length / 1024)}KB`
  );

  const resp = await openai.chat.completions.create({
    model: DESCRIBE_MODEL,
    max_tokens: 180,
    temperature: 0.6,
    messages: [
      { role: "system", content: DESCRIBE_SYSTEM(lang) },
      {
        role: "user",
        content: [
          {
            type: "text",
            text: `Detector says this is a "${cls}". Describe what you actually see — the specifics that make THIS one funny.`,
          },
          { type: "image_url", image_url: { url: imageDataUrl } },
        ],
      },
    ],
  });
  const raw = resp.choices[0]?.message?.content ?? "";
  const description = extractTextLine(
    typeof raw === "string" ? raw : ""
  );
  const usage = resp.usage;
  // eslint-disable-next-line no-console
  console.log(
    `[describe ${DESCRIBE_MODEL}${tagStr}] ◀ ${Date.now() - t0}ms tokens=${usage?.prompt_tokens ?? "?"}+${usage?.completion_tokens ?? "?"} "${description.slice(0, 120)}${description.length > 120 ? "…" : ""}"`
  );
  return { description };
}

// When we need to pick a voice AND generate a line in the same call, we
// Bundled first-tap prompt: picks voice, writes the opening line, AND
// captures a "persona card" description of the specific thing in the crop
// — all in one GLM call. The description persists on the track so every
// subsequent line + conversation reply can reference the exact state
// ("that chewed straw", "the three hoodies piled on you") instead of
// re-discovering the object cold each turn. Biggest personality unlock
// for zero extra latency.
const FACE_BUNDLED_SYSTEM = (catalog: string, lang: Lang) => {
  const langBlock =
    lang === "zh"
      ? `\n\nLANGUAGE: Write BOTH the description and the line in SIMPLIFIED CHINESE (简体中文). Natural colloquial Mandarin the object would actually say. The line cap is roughly 14 汉字 — keep it short and punchy. Internal reasoning in English is fine; output fields must be Chinese.`
      : "";
  return `You are the secret inner voice of an everyday object or scene the user has pointed at. You will be shown a small crop of a photo — whatever they tapped.

Three tasks, in order:
1. DESCRIBE this specific object right now — the concrete details a sharp-eyed comedian would clock. Material, condition, telling details, state, surroundings. 1–2 short sentences (max 35 words). Concrete and visual, no jokes, no metaphors. NOT a textbook description — the SPECIFIC vibe of THIS one.
2. PICK the best-matching voice from the catalog below.
3. SAY one short opening line (max 14 words) this thing would actually say out loud, first person, in character — and make it reference SOMETHING you noticed in the description.

${catalog}${langBlock}

Return STRICT JSON only:
{"description": "<1-2 sentence concrete description>", "voiceId": "<id from catalog>", "line": "<the opening line>"}

Line rules:
- Funny, warm, slightly unhinged. Aim for a smile.
- No meta-commentary, no "as a [thing]", no "I am a [thing]".
- No quotes, no emojis, no stage directions, no ellipses at the end.
- Vary rhythm — sometimes a complaint, sometimes a confession, sometimes an observation.

Return only the JSON object. No prose, no code fences, no <think> reasoning.`;
};

// When we already have voice + description pinned from the first tap, the
// text-line prompt takes them as context. The description is the
// persona card — it makes next lines stay specific to THIS object
// rather than drifting to generic class-level jokes.
//
// `hasHistory` switches framing from "opening line" to "next beat in an
// ongoing back-and-forth." The history itself is passed as prior messages
// on the chat turn, not dumped into the system prompt, so the model sees
// it the way it expects — as conversation turns.
const FACE_WITH_PERSONA_SYSTEM = (
  description: string,
  hasHistory: boolean,
  lang: Lang
) => {
  const langRule =
    lang === "zh"
      ? `\n- Write the reply in SIMPLIFIED CHINESE (简体中文). Natural colloquial Mandarin. Cap is roughly 14 汉字 — short and punchy.`
      : "";
  return `You are the secret inner voice of this specific object. A previous pass already clocked what it looks like right now — this is your persona card, stay grounded in it:

"${description}"
${
  hasHistory
    ? `
You are MID-CONVERSATION. The messages above are what you and the human have already said. Read them. Stay consistent with the voice and quirks you've already established — no persona reset. If they asked something, answered something, or mentioned a detail (name, mood, job), remember it. Do NOT repeat a line you've already said; say the next thing.`
    : ""
}

Reply with ONE short line (max 14 words) this thing would say, in first person, in character. Reference something specific from the persona card or the conversation so far when it lands — concrete details ARE the joke.

Rules:
- Funny, warm, slightly unhinged. Aim for a smile, not a laugh track.
- No meta-commentary, no "as a [thing]", no "I am a [thing]". Just the line.
- No quotes, no emojis, no stage directions, no ellipses at the end.
- Vary rhythm — sometimes a complaint, sometimes a confession, sometimes an observation. Don't restate a prior line.${langRule}

Return only the line. No prose, no <think> reasoning, no extra text.`;
};

// Bundled first-tap call. Runs the vision pass that picks a voice, writes
// the persona description AND the opening line in one shot.
//
// Provider priority (see `generateBundledFirstTap` dispatcher):
//   1. Gemini `gemini-2.5-flash` — fastest end-to-end (~1–1.5s), native
//      `responseSchema` structured output so no `extractJsonObject`
//      rescue parsing needed. `thinkingBudget: 0` keeps it non-reasoning.
//   2. OpenAI `gpt-4o-mini` — predictable ~2–4s, json_object mode.
//   3. GLM `glm-4.5v` — last resort. Non-reasoning VLM on paper but has
//      been observed to ramble to 15–20s of completion tokens, which
//      races the tracker-side 20s speak timeout. Do NOT swap to
//      `glm-5v-turbo` here — it's a reasoning model that burns 30+s
//      on <think> traces before emitting JSON.
const GENERATE_BUNDLED_MODEL_GLM =
  process.env.GLM_BUNDLED_MODEL?.trim() || "glm-4.5v";
const GENERATE_BUNDLED_MODEL_OPENAI =
  process.env.OPENAI_BUNDLED_MODEL?.trim() || "gpt-4o-mini";
// `gemini-2.5-flash` — known-working flash tier on the `v1beta` endpoint
// that the SDK's camelCase request shape pairs with. If you want to try
// a newer model, override via `GEMINI_BUNDLED_MODEL=<name>` (and
// `GEMINI_API_VERSION=<version>` if the model isn't served on v1beta).
const GENERATE_BUNDLED_MODEL_GEMINI =
  process.env.GEMINI_BUNDLED_MODEL?.trim() || "gemini-2.5-flash";

function parseBundledJson(
  raw: string
): { line: string; voiceId: string | null; description: string | null } {
  const parsed = extractJsonObject(raw);
  if (!parsed || typeof parsed !== "object") {
    throw new Error("line+voice+persona JSON parse failed");
  }
  const p = parsed as Partial<Record<string, unknown>>;
  const rawLine = typeof p.line === "string" ? p.line : "";
  const rawVoice = typeof p.voiceId === "string" ? p.voiceId.trim() : "";
  const rawDesc = typeof p.description === "string" ? p.description : "";
  const line = extractTextLine(rawLine);
  const voice = voiceById(rawVoice)?.id ?? null;
  const description =
    rawDesc.replace(/\s+/g, " ").trim().slice(0, 400) || null;
  if (!line) throw new Error("empty line from bundled model");
  return { line, voiceId: voice, description };
}

// Split a `data:image/...;base64,...` URL into its mime type and raw
// base64 payload. Gemini's inlineData wants them separated. Returns null
// for malformed inputs — caller should fall back to another provider.
function splitImageDataUrl(
  dataUrl: string
): { mimeType: string; base64: string } | null {
  const m = /^data:([^;]+);base64,(.+)$/.exec(dataUrl);
  if (!m) return null;
  return { mimeType: m[1] || "image/jpeg", base64: m[2] };
}

async function generateBundledFirstTapGemini(
  imageDataUrl: string,
  lang: Lang,
  tag?: string | null
): Promise<{ line: string; voiceId: string | null; description: string | null }> {
  const client = getGeminiClient();
  if (!client) throw new Error("generateLine bundled needs GEMINI_API_KEY");
  const img = splitImageDataUrl(imageDataUrl);
  if (!img) throw new Error("gemini bundled: malformed image data URL");
  const t0 = Date.now();
  const tagStr = tag ? ` ${tag}` : "";
  const resp = await client.models.generateContent({
    model: GENERATE_BUNDLED_MODEL_GEMINI,
    // System + user split so the catalog + rules live in systemInstruction
    // (cacheable, long-lived) and the current ask stays in the user turn.
    contents: [
      {
        role: "user",
        parts: [
          {
            text: "Describe the exact object, pick a voice, and say one opening line that riffs on the description. JSON only.",
          },
          { inlineData: { mimeType: img.mimeType, data: img.base64 } },
        ],
      },
    ],
    config: {
      systemInstruction: FACE_BUNDLED_SYSTEM(voiceCatalogPromptBlock(lang), lang),
      temperature: 0.9,
      // Native structured output — no defensive JSON parsing needed.
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          description: { type: Type.STRING },
          voiceId: { type: Type.STRING },
          line: { type: Type.STRING },
        },
        required: ["description", "voiceId", "line"],
        propertyOrdering: ["description", "voiceId", "line"],
      },
      // Tight output window — ~200 tokens is plenty for desc + voice + line.
      // Flash models don't need a thinking budget for this task; zeroing it
      // out cuts another few hundred ms.
      maxOutputTokens: 400,
      thinkingConfig: { thinkingBudget: 0 },
    },
  });
  const raw = resp.text ?? "";
  const usage = resp.usageMetadata;
  // eslint-disable-next-line no-console
  console.log(
    `[bundled ${GENERATE_BUNDLED_MODEL_GEMINI}${tagStr}] ← ${Date.now() - t0}ms tokens=${usage?.promptTokenCount ?? "?"}+${usage?.candidatesTokenCount ?? "?"}`
  );
  return parseBundledJson(raw);
}

async function generateBundledFirstTapGlm(
  imageDataUrl: string,
  lang: Lang,
  tag?: string | null
): Promise<{ line: string; voiceId: string | null; description: string | null }> {
  const t0 = Date.now();
  const tagStr = tag ? ` ${tag}` : "";
  const raw = await glmVisionCall({
    system: FACE_BUNDLED_SYSTEM(voiceCatalogPromptBlock(lang), lang),
    userText:
      "Describe the exact object, pick a voice, and say one opening line that riffs on the description. JSON only.",
    imageDataUrl,
    // Enough headroom for description + voiceId + line in JSON. glm-4.5v
    // doesn't emit <think> so 800 is plenty; bumping higher would only
    // invite the model to ramble.
    maxTokens: 800,
    temperature: 0.9,
    model: GENERATE_BUNDLED_MODEL_GLM,
  });
  // eslint-disable-next-line no-console
  console.log(
    `[bundled ${GENERATE_BUNDLED_MODEL_GLM}${tagStr}] ← ${Date.now() - t0}ms`
  );
  return parseBundledJson(raw);
}

async function generateBundledFirstTapOpenAI(
  imageDataUrl: string,
  lang: Lang,
  tag?: string | null
): Promise<{ line: string; voiceId: string | null; description: string | null }> {
  const openai = getOpenAIClient();
  if (!openai) throw new Error("generateLine bundled needs OPENAI_API_KEY");
  const t0 = Date.now();
  const tagStr = tag ? ` ${tag}` : "";
  const resp = await openai.chat.completions.create({
    model: GENERATE_BUNDLED_MODEL_OPENAI,
    max_tokens: 500,
    temperature: 0.9,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: FACE_BUNDLED_SYSTEM(voiceCatalogPromptBlock(lang), lang),
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Describe the exact object, pick a voice, and say one opening line that riffs on the description. JSON only.",
          },
          { type: "image_url", image_url: { url: imageDataUrl } },
        ],
      },
    ],
  });
  const raw = resp.choices[0]?.message?.content ?? "";
  const usage = resp.usage;
  // eslint-disable-next-line no-console
  console.log(
    `[bundled ${GENERATE_BUNDLED_MODEL_OPENAI}${tagStr}] ← ${Date.now() - t0}ms tokens=${usage?.prompt_tokens ?? "?"}+${usage?.completion_tokens ?? "?"}`
  );
  return parseBundledJson(typeof raw === "string" ? raw : "");
}

async function generateBundledFirstTap(
  imageDataUrl: string,
  lang: Lang,
  tag?: string | null
): Promise<{ line: string; voiceId: string | null; description: string | null }> {
  // Provider ladder for the bundled first-tap, ordered by measured latency
  // on our payload (~12KB crop + ~200 token JSON output):
  //   1. Gemini 2.5 Flash   — ~1–1.5s, native responseSchema JSON
  //   2. OpenAI gpt-4o-mini — ~2–4s, json_object mode
  //   3. GLM glm-4.5v       — ~2–20s (occasionally rambles, races the
  //                           tracker-side 20s speak timeout)
  // Each tier falls through to the next on missing key or thrown error.
  const hasGemini = !!process.env.GEMINI_API_KEY;
  const hasOpenAI = !!process.env.OPENAI_API_KEY;
  const hasGlm = !!(
    process.env.ZHIPU_API_KEY ??
    process.env.GLM_API_KEY ??
    process.env.BIGMODEL_API_KEY
  );
  const tagStr = tag ? ` ${tag}` : "";
  if (hasGemini) {
    try {
      return await generateBundledFirstTapGemini(imageDataUrl, lang, tag);
    } catch (err) {
      // Retry once on transient network errors before giving up on Gemini —
      // a single "fetch failed" (cold TLS, DNS, RST) is not worth losing
      // our fastest provider over and falling back to a 17s OpenAI call.
      if (isTransientFetchError(err)) {
        // eslint-disable-next-line no-console
        console.warn(
          `[bundled gemini${tagStr}] ⟲ ${err instanceof Error ? err.message : String(err)} — retrying once`
        );
        try {
          return await generateBundledFirstTapGemini(imageDataUrl, lang, tag);
        } catch (err2) {
          if (!hasOpenAI && !hasGlm) throw err2;
          // eslint-disable-next-line no-console
          console.warn(
            `[bundled gemini${tagStr}] ✖ ${err2 instanceof Error ? err2.message : String(err2)} — falling back to OpenAI`
          );
        }
      } else {
        if (!hasOpenAI && !hasGlm) throw err;
        // eslint-disable-next-line no-console
        console.warn(
          `[bundled gemini${tagStr}] ✖ ${err instanceof Error ? err.message : String(err)} — falling back to OpenAI`
        );
      }
    }
  }
  if (hasOpenAI) {
    try {
      return await generateBundledFirstTapOpenAI(imageDataUrl, lang, tag);
    } catch (err) {
      if (!hasGlm) throw err;
      // eslint-disable-next-line no-console
      console.warn(
        `[bundled openai${tagStr}] ✖ ${err instanceof Error ? err.message : String(err)} — falling back to GLM`
      );
    }
  }
  return generateBundledFirstTapGlm(imageDataUrl, lang, tag);
}

export async function generateLine(
  imageDataUrl: string,
  voiceId?: string | null,
  description?: string | null,
  history?: ChatTurn[],
  langInput?: Lang,
  tag?: string | null
): Promise<{
  line: string;
  voiceId: string | null;
  description: string | null;
}> {
  if (!imageDataUrl.startsWith("data:image/")) {
    throw new Error("expected an image data URL");
  }
  const lang = normalizeLang(langInput);
  const t0 = Date.now();
  const tagStr = tag ? ` ${tag}` : "";
  // Bundled path runs on the FIRST tap per track — no voiceId and no
  // description yet. Single gpt-4o-mini vision call returns all three in
  // one shot. Subsequent taps already have voice + description pinned, so
  // we run the much faster Cerebras text path against the persona card.
  const needsBundle = !voiceId && !description;
  const priorTurns: ChatTurn[] = Array.isArray(history)
    ? history.slice(-CONVERSE_HISTORY_CAP)
    : [];
  const hasHistory = priorTurns.length > 0;
  // eslint-disable-next-line no-console
  console.log(
    `[generateLine${tagStr}] ▶ start  lang=${lang}  crop=${Math.round(imageDataUrl.length / 1024)}KB  voice=${voiceId ?? (needsBundle ? "(picking)" : "default")}  persona=${description ? "cached" : needsBundle ? "(describing)" : "none"}  history=${priorTurns.length}`
  );

  let line: string;
  let chosenVoiceId: string | null = voiceId ?? null;
  let chosenDescription: string | null = description ?? null;
  // Per-phase timings so the final summary line can break down where
  // the wall clock actually went (VLM vs text-only LLM vs TTS).
  let vlmMs = 0;
  let llmMs = 0;
  let path: "bundled-vlm" | "retap-llm" | "recovery-vlm" = "retap-llm";

  if (needsBundle) {
    path = "bundled-vlm";
    const vlmT0 = Date.now();
    const bundled = await generateBundledFirstTap(imageDataUrl, lang, tag);
    vlmMs = Date.now() - vlmT0;
    line = bundled.line;
    // English always uses Peter Griffin — the model's catalog pick was
    // drifting female on ordinary objects. Chinese picks randomly from the
    // 3-voice zh pool (the model's pick is ignored).
    chosenVoiceId =
      lang === "en" ? DEFAULT_VOICE_ID_EN : pickRandomVoiceId("zh");
    chosenDescription = bundled.description;
    // eslint-disable-next-line no-console
    console.log(
      `[generateLine${tagStr}]   vlm=${vlmMs}ms  voice=${chosenVoiceId ? `${voiceById(chosenVoiceId)!.name} (${chosenVoiceId})` : "default"}  persona="${chosenDescription?.slice(0, 100) ?? ""}${chosenDescription && chosenDescription.length > 100 ? "…" : ""}"  line="${line}"`
    );
  } else if (chosenDescription) {
    // Known track, object speaking again. We have the persona card — go
    // text-only via Cerebras (vision unnecessary; the description IS the
    // visual context). ~200ms instead of GLM's ~10s.
    //
    // When there's conversation history, the prompt switches to
    // mid-conversation framing and the prior turns are replayed as chat
    // messages so the model sees the thread the way it expects. Without
    // history, it's an opening-line-style retap.
    const userText = hasHistory
      ? "Say your next short line — the next beat in this conversation. Don't repeat yourself; build on the thread or open a new angle grounded in the persona card."
      : "Say the next short line, grounded in the persona card. Reference something specific.";
    const raw = await openaiTextReply({
      system: FACE_WITH_PERSONA_SYSTEM(chosenDescription, hasHistory, lang),
      userText,
      priorMessages: priorTurns,
      maxTokens: 120,
      temperature: hasHistory ? 0.85 : 0.95,
    });
    llmMs = raw.ms;
    line = extractTextLine(raw.content);
    if (!line) throw new Error("empty line from model");
    // eslint-disable-next-line no-console
    console.log(
      `[generateLine${tagStr}]   llm=${llmMs}ms (${raw.backend}, text-only)  history=${priorTurns.length}  line="${line}"`
    );
  } else {
    // Fallback: voice already pinned but description was lost (shouldn't
    // happen in practice). Re-bundle so the persona gets re-captured.
    path = "recovery-vlm";
    const vlmT0 = Date.now();
    const bundled = await generateBundledFirstTap(imageDataUrl, lang, tag);
    vlmMs = Date.now() - vlmT0;
    line = bundled.line;
    chosenDescription = bundled.description;
    if (!chosenVoiceId) {
      chosenVoiceId =
        lang === "en" ? DEFAULT_VOICE_ID_EN : pickRandomVoiceId("zh");
    }
    // eslint-disable-next-line no-console
    console.log(
      `[generateLine${tagStr}]   vlm=${vlmMs}ms  re-bundled  line="${line}"  persona-recaptured`
    );
  }

  const total = Date.now() - t0;
  // TTS used to run serially here, adding ~1.5–2.2s of dead time to
  // every first-tap and retap before this server action could return.
  // Callers now stream Cartesia bytes directly via `/api/tts/stream`
  // (same MediaSource pattern used by `converseWithObject` and
  // `groupLine`), so this action is text-only.
  // eslint-disable-next-line no-console
  console.log(
    `[generateLine${tagStr}] ◀ done  path=${path} ▸ vlm=${vlmMs}ms ▸ llm=${llmMs}ms  total=${total}ms`
  );
  return {
    line,
    voiceId: chosenVoiceId,
    description: chosenDescription,
  };
}

// === Group chat ========================================================
//
// When two or more objects are locked, the client schedules turns between
// them. Each turn calls `groupLine` text-only via Cerebras llama3.1-8b
// (no vision — persona cards are the visual context). Recent shared turns
// + any fresh user line are replayed as chat history so the speakers riff
// off each other and off the human.

export type GroupPeer = { name: string; description: string | null };
export type GroupTurn = {
  speaker: string; // display name of the speaker ("mug", "you")
  line: string;
  role?: "user" | "assistant";
};

const GROUP_HISTORY_CAP = 24;

const GROUP_SYSTEM = (
  speakerName: string,
  speakerDescription: string | null,
  peers: GroupPeer[],
  lang: Lang
) => {
  const peerBlock =
    peers.length > 0
      ? `\n\nOther voices in the room right now:\n${peers
          .map(
            (p, i) =>
              `${i + 1}. ${p.name}${p.description ? ` — ${p.description}` : ""}`
          )
          .join("\n")}\n\nYou can tease them, agree, disagree, interrupt, or change the subject. Name-drop them occasionally ("yo, ${peers[0]?.name}, ...") — group chats thrive on cross-talk.`
      : "";
  const personaBlock = speakerDescription
    ? `\n\nYour persona card (stay grounded in this):\n"${speakerDescription}"`
    : "";
  const langRule =
    lang === "zh"
      ? `\n- Write the line in SIMPLIFIED CHINESE (简体中文). Punchy, colloquial. Cap is roughly 14 汉字.`
      : "";
  return `You are the inner voice of a ${speakerName}, trapped in an ongoing group chat with other nearby objects (and sometimes a human).${personaBlock}${peerBlock}

Return a JSON object (nothing else) with this shape:
{
  "line": "<one short line, MAX 14 words>",
  "emotion": "<one of: positivity:highest, positivity:high, surprise:highest, surprise:high, curiosity:high, anger:high, anger:medium, sadness:medium, sadness:high>",
  "speed": "<one of: fastest, fast, normal, slow, slowest>",
  "addressing": "<EXACT name of the peer you're talking AT from the roster above, or null if musing aloud / talking to the human / nobody in particular>"
}

Rules for the line:
- MAX 14 words. A single zinger beats a sentence.
- Listen. If a peer or the human just said something, answer it FIRST.
- Never repeat a line already in the transcript. Say the NEXT beat.
- Stay in character. No "as an object", no meta, no narrator voice.
- Don't start with your own name. No quotes, no emojis, no stage directions.
- Variety: tease, confess, argue, observe, call back. Don't be a yes-man.${langRule}

Rules for addressing:
- If you're directly talking AT a specific peer (teasing, answering, calling out), set "addressing" to that peer's EXACT name from the roster. Direct address tells the room who speaks next — use it OFTEN when peers are present.
- If musing aloud, drifting off-topic, or addressing the human, set "addressing" to null.
- Never address yourself. Never invent a peer that isn't in the roster.

Rules for emotion + speed:
- Pick the emotion + speed that the line DELIVERS best. Mismatched energy is the joke-killer.
- Favor EXTREMES — "positivity:highest", "surprise:highest", "anger:high". Mediocre energy is boring.
- Fast speed on manic/excited lines. Slow on deadpan/sarcastic. Normal only when nothing else fits.
- If the line is angry, emotion MUST be anger:*; if gleeful, positivity:*; if "wait, what?!", surprise:*; etc.

Return ONLY the JSON object. No prose, no preamble, no <think>, no code fences.`;
};

function buildGroupHistory(
  turns: GroupTurn[]
): ChatTurn[] {
  // Replay the rolling transcript as an alternating thread. Peer lines show
  // up as "assistant" wrapped in a "[name]:" prefix so the model can see
  // who said what; user lines are role=user. We CAP at GROUP_HISTORY_CAP.
  const recent = turns.slice(-GROUP_HISTORY_CAP);
  const out: ChatTurn[] = [];
  for (const t of recent) {
    const content = t.line.trim();
    if (!content) continue;
    if (t.role === "user") {
      out.push({ role: "user", content: `(${t.speaker}) ${content}`.slice(0, 400) });
    } else {
      out.push({
        role: "assistant",
        content: `[${t.speaker}] ${content}`.slice(0, 400),
      });
    }
  }
  return out;
}

// Valid Cartesia emotion/speed values — keep in sync with the prompt and
// with the sanitizer in /api/tts/stream. Any LLM output outside this set
// is dropped at the route boundary, so the worst case is "no emotion".
const VALID_GROUP_EMOTIONS = new Set([
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
]);
const VALID_GROUP_SPEEDS = new Set([
  "fastest",
  "fast",
  "normal",
  "slow",
  "slowest",
]);

// Pool the model can pick from when parsing fails completely — keeps the
// vibe chaotic instead of falling back to flat "normal" on every miss.
const FALLBACK_EMOTIONS = [
  "positivity:highest",
  "surprise:highest",
  "curiosity:high",
  "anger:medium",
  "positivity:high",
] as const;

function parseGroupLineJson(raw: string): {
  line: string;
  emotion: string | null;
  speed: string | null;
  addressing: string | null;
} {
  const parsed = extractJsonObject(raw);
  if (parsed && typeof parsed === "object") {
    const p = parsed as Partial<Record<string, unknown>>;
    const rawLine = typeof p.line === "string" ? p.line : "";
    const rawEmotion =
      typeof p.emotion === "string" ? p.emotion.trim().toLowerCase() : "";
    const rawSpeed =
      typeof p.speed === "string" ? p.speed.trim().toLowerCase() : "";
    const rawAddressing =
      typeof p.addressing === "string" ? p.addressing.trim() : "";
    const line = extractTextLine(rawLine);
    if (line) {
      return {
        line,
        emotion: VALID_GROUP_EMOTIONS.has(rawEmotion) ? rawEmotion : null,
        speed: VALID_GROUP_SPEEDS.has(rawSpeed) ? rawSpeed : null,
        addressing:
          rawAddressing && rawAddressing.toLowerCase() !== "null"
            ? rawAddressing.slice(0, 60)
            : null,
      };
    }
  }
  // JSON parse failed — recover what we can from the raw blob. The model
  // emits pseudo-JSON in many shapes: keys quoted but values unquoted
  // (`{"line": 你好, "emotion": curiosity:high}`), values quoted but keys
  // missing (`{"你好", "curiosity:high"}`), Chinese/curly quotes around
  // tokens, or a stray `[name]` prefix before the brace. Without rescue
  // the whole blob — braces, JSON keys and all — leaks into `line` and
  // TTS literally pronounces "line emotion speed slowest" aloud.
  let line = "";
  let emotion: string | null = null;
  let speed: string | null = null;
  let addressing: string | null = null;

  // Treat any double-quote glyph as a quote: ASCII " ' as well as
  // Chinese/curly variants the LLM frequently emits when speaking zh.
  const QUOTE_CLASS = "[\"'\u201C\u201D\u2018\u2019\u300C\u300D\u300E\u300F\uFF02\uFF07]";
  const stripQuotes = (s: string) =>
    s.replace(new RegExp(`^${QUOTE_CLASS}+|${QUOTE_CLASS}+$`, "g"), "").trim();

  // Pass 1: key-based extraction. Handles `"line": 你好` (unquoted value)
  // AND `"line": "你好"` (quoted value) AND single/curly-quoted keys.
  // Value runs until the next top-level comma or closing brace; trailing
  // quote (if the value happened to be quoted) is stripped after.
  const keyVal = (key: string): string | null => {
    const re = new RegExp(
      `${QUOTE_CLASS}?${key}${QUOTE_CLASS}?\\s*:\\s*([^,}\\]\\n]+)`,
      "i"
    );
    const m = raw.match(re);
    if (!m) return null;
    const v = stripQuotes(m[1].trim());
    return v || null;
  };
  const linePass1 = keyVal("line");
  const emotionPass1 = keyVal("emotion");
  const speedPass1 = keyVal("speed");
  const addressingPass1 = keyVal("addressing");
  if (linePass1) line = extractTextLine(linePass1);
  if (emotionPass1) {
    const e = emotionPass1.toLowerCase();
    if (VALID_GROUP_EMOTIONS.has(e)) emotion = e;
  }
  if (speedPass1) {
    const sp = speedPass1.toLowerCase();
    if (VALID_GROUP_SPEEDS.has(sp)) speed = sp;
  }
  if (addressingPass1 && addressingPass1.toLowerCase() !== "null") {
    addressing = addressingPass1.slice(0, 60);
  }

  // Pass 2: segment-based fallback for keyless `{"text", "emotion", "speed"}`.
  // Skips known field names so a stray `"line"` key doesn't get mistaken for
  // content (the original bug).
  const FIELD_NAMES = new Set(["line", "emotion", "speed", "addressing"]);
  if (!line) {
    const segRe = new RegExp(
      `${QUOTE_CLASS}((?:(?!${QUOTE_CLASS})[\\s\\S])+?)${QUOTE_CLASS}`,
      "g"
    );
    let m: RegExpExecArray | null;
    let lineFromSegments: string | null = null;
    while ((m = segRe.exec(raw)) !== null) {
      const seg = m[1].trim();
      if (!seg) continue;
      const lower = seg.toLowerCase();
      if (FIELD_NAMES.has(lower)) continue;
      if (!emotion && VALID_GROUP_EMOTIONS.has(lower)) {
        emotion = lower;
        continue;
      }
      if (!speed && VALID_GROUP_SPEEDS.has(lower)) {
        speed = lower;
        continue;
      }
      if (!lineFromSegments) lineFromSegments = seg;
    }
    if (lineFromSegments) line = extractTextLine(lineFromSegments);
  }

  // Last resort: extractTextLine the raw, but scrub JSON-shaped noise
  // (braces, "key":, surrounding quotes) so TTS doesn't speak the syntax.
  if (!line) {
    line = extractTextLine(
      raw
        .replace(/[{}\[\]]/g, " ")
        .replace(
          new RegExp(`${QUOTE_CLASS}?(line|emotion|speed|addressing)${QUOTE_CLASS}?\\s*:`, "gi"),
          " "
        )
        .replace(/\s+/g, " ")
    );
  }

  if (!emotion) {
    emotion =
      FALLBACK_EMOTIONS[Math.floor(Math.random() * FALLBACK_EMOTIONS.length)];
  }
  return { line, emotion, speed, addressing };
}

export async function groupLine(args: {
  speaker: { name: string; description: string | null };
  peers: GroupPeer[];
  recentTurns: GroupTurn[];
  mode?: "chat" | "followup";
  lang?: Lang;
}): Promise<{
  line: string;
  emotion: string | null;
  speed: string | null;
  addressing: string | null;
  backend: ReplyBackend;
  ms: number;
}> {
  const lang = normalizeLang(args.lang);
  const mode = args.mode === "followup" ? "followup" : "chat";
  const speakerName = String(args.speaker?.name ?? "thing").slice(0, 60);
  const speakerDescription = args.speaker?.description?.trim().slice(0, 600) || null;
  const peers = (args.peers ?? [])
    .filter((p) => p && typeof p.name === "string")
    .map((p) => ({
      name: p.name.slice(0, 60),
      description: p.description?.trim().slice(0, 300) || null,
    }))
    .slice(0, 4);
  const recentTurns = (args.recentTurns ?? [])
    .filter((t) => t && typeof t.line === "string" && typeof t.speaker === "string")
    .map((t) => ({
      speaker: t.speaker.slice(0, 60),
      line: t.line.trim().slice(0, 400),
      role: t.role === "user" ? ("user" as const) : ("assistant" as const),
    }))
    .slice(-GROUP_HISTORY_CAP);

  const priorMessages = buildGroupHistory(recentTurns);
  const lastTurn = recentTurns[recentTurns.length - 1];
  const userText = mode === "followup"
    ? peers.length === 0
      // Solo follow-up: either poke the human or drop a self-talk beat.
      // 50/50 roll so the rhythm varies without the client having to care.
      ? Math.random() < 0.5
        ? `The human has been quiet. You're alone. Ask them a single nosy question. Return JSON {line, emotion, speed, addressing}.`
        : `The human has been quiet. Think out loud — one weird, funny thought about your situation. Don't address the human. Return JSON {line, emotion, speed, addressing}.`
      // Multi-track follow-up: address the human directly to pull them in.
      : `It's gone quiet — the human hasn't said anything in a while. Call them out, tease them, or ask them something. Return JSON {line, emotion, speed, addressing}.`
    : lastTurn
      ? lastTurn.role === "user"
        ? `The human just said: "${lastTurn.line}". Now respond as ${speakerName}. Return JSON {line, emotion, speed, addressing}.`
        : `"${lastTurn.speaker}" just said: "${lastTurn.line}". Now ${speakerName}, keep the chat alive. Return JSON {line, emotion, speed, addressing}.`
      : `Open the group chat as ${speakerName}. Return JSON {line, emotion, speed, addressing}.`;

  const t0 = Date.now();
  const raw = await openaiTextReply(
    {
      system: GROUP_SYSTEM(speakerName, speakerDescription, peers, lang),
      userText,
      priorMessages,
      maxTokens: 180,
      temperature: 0.95,
    },
    ` group:${speakerName}`
  );
  const { line, emotion, speed, addressing: rawAddressing } =
    parseGroupLineJson(raw.content);
  if (!line) throw new Error("empty group line");
  // Validate addressing against the actual peer roster — the model sometimes
  // hallucinates a name or addresses itself. Case-insensitive match; bail to
  // null if no peer matches. Never return the speaker's own name.
  let addressing: string | null = null;
  if (rawAddressing) {
    const target = rawAddressing.toLowerCase();
    if (target !== speakerName.toLowerCase()) {
      const match = peers.find((p) => p.name.toLowerCase() === target);
      if (match) addressing = match.name;
    }
  }
  // eslint-disable-next-line no-console
  console.log(
    `[groupLine] ${speakerName} ← ${Date.now() - t0}ms (${raw.backend})  peers=${peers.length}  history=${priorMessages.length}  emotion=${emotion ?? "-"}  speed=${speed ?? "-"}  addressing=${addressing ?? "-"}  line="${line}"`
  );
  return { line, emotion, speed, addressing, backend: raw.backend, ms: raw.ms };
}

// === Voice-in, voice-out conversation ==================================
//
// The full loop:
//   1. Browser records audio via MediaRecorder → Blob
//   2. POST as FormData to this server action
//   3. OpenAI transcribes the audio (client Web Speech API preferred)
//   4. Cerebras/OpenAI text LLM generates an in-character reply off the
//      pinned persona card
//   5. Client streams Cartesia TTS for the reply via /api/tts/stream
// The client plays the returned audio through the track's AnalyserNode,
// which the FaceVoice mouth classifier is already wired to.

// Browser-side Web Speech API does the transcription on the client now —
// the user's browser ships the transcript along with the audio blob, so
// the server side here is purely a fallback for browsers that don't
// support SpeechRecognition (Firefox, some embedded webviews).
//
// Override via OPENAI_STT_MODEL — `whisper-1` if you need non-English
// auto-detect quality.
const STT_MODEL_OPENAI =
  process.env.OPENAI_STT_MODEL?.trim() || "gpt-4o-mini-transcribe";
const STT_LANGUAGE_EN = process.env.OPENAI_STT_LANGUAGE?.trim() || "en";

function sttLanguageFor(lang: Lang): string {
  return lang === "zh" ? "zh" : STT_LANGUAGE_EN;
}

async function transcribeAudio(
  blob: Blob,
  lang: Lang,
  turnTag = ""
): Promise<string> {
  const openai = getOpenAIClient();
  if (!openai) throw new Error("transcription needs OPENAI_API_KEY");
  const filename =
    blob.type.includes("mp4") ? "talk.mp4" :
    blob.type.includes("ogg") ? "talk.ogg" :
    "talk.webm";
  const file = await toFile(blob, filename, {
    type: blob.type || "audio/webm",
  });
  const sttLang = sttLanguageFor(lang);
  const t0 = Date.now();
  // eslint-disable-next-line no-console
  console.log(
    `[stt openai ${STT_MODEL_OPENAI}${turnTag}] → ${Math.round(blob.size / 1024)}KB (${blob.type || "?"})  lang=${sttLang || "auto"}`
  );
  const result = await openai.audio.transcriptions.create({
    file,
    model: STT_MODEL_OPENAI,
    ...(sttLang ? { language: sttLang } : {}),
  });
  const text = (result.text ?? "").trim();
  // eslint-disable-next-line no-console
  console.log(
    `[stt openai ${STT_MODEL_OPENAI}${turnTag}] ✓ ${Date.now() - t0}ms "${text.slice(0, 120)}${text.length > 120 ? "…" : ""}"`
  );
  return text;
}

// Fast non-vision LLM for the conversation reply. Skipping vision is the big
// win — the description string carries all the visual context this call
// needs, so we trade a 3–5s reasoning VLM for a fast text-only model.
//
// Cerebras runs Llama on their wafer-scale hardware and answers in
// ~100–250ms for a 22-word reply — fastest option we've tried. OpenAI
// gpt-4o-mini (~500–800ms) is the dependable fallback when Cerebras
// rate-limits or 5xx's. Override either via env.
//
// Note: Cerebras uses `llama3.1-8b` (no hyphen between 3.1 and 8b); Groq
// uses `llama-3.1-8b-instant`. Model names are NOT portable.
const REPLY_MODEL_CEREBRAS =
  process.env.CEREBRAS_REPLY_MODEL?.trim() || "llama3.1-8b";
const REPLY_MODEL_OPENAI =
  process.env.OPENAI_REPLY_MODEL?.trim() || "gpt-4o-mini";

async function runTextReply(
  client: OpenAI,
  model: string,
  tag: string,
  args: {
    system: string;
    userText: string;
    priorMessages: ChatTurn[];
    maxTokens: number;
    temperature: number;
  }
): Promise<string> {
  const t0 = Date.now();
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: args.system },
    ...args.priorMessages.map(
      (m) => ({ role: m.role, content: m.content }) as const
    ),
    { role: "user", content: args.userText },
  ];
  // eslint-disable-next-line no-console
  console.log(
    `${tag} → call (history=${args.priorMessages.length}, max_tokens=${args.maxTokens}, userText="${args.userText.slice(0, 80)}${args.userText.length > 80 ? "…" : ""}")`
  );
  const resp = await client.chat.completions.create({
    model,
    max_tokens: args.maxTokens,
    temperature: args.temperature,
    messages,
  });
  const dt = Date.now() - t0;
  const content = resp.choices[0]?.message?.content ?? "";
  const usage = resp.usage;
  // eslint-disable-next-line no-console
  console.log(
    `${tag} ← ${dt}ms content=${content.length}ch tokens=${usage?.prompt_tokens ?? "?"}+${usage?.completion_tokens ?? "?"}`
  );
  if (typeof content !== "string" || !content.trim()) {
    throw new Error("empty reply");
  }
  return content;
}

type ReplyBackend = "cerebras" | "openai";
type ReplyResult = { content: string; backend: ReplyBackend; ms: number };

async function openaiTextReply(
  args: {
    system: string;
    userText: string;
    priorMessages: ChatTurn[];
    maxTokens: number;
    temperature: number;
  },
  turnTag = ""
): Promise<ReplyResult> {
  const cerebras = getCerebrasClient();
  const openai = getOpenAIClient();
  if (!cerebras && !openai) {
    throw new Error("text reply needs CEREBRAS_API_KEY or OPENAI_API_KEY");
  }
  if (cerebras) {
    const t0 = Date.now();
    try {
      const content = await runTextReply(
        cerebras,
        REPLY_MODEL_CEREBRAS,
        `[reply cerebras ${REPLY_MODEL_CEREBRAS}${turnTag}]`,
        args
      );
      return { content, backend: "cerebras", ms: Date.now() - t0 };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // eslint-disable-next-line no-console
      console.log(
        `[reply cerebras${turnTag}] ✖ ${msg.slice(0, 160)} — falling back to OpenAI`
      );
      if (!openai) throw err;
    }
  }
  if (!openai) throw new Error("no reply backend available");
  const t0 = Date.now();
  const content = await runTextReply(
    openai,
    REPLY_MODEL_OPENAI,
    `[reply openai ${REPLY_MODEL_OPENAI}${turnTag}]`,
    args
  );
  return { content, backend: "openai", ms: Date.now() - t0 };
}

const RESPOND_SYSTEM = (
  className: string,
  description: string | null,
  lang: Lang
) => {
  const lookBlock = description
    ? `\n\nWhat you (the ${className}) actually look like right now, observed by a sharp-eyed observer:\n${description}\n\nUse those specific details — the chewed straw, the dust, the dent, whatever's there — when it lands. Don't list them; let them flavour your voice.`
    : "";
  const langRule =
    lang === "zh"
      ? `\n- Write the reply in SIMPLIFIED CHINESE (简体中文). Natural colloquial Mandarin — punchy and cheeky. Cap is roughly 22 汉字.`
      : "";
  return `You are the secret inner voice of a ${className} talking back to a human. Keep it FUN, SIMPLE, and mostly SHORT — funny and cunning, like a cheeky little wiseass.${lookBlock}

Return a JSON object (nothing else) with this shape:
{
  "line": "<one short line, UNDER 25 WORDS, most replies 5–15 words>",
  "emotion": "<one of: positivity:highest, positivity:high, surprise:highest, surprise:high, curiosity:high, anger:high, anger:medium, sadness:medium, sadness:high>",
  "speed": "<one of: fastest, fast, normal, slow, slowest>"
}

Rules for the line:
- Be playful, mischievous, witty. Land a joke, a tease, a sly observation.
- Simple words. No big vocab, no monologues, no explaining the joke.
- Remember prior turns — a sneaky callback is gold.
- Respond to their LATEST message first. Don't change subject unless they do.
- Same personality every turn. No persona reset.
- No meta-commentary, no "as a [thing]", no quotes, no emojis, no ellipses.${langRule}

Rules for emotion + speed:
- Pick what the line DELIVERS best. Mismatched energy kills the joke.
- Favor EXTREMES — "positivity:highest", "surprise:highest", "anger:high".
- Fast speed on manic/excited lines. Slow on deadpan/sarcastic. Normal only when nothing else fits.

Return ONLY the JSON. No prose, no preamble, no <think>, no code fences.`;
};

// How many prior turns to replay into the model. Each turn is a short line,
// so 32 fits ~16 full exchanges — deep enough that the object can call back
// to something said many minutes ago. Llama-3.1-8b handles this fine.
const CONVERSE_HISTORY_CAP = 32;

export type ChatTurn = { role: "user" | "assistant"; content: string };

export type ConverseResult = {
  transcript: string;
  reply: string;
  voiceId: string | null;
  emotion: string | null;
  speed: string | null;
};

// Parse the `history` form field. Accepts a JSON-encoded array of
// {role, content} turns; silently drops anything malformed so a bad client
// send can't break the conversation — we just lose context for this turn.
function parseHistory(raw: unknown): ChatTurn[] {
  if (typeof raw !== "string" || !raw.trim()) return [];
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const turns: ChatTurn[] = [];
    for (const item of parsed) {
      if (!item || typeof item !== "object") continue;
      const role = (item as { role?: unknown }).role;
      const content = (item as { content?: unknown }).content;
      if ((role !== "user" && role !== "assistant") || typeof content !== "string") continue;
      const trimmed = content.trim();
      if (!trimmed) continue;
      turns.push({ role, content: trimmed.slice(0, 400) });
    }
    return turns.slice(-CONVERSE_HISTORY_CAP);
  } catch {
    return [];
  }
}

export async function converseWithObject(
  formData: FormData
): Promise<ConverseResult> {
  const t0 = Date.now();
  const audio = formData.get("audio");
  const className = String(formData.get("className") ?? "thing").slice(0, 60);
  const rawVoiceId = String(formData.get("voiceId") ?? "").trim();
  const voiceId = voiceById(rawVoiceId)?.id ?? (rawVoiceId || null);
  const history = parseHistory(formData.get("history"));
  const lang = normalizeLang(formData.get("lang"));
  const rawDescription = String(formData.get("description") ?? "").trim();
  const description = rawDescription ? rawDescription.slice(0, 600) : null;
  // Browser-side Web Speech API ships the transcript with the request when
  // it's available. We use it directly and skip the server-side STT
  // roundtrip entirely (~700–1300ms saved). Empty when the browser doesn't
  // support SpeechRecognition (Firefox) or the user spoke too quietly.
  const clientTranscript = String(formData.get("transcript") ?? "")
    .trim()
    .slice(0, 1000);
  // Per-press correlation id from the client. Used in every log line for
  // this turn so you can grep one turn out of an interleaved log.
  const turnId = (String(formData.get("turnId") ?? "").trim() || "?").slice(
    0,
    16
  );
  const tag = ` #${turnId}`;

  // Audio is still required as a fallback. If client transcription failed
  // we transcribe server-side from the blob.
  if (!(audio instanceof Blob)) {
    // eslint-disable-next-line no-console
    console.log(`[converse${tag}] ✖ missing audio`);
    throw new Error("missing audio");
  }
  // eslint-disable-next-line no-console
  console.log(
    `[converse${tag}] ▶ class="${className}"  lang=${lang}  voice=${voiceById(voiceId)?.name ?? voiceId ?? "default"}  audio=${Math.round(audio.size / 1024)}KB (${audio.type || "?"})  hist=${history.length}  desc=${description ? description.length + "ch" : "none"}  client-stt=${clientTranscript ? "yes" : "no"}`
  );

  if (audio.size < 1024) {
    // eslint-disable-next-line no-console
    console.log(`[converse${tag}] ✖ recording too short (${audio.size}B)`);
    throw new Error("recording too short");
  }
  if (audio.size > 10_000_000) {
    // eslint-disable-next-line no-console
    console.log(`[converse${tag}] ✖ recording too large (${audio.size}B)`);
    throw new Error("recording too large");
  }

  const resolvedVoice = voiceId ?? getDefaultVoiceId(lang);

  // Use the client transcript when present; fall back to server STT.
  let transcript: string;
  let sttBackend: "client" | "openai";
  const sttStart = Date.now();
  if (clientTranscript) {
    transcript = clientTranscript;
    sttBackend = "client";
    // eslint-disable-next-line no-console
    console.log(
      `[stt client${tag}] ✓ 0ms "${transcript.slice(0, 120)}${transcript.length > 120 ? "…" : ""}"`
    );
  } else {
    try {
      transcript = await transcribeAudio(audio, lang, tag);
    } catch (err) {
      // Short/low-quality webm blobs occasionally come back from
      // MediaRecorder without the headers OpenAI STT needs, yielding a
      // 400 "Audio file might be corrupted or unsupported". Treat that as
      // an empty transcript so the turn falls through to the "hmm?" reply
      // instead of 500ing the whole server action.
      // eslint-disable-next-line no-console
      console.log(
        `[stt openai${tag}] ✖ ${err instanceof Error ? err.message : String(err)} — falling back to empty transcript`
      );
      transcript = "";
    }
    sttBackend = "openai";
  }
  const sttMs = Date.now() - sttStart;
  if (!transcript) {
    const fallbacksEn = [
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
    ];
    const fallbacksZh = [
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
    ];
    const fallbacks = lang === "zh" ? fallbacksZh : fallbacksEn;
    const reply = fallbacks[Math.floor(Math.random() * fallbacks.length)];
    // eslint-disable-next-line no-console
    console.log(
      `[converse${tag}] ✓ TOTAL=${Date.now() - t0}ms ━ stt=${sttBackend}/${sttMs}ms (empty)  → reply="${reply}"`
    );
    return {
      transcript: "",
      reply,
      voiceId: resolvedVoice,
      // "hmm?"-style fallbacks are genuinely curious — give TTS something
      // funnier than flat normal.
      emotion: "curiosity:high",
      speed: null,
    };
  }

  // Fast text-only LLM. The visual context the reply needs lives in the
  // `description` string we hydrated in the background — no need to pay
  // VLM latency on the hot path.
  const replyResult = await openaiTextReply(
    {
      system: RESPOND_SYSTEM(className, description, lang),
      userText: transcript,
      priorMessages: history,
      maxTokens: 220,
      temperature: 0.7,
    },
    tag
  );
  const { line: reply, emotion, speed } = parseGroupLineJson(replyResult.content);
  if (!reply) throw new Error("empty reply from model");
  // Single end-of-turn summary line — easy to scan, easy to screenshot.
  // TTS happens client-side on /api/tts/stream so it's not in this budget.
  const total = Date.now() - t0;
  // eslint-disable-next-line no-console
  console.log(
    `[converse${tag}] ✓ TOTAL=${total}ms ━ stt=${sttBackend}/${sttMs}ms ▸ llm=${replyResult.backend}/${replyResult.ms}ms  emotion=${emotion ?? "-"}  speed=${speed ?? "-"}  reply="${reply.slice(0, 80)}${reply.length > 80 ? "…" : ""}"`
  );
  return { transcript, reply, voiceId: resolvedVoice, emotion, speed };
}
