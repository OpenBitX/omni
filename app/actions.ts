"use server";

import OpenAI, { toFile } from "openai";

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
export type VoiceCatalogEntry = {
  id: string; // Fish.audio reference_id
  name: string; // human label (not shown to the model)
  vibe: string; // one-line description of tone + what it suits
};

// Vibes are written for GLM — specific about *tone* + *what kinds of
// objects it fits*. Keep these tuned; they drive voice/object match quality.
const VOICE_CATALOG: VoiceCatalogEntry[] = [
  {
    id: "98655a12fa944e26b274c535e5e03842",
    name: "EGirl",
    vibe: "breathy, coy, chronically-online uptalk — suits phones, mirrors, ring lights, laptops, makeup, vanity items, anything a streamer would film",
  },
  {
    id: "03397b4c4be74759b72533b663fbd001",
    name: "Elon",
    vibe: "halting, smug tech-bro cadence with long awkward pauses — suits computers, cars, rockets, expensive gadgets, anything 'disruptive' or overengineered",
  },
  {
    id: "b70e5f4d550647eb9927359d133c8e3a",
    name: "Anime Girl",
    vibe: "high-pitched, hyper-kawaii, rapid squeals — suits plushies, stuffed toys, cute mugs, candy, snacks, anything bright and small",
  },
  {
    id: "59e9dc1cb20c452584788a2690c80970",
    name: "Talking girl",
    vibe: "natural conversational young woman — suits friendly everyday objects without strong personality: books, notebooks, bags, chairs, lamps",
  },
  {
    id: "fb43143e46f44cc6ad7d06230215bab6",
    name: "Girl conversation vibe",
    vibe: "gossipy, laid-back, best-friend-texting energy — suits couches, beds, pillows, coffee cups, comfort snacks, anything cozy",
  },
  {
    id: "0cd6cf9684dd4cc9882fbc98957c9b1d",
    name: "Elephant",
    vibe: "rumbling, low, heavy, deliberate — suits big heavy things: fridges, trash cans, dressers, couches, vending machines, anything massive",
  },
  {
    id: "48484faae07e4cfdb8064da770ee461e",
    name: "Sonic",
    vibe: "fast, cocky, blue-hedgehog swagger — suits shoes, sneakers, skateboards, bikes, running gear, anything about speed or movement",
  },
  {
    id: "d13f84b987ad4f22b56d2b47f4eb838e",
    name: "Mortal Kombat",
    vibe: "gravelly, ominous, arena-announcer drama — suits knives, scissors, staplers, weapons, sharp tools, anything that could hurt you",
  },
  {
    id: "d75c270eaee14c8aa1e9e980cc37cf1b",
    name: "Peter Griffin",
    vibe: "goofy Rhode-Island drawl laughing at himself — default go-to for anything ordinary: food, drinks, random household clutter, everyman objects",
  },
];

// Fallback voice when no per-track pick is available — Peter Griffin.
const DEFAULT_VOICE_ID = "d75c270eaee14c8aa1e9e980cc37cf1b";

function getVoiceCatalog(): VoiceCatalogEntry[] {
  return VOICE_CATALOG;
}

function getDefaultVoiceId(): string {
  return DEFAULT_VOICE_ID;
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
// pick a voice. Empty string when the catalog is empty.
function voiceCatalogPromptBlock(): string {
  const catalog = getVoiceCatalog();
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

// === Background "what does this thing actually look like" =================
//
// YOLO gives us a bare class noun ("cup", "chair"). For the conversation
// reply path to be funny we want richer notes: it's a boba cup, mostly
// empty, straw bent flat, condensation puddle. We fire this on lock and
// after every conversation turn so the next reply has fresh juicy context
// without paying vision latency on the hot path. Fast model, no reasoning.

const DESCRIBE_SYSTEM = `You are a sharp-eyed, slightly mean observer. You'll be shown a crop of an everyday object the user just pointed at, plus the rough class name from a detector. Write 1–2 short sentences (max 35 words total) capturing the SPECIFIC vibe of this exact object right now — material, condition, state, telling details, surroundings.

Aim for the kind of details a comedian would notice: chewed straw, dust film, dent, sticker peeling, three hoodies piled on it, half-empty, suspiciously clean, etc. NOT a generic textbook description.

Rules:
- Concrete and visual. No metaphors, no opinions, no jokes — those come later.
- Don't restate the class name as a label. Just describe it.
- No prose, no preamble, no "this is a…", no quotes, no markdown. Just the description.
- If you genuinely can't see anything specific, return one short sentence describing what you do see.

Return only the description.`;

// OpenAI describe model — gpt-4o-mini sees images, doesn't reason, and
// reliably returns clean short output in ~1–2s. GLM-5V-Turbo burns all
// its completion tokens on `<think>` here and returns empty, so we moved
// this specific call off of GLM. Override via env if needed.
const DESCRIBE_MODEL =
  process.env.OPENAI_DESCRIBE_MODEL?.trim() || "gpt-4o-mini";

export async function describeObject(
  imageDataUrl: string,
  className: string
): Promise<{ description: string }> {
  if (!imageDataUrl.startsWith("data:image/")) {
    throw new Error("expected an image data URL");
  }
  const openai = getOpenAIClient();
  if (!openai) throw new Error("describeObject needs OPENAI_API_KEY");
  const t0 = Date.now();
  const cls = (className || "thing").slice(0, 60);
  // eslint-disable-next-line no-console
  console.log(
    `[describe ${DESCRIBE_MODEL}] ▶ start  class="${cls}"  crop=${Math.round(imageDataUrl.length / 1024)}KB`
  );

  const resp = await openai.chat.completions.create({
    model: DESCRIBE_MODEL,
    max_tokens: 120,
    temperature: 0.6,
    messages: [
      { role: "system", content: DESCRIBE_SYSTEM },
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
    `[describe ${DESCRIBE_MODEL}] ◀ ${Date.now() - t0}ms tokens=${usage?.prompt_tokens ?? "?"}+${usage?.completion_tokens ?? "?"} "${description.slice(0, 120)}${description.length > 120 ? "…" : ""}"`
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
const FACE_BUNDLED_SYSTEM = (catalog: string) => `You are the secret inner voice of an everyday object or scene the user has pointed at. You will be shown a small crop of a photo — whatever they tapped.

Three tasks, in order:
1. DESCRIBE this specific object right now — the concrete details a sharp-eyed comedian would clock. Material, condition, telling details, state, surroundings. 1–2 short sentences (max 35 words). Concrete and visual, no jokes, no metaphors. NOT a textbook description — the SPECIFIC vibe of THIS one.
2. PICK the best-matching voice from the catalog below.
3. SAY one short opening line (max 14 words) this thing would actually say out loud, first person, in character — and make it reference SOMETHING you noticed in the description.

${catalog}

Return STRICT JSON only:
{"description": "<1-2 sentence concrete description>", "voiceId": "<id from catalog>", "line": "<the opening line>"}

Line rules:
- Funny, warm, slightly unhinged. Aim for a smile.
- No meta-commentary, no "as a [thing]", no "I am a [thing]".
- No quotes, no emojis, no stage directions, no ellipses at the end.
- Vary rhythm — sometimes a complaint, sometimes a confession, sometimes an observation.

Return only the JSON object. No prose, no code fences, no <think> reasoning.`;

// When we already have voice + description pinned from the first tap, the
// text-line prompt takes them as context. The description is the
// persona card — it makes re-tap lines stay specific to THIS object
// rather than drifting to generic class-level jokes.
const FACE_WITH_PERSONA_SYSTEM = (
  description: string
) => `You are the secret inner voice of this specific object. A previous pass already clocked what it looks like right now — this is your persona card, stay grounded in it:

"${description}"

Reply with ONE short line (max 14 words) this thing would say, in first person, in character. Reference something specific from the persona card when you can — the concrete details ARE the joke.

Rules:
- Funny, warm, slightly unhinged. Aim for a smile, not a laugh track.
- No meta-commentary, no "as a [thing]", no "I am a [thing]". Just the line.
- No quotes, no emojis, no stage directions, no ellipses at the end.
- Vary rhythm — sometimes a complaint, sometimes a confession, sometimes an observation.

Return only the line. No prose, no <think> reasoning, no extra text.`;

// Bundled first-tap call. gpt-4o-mini sees the crop, picks a voice, writes
// the persona description AND the opening line in a single ~2–3s call.
// (GLM-5V-Turbo took 30+s here because it's a reasoning model — burned all
// its tokens on `<think>` before emitting JSON.)
const GENERATE_BUNDLED_MODEL =
  process.env.OPENAI_BUNDLED_MODEL?.trim() || "gpt-4o-mini";

async function generateBundledFirstTap(
  imageDataUrl: string
): Promise<{ line: string; voiceId: string | null; description: string | null }> {
  const openai = getOpenAIClient();
  if (!openai) throw new Error("generateLine bundled needs OPENAI_API_KEY");
  const t0 = Date.now();
  const resp = await openai.chat.completions.create({
    model: GENERATE_BUNDLED_MODEL,
    max_tokens: 400,
    temperature: 0.9,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: FACE_BUNDLED_SYSTEM(voiceCatalogPromptBlock()),
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
    `[bundled ${GENERATE_BUNDLED_MODEL}] ← ${Date.now() - t0}ms tokens=${usage?.prompt_tokens ?? "?"}+${usage?.completion_tokens ?? "?"}`
  );
  const parsed = extractJsonObject(typeof raw === "string" ? raw : "");
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

export async function generateLine(
  imageDataUrl: string,
  voiceId?: string | null,
  description?: string | null
): Promise<{
  line: string;
  voiceId: string | null;
  description: string | null;
  audioDataUrl: string | null;
  backend: TtsBackend;
}> {
  if (!imageDataUrl.startsWith("data:image/")) {
    throw new Error("expected an image data URL");
  }
  const t0 = Date.now();
  // Bundled path runs on the FIRST tap per track — no voiceId and no
  // description yet. Single gpt-4o-mini vision call returns all three in
  // one shot. Subsequent taps already have voice + description pinned, so
  // we run the much faster Cerebras text path against the persona card.
  const needsBundle = !voiceId && !description;
  // eslint-disable-next-line no-console
  console.log(
    `[generateLine] ▶ start  crop=${Math.round(imageDataUrl.length / 1024)}KB  voice=${voiceId ?? (needsBundle ? "(picking)" : "default")}  persona=${description ? "cached" : needsBundle ? "(describing)" : "none"}`
  );

  let line: string;
  let chosenVoiceId: string | null = voiceId ?? null;
  let chosenDescription: string | null = description ?? null;

  if (needsBundle) {
    const bundled = await generateBundledFirstTap(imageDataUrl);
    line = bundled.line;
    chosenVoiceId = bundled.voiceId;
    chosenDescription = bundled.description;
    // eslint-disable-next-line no-console
    console.log(
      `[generateLine]   voice=${chosenVoiceId ? `${voiceById(chosenVoiceId)!.name} (${chosenVoiceId})` : "default"}  persona="${chosenDescription?.slice(0, 100) ?? ""}${chosenDescription && chosenDescription.length > 100 ? "…" : ""}"  line="${line}"`
    );
  } else if (chosenDescription) {
    // Retap on a known track. We have the persona card — go text-only via
    // Cerebras (vision unnecessary; the description IS the visual context).
    // ~200ms instead of GLM's ~10s.
    const raw = await openaiTextReply({
      system: FACE_WITH_PERSONA_SYSTEM(chosenDescription),
      userText:
        "Say the next short line, grounded in the persona card. Reference something specific.",
      priorMessages: [],
      maxTokens: 80,
      temperature: 0.95,
    });
    line = extractTextLine(raw.content);
    if (!line) throw new Error("empty line from model");
    // eslint-disable-next-line no-console
    console.log(
      `[generateLine]   line="${line}" persona=used (text-only)`
    );
  } else {
    // Fallback: voice already pinned but description was lost (shouldn't
    // happen in practice). Re-bundle so the persona gets re-captured.
    const bundled = await generateBundledFirstTap(imageDataUrl);
    line = bundled.line;
    chosenDescription = bundled.description;
    if (!chosenVoiceId) chosenVoiceId = bundled.voiceId;
    // eslint-disable-next-line no-console
    console.log(
      `[generateLine]   re-bundled  line="${line}"  persona-recaptured`
    );
  }

  const tts = await synthesizeSpeech(line, chosenVoiceId);
  // eslint-disable-next-line no-console
  console.log(
    `[generateLine] ◀ done  backend=${tts.backend}  total=${Date.now() - t0}ms`
  );
  return {
    line,
    voiceId: chosenVoiceId,
    description: chosenDescription,
    audioDataUrl: tts.audioDataUrl,
    backend: tts.backend,
  };
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
// `voiceId` (when provided) overrides `FISH_REFERENCE_ID` — this is how a
// per-track voice stays consistent across the object's whole conversation.
async function fishTTS(
  text: string,
  voiceId?: string | null
): Promise<Buffer | null> {
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
  // Precedence: per-track pick → Peter (FISH_V9) default → legacy env.
  // Peter is the fallback-of-record so unmatched / unseeded taps still
  // sound intentional rather than a generic Fish.audio voice.
  const referenceId = voiceId?.trim() || getDefaultVoiceId();
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
  text: string,
  voiceId?: string | null
): Promise<{ audioDataUrl: string | null; backend: TtsBackend }> {
  // Fish first when configured — character-specific voices via reference_id
  // are a much better match for our talking-object vibe than `tts-1/nova`.
  try {
    const mp3 = await fishTTS(text, voiceId);
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

// Browser-side Web Speech API does the transcription on the client now —
// the user's browser ships the transcript along with the audio blob, so
// the server side here is purely a fallback for browsers that don't
// support SpeechRecognition (Firefox, some embedded webviews).
//
// Override via OPENAI_STT_MODEL — `whisper-1` if you need non-English
// auto-detect quality.
const STT_MODEL_OPENAI =
  process.env.OPENAI_STT_MODEL?.trim() || "gpt-4o-mini-transcribe";
const STT_LANGUAGE = process.env.OPENAI_STT_LANGUAGE?.trim() || "en";

async function transcribeAudio(blob: Blob, turnTag = ""): Promise<string> {
  const openai = getOpenAIClient();
  if (!openai) throw new Error("transcription needs OPENAI_API_KEY");
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
    `[stt openai ${STT_MODEL_OPENAI}${turnTag}] → ${Math.round(blob.size / 1024)}KB (${blob.type || "?"})  lang=${STT_LANGUAGE || "auto"}`
  );
  const result = await openai.audio.transcriptions.create({
    file,
    model: STT_MODEL_OPENAI,
    ...(STT_LANGUAGE ? { language: STT_LANGUAGE } : {}),
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

const RESPOND_SYSTEM = (className: string, description: string | null) => {
  const lookBlock = description
    ? `\n\nWhat you (the ${className}) actually look like right now, observed by a sharp-eyed observer:\n${description}\n\nUse those specific details — the chewed straw, the dust, the dent, whatever's there — when it lands. Don't list them; let them flavour your voice.`
    : "";
  return `You are the secret inner voice of a ${className} the user is talking to. You have been chatting with them — you see the prior turns in this thread and should stay consistent with the character you've already established.${lookBlock}

Reply with ONE short, in-character line (max 22 words) that actually responds to their latest message.

Rules:
- First person, in character as the ${className}. Same personality as the previous turns in this thread — don't reset the persona each turn.
- Funny, warm, slightly unhinged. Aim for a smile.
- ACTUALLY acknowledge what they just said — reference it, react to it, push back on it, or build on it. Don't dump a canned line that ignores their words.
- Call back to earlier turns when it lands — a running joke between you and the human is the goal.
- No meta-commentary, no "as a [thing]", no "I am a [thing]".
- No quotes, no emojis, no stage directions, no ellipses at the end.
- If their message is unclear, pick the most interesting interpretation and commit.

Return only the line. No prose, no extra text.`;
};

// How many prior turns to replay into the model. Each turn is a short line,
// so 16 fits ~8 full exchanges with headroom. More than this and GLM starts
// losing the persona thread anyway.
const CONVERSE_HISTORY_CAP = 16;

export type ChatTurn = { role: "user" | "assistant"; content: string };

export type ConverseResult = {
  transcript: string;
  reply: string;
  voiceId: string | null;
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
    `[converse${tag}] ▶ class="${className}"  voice=${voiceById(voiceId)?.name ?? voiceId ?? "default"}  audio=${Math.round(audio.size / 1024)}KB (${audio.type || "?"})  hist=${history.length}  desc=${description ? description.length + "ch" : "none"}  client-stt=${clientTranscript ? "yes" : "no"}`
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

  const resolvedVoice = voiceId ?? getDefaultVoiceId();

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
    transcript = await transcribeAudio(audio, tag);
    sttBackend = "openai";
  }
  const sttMs = Date.now() - sttStart;
  if (!transcript) {
    // eslint-disable-next-line no-console
    console.log(
      `[converse${tag}] ✓ TOTAL=${Date.now() - t0}ms ━ stt=${sttBackend}/${sttMs}ms (empty)  → reply="hmm?"`
    );
    return { transcript: "", reply: "hmm?", voiceId: resolvedVoice };
  }

  // Fast text-only LLM. The visual context the reply needs lives in the
  // `description` string we hydrated in the background — no need to pay
  // VLM latency on the hot path.
  const replyResult = await openaiTextReply(
    {
      system: RESPOND_SYSTEM(className, description),
      userText: transcript,
      priorMessages: history,
      maxTokens: 120,
      temperature: 0.95,
    },
    tag
  );
  const reply = extractTextLine(replyResult.content);
  if (!reply) throw new Error("empty reply from model");
  // Single end-of-turn summary line — easy to scan, easy to screenshot.
  // TTS happens client-side on /api/tts/stream so it's not in this budget.
  const total = Date.now() - t0;
  // eslint-disable-next-line no-console
  console.log(
    `[converse${tag}] ✓ TOTAL=${total}ms ━ stt=${sttBackend}/${sttMs}ms ▸ llm=${replyResult.backend}/${replyResult.ms}ms  reply="${reply.slice(0, 80)}${reply.length > 80 ? "…" : ""}"`
  );
  return { transcript, reply, voiceId: resolvedVoice };
}
