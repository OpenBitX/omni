// Runware image generation client + prompt builder.
//
// The /gallery page displays comic-book style illustrations of the physical
// objects the user tapped in the tracker. We feed Runware a prompt built
// from the card's className + persona description + a tiny summary of the
// conversation topic, and return one 512x512 JPEG URL.
//
// Keep this file server-only — `generateComicImage` reads the API key from
// process.env and is invoked from the /api/runware/generate route.

import OpenAI from "openai";
import type { ChatTurn } from "@/lib/session-cards";

export type RunwareGenerateInput = {
  className: string;
  description: string;
  // Recent conversation turns, newest last, already truncated by caller.
  history: ChatTurn[];
  spokenLang?: "en" | "zh";
  learnLang?: "en" | "zh";
  // Optional data URL of the cropped object. When provided + OPENAI_API_KEY
  // is set, we ask gpt-4o-mini to craft the comic prompt directly from the
  // image (it has seen the actual object, including all the specific
  // details a heuristic prompt misses — chewed straw, dust on the screen,
  // a peeling sticker). Falls back to the heuristic if anything goes wrong.
  imageDataUrl?: string;
};

export type RunwareGenerateResult =
  | { ok: true; imageUrl: string; prompt: string; promptSource: "vlm" | "heuristic" }
  | { ok: false; error: string; promptSource?: "vlm" | "heuristic" };

const RUNWARE_ENDPOINT = "https://api.runware.ai/v1";
const RUNWARE_MODEL = "runware:100@1";
const REQUEST_TIMEOUT_MS = 20_000;

// Max allowed image data URL size (base64 inflation ~1.37x). 3 MB covers
// every crop the tracker produces (typical: 30–180 KB). Anything bigger is
// almost certainly a mistake — fail fast rather than ship it to OpenAI.
const MAX_IMAGE_DATA_URL_BYTES = 3_000_000;

// Art-style opener we expect every prompt to start with. If the VLM
// forgets it (it sometimes does), we prepend it so Runware stays on-brand.
// Gallery page composites these onto a wooden bookshelf via mix-blend-mode:
// multiply, so solid pure-white backgrounds disappear into the wood tone.
const STYLE_OPENER =
  "bold comic-book illustration, thick black outlines, flat cel-shading, retro manga × kawaii energy, vivid pastel palette.";

// Tail appended to every prompt — forces the die-cut silhouette so the
// composited thumbnail reads as a sticker sitting on the shelf, not a
// framed painting.
const DIE_CUT_TAIL =
  "full character portrait, die-cut sticker, centered, isolated on solid pure white (#ffffff) background, no ground, no shadow, no scenery, clean product-shot lighting. No text, no speech bubbles, no watermarks.";

// --- Prompt construction --------------------------------------------------

// Strip/normalize a snippet for inclusion in the prompt. We don't want
// stray quotes or newlines confusing the image model.
function sanitizeSnippet(s: string): string {
  return s
    .replace(/\s+/g, " ")
    .replace(/["'`]/g, "")
    .trim();
}

// Deterministic one-sentence summary of the conversation topic.
// If there are no user turns yet, fall back to the generic intro line.
// Otherwise compress the last user + last assistant turn into ~120 chars.
function summarizeHistory(history: ChatTurn[]): string {
  const hasUser = history.some((t) => t.role === "user");
  if (!hasUser) return "introducing itself, looking playful.";

  // Walk from the tail to find the most recent user and assistant turns.
  let lastUser: string | null = null;
  let lastAssistant: string | null = null;
  for (let i = history.length - 1; i >= 0; i--) {
    const t = history[i];
    if (!t || !t.content) continue;
    if (!lastUser && t.role === "user") lastUser = t.content;
    else if (!lastAssistant && t.role === "assistant") lastAssistant = t.content;
    if (lastUser && lastAssistant) break;
  }

  const parts: string[] = [];
  if (lastUser) parts.push(`user says ${sanitizeSnippet(lastUser)}`);
  if (lastAssistant) parts.push(`it replies ${sanitizeSnippet(lastAssistant)}`);
  const joined = parts.join("; ");
  if (!joined) return "introducing itself, looking playful.";

  const capped = joined.length > 120 ? `${joined.slice(0, 117).trimEnd()}...` : joined;
  // Ensure the scene context reads as a sentence fragment we can slot in.
  return capped.endsWith(".") ? capped : `${capped}.`;
}

export function buildComicPrompt(input: RunwareGenerateInput): string {
  const className = sanitizeSnippet(input.className) || "object";
  const description = sanitizeSnippet(input.description) || "expressive character";
  const scene = summarizeHistory(input.history ?? []);

  return [
    STYLE_OPENER,
    `Center subject: ${className}, ${description}.`,
    `Scene context: ${scene}`,
    "Anthropomorphized — give it a face, eyes, mouth, expressive posture.",
    DIE_CUT_TAIL,
  ].join(" ");
}

// --- VLM-crafted prompt (lets the model that saw the object write the
// prompt). We feed it the actual crop + the persona card + the latest
// conversation turn, and ask for a specific, fun, comic-style Runware
// prompt. Falls back to the heuristic when the VLM errors or the JSON
// comes back malformed.
// --------------------------------------------------------------------------

const VLM_PROMPT_MODEL =
  process.env.RUNWARE_PROMPT_MODEL?.trim() || "gpt-4o-mini";
const VLM_PROMPT_TIMEOUT_MS = 12_000;

const VLM_PROMPT_SYSTEM = `You are a comic-book art director. You will see a cropped photo of a real physical object and a short persona card about it. You write ONE image prompt for a text-to-image model that will produce a fun, specific cartoon portrait of THIS object — not a generic one. The result will be composited onto a wooden bookshelf as a die-cut sticker, so the background MUST be solid pure white.

Rules:
- 55–90 words. One paragraph. No line breaks.
- Start with the art style: "bold comic-book illustration, thick black outlines, flat cel-shading, retro manga × kawaii energy, vivid pastel palette."
- Then describe the subject. Lock in 2–3 hyper-specific visual details you can actually see in the photo (a chip, a smudge, a sticker, a worn edge, a color, a reflection). These details are what make the image feel like a portrait of THIS object instead of a stock cartoon.
- Give the object a face + posture + mood that matches the persona card and the conversation topic. Anthropomorphize: give it eyes, mouth, little arms if it fits.
- Centered composition, full figure, subject fills the frame, expressive, alive.
- End with EXACTLY: "full character portrait, die-cut sticker, centered, isolated on solid pure white (#ffffff) background, no ground, no shadow, no scenery, clean product-shot lighting. No text, no speech bubbles, no watermarks."
- Do NOT include real brand names or logos. Paraphrase ("sleek black phone" not "iPhone 15 Pro").
- Return JSON: {"prompt": "..."}.`;

function extractPromptField(raw: string | null | undefined): string | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  // Strip code fences if present.
  const unfenced = trimmed.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "");
  // Find the first {...} block.
  const start = unfenced.indexOf("{");
  const end = unfenced.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  try {
    const obj = JSON.parse(unfenced.slice(start, end + 1)) as {
      prompt?: unknown;
    };
    if (typeof obj.prompt === "string" && obj.prompt.trim().length > 20) {
      return obj.prompt.trim();
    }
  } catch {
    // fall through
  }
  return null;
}

// Module-level OpenAI client; created once per process on first use so we
// don't pay connection churn per request.
let _openai: OpenAI | null = null;
function getOpenAI(): OpenAI | null {
  if (_openai) return _openai;
  const key = process.env.OPENAI_API_KEY;
  if (!key) return null;
  _openai = new OpenAI({ apiKey: key, timeout: VLM_PROMPT_TIMEOUT_MS });
  return _openai;
}

// Ensure the crafted prompt still starts with the brand art-style opener
// and ends with the die-cut / safety clause. VLMs sometimes drop one or
// the other when they get creative.
function normalizeCraftedPrompt(raw: string): string {
  const compact = raw.replace(/\s+/g, " ").trim();
  const withOpener = /^bold comic-book/i.test(compact)
    ? compact
    : `${STYLE_OPENER} ${compact}`;
  const withDieCut = /die-cut|isolated on .*white|pure white.*background/i.test(
    withOpener
  )
    ? withOpener
    : `${withOpener} ${DIE_CUT_TAIL}`;
  const withSafety = /no text|no speech bubbles|no watermarks/i.test(withDieCut)
    ? withDieCut
    : `${withDieCut} No text, no speech bubbles, no watermarks.`;
  return withSafety.length > 1400 ? withSafety.slice(0, 1400) : withSafety;
}

async function craftPromptFromImage(
  input: RunwareGenerateInput,
  signal?: AbortSignal
): Promise<string | null> {
  const client = getOpenAI();
  if (!client) return null;
  if (!input.imageDataUrl || !input.imageDataUrl.startsWith("data:image/")) {
    return null;
  }
  if (input.imageDataUrl.length > MAX_IMAGE_DATA_URL_BYTES) {
    // Too large — the crop is absurd, don't pay the OpenAI vision bill.
    return null;
  }

  const description =
    sanitizeSnippet(input.description) || "expressive character";
  const className = sanitizeSnippet(input.className) || "object";
  const topic = summarizeHistory(input.history ?? []);

  const userText = [
    `Object class (YOLO guess): ${className}`,
    `Persona card: ${description}`,
    `Conversation so far: ${topic}`,
    "",
    "Look at the image. Pick 2–3 specific visual details you see. Then write the comic prompt per the system rules. Return JSON.",
  ].join("\n");

  try {
    const resp = await client.chat.completions.create(
      {
        model: VLM_PROMPT_MODEL,
        response_format: { type: "json_object" },
        max_tokens: 420,
        temperature: 0.85,
        messages: [
          { role: "system", content: VLM_PROMPT_SYSTEM },
          {
            role: "user",
            content: [
              { type: "text", text: userText },
              {
                type: "image_url",
                image_url: { url: input.imageDataUrl, detail: "low" },
              },
            ],
          },
        ],
      },
      signal ? { signal } : undefined
    );
    const raw = resp.choices?.[0]?.message?.content;
    const prompt = extractPromptField(typeof raw === "string" ? raw : null);
    if (!prompt) return null;
    return normalizeCraftedPrompt(prompt);
  } catch {
    return null;
  }
}

// --- UUID v4 (no crypto.randomUUID dependency on older Node runtimes) -----

function uuidv4(): string {
  // Prefer the runtime's own implementation when available.
  const g = globalThis as unknown as {
    crypto?: { randomUUID?: () => string };
  };
  if (g.crypto?.randomUUID) return g.crypto.randomUUID();

  const bytes = new Uint8Array(16);
  // Node 18+ exposes globalThis.crypto.getRandomValues.
  if (g.crypto && typeof (g.crypto as { getRandomValues?: Function }).getRandomValues === "function") {
    (g.crypto as unknown as Crypto).getRandomValues(bytes);
  } else {
    for (let i = 0; i < 16; i++) bytes[i] = Math.floor(Math.random() * 256);
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

// --- Runware HTTP call ----------------------------------------------------

type RunwareSuccessItem = {
  taskType?: string;
  taskUUID?: string;
  imageURL?: string;
  imageUUID?: string;
};

type RunwareErrorItem = {
  message?: string;
  code?: string;
};

type RunwareResponse = {
  data?: RunwareSuccessItem[];
  errors?: RunwareErrorItem[];
};

export async function generateComicImage(
  input: RunwareGenerateInput,
  signal?: AbortSignal
): Promise<RunwareGenerateResult> {
  const apiKey = process.env.RUNWARE_API_KEY;
  if (!apiKey) {
    return { ok: false, error: "RUNWARE_API_KEY missing" };
  }

  // Prefer the VLM-crafted prompt when we have the image crop; that model
  // can lock in specific visible details ("chipped yellow lid", "peeling
  // sticker on the back") that the heuristic prompt can't reach.
  const crafted = await craftPromptFromImage(input, signal);
  const prompt = crafted ?? buildComicPrompt(input);
  const promptSource: "vlm" | "heuristic" = crafted ? "vlm" : "heuristic";
  const taskUUID = uuidv4();

  const body = [
    {
      taskType: "imageInference",
      taskUUID,
      positivePrompt: prompt,
      model: RUNWARE_MODEL,
      width: 384,
      height: 384,
      numberResults: 1,
      outputType: "URL",
      // WEBP at moderate quality keeps the file well under ~40 KB so the
      // gallery tile renders immediately instead of showing the alt-text
      // filename while a ~1 MB PNG streams in. The whiteToAlpha SVG
      // filter in globals.css thresholds near-white pixels, so mild
      // compression artefacts near the silhouette still get clipped.
      outputFormat: "WEBP",
      outputQuality: 72,
      checkNSFW: false,
    },
  ];

  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), REQUEST_TIMEOUT_MS);
  // Chain the caller's signal so a client disconnect or overall budget
  // expiry cancels the fetch promptly.
  const onCallerAbort = () => ac.abort();
  if (signal) {
    if (signal.aborted) ac.abort();
    else signal.addEventListener("abort", onCallerAbort, { once: true });
  }

  let res: Response;
  try {
    res = await fetch(RUNWARE_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
      signal: ac.signal,
    });
  } catch (err) {
    clearTimeout(timer);
    if (signal) signal.removeEventListener("abort", onCallerAbort);
    const aborted =
      (err instanceof Error && err.name === "AbortError") ||
      (err as { name?: string } | null)?.name === "AbortError";
    if (aborted) {
      return {
        ok: false,
        error: signal?.aborted ? "cancelled" : "timeout",
        promptSource,
      };
    }
    const message = err instanceof Error ? err.message : "network error";
    return { ok: false, error: message, promptSource };
  }
  clearTimeout(timer);
  if (signal) signal.removeEventListener("abort", onCallerAbort);

  if (!res.ok) {
    // Try to surface the provider's error message but don't crash on bad JSON.
    let detail = `runware http ${res.status}`;
    try {
      const j = (await res.json()) as RunwareResponse;
      const msg = j?.errors?.[0]?.message;
      if (msg) detail = msg;
    } catch {
      try {
        const t = await res.text();
        if (t) detail = t.slice(0, 200);
      } catch {
        // swallow
      }
    }
    return { ok: false, error: detail, promptSource };
  }

  let json: RunwareResponse;
  try {
    json = (await res.json()) as RunwareResponse;
  } catch {
    return { ok: false, error: "invalid runware response", promptSource };
  }

  if (json.errors && json.errors.length > 0) {
    const msg = json.errors[0]?.message || json.errors[0]?.code || "runware error";
    return { ok: false, error: msg, promptSource };
  }

  const imageUrl = json.data?.find((d) => d?.imageURL)?.imageURL;
  if (!imageUrl) {
    return { ok: false, error: "no image url in response", promptSource };
  }

  return { ok: true, imageUrl, prompt, promptSource };
}
