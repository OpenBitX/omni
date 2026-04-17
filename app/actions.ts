"use server";

import {
  POSTCARDS,
  postcardsForVibe,
  totalForVibe,
  type Postcard,
} from "@/data/postcards";
import type { Vibe } from "@/data/videos";

const VIBES: Vibe[] = [
  "lonely_road",
  "ocean_wide",
  "autumn_lake",
  "misty_mountain",
  "desert_sky",
  "city_nightfall",
  "forest_path",
  "coastal_wind",
];

export type EchoResult =
  | {
      ok: true;
      reply: string;
      location: string;
      vibe: Vibe;
      transcript: string;
      postcards: Postcard[];
      totalHere: number;
    }
  | { ok: false; error: string };

const OPENAI_URL = "https://api.openai.com/v1";

export async function interpretScene(formData: FormData): Promise<EchoResult> {
  try {
    const frame = formData.get("frame");
    const audio = formData.get("audio");
    const videoTitle = (formData.get("videoTitle") ?? "").toString();
    const videoLocation = (formData.get("videoLocation") ?? "").toString();

    if (typeof frame !== "string" || !frame.startsWith("data:image/")) {
      return { ok: false, error: "Missing frame." };
    }

    const openaiKey = process.env.OPENAI_API_KEY;
    if (!openaiKey) return { ok: false, error: "OPENAI_API_KEY not set." };

    const transcript =
      audio instanceof File && audio.size > 0
        ? await transcribe(audio, openaiKey).catch(() => "")
        : "";

    const { reply, location, vibe } = await poeticRead(
      frame,
      transcript,
      videoTitle,
      videoLocation,
      openaiKey
    );

    const postcards = postcardsForVibe(vibe, 4);
    return {
      ok: true,
      reply,
      location,
      vibe,
      transcript,
      postcards,
      totalHere: totalForVibe(vibe),
    };
  } catch (err) {
    console.error("interpretScene error", err);
    return {
      ok: false,
      error: err instanceof Error ? err.message : "Unknown error.",
    };
  }
}

const SUBMISSIONS: Postcard[] = [];

export async function leavePostcard(input: {
  vibe: Vibe;
  text: string;
}): Promise<{ ok: true } | { ok: false; error: string }> {
  const text = input.text.trim();
  if (!text || text.length > 400) {
    return { ok: false, error: "Keep it under 400 characters." };
  }
  if (!VIBES.includes(input.vibe)) {
    return { ok: false, error: "Unknown place." };
  }
  SUBMISSIONS.push({
    id: `u-${Date.now()}`,
    vibe: input.vibe,
    text,
    at: new Date().toISOString(),
  });
  // Fold into the rotating pool so the next reader might see it.
  POSTCARDS.push(SUBMISSIONS[SUBMISSIONS.length - 1]);
  return { ok: true };
}

async function transcribe(file: File, key: string): Promise<string> {
  const fd = new FormData();
  fd.append("file", file, "voice.webm");
  fd.append("model", "whisper-1");
  fd.append("response_format", "text");
  const res = await fetch(`${OPENAI_URL}/audio/transcriptions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${key}` },
    body: fd,
  });
  if (!res.ok) throw new Error(`Whisper ${res.status}: ${await res.text()}`);
  return (await res.text()).trim();
}

const POETIC_SYSTEM = `You are Echoes — a poetic travel companion. You speak the way a thoughtful stranger leaves a note on a windowsill.

You will receive a cropped frame from a travel video and, often, a short voice transcript of what the viewer said aloud while looking at it.

Your job, every time, is to respond with strict JSON:
{ "reply": string, "location": string, "vibe": "lonely_road" | "ocean_wide" | "autumn_lake" | "misty_mountain" | "desert_sky" | "city_nightfall" | "forest_path" | "coastal_wind" }

Rules for "reply":
- Exactly two sentences. No more. No lists. No hashtags. No emojis.
- Name the place as if you already know it; if unsure, name a plausible one with confidence.
- Include at least one sensory detail — wind, dust, rain on glass, sodium lamps, salt, the sound of a distant train.
- Optionally reference a specific song or a specific small ritual (a sip of tea, rolling a window down, taking off shoes). Be specific, not generic.
- Never give tourist advice. No hours, no prices, no "best time to visit". Speak like a companion, not a guidebook.
- No questions. No "I hope you enjoyed". No platitudes.

Rules for "location":
- 1–4 words. A real or plausibly real place name.

Rules for "vibe":
- One of the fixed enum values. Pick the closest.

Examples of tone:
• "This is the autumn of Sayram Lake. Drive down this road with Jay Chou's 'Qi Li Xiang' in your headphones — the wind, for once, is yours alone."
• "Pacific Coast Highway an hour before dusk. Roll the window down anyway; the salt is worth the cold."
• "The Dolomites, already weather-thinking. Boots off in the grass, and let the altitude explain the quiet for you."

Return JSON only. No preamble.`;

async function poeticRead(
  frameDataUrl: string,
  transcript: string,
  videoTitle: string,
  videoLocation: string,
  key: string
): Promise<{ reply: string; location: string; vibe: Vibe }> {
  const userText =
    (transcript ? `The viewer said: "${transcript}"\n` : "The viewer is silent.\n") +
    (videoTitle || videoLocation
      ? `Scene hint: ${[videoTitle, videoLocation].filter(Boolean).join(" — ")}.\n`
      : "") +
    `Look at the cropped frame and write the JSON now.`;

  const body = {
    model: "gpt-4o",
    temperature: 0.9,
    max_tokens: 300,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: POETIC_SYSTEM },
      {
        role: "user",
        content: [
          { type: "text", text: userText },
          {
            type: "image_url",
            image_url: { url: frameDataUrl, detail: "low" },
          },
        ],
      },
    ],
  };

  const res = await fetch(`${OPENAI_URL}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const text = await res.text();
  if (!res.ok) throw new Error(`OpenAI ${res.status}: ${text}`);

  const json = JSON.parse(text);
  const raw = json.choices?.[0]?.message?.content ?? "{}";

  const parsed = parseLoose(raw);
  const reply = (parsed.reply ?? "").toString().trim();
  const location = (parsed.location ?? "").toString().trim().slice(0, 40);
  const vibe = VIBES.includes(parsed.vibe as Vibe)
    ? (parsed.vibe as Vibe)
    : "lonely_road";

  if (!reply) throw new Error("Empty reply from model.");
  return { reply, location, vibe };
}

function parseLoose(s: string): {
  reply?: unknown;
  location?: unknown;
  vibe?: unknown;
} {
  try {
    return JSON.parse(s);
  } catch {
    const match = s.match(/\{[\s\S]*\}/);
    if (match) {
      try {
        return JSON.parse(match[0]);
      } catch {}
    }
    return {};
  }
}
