"use server";

import { randomUUID } from "crypto";

export type MemeResult =
  | { ok: true; imageUrl: string; caption: string; transcript: string }
  | { ok: false; error: string };

const RUNWARE_URL = "https://api.runware.ai/v1";
const OPENAI_URL = "https://api.openai.com/v1";

export async function generateMeme(formData: FormData): Promise<MemeResult> {
  try {
    const photo = formData.get("photo");
    const audio = formData.get("audio");
    if (typeof photo !== "string" || !photo.startsWith("data:image/")) {
      return { ok: false, error: "Missing photo." };
    }

    const runwareKey = process.env.RUNWARE_API_KEY;
    const openaiKey = process.env.OPENAI_API_KEY;
    if (!runwareKey) return { ok: false, error: "RUNWARE_API_KEY not set." };
    if (!openaiKey) return { ok: false, error: "OPENAI_API_KEY not set." };

    const transcript =
      audio instanceof File && audio.size > 0
        ? await transcribe(audio, openaiKey)
        : "";

    const concept = await craftConcept(photo, transcript, openaiKey);
    const imageUrl = await runware(concept.prompt, runwareKey);

    return {
      ok: true,
      imageUrl,
      caption: concept.caption,
      transcript,
    };
  } catch (err) {
    console.error("generateMeme error", err);
    return {
      ok: false,
      error: err instanceof Error ? err.message : "Unknown error.",
    };
  }
}

async function transcribe(file: File, key: string): Promise<string> {
  const fd = new FormData();
  fd.append("file", file, "vent.webm");
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

type Concept = { caption: string; prompt: string };

async function craftConcept(
  photoDataUrl: string,
  transcript: string,
  key: string
): Promise<Concept> {
  const system = `You turn a photo and a spoken vent into a single, punchy meme.
Return strict JSON: { "caption": string, "prompt": string }.

- "caption": one short line (<= 80 chars), the meme's text. No quotes, no hashtags, no emojis. Dry, observational, internet-native humor.
- "prompt": a vivid image-generation prompt for a model. Describe a funny, exaggerated visual metaphor that riffs on what's in the photo and the vent. Include style cues (e.g. "cinematic 35mm", "hyperreal", "dramatic lighting"). Do NOT ask for on-image text — keep it purely visual. Avoid names, logos, or copyrighted characters.`;

  const userText = transcript
    ? `The person said: "${transcript}"\nThe photo shows the context. Make the meme about the collision between expectation and reality.`
    : `No words — just read the photo. Invent a relatable, slightly self-deprecating meme from what you see.`;

  const res = await fetch(`${OPENAI_URL}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      temperature: 0.9,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: system },
        {
          role: "user",
          content: [
            { type: "text", text: userText },
            { type: "image_url", image_url: { url: photoDataUrl, detail: "low" } },
          ],
        },
      ],
    }),
  });

  if (!res.ok) throw new Error(`Concept ${res.status}: ${await res.text()}`);
  const json = await res.json();
  const raw = json.choices?.[0]?.message?.content ?? "{}";
  const parsed = JSON.parse(raw) as Partial<Concept>;
  const caption = (parsed.caption ?? "").toString().slice(0, 120).trim();
  const prompt = (parsed.prompt ?? "").toString().trim();
  if (!prompt) throw new Error("Empty prompt from concept step.");
  return { caption, prompt };
}

async function runware(prompt: string, key: string): Promise<string> {
  const taskUUID = randomUUID();
  const body = [
    {
      taskType: "imageInference",
      taskUUID,
      positivePrompt: prompt,
      model: "runware:100@1",
      width: 1024,
      height: 1024,
      numberResults: 1,
      outputFormat: "WEBP",
      includeCost: false,
    },
  ];

  const res = await fetch(RUNWARE_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const text = await res.text();
  if (!res.ok) throw new Error(`Runware ${res.status}: ${text}`);

  const data = JSON.parse(text);
  const items = data.data ?? data.results ?? [];
  const img = items[0];
  const url = img?.imageURL ?? img?.imageUrl ?? img?.URL;
  if (!url) throw new Error(`Runware returned no imageURL: ${text.slice(0, 200)}`);
  return url as string;
}
