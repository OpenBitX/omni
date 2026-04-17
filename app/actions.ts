"use server";

import OpenAI from "openai";

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
- a stapler: "I bite because I love"`;

function getClient(): OpenAI {
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error("OPENAI_API_KEY not set");
  return new OpenAI({ apiKey: key });
}

export async function generateLine(
  imageDataUrl: string
): Promise<{ line: string; audioDataUrl: string }> {
  if (!imageDataUrl.startsWith("data:image/")) {
    throw new Error("expected an image data URL");
  }

  const client = getClient();

  const chat = await client.chat.completions.create({
    model: "gpt-4o",
    max_tokens: 60,
    temperature: 0.95,
    messages: [
      { role: "system", content: FACE_SYSTEM },
      {
        role: "user",
        content: [
          { type: "text", text: "What does this thing say?" },
          { type: "image_url", image_url: { url: imageDataUrl, detail: "low" } },
        ],
      },
    ],
  });

  const raw = chat.choices[0]?.message?.content?.trim() ?? "";
  const line = raw
    .replace(/^["'`]+|["'`]+$/g, "")
    .replace(/\s+/g, " ")
    .slice(0, 180);
  if (!line) throw new Error("empty line from model");

  const speech = await client.audio.speech.create({
    model: "tts-1",
    voice: "nova",
    input: line,
    response_format: "mp3",
  });
  const buf = Buffer.from(await speech.arrayBuffer());
  const audioDataUrl = `data:audio/mpeg;base64,${buf.toString("base64")}`;

  return { line, audioDataUrl };
}
