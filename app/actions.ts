"use server";

import { randomUUID } from "crypto";

export type MemeFormat = "top-bottom" | "bottom-only" | "top-only";

export type MemeResult =
  | {
      ok: true;
      imageDataUrl: string;
      topText: string;
      bottomText: string;
      format: MemeFormat;
      caption: string;
      transcript: string;
    }
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
        ? await transcribe(audio, openaiKey).catch(() => "")
        : "";

    const concept = await craftMeme(photo, transcript, openaiKey);
    const imageUrl = await runware(concept.imagePrompt, runwareKey);
    const imageDataUrl = await fetchAsDataUrl(imageUrl);

    return {
      ok: true,
      imageDataUrl,
      topText: concept.topText,
      bottomText: concept.bottomText,
      format: concept.format,
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

type Concept = {
  topText: string;
  bottomText: string;
  format: MemeFormat;
  caption: string;
  imagePrompt: string;
};

const MEME_SYSTEM = `You are a meme writer. Not a children's-book writer, not a corporate copywriter — a meme writer. Your job is to be FUNNY. Real funny. The kind of meme a 24-year-old would actually screenshot and send to the group chat at 11pm.

You get a photo (what the person is looking at) and optionally a transcript of them venting. Turn it into a single meme.

Return STRICT JSON only:
{
  "format": "top-bottom" | "bottom-only" | "top-only",
  "topText": string,
  "bottomText": string,
  "caption": string,
  "imagePrompt": string
}

RULES FOR THE MEME TEXT — these are non-negotiable:
- ALL CAPS. No exceptions. Impact-font style.
- Each line ≤ 45 characters. SHORTER IS FUNNIER. Cut every word that isn't pulling weight.
- No quotes, no hashtags, no emojis, no "lol", no "literally".
- Funniest formats:
  • SETUP / PUNCHLINE — topText sets expectation, bottomText subverts it.
  • POV — topText: "POV: [specific, cringe situation]", bottomText: the damning detail.
  • ME / ALSO ME — topText: "ME: [what I said]", bottomText: "ALSO ME: [what I did]".
  • NOBODY / ME — topText: "NOBODY:" or "LITERALLY NO ONE:", bottomText: the unhinged thing.
  • EXPECTATION / REALITY — sparingly, only if the contrast is razor-sharp.
- Use "bottom-only" for a single punchline delivered over an absurd image.
- Use "top-only" for a deadpan observation.
- SPECIFIC beats GENERIC. "My Google Calendar at 4pm on a Thursday" destroys "my schedule".
- PUNCH DOWN AT THE SITUATION, NEVER AT A PERSON. Self-deprecation about the user's own moment is gold.
- Dry. Deadpan. Internet-native. No dad jokes. No puns unless the pun is savage.

RULES FOR imagePrompt:
- A vivid, slightly absurd, cinematic scene that AMPLIFIES the meme's joke visually.
- Do NOT ask for any text in the image — text is added separately.
- Include style direction: "shot on 35mm film", "blown-out flash photography", "dramatic cinematic lighting", "wide-angle", "hyperreal detail", "backlit golden hour".
- Be specific about subject, setting, mood. 40–80 words.
- No real people, no logos, no copyrighted characters.

RULES FOR caption:
- One short line (<60 chars) for social sharing. Can be more conversational than the meme text. Still funny.

EXAMPLES (study the voice):

{"format":"top-bottom","topText":"ME: I'M GOING TO BED AT 10","bottomText":"ME AT 2:47AM WATCHING A 40-MIN VIDEO ON THE FALL OF CONSTANTINOPLE","caption":"sleep is a construct","imagePrompt":"A person alone in bed at 3am, face lit only by a phone screen, eyes wide and unblinking, room otherwise pitch black, shot on 35mm film, claustrophobic close-up, moody cinematic lighting"}

{"format":"top-bottom","topText":"THE RECIPE: 45 MINUTES","bottomText":"ME, 3 HOURS IN, COVERED IN FLOUR, CRYING","caption":"cooking is fake","imagePrompt":"A disaster-zone home kitchen at dusk, flour everywhere, a single sad lump of dough, one defeated person sitting on the floor staring into middle distance, warm tungsten lighting, shot on 35mm"}

{"format":"bottom-only","topText":"","bottomText":"ME EXPLAINING TO MY THERAPIST WHY I THINK THE BARISTA HATES ME","caption":"the barista is fine actually","imagePrompt":"An intensely earnest young person sitting forward on a therapist's couch, gesturing emphatically, late afternoon sun through blinds, hyperreal cinematic detail, shallow depth of field"}

{"format":"top-bottom","topText":"NOBODY:","bottomText":"MY BRAIN AT 3AM: REMEMBER THAT THING YOU SAID IN 2014?","caption":"","imagePrompt":"A dark bedroom, single figure lying flat on back staring at the ceiling with an expression of quiet horror, cold blue moonlight, wide shot, cinematic 35mm"}

Now write a meme that would make a jaded zillennial snort-laugh. Be specific. Be mean to the situation. Be FUNNY.`;

async function craftMeme(
  photoDataUrl: string,
  transcript: string,
  key: string
): Promise<Concept> {
  const userText = transcript
    ? `They said (verbatim): "${transcript}"\n\nThe photo shows what they're looking at. Write the meme.`
    : `No words — read the photo. What is the universal, slightly miserable, very funny truth hiding in this image? Write the meme.`;

  const call = async (model: string) =>
    fetch(`${OPENAI_URL}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        temperature: 1.0,
        top_p: 0.95,
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: MEME_SYSTEM },
          {
            role: "user",
            content: [
              { type: "text", text: userText },
              {
                type: "image_url",
                image_url: { url: photoDataUrl, detail: "low" },
              },
            ],
          },
        ],
      }),
    });

  let res = await call("gpt-4o");
  if (!res.ok && res.status !== 400) res = await call("gpt-4o-mini");
  if (!res.ok) throw new Error(`Concept ${res.status}: ${await res.text()}`);

  const json = await res.json();
  const raw = json.choices?.[0]?.message?.content ?? "{}";
  const parsed = JSON.parse(raw) as Partial<Concept>;

  const format: MemeFormat =
    parsed.format === "bottom-only" || parsed.format === "top-only"
      ? parsed.format
      : "top-bottom";
  const topText = sanitize(parsed.topText, 60);
  const bottomText = sanitize(parsed.bottomText, 60);
  const caption = (parsed.caption ?? "").toString().slice(0, 120).trim();
  const imagePrompt = (parsed.imagePrompt ?? "").toString().trim();

  if (!imagePrompt) throw new Error("Empty image prompt from concept step.");
  if (!topText && !bottomText) throw new Error("Empty meme text.");

  return { topText, bottomText, format, caption, imagePrompt };
}

function sanitize(s: unknown, maxLen: number): string {
  return (s ?? "")
    .toString()
    .replace(/["“”]/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .toUpperCase()
    .slice(0, maxLen);
}

async function runware(prompt: string, key: string): Promise<string> {
  const taskUUID = randomUUID();
  const body = [
    {
      taskType: "imageInference",
      taskUUID,
      positivePrompt: prompt,
      negativePrompt:
        "text, letters, watermark, signature, caption, subtitles, typography, logo, blurry, low quality",
      model: "runware:100@1",
      width: 768,
      height: 1344,
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

async function fetchAsDataUrl(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Image fetch ${res.status}`);
  const contentType = res.headers.get("content-type") ?? "image/webp";
  const buffer = Buffer.from(await res.arrayBuffer());
  return `data:${contentType};base64,${buffer.toString("base64")}`;
}
