// End-to-end test of the GLM call that app/actions.ts makes.
// Verifies: (a) model answers, (b) JSON extraction for assess, (c) line for gen.
import OpenAI from "openai";
import { readFileSync } from "node:fs";

const env = readFileSync("/Users/harryedwards/hackathon/.env.local", "utf8");
for (const line of env.split("\n")) {
  const m = line.match(/^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$/);
  if (m) process.env[m[1]] = m[2].trim();
}
const key = process.env.ZHIPU_API_KEY;
const client = new OpenAI({
  apiKey: key,
  baseURL: "https://open.bigmodel.cn/api/paas/v4/",
  timeout: 120_000,
});

const img = readFileSync("/tmp/bus.jpg");
const imageDataUrl = `data:image/jpeg;base64,${img.toString("base64")}`;

const MODEL = "glm-5v-turbo";

const ASSESS_SYSTEM = `You place a cartoon face on whatever the user tapped.
Return STRICT JSON only:
{"suitable": boolean, "cx": number, "cy": number, "bbox": [number, number, number, number], "reason": string}
DEFAULT TO suitable=true. Only reject for: human, empty. cx/cy/bbox normalized to crop.
reason: max 10 words.`;

const FACE_SYSTEM = `Secret inner voice of a thing. ONE short line (max 14 words) in first person, in character. No meta, no quotes, no emojis.`;

function extractJsonObject(text) {
  if (!text) return null;
  let s = text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  const fence = s.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fence) s = fence[1].trim();
  const first = s.indexOf("{"), last = s.lastIndexOf("}");
  if (first === -1 || last === -1 || last <= first) return null;
  try { return JSON.parse(s.slice(first, last + 1)); } catch { return null; }
}
function extractTextLine(text) {
  if (!text) return "";
  let s = text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  const fence = s.match(/```[\w-]*\s*([\s\S]*?)\s*```/);
  if (fence) s = fence[1].trim();
  return s.replace(/^["'`]+|["'`]+$/g, "").replace(/\s+/g, " ").trim().slice(0, 180);
}

// 1) assess
console.log("\n=== assess ===");
{
  const r = await client.chat.completions.create({
    model: MODEL,
    max_tokens: 1536,
    temperature: 0.2,
    messages: [
      { role: "system", content: ASSESS_SYSTEM },
      { role: "user", content: [
        { type: "text", text: "Tap at (0.5, 0.5). Return JSON." },
        { type: "image_url", image_url: { url: imageDataUrl } },
      ]},
    ],
  });
  const content = r.choices[0]?.message?.content ?? "";
  console.log("raw content:", JSON.stringify(content).slice(0, 300));
  console.log("parsed:", extractJsonObject(content));
}

// 2) line
console.log("\n=== line ===");
{
  const r = await client.chat.completions.create({
    model: MODEL,
    max_tokens: 1024,
    temperature: 0.95,
    messages: [
      { role: "system", content: FACE_SYSTEM },
      { role: "user", content: [
        { type: "text", text: "What does this thing say?" },
        { type: "image_url", image_url: { url: imageDataUrl } },
      ]},
    ],
  });
  const content = r.choices[0]?.message?.content ?? "";
  console.log("raw content:", JSON.stringify(content).slice(0, 300));
  console.log("line:", extractTextLine(content));
}
