// Smoke-test the converse path's GLM call with the corrected model.
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
const MODEL = "glm-4.5v";

function extract(text) {
  if (!text) return "";
  let s = text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  const fence = s.match(/```[\w-]*\s*([\s\S]*?)\s*```/);
  if (fence) s = fence[1].trim();
  return s.replace(/^["'`]+|["'`]+$/g, "").replace(/\s+/g, " ").trim().slice(0, 180);
}

const className = "bus";
const system = `You are the secret inner voice of a ${className}. Reply with ONE short, in-character line (max 22 words) that actually responds to what they said. First person. Funny. No meta. No quotes.`;
const userText = `They said: "how's your day going?". Reply in character — one short line.`;

const t0 = Date.now();
const r = await client.chat.completions.create({
  model: MODEL,
  max_tokens: 1024,
  temperature: 0.95,
  messages: [
    { role: "system", content: system },
    { role: "user", content: [
      { type: "text", text: userText },
      { type: "image_url", image_url: { url: imageDataUrl } },
    ]},
  ],
});
console.log(`model: ${MODEL}  ${Date.now() - t0} ms`);
const content = r.choices[0]?.message?.content ?? "";
console.log("raw:", JSON.stringify(content).slice(0, 400));
console.log("line:", extract(content));
