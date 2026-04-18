// Probe which GLM vision models actually work on this account right now.
// Used to pick sensible GLM_MODEL_FAST / GLM_MODEL_DEEP defaults.
import OpenAI from "openai";
import { readFileSync } from "node:fs";

const env = readFileSync("/Users/harryedwards/hackathon/.env.local", "utf8");
for (const line of env.split("\n")) {
  const m = line.match(/^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$/);
  if (m) process.env[m[1]] = m[2].trim();
}
const key = process.env.ZHIPU_API_KEY;
if (!key) { console.error("no ZHIPU_API_KEY"); process.exit(2); }

const client = new OpenAI({
  apiKey: key,
  baseURL: "https://open.bigmodel.cn/api/paas/v4/",
  timeout: 90_000,
});

const img = readFileSync("/tmp/bus.jpg");
const imageDataUrl = `data:image/jpeg;base64,${img.toString("base64")}`;

const candidates = [
  "glm-4v",
  "glm-4v-plus",
  "glm-4v-flash",
  "glm-4v-plus-0111",
  "glm-4.5v",
  "glm-4.1v-thinking-flash",
  "glm-5v-turbo",
];

for (const model of candidates) {
  try {
    const t0 = Date.now();
    const r = await client.chat.completions.create({
      model,
      max_tokens: 160,
      temperature: 0.8,
      messages: [
        { role: "system", content: "Reply with one short first-person line as the object. Max 14 words." },
        { role: "user", content: [
          { type: "text", text: "What does this thing say?" },
          { type: "image_url", image_url: { url: imageDataUrl } },
        ]},
      ],
    });
    const content = r.choices[0]?.message?.content ?? "";
    console.log(`  ✅ ${model.padEnd(26)} ${Date.now() - t0} ms  → ${JSON.stringify(content).slice(0, 100)}`);
  } catch (e) {
    const msg = e?.error?.message || e?.message || String(e);
    console.log(`  ❌ ${model.padEnd(26)} ${msg.slice(0, 100)}`);
  }
}
