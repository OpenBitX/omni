// Probe GLM-5V-Turbo's real response shape to understand why content is empty.
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

// Load a real test image (the bus.jpg that test-yolo.mjs uses)
let imageDataUrl = null;
try {
  const img = readFileSync("/tmp/bus.jpg");
  imageDataUrl = `data:image/jpeg;base64,${img.toString("base64")}`;
  console.log("loaded /tmp/bus.jpg:", img.length, "bytes");
} catch {
  console.log("no /tmp/bus.jpg — will test text-only");
}

for (const model of ["glm-4.5v", "glm-5v-turbo"]) {
  console.log(`\n--- ${model} ---`);
  try {
    const userContent = [{ type: "text", text: "What does this thing say out loud, in first person, in 14 words max? Just the line." }];
    if (imageDataUrl) userContent.push({ type: "image_url", image_url: { url: imageDataUrl } });
    const r = await client.chat.completions.create({
      model,
      max_tokens: 1024,
      temperature: 0.8,
      messages: [
        { role: "system", content: "You are the voice of an object. One short first-person line." },
        { role: "user", content: userContent },
      ],
    });
    console.log("full choice:", JSON.stringify(r.choices[0], null, 2).slice(0, 2000));
  } catch (e) {
    console.error("err:", e?.error?.message || e?.message);
  }
}
