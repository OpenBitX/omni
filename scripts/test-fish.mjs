// Smoke-test Fish.audio TTS with the key in .env.local.
// Mirrors the request shape in app/actions.ts so any divergence is caught here.
import { readFileSync, writeFileSync } from "node:fs";

const env = readFileSync("/Users/harryedwards/hackathon/.env.local", "utf8");
for (const line of env.split("\n")) {
  const m = line.match(/^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$/);
  if (m) process.env[m[1]] = m[2].trim();
}
const key = process.env.FISH_API_KEY;
if (!key) {
  console.error("no FISH_API_KEY");
  process.exit(2);
}
console.log("key:", key.slice(0, 8) + "…");

const body = {
  text: "hello, I am a small stapler. I bite because I love.",
  format: "mp3",
  mp3_bitrate: 128,
  normalize: true,
  latency: "normal",
  chunk_length: 200,
};
if (process.env.FISH_REFERENCE_ID) body.reference_id = process.env.FISH_REFERENCE_ID;

const t0 = Date.now();
const resp = await fetch("https://api.fish.audio/v1/tts", {
  method: "POST",
  headers: {
    Authorization: `Bearer ${key}`,
    "Content-Type": "application/json",
    model: process.env.FISH_MODEL?.trim() || "s1",
  },
  body: JSON.stringify(body),
});
console.log("status:", resp.status, "in", Date.now() - t0, "ms");
console.log("content-type:", resp.headers.get("content-type"));

if (!resp.ok) {
  const text = await resp.text().catch(() => "");
  console.error("body:", text.slice(0, 500));
  process.exit(1);
}
const buf = Buffer.from(await resp.arrayBuffer());
console.log("bytes:", buf.length);
writeFileSync("/tmp/fish-test.mp3", buf);
console.log("wrote /tmp/fish-test.mp3 — try: afplay /tmp/fish-test.mp3");
