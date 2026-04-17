// Copies onnxruntime-web's WASM runtime files into public/ort so the browser
// can load them same-origin. Runs on postinstall so fresh checkouts are ready
// without committing ~36 MB of vendor binaries to git.
import fs from "node:fs";
import path from "node:path";

const FILES = [
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
  "ort-wasm-simd-threaded.jsep.mjs",
];

const root = process.cwd();
const srcDir = path.join(root, "node_modules", "onnxruntime-web", "dist");
const dstDir = path.join(root, "public", "ort");

if (!fs.existsSync(srcDir)) {
  console.warn("[setup-ort] onnxruntime-web not installed, skipping");
  process.exit(0);
}

fs.mkdirSync(dstDir, { recursive: true });
let copied = 0;
for (const name of FILES) {
  const src = path.join(srcDir, name);
  const dst = path.join(dstDir, name);
  if (!fs.existsSync(src)) {
    console.warn(`[setup-ort] missing ${name} in ${srcDir}`);
    continue;
  }
  fs.copyFileSync(src, dst);
  copied++;
}
console.log(`[setup-ort] copied ${copied}/${FILES.length} runtime files → public/ort/`);
