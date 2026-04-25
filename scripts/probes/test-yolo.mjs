// Sanity test: load YOLO26n, run on a real image, print top-K detections.
// Verifies (a) the model loads, (b) input/output shapes match expectations,
// (c) the coordinate system we use in lib/yolo.ts produces sane boxes.

import * as ort from "onnxruntime-node";
import sharp from "sharp";
import { readFileSync } from "node:fs";

const COCO = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
  "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
  "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
  "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
  "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
  "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
  "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
  "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
  "scissors", "teddy bear", "hair drier", "toothbrush",
];

const MODEL = "/Users/harryedwards/hackathon/public/models/yolo26n.onnx";
const IMG = process.argv[2] ?? "/tmp/bus.jpg";
const SIZE = 640;
const CONF = 0.15;

async function main() {
  const raw = readFileSync(IMG);
  const meta = await sharp(raw).metadata();
  const srcW = meta.width, srcH = meta.height;
  console.log(`image: ${IMG}  ${srcW}x${srcH}`);

  // Letterbox into SIZE×SIZE with gray pad. Match what lib/yolo.ts does.
  const scale = Math.min(SIZE / srcW, SIZE / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = Math.floor((SIZE - newW) / 2);
  const padY = Math.floor((SIZE - newH) / 2);

  const resized = await sharp(raw)
    .resize(newW, newH, { fit: "fill" })
    .raw()
    .toBuffer();

  const tensorData = new Float32Array(3 * SIZE * SIZE).fill(114 / 255); // gray pad
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const src = (y * newW + x) * 3;
      const dx = padX + x;
      const dy = padY + y;
      const idxR = 0 * SIZE * SIZE + dy * SIZE + dx;
      const idxG = 1 * SIZE * SIZE + dy * SIZE + dx;
      const idxB = 2 * SIZE * SIZE + dy * SIZE + dx;
      tensorData[idxR] = resized[src] / 255;
      tensorData[idxG] = resized[src + 1] / 255;
      tensorData[idxB] = resized[src + 2] / 255;
    }
  }

  const session = await ort.InferenceSession.create(MODEL);
  console.log("inputs:", session.inputNames);
  console.log("outputs:", session.outputNames);

  const feeds = {
    [session.inputNames[0]]: new ort.Tensor("float32", tensorData, [1, 3, SIZE, SIZE]),
  };
  const outs = await session.run(feeds);
  const logits = outs[session.outputNames[0] === "logits" ? "logits" : "pred_logits"] ?? outs["logits"];
  const boxes = outs["pred_boxes"];
  console.log("logits shape:", logits.dims);
  console.log("boxes  shape:", boxes.dims);

  const nQ = logits.dims[1];
  const nC = logits.dims[2];
  const L = logits.data;
  const B = boxes.data;

  const sig = (x) => 1 / (1 + Math.exp(-x));

  const detections = [];
  for (let q = 0; q < nQ; q++) {
    let bestL = -Infinity, bestC = -1;
    for (let c = 0; c < nC; c++) {
      const v = L[q * nC + c];
      if (v > bestL) { bestL = v; bestC = c; }
    }
    const score = sig(bestL);
    if (score < CONF) continue;
    const cx = B[q * 4], cy = B[q * 4 + 1], w = B[q * 4 + 2], h = B[q * 4 + 3];
    detections.push({ q, score, classId: bestC, name: COCO[bestC] ?? "?", cx, cy, w, h });
  }
  detections.sort((a, b) => b.score - a.score);

  console.log(`\nraw box value distribution (normalization check):`);
  const cxs = detections.map(d => d.cx);
  if (cxs.length) {
    console.log(`  cx range: ${Math.min(...cxs).toFixed(3)} .. ${Math.max(...cxs).toFixed(3)}`);
    const ws = detections.map(d => d.w);
    console.log(`  w  range: ${Math.min(...ws).toFixed(3)} .. ${Math.max(...ws).toFixed(3)}`);
  }

  console.log(`\ntop ${Math.min(10, detections.length)} detections (conf >= ${CONF}):`);
  for (const d of detections.slice(0, 10)) {
    // Interpret boxes as [0,1] of the 640×640 letterbox frame.
    const cx640 = d.cx * SIZE, cy640 = d.cy * SIZE;
    const w640 = d.w * SIZE, h640 = d.h * SIZE;
    const ox = (cx640 - padX) / scale;
    const oy = (cy640 - padY) / scale;
    const ow = w640 / scale;
    const oh = h640 / scale;
    console.log(
      `  q${String(d.q).padStart(3)}  ${(d.score * 100).toFixed(1).padStart(5)}%  ` +
      `${d.name.padEnd(16)}  raw=(${d.cx.toFixed(3)},${d.cy.toFixed(3)},${d.w.toFixed(3)},${d.h.toFixed(3)})  ` +
      `orig=(${ox.toFixed(0)},${oy.toFixed(0)},${ow.toFixed(0)}x${oh.toFixed(0)})`
    );
  }
  console.log(`\nimage dim: ${srcW}x${srcH} — a bus image should produce boxes roughly filling the image.`);
}

main().catch((e) => { console.error(e); process.exit(1); });
