// Sanity test for yolo26n-seg: verify output shape + probe row layout by
// looking at the first row's values. Prints top detections with decoded box
// + score + class + mask summary so we know the post-proc math is correct.
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

const MODEL = "/Users/harryedwards/hackathon/public/models/yolo26n-seg.onnx";
const IMG = process.argv[2] ?? "/tmp/bus.jpg";
const SIZE = 640;
const CONF = 0.25;

async function main() {
  const raw = readFileSync(IMG);
  const meta = await sharp(raw).metadata();
  const srcW = meta.width, srcH = meta.height;

  const scale = Math.min(SIZE / srcW, SIZE / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = Math.floor((SIZE - newW) / 2);
  const padY = Math.floor((SIZE - newH) / 2);

  const resized = await sharp(raw).resize(newW, newH, { fit: "fill" }).raw().toBuffer();

  const tensorData = new Float32Array(3 * SIZE * SIZE).fill(114 / 255);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const src = (y * newW + x) * 3;
      const dx = padX + x;
      const dy = padY + y;
      tensorData[0 * SIZE * SIZE + dy * SIZE + dx] = resized[src] / 255;
      tensorData[1 * SIZE * SIZE + dy * SIZE + dx] = resized[src + 1] / 255;
      tensorData[2 * SIZE * SIZE + dy * SIZE + dx] = resized[src + 2] / 255;
    }
  }

  const session = await ort.InferenceSession.create(MODEL);
  console.log("inputs :", session.inputNames);
  console.log("outputs:", session.outputNames);

  const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", tensorData, [1, 3, SIZE, SIZE]) };
  const out = await session.run(feeds);
  const out0 = out[session.outputNames[0]];
  const out1 = out[session.outputNames[1]];
  console.log("output0 dims:", out0.dims);
  console.log("output1 dims:", out1.dims);

  // Dump a few rows from output0 so we can see the layout.
  const D = out0.data;
  const [, N, F] = out0.dims; // N=300, F=38
  console.log(`\nfirst 3 rows (${F} features each):`);
  for (let i = 0; i < 3; i++) {
    const row = Array.from(D.slice(i * F, i * F + F));
    // Compact print: first 6 + last 6 of mask coefs
    console.log(`  q${i}:`, row.slice(0, 6).map((v) => v.toFixed(3)).join(", "), "| …mask coefs…");
  }

  // Try the hypothesis: [0..3]=box (normalized cxcywh), [4]=score, [5]=class, [6..37]=32 mask coefs
  console.log(`\nassuming: box=[0..3], score=[4], class=[5], coefs=[6..37]`);
  const kept = [];
  for (let i = 0; i < N; i++) {
    const base = i * F;
    const score = D[base + 4];
    const cls = D[base + 5];
    if (score < CONF) continue;
    kept.push({
      q: i,
      score,
      classId: Math.round(cls),
      cx: D[base + 0],
      cy: D[base + 1],
      w: D[base + 2],
      h: D[base + 3],
      coefs: D.slice(base + 6, base + 38),
    });
  }
  kept.sort((a, b) => b.score - a.score);
  console.log(`\n${kept.length} detections ≥ ${CONF}:`);
  for (const d of kept.slice(0, 6)) {
    console.log(
      `  q${String(d.q).padStart(3)}  ${(d.score * 100).toFixed(1).padStart(5)}%  cls=${d.classId} ${COCO[d.classId] ?? "?"}  ` +
      `raw_box=(${d.cx.toFixed(3)}, ${d.cy.toFixed(3)}, ${d.w.toFixed(3)}, ${d.h.toFixed(3)})`
    );
  }

  // Compute the first detection's mask and report centroid
  if (kept[0]) {
    const d = kept[0];
    const protos = out1.data; // [1, 32, 160, 160]
    const MH = 160, MW = 160, NC = 32;
    const mask = new Float32Array(MH * MW);
    for (let c = 0; c < NC; c++) {
      const coef = d.coefs[c];
      for (let y = 0; y < MH; y++) {
        for (let x = 0; x < MW; x++) {
          mask[y * MW + x] += coef * protos[c * MH * MW + y * MW + x];
        }
      }
    }
    // Sigmoid + threshold
    let sumX = 0, sumY = 0, sumM = 0, maxV = -Infinity, minV = Infinity;
    for (let y = 0; y < MH; y++) {
      for (let x = 0; x < MW; x++) {
        const v = 1 / (1 + Math.exp(-mask[y * MW + x]));
        if (v > maxV) maxV = v;
        if (v < minV) minV = v;
        if (v > 0.5) {
          sumM += v;
          sumX += x * v;
          sumY += y * v;
        }
      }
    }
    console.log(
      `\ntop-det mask check: min=${minV.toFixed(3)} max=${maxV.toFixed(3)} area=${Math.round(sumM)}` +
      `  centroid (in 160×160 space): (${(sumX / sumM).toFixed(1)}, ${(sumY / sumM).toFixed(1)})`
    );
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
