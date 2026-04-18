// Browser-side object detector on top of onnxruntime-web.
//
// Default model: YOLO26n (onnx-community/yolo26n-ONNX), 9.4 MB, DETR-style
// outputs (logits [1,N,80] + pred_boxes [1,N,4] normalized cxcywh). The same
// post-processing path also handles RF-DETR Nano once the class remap is
// wired. Legacy YOLOv8-style anchor-grid outputs ([1, 84, 8400]) are
// supported via the "yolov8-head" mode for fallback.
//
// Pipeline: letterbox → CHW float32 [0,1] → inference →
//   DETR head   : per-query sigmoid(logits) argmax, threshold, done (no NMS)
//   YOLOv8 head : per-cell class argmax, threshold, NMS
// Boxes are returned in the ORIGINAL video's pixel space.
import * as ort from "onnxruntime-web";

export const COCO_CLASSES: readonly string[] = [
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

// COCO class id for "person" — the one class this app explicitly does not
// want to talk to. Matches the policy in the ASSESS_SYSTEM prompt.
export const PERSON_CLASS_ID = 0;

export type Detection = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
  score: number;
  classId: number;
  className: string;
  // Populated only by the seg head. Centroid is in source (video) pixels;
  // maskArea is in 160×160 prototype pixels — useful as an occlusion signal
  // (sudden drop ⇒ something covered the object).
  maskCentroid?: { x: number; y: number };
  maskArea?: number;
  // Binary silhouette at prototype resolution, cropped to the detection's
  // box. Bytes are 0 (outside) or 255 (inside). Row-major, stride = w.
  // The mask visually corresponds to the detection's {x1,y1,x2,y2} source
  // rect. Populated only by the seg head. Used for clipping the talking
  // face to the object's silhouette so it reads as painted-on, not pasted.
  mask?: { data: Uint8Array; w: number; h: number };
  // Orientation of the mask's long axis, in radians, in screen space (y
  // axis points down). Computed via PCA on the binary silhouette in the
  // same pass as mask decode. Angle is in [-π/2, π/2] — principal axis is
  // 180° ambiguous, so we fold to that range.
  //   axisRatio = sqrt(lambda1 / lambda2) — 1.0 for a circle, growing with
  //   elongation. Callers should only rotate when axisRatio is comfortably
  //   above 1 (round objects have a meaningless principal angle).
  principalAngle?: number;
  axisRatio?: number;
};

// Head type selects the post-processing branch.
//   - "yolo-detr": outputs logits [1,N,C] + pred_boxes [1,N,4] normalized
//     cxcywh. YOLO26n + RF-DETR Nano use this.
//   - "yolov8-head": legacy anchor-grid output ([1, 4+C, N]). Used by raw
//     Ultralytics YOLOv8/v11 exports.
//   - "yolo-seg-detr": DETR-style seg. Output0 [1, N, 6+32] rows of
//     (x1, y1, x2, y2, score, classId, 32 mask coefs). Output1 [1, 32, H, W]
//     prototype masks. Per detection we compute the binary mask and its
//     centroid for anchor stability.
export type ModelHead = "yolo-detr" | "yolov8-head" | "yolo-seg-detr";

// Per-model native input sizes, confirmed empirically on each ONNX.
//
// ⚠️ IMPORTANT: RF-DETR is optimized per-variant via Neural Architecture
// Search. Nano is tuned for a specific resolution and its compute scales
// with the **4th power of resolution** — running nano at 640 instead of
// its native ~384 isn't just slower, it's catastrophically slower AND
// degrades accuracy. If you want higher resolution with RF-DETR, move up
// the ladder (small → medium → base), don't upscale nano. (Author guidance,
// 2026-04-18.)
export const MODEL_PRESETS = {
  "yolo26n": { head: "yolo-detr" as ModelHead, inputSize: 640, classIdMap: null as Int32Array | null },
  "yolo26n-seg": { head: "yolo-seg-detr" as ModelHead, inputSize: 640, classIdMap: null as Int32Array | null },
  // RF-DETR Nano's exported ONNX only accepts 384×384 — the detection nano's
  // NAS-native size. We also let anyone explicitly pass a larger RF-DETR
  // variant + matching inputSize if they want more accuracy.
  "rf-detr-nano": { head: "yolo-detr" as ModelHead, inputSize: 384, classIdMap: null as Int32Array | null },
  "yolov8n": { head: "yolov8-head" as ModelHead, inputSize: 640, classIdMap: null as Int32Array | null },
} as const;

// Best-effort preset lookup from a model URL. Returns null if unrecognized,
// letting callers pass explicit head/inputSize for anything custom.
function presetForUrl(url: string): (typeof MODEL_PRESETS)[keyof typeof MODEL_PRESETS] | null {
  const basename = url.split("/").pop()?.toLowerCase() ?? "";
  if (basename.startsWith("yolo26n-seg")) return MODEL_PRESETS["yolo26n-seg"];
  if (basename.startsWith("yolo26n")) return MODEL_PRESETS["yolo26n"];
  if (basename.startsWith("rf-detr-nano")) return MODEL_PRESETS["rf-detr-nano"];
  if (basename.startsWith("yolov8n") || basename.startsWith("yolov11n")) return MODEL_PRESETS["yolov8n"];
  return null;
}

export type YoloInitOptions = {
  modelUrl?: string;
  head?: ModelHead;
  inputSize?: number;
  /** Map raw class id → COCO-80 id, or -1 to drop. Used for RF-DETR which
   *  emits 91 classes including N/A gaps. Omit for YOLO26n (direct 80-id). */
  classIdMap?: Int32Array | null;
  wasmPaths?: string;
  preferWebGPU?: boolean;
};

type Config = Required<Pick<YoloInitOptions, "modelUrl" | "head" | "inputSize">> & {
  classIdMap: Int32Array | null;
};

let session: ort.InferenceSession | null = null;
let cfg: Config | null = null;
let initPromise: Promise<ort.InferenceSession> | null = null;

// Observable status — exposed so the UI can render progress, backend, and
// any fatal error inline instead of staring at a blank camera when something
// fails to load.
export type YoloStatus = {
  stage: "idle" | "downloading" | "compiling" | "ready" | "error";
  backend: "webgpu" | "wasm" | null;
  /** Download progress ratio [0, 1]. -1 while unknown (no content-length). */
  progress: number;
  /** Bytes downloaded, for HUD display. */
  bytesLoaded: number;
  bytesTotal: number;
  error: string | null;
  modelUrl: string | null;
};

const listeners = new Set<(s: YoloStatus) => void>();
let status: YoloStatus = {
  stage: "idle",
  backend: null,
  progress: 0,
  bytesLoaded: 0,
  bytesTotal: 0,
  error: null,
  modelUrl: null,
};

function setStatus(patch: Partial<YoloStatus>) {
  status = { ...status, ...patch };
  for (const l of listeners) {
    try {
      l(status);
    } catch {
      // subscriber blew up — not our problem, don't take down the pipeline
    }
  }
}

export function getYoloStatus(): YoloStatus {
  return status;
}

export function subscribeYoloStatus(cb: (s: YoloStatus) => void): () => void {
  listeners.add(cb);
  cb(status);
  return () => {
    listeners.delete(cb);
  };
}

// Build the runtime config. Explicit opts win; the rest are auto-picked from
// a URL-based preset so nobody accidentally runs a NAS-tuned model at the
// wrong resolution (see MODEL_PRESETS comment).
//
// Default is the seg model — it gives us the mask-centroid anchor which is
// meaningfully more stable than bbox-center for asymmetric objects (the
// mug-handle problem). Seg inference is ~2× the work of plain detection but
// stays well inside real-time on WebGPU, and the centroid drops the
// visible jitter enough to be worth it.
function defaultConfig(opts: YoloInitOptions): Config {
  const modelUrl = opts.modelUrl ?? "/models/yolo26n-seg.onnx";
  const preset = presetForUrl(modelUrl);
  return {
    modelUrl,
    head: opts.head ?? preset?.head ?? "yolo-seg-detr",
    inputSize: opts.inputSize ?? preset?.inputSize ?? 640,
    classIdMap: opts.classIdMap ?? preset?.classIdMap ?? null,
  };
}

// Fetch the ONNX model with progress reporting so the UI can show "32%
// downloaded" instead of a mystery delay. Falls back to a single `fetch` +
// `arrayBuffer()` if the browser can't stream (very old browsers).
async function fetchModelWithProgress(url: string): Promise<ArrayBuffer> {
  const resp = await fetch(url, { cache: "force-cache" });
  if (!resp.ok) {
    throw new Error(`model fetch failed: ${resp.status} ${resp.statusText}`);
  }
  const total = Number(resp.headers.get("content-length") ?? 0);
  setStatus({ stage: "downloading", bytesLoaded: 0, bytesTotal: total, progress: total > 0 ? 0 : -1 });

  if (!resp.body || !("getReader" in resp.body)) {
    const buf = await resp.arrayBuffer();
    setStatus({ bytesLoaded: buf.byteLength, bytesTotal: buf.byteLength, progress: 1 });
    return buf;
  }
  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loaded += value.byteLength;
      setStatus({
        bytesLoaded: loaded,
        bytesTotal: total,
        progress: total > 0 ? loaded / total : -1,
      });
    }
  }
  const merged = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    merged.set(c, off);
    off += c.byteLength;
  }
  return merged.buffer;
}

async function tryCreateSession(
  model: ArrayBuffer,
  providers: ("webgpu" | "wasm")[]
): Promise<{ session: ort.InferenceSession; backend: "webgpu" | "wasm" }> {
  const errors: string[] = [];
  for (const p of providers) {
    try {
      const s = await ort.InferenceSession.create(model, {
        executionProviders: [p] as ort.InferenceSession.ExecutionProviderConfig[],
        graphOptimizationLevel: "all",
        // Keep this session as quiet as possible — session-level logLevel
        // defaults to warning, which Next.js dev overlays as errors.
        logSeverityLevel: 3,
      });
      // eslint-disable-next-line no-console
      console.log(`[yolo] session created, backend=${p}`);
      return { session: s, backend: p };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      errors.push(`${p}: ${msg}`);
      // Use console.log so the Next.js dev overlay doesn't promote this to
      // an error card. Real failures throw below after all providers fail.
      // eslint-disable-next-line no-console
      console.log(`[yolo] backend ${p} unavailable, trying next:`, msg);
    }
  }
  throw new Error(`all backends failed — ${errors.join(" | ")}`);
}

// Load the ONNX model + runtime. Idempotent — concurrent callers share the
// same promise. WebGPU is attempted first when available, WASM otherwise.
//
// Publishes progress/backend/error via the status subscription so the UI can
// render a load bar and surface failures inline.
export async function initYolo(
  opts: YoloInitOptions = {}
): Promise<ort.InferenceSession> {
  if (session && cfg && cfg.modelUrl === (opts.modelUrl ?? cfg.modelUrl)) {
    return session;
  }
  if (initPromise) return initPromise;

  cfg = defaultConfig(opts);
  const wasmPaths = opts.wasmPaths ?? "/ort/";
  const preferWebGPU = opts.preferWebGPU ?? true;

  setStatus({
    stage: "downloading",
    backend: null,
    progress: 0,
    bytesLoaded: 0,
    bytesTotal: 0,
    error: null,
    modelUrl: cfg.modelUrl,
  });

  ort.env.wasm.wasmPaths = wasmPaths;
  // Stay single-threaded by default. Multi-threaded WASM needs COOP/COEP
  // headers (SharedArrayBuffer) which are brittle on localhost + Safari; the
  // single-threaded SIMD path is plenty fast for YOLO26n and it Just Works
  // everywhere.
  ort.env.wasm.numThreads = 1;
  // Disable the proxy worker path — it's finicky across Next.js dev and
  // Safari, and single-threaded in-process inference is reliable.
  ort.env.wasm.proxy = false;
  // ORT is chatty at warning level (partial EP assignment, etc.) and Next.js
  // dev promotes every warning to a fullscreen error overlay that hides the
  // camera feed. Bump to error-only — actual failures still surface.
  ort.env.logLevel = "error";

  initPromise = (async () => {
    try {
      const modelBuf = await fetchModelWithProgress(cfg!.modelUrl);
      setStatus({ stage: "compiling", progress: 1 });
      const providers: ("webgpu" | "wasm")[] = preferWebGPU
        ? ["webgpu", "wasm"]
        : ["wasm"];
      const { session: s, backend } = await tryCreateSession(modelBuf, providers);
      session = s;
      setStatus({ stage: "ready", backend, error: null });
      return s;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // console.log, not console.error — Next.js dev overlay hijacks
      // console.error into a fullscreen card. The error is surfaced via the
      // status observable / error banner already.
      // eslint-disable-next-line no-console
      console.log("[yolo] init failed:", err);
      setStatus({ stage: "error", error: msg });
      initPromise = null; // let callers retry
      throw err;
    }
  })();
  return initPromise;
}

// Tear everything down so we can call initYolo again with a fresh try. Used
// by the retry button in the UI.
export function resetYolo(): void {
  session = null;
  cfg = null;
  initPromise = null;
  setStatus({
    stage: "idle",
    backend: null,
    progress: 0,
    bytesLoaded: 0,
    bytesTotal: 0,
    error: null,
    modelUrl: null,
  });
}

export function isYoloReady(): boolean {
  return session !== null;
}

// Letterbox into SIZE×SIZE preserving aspect ratio, pad with RGB(114,114,114).
// Returns the tensor plus the transform needed to map predictions back to
// the source's pixel space.
type LetterboxResult = {
  tensor: ort.Tensor;
  scale: number;
  padX: number;
  padY: number;
  size: number;
};

function letterbox(
  source: CanvasImageSource,
  srcW: number,
  srcH: number,
  scratch: HTMLCanvasElement,
  size: number
): LetterboxResult | null {
  const scale = Math.min(size / srcW, size / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = Math.floor((size - newW) / 2);
  const padY = Math.floor((size - newH) / 2);

  scratch.width = size;
  scratch.height = size;
  const ctx = scratch.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(source, padX, padY, newW, newH);

  const { data } = ctx.getImageData(0, 0, size, size);
  const pixels = size * size;
  const tensor = new Float32Array(3 * pixels);
  const rOff = 0;
  const gOff = pixels;
  const bOff = 2 * pixels;
  for (let i = 0; i < pixels; i++) {
    const src = i * 4;
    tensor[rOff + i] = data[src] / 255;
    tensor[gOff + i] = data[src + 1] / 255;
    tensor[bOff + i] = data[src + 2] / 255;
  }

  return {
    tensor: new ort.Tensor("float32", tensor, [1, 3, size, size]),
    scale,
    padX,
    padY,
    size,
  };
}

// Sigmoid with mild precision guard. Used per-logit in the DETR head only
// (thousands of calls per frame; hand-inlined rather than relying on Math).
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function iouBox(
  ax1: number, ay1: number, ax2: number, ay2: number,
  bx1: number, by1: number, bx2: number, by2: number
): number {
  const x1 = Math.max(ax1, bx1);
  const y1 = Math.max(ay1, by1);
  const x2 = Math.min(ax2, bx2);
  const y2 = Math.min(ay2, by2);
  if (x2 <= x1 || y2 <= y1) return 0;
  const inter = (x2 - x1) * (y2 - y1);
  const areaA = Math.max(0, (ax2 - ax1) * (ay2 - ay1));
  const areaB = Math.max(0, (bx2 - bx1) * (by2 - by1));
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

function nms(dets: Detection[], iouThreshold: number): Detection[] {
  if (dets.length < 2) return dets;
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const keep: Detection[] = [];
  const suppressed = new Uint8Array(sorted.length);
  for (let i = 0; i < sorted.length; i++) {
    if (suppressed[i]) continue;
    const a = sorted[i];
    keep.push(a);
    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed[j]) continue;
      const b = sorted[j];
      const ov = iouBox(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2);
      const t = a.classId === b.classId ? iouThreshold : 0.85;
      if (ov > t) suppressed[j] = 1;
    }
  }
  return keep;
}

export type DetectOptions = {
  confThreshold?: number;
  iouThreshold?: number;
  classFilter?: (id: number) => boolean;
  maxDetections?: number;
};

function mapToCoco80(
  rawClassId: number,
  classIdMap: Int32Array | null
): number {
  if (!classIdMap) return rawClassId;
  if (rawClassId < 0 || rawClassId >= classIdMap.length) return -1;
  return classIdMap[rawClassId];
}

// Build a Detection from normalized cxcywh (DETR) in the letterbox-input
// coordinate frame. Handles the letterbox inverse and clamps to the source.
function detectionFromDetrBox(
  cxN: number,
  cyN: number,
  wN: number,
  hN: number,
  score: number,
  coco80: number,
  pre: LetterboxResult,
  vw: number,
  vh: number
): Detection | null {
  if (coco80 < 0 || coco80 >= COCO_CLASSES.length) return null;
  const cx640 = cxN * pre.size;
  const cy640 = cyN * pre.size;
  const w640 = wN * pre.size;
  const h640 = hN * pre.size;
  const cx = (cx640 - pre.padX) / pre.scale;
  const cy = (cy640 - pre.padY) / pre.scale;
  const w = w640 / pre.scale;
  const h = h640 / pre.scale;
  const x1 = Math.max(0, cx - w / 2);
  const y1 = Math.max(0, cy - h / 2);
  const x2 = Math.min(vw, cx + w / 2);
  const y2 = Math.min(vh, cy + h / 2);
  if (x2 - x1 < 4 || y2 - y1 < 4) return null;
  return {
    x1, y1, x2, y2,
    cx, cy, w, h,
    score,
    classId: coco80,
    className: COCO_CLASSES[coco80] ?? "?",
  };
}

// DETR-style post-processing: iterate N queries, take sigmoid-max class per
// query, threshold, convert boxes. No NMS — set prediction handles dedup.
function postprocessDetr(
  outputs: Record<string, ort.Tensor>,
  pre: LetterboxResult,
  vw: number,
  vh: number,
  conf: number,
  classIdMap: Int32Array | null,
  classFilter?: (id: number) => boolean
): Detection[] {
  // Tolerate both name pairs: YOLO26n uses (logits, pred_boxes); RF-DETR
  // flavors often use (pred_logits, pred_boxes).
  const logitsTensor =
    outputs["logits"] ?? outputs["pred_logits"] ?? null;
  const boxesTensor = outputs["pred_boxes"] ?? null;
  if (!logitsTensor || !boxesTensor) return [];
  const logits = logitsTensor.data as Float32Array;
  const boxes = boxesTensor.data as Float32Array;
  const [, nQueries, nClasses] = logitsTensor.dims as [number, number, number];

  const out: Detection[] = [];
  for (let q = 0; q < nQueries; q++) {
    let bestLogit = -Infinity;
    let bestCls = -1;
    const base = q * nClasses;
    for (let c = 0; c < nClasses; c++) {
      const v = logits[base + c];
      if (v > bestLogit) {
        bestLogit = v;
        bestCls = c;
      }
    }
    const score = sigmoid(bestLogit);
    if (score < conf) continue;

    const coco80 = mapToCoco80(bestCls, classIdMap);
    if (coco80 < 0) continue;
    if (classFilter && !classFilter(coco80)) continue;

    const bi = q * 4;
    const det = detectionFromDetrBox(
      boxes[bi], boxes[bi + 1], boxes[bi + 2], boxes[bi + 3],
      score, coco80, pre, vw, vh
    );
    if (det) out.push(det);
  }
  return out;
}

// YOLOv8-head post-processing: [1, 4+C, N] with rows 0..3 = cxcywh in input
// pixels. Classic anchor-grid output — needs NMS.
function postprocessYoloV8(
  outputs: Record<string, ort.Tensor>,
  pre: LetterboxResult,
  vw: number,
  vh: number,
  conf: number,
  iouT: number,
  classFilter?: (id: number) => boolean
): Detection[] {
  const key = Object.keys(outputs)[0];
  const out = outputs[key];
  const [, nFeatures, nBoxes] = out.dims as [number, number, number];
  const data = out.data as Float32Array;
  const nClasses = nFeatures - 4;

  const candidates: Detection[] = [];
  for (let i = 0; i < nBoxes; i++) {
    let bestScore = 0;
    let bestClass = -1;
    for (let c = 0; c < nClasses; c++) {
      const s = data[(4 + c) * nBoxes + i];
      if (s > bestScore) {
        bestScore = s;
        bestClass = c;
      }
    }
    if (bestScore < conf) continue;
    if (classFilter && !classFilter(bestClass)) continue;

    const cx640 = data[0 * nBoxes + i];
    const cy640 = data[1 * nBoxes + i];
    const w640 = data[2 * nBoxes + i];
    const h640 = data[3 * nBoxes + i];

    const cx = (cx640 - pre.padX) / pre.scale;
    const cy = (cy640 - pre.padY) / pre.scale;
    const w = w640 / pre.scale;
    const h = h640 / pre.scale;

    const x1 = Math.max(0, cx - w / 2);
    const y1 = Math.max(0, cy - h / 2);
    const x2 = Math.min(vw, cx + w / 2);
    const y2 = Math.min(vh, cy + h / 2);
    if (x2 - x1 < 4 || y2 - y1 < 4) continue;

    candidates.push({
      x1, y1, x2, y2,
      cx, cy, w, h,
      score: bestScore,
      classId: bestClass,
      className: COCO_CLASSES[bestClass] ?? "?",
    });
  }
  return nms(candidates, iouT);
}

// DETR-style seg post-processing.
// Output0 rows: [x1, y1, x2, y2, score, classId, 32 mask coefficients] in
// input-pixel space (not normalized). Output1 is [1, 32, H, W] prototype
// masks at a quarter of the input resolution.
//
// Per kept detection we do `sigmoid(coefs @ protos)` to recover a dense
// mask, crop it to the box, threshold at 0.5, and compute a centroid. That
// centroid is what the tracker uses as the stable face anchor — a mug's
// mask centroid lives inside the mug body, not sheared by the handle the
// way the bbox center is.
//
// Cost: 32 × H × W multiply-adds per detection (≈820k for 160×160), plus
// sigmoid. Under 3 ms per detection on modern CPUs; we cap the number of
// detections we bother decoding so cost stays bounded on mobile.
function postprocessYoloSegDetr(
  outputs: Record<string, ort.Tensor>,
  pre: LetterboxResult,
  vw: number,
  vh: number,
  conf: number,
  classIdMap: Int32Array | null,
  classFilter?: (id: number) => boolean,
  maxMaskDetections = 8
): Detection[] {
  const keys = Object.keys(outputs);
  // Output order isn't guaranteed — pick by rank.
  let out0: ort.Tensor | null = null;
  let out1: ort.Tensor | null = null;
  for (const k of keys) {
    const t = outputs[k];
    if (t.dims.length === 3 && t.dims[2] > 6) out0 = t;
    else if (t.dims.length === 4) out1 = t;
  }
  if (!out0 || !out1) return [];

  const [, nQ, nFeat] = out0.dims as [number, number, number];
  const [, nMasks, mh, mw] = out1.dims as [number, number, number, number];
  if (nFeat < 6 + nMasks) return [];

  const d0 = out0.data as Float32Array;
  const d1 = out1.data as Float32Array;

  // First pass: cheaply collect qualifying detections WITHOUT mask decode.
  type Pending = { q: number; x1: number; y1: number; x2: number; y2: number; score: number; classRaw: number; coefOff: number };
  const pending: Pending[] = [];
  for (let i = 0; i < nQ; i++) {
    const base = i * nFeat;
    const score = d0[base + 4];
    if (score < conf) continue;
    const classRaw = Math.round(d0[base + 5]);
    const coco80 = classIdMap
      ? (classRaw >= 0 && classRaw < classIdMap.length ? classIdMap[classRaw] : -1)
      : classRaw;
    if (coco80 < 0 || coco80 >= COCO_CLASSES.length) continue;
    if (classFilter && !classFilter(coco80)) continue;
    pending.push({
      q: i,
      x1: d0[base + 0],
      y1: d0[base + 1],
      x2: d0[base + 2],
      y2: d0[base + 3],
      score,
      classRaw: coco80,
      coefOff: base + 6,
    });
  }

  // Keep the top-K by score for full mask decode. More than 8 masks per
  // frame gets costly on mobile and our UI only ever uses the tight
  // top-of-list for tapping/locking.
  pending.sort((a, b) => b.score - a.score);
  const keep = pending.slice(0, maxMaskDetections);

  const protoArea = mh * mw;
  const out: Detection[] = [];

  for (const p of keep) {
    // Input-space box → source space (letterbox inverse).
    const cx640 = (p.x1 + p.x2) * 0.5;
    const cy640 = (p.y1 + p.y2) * 0.5;
    const w640 = p.x2 - p.x1;
    const h640 = p.y2 - p.y1;
    const cx = (cx640 - pre.padX) / pre.scale;
    const cy = (cy640 - pre.padY) / pre.scale;
    const w = w640 / pre.scale;
    const h = h640 / pre.scale;
    const sx1 = Math.max(0, cx - w / 2);
    const sy1 = Math.max(0, cy - h / 2);
    const sx2 = Math.min(vw, cx + w / 2);
    const sy2 = Math.min(vh, cy + h / 2);
    if (sx2 - sx1 < 4 || sy2 - sy1 < 4) continue;

    // Mask decode over the intersection of the box with the 160×160 proto
    // grid — huge perf win over decoding the full 160×160 per detection.
    // Proto grid covers the same 640×640 letterbox as the input, at mh×mw
    // resolution. So a mask pixel (mx, my) corresponds to input (mx * 640/mw, my * 640/mh).
    const inputToProtoX = mw / pre.size;
    const inputToProtoY = mh / pre.size;
    const px1 = Math.max(0, Math.floor(p.x1 * inputToProtoX));
    const py1 = Math.max(0, Math.floor(p.y1 * inputToProtoY));
    const px2 = Math.min(mw, Math.ceil(p.x2 * inputToProtoX));
    const py2 = Math.min(mh, Math.ceil(p.y2 * inputToProtoY));
    if (px2 <= px1 || py2 <= py1) continue;

    // Capture the binary silhouette alongside the centroid decode so we
    // only touch each mask pixel once. Dimensions in proto space.
    const maskW = px2 - px1;
    const maskH = py2 - py1;
    const maskData = new Uint8Array(maskW * maskH);

    let centroidX = 0;
    let centroidY = 0;
    let totalMass = 0;
    let abovePixels = 0;
    // PCA sums (unweighted, over above-threshold pixels). Used for
    // principal-axis orientation so the face tilts to match a banana on
    // its side. Same pass — negligible extra cost.
    let sumX = 0;
    let sumY = 0;
    let sumXX = 0;
    let sumYY = 0;
    let sumXY = 0;
    for (let y = py1; y < py2; y++) {
      const rowOff = y * mw;
      const maskRowOff = (y - py1) * maskW;
      for (let x = px1; x < px2; x++) {
        // Dot product across the 32 coefs × 32 protos at this pixel.
        let s = 0;
        for (let c = 0; c < nMasks; c++) {
          s += d0[p.coefOff + c] * d1[c * protoArea + rowOff + x];
        }
        // Sigmoid + threshold 0.5. Since sigmoid(0) = 0.5, we just check s > 0.
        if (s > 0) {
          const v = 1 / (1 + Math.exp(-s));
          if (v > 0.5) {
            centroidX += x * v;
            centroidY += y * v;
            totalMass += v;
            abovePixels++;
            maskData[maskRowOff + (x - px1)] = 255;
            sumX += x;
            sumY += y;
            sumXX += x * x;
            sumYY += y * y;
            sumXY += x * y;
          }
        }
      }
    }

    // Orientation via PCA on the binary silhouette. Closed-form angle for
    // a 2x2 symmetric covariance matrix avoids eigenvector branching; the
    // ratio is sqrt(λ1/λ2) for the usual major/minor axis scaling.
    // Needs enough pixels to be statistically meaningful — below ~20 the
    // covariance is dominated by quantization noise.
    let principalAngle: number | undefined;
    let axisRatio: number | undefined;
    if (abovePixels > 20) {
      const invN = 1 / abovePixels;
      const meanX = sumX * invN;
      const meanY = sumY * invN;
      const covXX = sumXX * invN - meanX * meanX;
      const covYY = sumYY * invN - meanY * meanY;
      const covXY = sumXY * invN - meanX * meanY;
      principalAngle = 0.5 * Math.atan2(2 * covXY, covXX - covYY);
      const trace = covXX + covYY;
      const disc = Math.max(0, trace * trace * 0.25 - (covXX * covYY - covXY * covXY));
      const lambda1 = trace * 0.5 + Math.sqrt(disc);
      const lambda2 = trace * 0.5 - Math.sqrt(disc);
      axisRatio = Math.sqrt(Math.max(1, lambda1) / Math.max(1, lambda2));
    }

    let maskCentroid: { x: number; y: number } | undefined;
    if (totalMass > 0) {
      // Centroid lives in proto coords; scale back to source pixels.
      const protoCx = centroidX / totalMass;
      const protoCy = centroidY / totalMass;
      const centroid640X = protoCx / inputToProtoX;
      const centroid640Y = protoCy / inputToProtoY;
      const centroidSrcX = (centroid640X - pre.padX) / pre.scale;
      const centroidSrcY = (centroid640Y - pre.padY) / pre.scale;
      // Clamp to the box — guards against rare degenerate masks that leak
      // outside the box (mostly from low-confidence prototype artifacts).
      maskCentroid = {
        x: Math.max(sx1, Math.min(sx2, centroidSrcX)),
        y: Math.max(sy1, Math.min(sy2, centroidSrcY)),
      };
    }

    out.push({
      x1: sx1, y1: sy1, x2: sx2, y2: sy2,
      cx, cy, w, h,
      score: p.score,
      classId: p.classRaw,
      className: COCO_CLASSES[p.classRaw] ?? "?",
      maskCentroid,
      maskArea: abovePixels,
      mask: abovePixels > 0 ? { data: maskData, w: maskW, h: maskH } : undefined,
      principalAngle,
      axisRatio,
    });
  }

  return out;
}

export async function detect(
  video: HTMLVideoElement,
  scratch: HTMLCanvasElement,
  opts: DetectOptions = {}
): Promise<Detection[]> {
  // Returning [] on "not initialized" rather than throwing is intentional: in
  // Next.js dev, HMR can reload this module (wiping `session`) while the
  // consumer component stays mounted and keeps calling detect(). Throwing
  // there fires console.error → Next's dev overlay → looks like a real bug.
  // Empty results until the caller re-inits is the correct idle behavior.
  if (!session || !cfg) return [];
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return [];

  const conf = opts.confThreshold ?? 0.35;
  const iouT = opts.iouThreshold ?? 0.45;
  const maxDet = opts.maxDetections ?? 50;

  const pre = letterbox(video, vw, vh, scratch, cfg.inputSize);
  if (!pre) return [];

  const inputName = session.inputNames[0];
  const feeds: Record<string, ort.Tensor> = { [inputName]: pre.tensor };
  const outputs = (await session.run(feeds)) as Record<string, ort.Tensor>;

  const results =
    cfg.head === "yolo-detr"
      ? postprocessDetr(outputs, pre, vw, vh, conf, cfg.classIdMap, opts.classFilter)
      : cfg.head === "yolo-seg-detr"
        ? postprocessYoloSegDetr(outputs, pre, vw, vh, conf, cfg.classIdMap, opts.classFilter, opts.maxDetections ?? 10)
        : postprocessYoloV8(outputs, pre, vw, vh, conf, iouT, opts.classFilter);

  return results.length > maxDet
    ? results.sort((a, b) => b.score - a.score).slice(0, maxDet)
    : results;
}

// COCO-91 → COCO-80 mapping used by RF-DETR exports (91 raw classes include
// N/A gaps from the original COCO JSON). Pre-computed constant — commented
// out until we flip RF-DETR on as the default, but ready to wire.
export const RF_DETR_COCO91_TO_COCO80: Int32Array = new Int32Array([
  // 0..90, -1 = drop (N/A). Derived from standard COCO 91→80 mapping.
  -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, 11, -1, -1, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, -1, 23, -1, -1, 24, 25, -1, 26, 27, 28, 29, 30, 31,
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, -1, 45, 46, 47, 48, 49, 50,
  51, 52, 53, 54, 55, 56, 57, 58, 59, 60, -1, 61, -1, -1, 62, -1, 63, 64, 65, 66,
  67, 68, 69, 70, 71, 72, -1, 73, 74, 75, 76, 77, 78, 79,
]);

// Debug helper — emits the letterboxed frame as a JPEG data URL.
export function debugLetterboxDataUrl(
  video: HTMLVideoElement,
  scratch: HTMLCanvasElement
): string | null {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh || !cfg) return null;
  if (!letterbox(video, vw, vh, scratch, cfg.inputSize)) return null;
  return scratch.toDataURL("image/jpeg", 0.7);
}
