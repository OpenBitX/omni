// WebSocket-backed YOLO detector. Drop-in replacement for `lib/yolo.ts`:
// same public API (initYolo, detect, resetYolo, subscribeYoloStatus, …) and
// same Detection shape. The heavy lifting (seg inference, mask decode, pole,
// PCA) happens in the Python server on MPS; this module just encodes frames,
// ships them over WS, and unpacks JSON responses.
//
// Lockstep: only one request is in flight at a time. A detect() call while
// another is mid-flight returns the previous frame's detections immediately
// (the same behaviour the old ORT path had when the caller hit its inference
// gate). This is fine because tracker.tsx already rate-limits to a max FPS
// and smooths boxes with an EMA between inferences.

import { wsUrl } from "@/lib/backend-url";

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
  maskCentroid?: { x: number; y: number };
  maskArea?: number;
  mask?: { data: Uint8Array; w: number; h: number };
  principalAngle?: number;
  axisRatio?: number;
  maskPole?: { x: number; y: number };
};

export type YoloStatus = {
  stage: "idle" | "downloading" | "compiling" | "ready" | "error";
  backend: "webgpu" | "wasm" | null;
  progress: number;
  bytesLoaded: number;
  bytesTotal: number;
  error: string | null;
  modelUrl: string | null;
};

export type YoloInitOptions = {
  // Kept for signature compatibility; ignored for WS. Set YOLO_WS_URL on the
  // window / env to override the server URL.
  modelUrl?: string;
  serverUrl?: string;
};

export type DetectOptions = {
  confThreshold?: number;
  iouThreshold?: number;
  classFilter?: (id: number) => boolean;
  maxDetections?: number;
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
      // subscriber blew up — ignore
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

// --- WS client state ------------------------------------------------------

type Pending = {
  seq: number;
  resolve: (dets: Detection[]) => void;
  timeoutHandle: ReturnType<typeof setTimeout> | null;
};

let ws: WebSocket | null = null;
let wsEndpointUrl: string | null = null;
let connecting: Promise<void> | null = null;
let ready = false;
let nextSeq = 1;
const pending = new Map<number, Pending>();
let lastDetections: Detection[] = [];
let inFlightSeq: number | null = null;
// Auto-reconnect state. `wantConnected` stays true from the first initYolo()
// call until resetYolo(), so transient drops (server restart, wifi blip,
// sleep/wake) heal without a user-visible "click retry" step.
let wantConnected = false;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let reconnectAttempts = 0;
// 500ms → 1s → 2s → 4s → 8s (Mirror uses the same curve).
const RECONNECT_BASE_MS = 500;
const RECONNECT_MAX_MS = 8000;

function resolveWsUrl(opts: YoloInitOptions): string {
  if (opts.serverUrl) return opts.serverUrl;
  if (typeof window !== "undefined") {
    const w = window as unknown as { YOLO_WS_URL?: string };
    if (typeof w.YOLO_WS_URL === "string") return w.YOLO_WS_URL;
  }
  if (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_YOLO_WS_URL) {
    return process.env.NEXT_PUBLIC_YOLO_WS_URL;
  }
  // Default: derive from NEXT_PUBLIC_BACKEND_URL (see lib/backend-url.ts).
  // This ensures true decoupling — the browser talks directly to the backend,
  // not through a Next.js rewrite proxy.
  return wsUrl("/ws/yolo");
}

function clearPending() {
  for (const p of pending.values()) {
    if (p.timeoutHandle) clearTimeout(p.timeoutHandle);
    p.resolve([]);
  }
  pending.clear();
  inFlightSeq = null;
}

function teardownSocket() {
  ready = false;
  clearPending();
  if (ws) {
    try { ws.close(); } catch { /* ignore */ }
  }
  ws = null;
}

function scheduleReconnect(reason: string) {
  if (!wantConnected) return;
  // Surface status, but don't flip to "error" — we're mid-recovery, which is
  // a different UX than "give up and click retry".
  setStatus({ stage: "downloading", error: `reconnecting: ${reason}` });
  if (reconnectTimer !== null) return;
  const delay = Math.min(
    RECONNECT_MAX_MS,
    RECONNECT_BASE_MS * Math.pow(2, Math.min(reconnectAttempts, 6))
  );
  reconnectAttempts++;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    if (!wantConnected) return;
    connect({}).catch(() => {
      // connect() schedules the next attempt on failure via onclose/onerror;
      // nothing to do here.
    });
  }, delay);
}

function handleMessage(ev: MessageEvent) {
  if (typeof ev.data !== "string") return;
  let msg: unknown;
  try {
    msg = JSON.parse(ev.data);
  } catch {
    return;
  }
  const data = msg as { type?: string; seq?: number; detections?: unknown[]; error?: string; model?: string; device?: string };
  if (data.type === "ready") {
    ready = true;
    setStatus({ stage: "ready", backend: null, error: null, modelUrl: data.model ?? null });
    return;
  }
  if (data.type === "error") {
    setStatus({ stage: "error", error: data.error ?? "server error" });
    return;
  }
  if (data.type === "detections") {
    const dets = Array.isArray(data.detections) ? data.detections.map(decodeDetection).filter(Boolean) as Detection[] : [];
    lastDetections = dets;
    const seq = typeof data.seq === "number" ? data.seq : null;
    if (seq !== null) {
      const p = pending.get(seq);
      if (p) {
        pending.delete(seq);
        if (p.timeoutHandle) clearTimeout(p.timeoutHandle);
        if (inFlightSeq === seq) inFlightSeq = null;
        p.resolve(dets);
      }
    }
  }
}

function decodeDetection(raw: unknown): Detection | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const num = (k: string) => (typeof r[k] === "number" ? (r[k] as number) : NaN);
  const x1 = num("x1"), y1 = num("y1"), x2 = num("x2"), y2 = num("y2");
  if (!Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(x2) || !Number.isFinite(y2)) return null;
  const det: Detection = {
    x1, y1, x2, y2,
    cx: num("cx"), cy: num("cy"), w: num("w"), h: num("h"),
    score: num("score"),
    classId: Math.round(num("classId")),
    className: typeof r.className === "string" ? r.className : "?",
  };
  const mc = r.maskCentroid as { x?: number; y?: number } | undefined;
  if (mc && typeof mc.x === "number" && typeof mc.y === "number") det.maskCentroid = { x: mc.x, y: mc.y };
  const mp = r.maskPole as { x?: number; y?: number } | undefined;
  if (mp && typeof mp.x === "number" && typeof mp.y === "number") det.maskPole = { x: mp.x, y: mp.y };
  if (typeof r.maskArea === "number") det.maskArea = r.maskArea;
  if (typeof r.principalAngle === "number") det.principalAngle = r.principalAngle;
  if (typeof r.axisRatio === "number") det.axisRatio = r.axisRatio;
  const m = r.mask as { w?: number; h?: number; data?: string; ox?: number; oy?: number; ow?: number; oh?: number } | undefined;
  if (m && typeof m.w === "number" && typeof m.h === "number" && typeof m.data === "string") {
    try {
      const bin = atob(m.data);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      det.mask = { data: bytes, w: m.w, h: m.h };
    } catch {
      // ignore malformed mask
    }
  }
  return det;
}

async function connect(opts: YoloInitOptions): Promise<void> {
  if (ready && ws && ws.readyState === WebSocket.OPEN) return;
  if (connecting) return connecting;
  // Drop any stale socket before opening a new one — this is important on
  // reconnect where the old sock is in a CLOSING/CLOSED state but still
  // attached.
  teardownSocket();
  const url = opts.serverUrl ? opts.serverUrl : (wsEndpointUrl ?? resolveWsUrl(opts));
  wsEndpointUrl = url;
  setStatus({
    stage: "downloading",
    backend: null,
    progress: 0,
    bytesLoaded: 0,
    bytesTotal: 0,
    error: reconnectAttempts > 0 ? `reconnecting (attempt ${reconnectAttempts})` : null,
    modelUrl: url,
  });
  connecting = new Promise<void>((resolve, reject) => {
    let sock: WebSocket;
    try {
      sock = new WebSocket(url);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus({ stage: "error", error: msg });
      scheduleReconnect(msg);
      reject(err);
      return;
    }
    ws = sock;
    let settled = false;

    const onOpen = () => {
      setStatus({ stage: "compiling", progress: 1 });
    };
    const onMessage = (ev: MessageEvent) => {
      if (!settled && typeof ev.data === "string") {
        try {
          const parsed = JSON.parse(ev.data) as { type?: string };
          if (parsed.type === "ready") {
            settled = true;
            reconnectAttempts = 0;
            resolve();
          }
        } catch {
          // non-JSON — ignored
        }
      }
      handleMessage(ev);
    };
    const onError = () => {
      // Don't mark status=error here — the subsequent onClose will schedule
      // reconnect. Browsers fire "error" then "close" in rapid succession on
      // network failures; we want the close path to own the recovery.
      if (!settled) {
        settled = true;
        reject(new Error(`websocket error (${url})`));
      }
    };
    const onClose = () => {
      const wasReady = ready;
      teardownSocket();
      if (!settled) {
        settled = true;
        reject(new Error("websocket closed before ready"));
      }
      scheduleReconnect(wasReady ? "connection dropped" : "handshake failed");
    };
    sock.addEventListener("open", onOpen);
    sock.addEventListener("message", onMessage);
    sock.addEventListener("error", onError);
    sock.addEventListener("close", onClose);
  });
  try {
    await connecting;
  } finally {
    connecting = null;
  }
}

export async function initYolo(opts: YoloInitOptions = {}): Promise<void> {
  wantConnected = true;
  reconnectAttempts = 0;
  await connect(opts);
}

export function resetYolo(): void {
  wantConnected = false;
  if (reconnectTimer !== null) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  reconnectAttempts = 0;
  lastDetections = [];
  teardownSocket();
  wsEndpointUrl = null;
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
  return ready && ws !== null && ws.readyState === WebSocket.OPEN;
}

// Encode the current video frame to JPEG bytes via the scratch canvas.
// JPEG quality 0.7 @ the video's native resolution is a sensible balance
// for LAN — higher quality barely helps detection and hurts RTT on WAN.
async function encodeFrame(video: HTMLVideoElement, scratch: HTMLCanvasElement): Promise<Blob | null> {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;
  // Cap longer side at 720 for transit. 480–720 is the sweet spot for COCO
  // pretrained models; going higher mostly wastes bandwidth.
  const MAX_SIDE = 640;
  const longer = Math.max(vw, vh);
  const scale = longer > MAX_SIDE ? MAX_SIDE / longer : 1;
  const dw = Math.round(vw * scale);
  const dh = Math.round(vh * scale);
  scratch.width = dw;
  scratch.height = dh;
  const ctx = scratch.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, dw, dh);
  return new Promise<Blob | null>((resolve) => {
    scratch.toBlob((b) => resolve(b), "image/jpeg", 0.7);
  });
}

export async function detect(
  video: HTMLVideoElement,
  scratch: HTMLCanvasElement,
  opts: DetectOptions = {}
): Promise<Detection[]> {
  if (!isYoloReady()) return lastDetections;
  if (inFlightSeq !== null) {
    // Another request is already in flight — surface the latest known dets
    // so the caller's render loop has something to chew on without queuing
    // a second concurrent inference. Matches the original module's
    // "return [] while not ready" semantics well enough.
    return lastDetections;
  }

  const blob = await encodeFrame(video, scratch);
  if (!blob) return lastDetections;

  // Scale detections from the encoded image size back to the video's native
  // pixel space, since tracker.tsx assumes source pixels. We do this server
  // side too (detections are in the encoded frame's pixel space), so we
  // pass the scale back client side.
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const scaleX = vw / scratch.width;
  const scaleY = vh / scratch.height;

  const seq = nextSeq++;
  inFlightSeq = seq;

  const preamble = {
    type: "opts" as const,
    seq,
    conf: opts.confThreshold ?? 0.35,
    iou: opts.iouThreshold ?? 0.45,
    maxDet: opts.maxDetections ?? 10,
  };

  const buf = await blob.arrayBuffer();
  const sock = ws;
  if (!sock || sock.readyState !== WebSocket.OPEN) {
    inFlightSeq = null;
    return lastDetections;
  }

  const done = new Promise<Detection[]>((resolve) => {
    const timeoutHandle = setTimeout(() => {
      const p = pending.get(seq);
      if (p) {
        pending.delete(seq);
        if (inFlightSeq === seq) inFlightSeq = null;
        resolve(lastDetections);
      }
    }, 5000);
    pending.set(seq, { seq, resolve, timeoutHandle });
  });

  try {
    sock.send(JSON.stringify(preamble));
    sock.send(buf);
  } catch {
    const p = pending.get(seq);
    if (p && p.timeoutHandle) clearTimeout(p.timeoutHandle);
    pending.delete(seq);
    inFlightSeq = null;
    return lastDetections;
  }

  const raw = await done;

  // Server returns detections in the frame's pixel space. If we downsampled
  // before sending, scale back to video pixels so the rest of the pipeline
  // (anchors, smoothing, rendering) works unchanged.
  const scaled = scaleX === 1 && scaleY === 1 ? raw : raw.map((d) => scaleDetection(d, scaleX, scaleY));

  // Apply client-side class filter (we don't ship the filter function to
  // the server — excludeClassIds covers the common case but general fn is
  // kept client-side).
  const filtered = opts.classFilter ? scaled.filter((d) => opts.classFilter!(d.classId)) : scaled;
  lastDetections = filtered;
  return filtered;
}

function scaleDetection(d: Detection, sx: number, sy: number): Detection {
  const out: Detection = {
    ...d,
    x1: d.x1 * sx,
    y1: d.y1 * sy,
    x2: d.x2 * sx,
    y2: d.y2 * sy,
    cx: d.cx * sx,
    cy: d.cy * sy,
    w: d.w * sx,
    h: d.h * sy,
  };
  if (d.maskCentroid) out.maskCentroid = { x: d.maskCentroid.x * sx, y: d.maskCentroid.y * sy };
  if (d.maskPole) out.maskPole = { x: d.maskPole.x * sx, y: d.maskPole.y * sy };
  // mask data itself stays in its own grid; tracker uses it as a silhouette
  // relative to the bbox, and the bbox is already in source space. The
  // original `lib/yolo.ts` never guarantees mask pixel scale matches source
  // pixels either — callers treat it as "the silhouette, cropped to the
  // bbox, at whatever resolution the prototype head produced".
  return out;
}

// Debug helper kept for API parity with the old module. Emits the encoded
// frame as a JPEG data URL via the scratch canvas.
export function debugLetterboxDataUrl(
  video: HTMLVideoElement,
  scratch: HTMLCanvasElement
): string | null {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;
  scratch.width = vw;
  scratch.height = vh;
  const ctx = scratch.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0);
  return scratch.toDataURL("image/jpeg", 0.7);
}
