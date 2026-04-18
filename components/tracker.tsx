"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { converseWithObject, describeObject, generateLine } from "@/app/actions";
import {
  FACE_VOICE_HEIGHT,
  FACE_VOICE_WIDTH,
  FaceVoice,
  classifyShapeSmooth,
  createLipSyncState,
  type LipSyncState,
  type MouthShape,
} from "@/components/face-voice";
import { SpeechBubble } from "@/components/speech-bubble";
import {
  COCO_CLASSES,
  PERSON_CLASS_ID,
  detect,
  getYoloStatus,
  initYolo,
  resetYolo,
  subscribeYoloStatus,
  type Detection,
  type YoloStatus,
} from "@/lib/yolo";
import {
  applyAnchor,
  makeBox,
  matchTarget,
  newBoxEMA,
  seedBoxEMA,
  smoothBox,
  type Anchor,
  type Box,
  type BoxEMA,
} from "@/lib/iou";

// === Web Speech API typing ==============================================
//
// The DOM lib doesn't always expose SpeechRecognition (it's
// implementation-defined) and Safari only ships the webkit-prefixed
// variant. We declare a minimal shape locally so the code compiles
// everywhere and feature-detect at runtime before using it.

interface SpeechRecognitionResultLike {
  isFinal: boolean;
  0: { transcript: string; confidence?: number };
}
interface SpeechRecognitionResultListLike {
  length: number;
  [index: number]: SpeechRecognitionResultLike;
}
interface SpeechRecognitionEventLike extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultListLike;
}
interface SpeechRecognitionErrorEventLike extends Event {
  error: string;
  message?: string;
}
interface SpeechRecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((e: SpeechRecognitionEventLike) => void) | null;
  onerror: ((e: SpeechRecognitionErrorEventLike) => void) | null;
  onend: ((e: Event) => void) | null;
  onstart: ((e: Event) => void) | null;
}
type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

function getSpeechRecognitionCtor(): SpeechRecognitionCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

// === Pipeline tuning knobs ===============================================

// Inference rate cap. YOLOv8n on mobile CPU-WASM sits around 3–8 FPS; on
// desktop WebGPU it hits 30+. The cap keeps us from burning battery for
// motion we can already handle with EMA between inferences.
const MAX_INFERENCE_FPS = 30;

// IoU gate for "this new box is the same instance I was tracking". Loose
// enough to survive shape wobble, tight enough that two adjacent cups don't
// share an identity. A same-class nearest-center fallback kicks in on
// consecutive misses (see WIDEN_MATCH_AFTER_MISSES).
const IDENTITY_IOU_MIN = 0.3;

// EMA alphas. Position alpha is high enough that the face stays glued to
// the latest detection without visible lag, but not so high that YOLO box
// jitter shows through as wobble. Size is slower again — bbox edges
// breathe more than centers do. Opacity fades fast so reacquisition and
// disappearance both feel instant.
const BOX_POS_ALPHA = 0.7;
const BOX_SIZE_ALPHA = 0.25;
const OPACITY_EMA_ALPHA = 0.4;

// "Lost" threshold. One concept, two consequences: (a) the face fades
// out — the object has actually left the frame, not just blinked for one
// inference; (b) the next match SNAPS the smoothed pose instead of EMA
// sliding — avoids a visible glide across the screen as the face fades
// back in at the new location. These MUST be the same number, otherwise
// there's a window where we fade out without snap-on-return (or vice
// versa) and the face visibly slides through the wrong path.
const LOST_AFTER_MISSES = 4;

// After this many misses, widen matching to same-class + closest-center.
// Camera pans, fast motion, and brief full occlusions can blow IoU to zero
// between frames even though a valid same-class detection is still in
// view — class+center recovers gracefully.
const WIDEN_MATCH_AFTER_MISSES = 3;

// Hold-last-good: reject position updates whose dimensions jumped this much
// vs. the previous smoothed box. Prevents the face from snapping onto a
// transient-but-larger neighbor of the same class during a noisy frame.
// The track keeps its identity and carries on — next inference usually
// gives a clean match.
const SUSPECT_SIZE_RATIO = 1.75;

// Velocity extrapolation. Between YOLO inferences (3–15 fps on mobile) we
// glide the face at 60 fps using per-track velocity, so the face stays
// pasted on a moving object instead of stuttering at the inference rate.
//
// EXTRAP_MAX_MS is sized to bridge a full inference gap at ~5 fps with a
// little headroom — at 10 fps it's well clear, at 5 fps it just covers
// the gap so the face doesn't visibly freeze. VELOCITY_EMA is high
// because the position EMA already filters YOLO box noise upstream;
// being responsive here is what actually keeps the face in front of the
// bottle as it moves. We also keep gliding through the first couple of
// misses (EXTRAP_MISS_LIMIT) — losing a single inference frame is the
// common case and freezing on it shows as jank.
const EXTRAP_MAX_MS = 220;
const EXTRAP_MISS_LIMIT = 2;
const VELOCITY_EMA = 0.75;
const VELOCITY_DECAY_PER_MISS = 0.6;


// Tap confidence is lower than continuous — the user's intent is a strong
// prior that something tappable sits under their finger. Continuous runs
// at a generous confidence so the breathing-box gallery feels alive.
const CONTINUOUS_CONF = 0.15;
const TAP_CONF = 0.08;
const CONTINUOUS_MAX_DET = 25;

// Face size = FACE_BBOX_FRACTION × min(box.w, box.h). <FaceVoice /> ships
// at FACE_VOICE_WIDTH CSS px at scale=1; the multiplier is target_px / that.
const FACE_BBOX_FRACTION = 0.92;
const FACE_NATIVE_PX = FACE_VOICE_WIDTH;

// Hard clamp on face scale — tiny boxes don't birth a grain-of-sand face,
// giant boxes don't drown the screen.
const FACE_SCALE_MIN = 0.25;
const FACE_SCALE_MAX = 3.0;

// CSS blend mode applied to the face when it rides inside an object's
// silhouette. `hard-light` adopts the object's color cast without crushing
// the face detail the way `multiply` does. Easy to tune — set to `normal`
// to turn the tint off entirely and just keep the silhouette clip.
const FACE_BLEND_MODE = "hard-light";

// Minimum eigenvalue ratio before we rotate the face to the object's long
// axis. Below this the object is roughly round and its "orientation" is
// quantization noise — a cup at 1.05 doesn't want a tilted face. Bananas,
// pens, laptops sit well above 2.
const ORIENTATION_MIN_RATIO = 1.25;

// Low-pass on the per-track rotation angle. PCA output jitters by a few
// degrees between inferences on non-rigid silhouettes; this smooths it
// without adding perceptible lag at 3–30 hz inference.
const ROTATION_EMA_ALPHA = 0.25;

// Classes the app explicitly refuses to put a face on — it's about things,
// not people. Mirrors the ASSESS_SYSTEM policy upstream.
const EXCLUDED_CLASS_IDS = new Set<number>([PERSON_CLASS_ID]);

// Tap-hit fallback cache — reuse the continuous loop's detections when
// fresh, otherwise fire a one-shot lower-threshold inference under the tap.
const TAP_CACHE_MAX_AGE_MS = 400;

// Visible tap-frame fallback dimensions (shown instantly while the real
// detection box is computed).
const TAP_FRAME_FRACTION = 0.55;

// How many objects can wear a face at once. Tap past this and the LRU
// (least-recently-tapped) track gets evicted to make room. Three is the
// sweet spot for demos: busy enough to feel alive, small enough to parse.
const MAX_FACES = 3;

// Adaptive waveform bar count. Symmetric layout around center; bumping this
// widens the strip but does not retune the bar-to-band mapping (see readLevel).
// Kept small so the strip fits cleanly inside the 84px mic button.
const WAVE_BARS = 14;

type Phase = "starting" | "ready" | "locked" | "error";

type ViewportBox = {
  left: number;
  top: number;
  width: number;
  height: number;
};

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

// Map a source-space axis-aligned box to the video element's displayed
// pixel space. The <video> uses object-cover, so there's letterboxing on
// one axis we need to account for.
// Write a detection's binary mask bitmap (0/255 alpha) into a reusable
// canvas and return a PNG data URL for CSS mask-image. Called only on
// inference ticks (3–30 hz), never per RAF — `canvas.toDataURL` is cheap
// at proto resolution (~1-3 ms for typical 40×60 patches) but not free.
// Reuses the provided canvas to avoid per-tick allocation of the pixel
// buffer. Alpha-only — RGB doesn't matter for mask-image.
function renderMaskToDataUrl(
  canvas: HTMLCanvasElement,
  mask: { data: Uint8Array; w: number; h: number }
): string | null {
  if (mask.w < 1 || mask.h < 1) return null;
  if (canvas.width !== mask.w) canvas.width = mask.w;
  if (canvas.height !== mask.h) canvas.height = mask.h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  // Pack the binary mask into an RGBA ImageData. White RGB + per-pixel
  // alpha = 0 or 255. The browser will smoothly scale this with CSS
  // mask-size: 100% 100% so low-res proto masks (~30×40) still look clean
  // stretched across a 300 px object.
  const img = ctx.createImageData(mask.w, mask.h);
  const out = img.data;
  const src = mask.data;
  for (let i = 0, j = 0; i < src.length; i++, j += 4) {
    out[j] = 255;
    out[j + 1] = 255;
    out[j + 2] = 255;
    out[j + 3] = src[i];
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}

function sourceBoxToElement(
  box: { x1: number; y1: number; x2: number; y2: number },
  video: HTMLVideoElement
): ViewportBox | null {
  const rect = video.getBoundingClientRect();
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;
  const elAspect = rect.width / rect.height;
  const vidAspect = vw / vh;
  let dispW: number;
  let dispH: number;
  let offX: number;
  let offY: number;
  if (vidAspect > elAspect) {
    dispH = vh;
    dispW = vh * elAspect;
    offX = (vw - dispW) / 2;
    offY = 0;
  } else {
    dispW = vw;
    dispH = vw / elAspect;
    offX = 0;
    offY = (vh - dispH) / 2;
  }
  const pxPerSrcX = rect.width / dispW;
  const pxPerSrcY = rect.height / dispH;
  return {
    left: (box.x1 - offX) * pxPerSrcX,
    top: (box.y1 - offY) * pxPerSrcY,
    width: (box.x2 - box.x1) * pxPerSrcX,
    height: (box.y2 - box.y1) * pxPerSrcY,
  };
}

function sourceToElementPoint(
  point: { x: number; y: number },
  video: HTMLVideoElement
): { clientX: number; clientY: number } | null {
  const rect = video.getBoundingClientRect();
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;
  const elAspect = rect.width / rect.height;
  const vidAspect = vw / vh;
  let dispW: number;
  let dispH: number;
  let offX: number;
  let offY: number;
  if (vidAspect > elAspect) {
    dispH = vh;
    dispW = vh * elAspect;
    offX = (vw - dispW) / 2;
    offY = 0;
  } else {
    dispW = vw;
    dispH = vw / elAspect;
    offX = 0;
    offY = (vh - dispH) / 2;
  }
  return {
    clientX: rect.left + ((point.x - offX) / dispW) * rect.width,
    clientY: rect.top + ((point.y - offY) / dispH) * rect.height,
  };
}

function elementPointToSource(
  clientX: number,
  clientY: number,
  video: HTMLVideoElement
): { x: number; y: number; vw: number; vh: number } | null {
  const rect = video.getBoundingClientRect();
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;
  const elAspect = rect.width / rect.height;
  const vidAspect = vw / vh;
  let dispW: number;
  let dispH: number;
  let offX: number;
  let offY: number;
  if (vidAspect > elAspect) {
    dispH = vh;
    dispW = vh * elAspect;
    offX = (vw - dispW) / 2;
    offY = 0;
  } else {
    dispW = vw;
    dispH = vw / elAspect;
    offX = 0;
    offY = (vh - dispH) / 2;
  }
  const ex = clientX - rect.left;
  const ey = clientY - rect.top;
  return {
    x: offX + (ex / rect.width) * dispW,
    y: offY + (ey / rect.height) * dispH,
    vw,
    vh,
  };
}

// Pick the smallest-area detection that contains the tap. Smaller boxes
// are usually the correct semantic target when two are nested (cup inside
// the region of a dining table).
function pickTappedDetection(
  dets: readonly Detection[],
  tapX: number,
  tapY: number
): Detection | null {
  let best: Detection | null = null;
  let bestArea = Infinity;
  for (const d of dets) {
    if (EXCLUDED_CLASS_IDS.has(d.classId)) continue;
    if (tapX < d.x1 || tapX > d.x2 || tapY < d.y1 || tapY > d.y2) continue;
    const area = (d.x2 - d.x1) * (d.y2 - d.y1);
    if (area < bestArea) {
      bestArea = area;
      best = d;
    }
  }
  if (best) return best;
  // No box contains the tap — fall back to nearest-center.
  let nearest: Detection | null = null;
  let nearestD = Infinity;
  for (const d of dets) {
    if (EXCLUDED_CLASS_IDS.has(d.classId)) continue;
    const dx = d.cx - tapX;
    const dy = d.cy - tapY;
    const dist = Math.hypot(dx, dy);
    if (dist < nearestD) {
      nearestD = dist;
      nearest = d;
    }
  }
  return nearest;
}

// === Per-track state =====================================================
//
// Each locked object gets its own TrackRefs (mutable, hot-path state written
// every RAF) plus its own TrackUI (React-rendered lifecycle/animation
// values). The two are kept in sync by id.

type TrackRefs = {
  id: string;
  classId: number;
  className: string;
  anchor: Anchor;
  // Tracking math
  boxEma: BoxEMA;
  smoothedBox: Box;
  missedFrames: number;
  opacity: number;
  lastTapAt: number;
  // Velocity for inter-inference glide (source pixels / ms), plus timestamp
  // of the last observation that moved the smoothed box. Extrapolation at
  // render time uses `now - lastUpdatedAt`, capped by EXTRAP_MAX_MS.
  vx: number;
  vy: number;
  lastUpdatedAt: number;
  // Per-track speak generation — a second tap on the same track increments
  // it, so an in-flight generateLine/TTS from the previous tap drops silent.
  speakGen: number;
  // Audio — each track has its own analyser so mouths sync to their own voice.
  // The gain node sits between analyser and destination and is driven from the
  // per-track opacity each RAF, so the voice fades with the face.
  analyser: AnalyserNode | null;
  gain: GainNode | null;
  freqData: Uint8Array<ArrayBuffer> | null;
  timeData: Uint8Array<ArrayBuffer> | null;
  source: AudioBufferSourceNode | null;
  shape: MouthShape;
  // Per-track lip-sync state (envelope follower + adaptive peak +
  // shape hysteresis). Reset at the start of every new line so the peak
  // calibrates to the new utterance's loudness, not the previous one's.
  lipSync: LipSyncState;
  // Auto-dismiss timer id for this track's caption once audio finishes.
  // Held per-track so a retap can cancel the prior dismissal.
  captionClearTimer: number | null;
  // Streaming audio element for the conversation reply path. Separate from
  // `source` (AudioBufferSourceNode) which handles the lock-time line —
  // the streaming path uses MediaSource so playback starts as the first
  // bytes arrive instead of waiting for the whole mp3. Stopping a track
  // must clear BOTH.
  streamingAudio: HTMLAudioElement | null;
  streamingUrl: string | null;
  // Fish.audio reference_id chosen by GLM on the first `generateLine` for
  // this track. Once set, it's reused for every subsequent line and reply
  // so the object's voice stays consistent for the whole session on it.
  // null until the first generateLine returns (or catalog empty → stays null).
  voiceId: string | null;
  // Conversation memory for this object. Populated in order: first line
  // from generateLine → user utterance (whisper transcript) → assistant
  // reply from converseWithObject → ... GLM sees this thread so the object
  // remembers what was said and replies coherently.
  history: { role: "user" | "assistant"; content: string }[];
  // Rich visual description of the object — hydrated in the background by
  // describeObject right after lock and refreshed after every conversation
  // turn. Lets converseWithObject stay text-only on the hot path while
  // still getting funnier-than-classname context (chewed straw, dust, etc.)
  description: string | null;
  // Bumped each time we kick off a describeObject call. The handler
  // compares its captured gen against the latest before storing, so a
  // slow describe response can't overwrite a fresher one.
  descriptionGen: number;
  // Silhouette clipping — the face is clipped to this shape so it reads
  // as painted onto the object instead of pasted over it. Updated only
  // on YOLO inference ticks (3–30 hz), never per RAF.
  //   maskCanvas  — offscreen canvas holding the alpha bitmap. Sized to
  //                 the detection's mask patch in proto resolution.
  //   maskDataUrl — PNG data URL of the canvas; used as CSS mask-image.
  //                 Regenerated each inference tick. Small (few KB).
  //   maskSrcBox  — source-pixel rect the mask covers; projected to
  //                 element space each render tick.
  maskCanvas: HTMLCanvasElement | null;
  maskDataUrl: string | null;
  maskSrcBox: { x1: number; y1: number; x2: number; y2: number } | null;
  // smoothedBox center at the moment the mask was captured. Render applies
  // (renderBox.cx - maskAnchor.cx) so the silhouette glides with the face
  // during inter-inference extrapolation instead of snapping at inference rate.
  maskAnchor: { cx: number; cy: number } | null;
  // Face rotation in radians. Target comes from the mask's principal-axis
  // angle; smoothed by ROTATION_EMA_ALPHA, gated by ORIENTATION_MIN_RATIO
  // so round objects stay upright. Applied on top of translate+scale in
  // the render transform so the face tilts with bananas, pens, laptops.
  rotation: number;
};

// How long a caption stays on screen after its voice line finishes. Long
// enough to re-read the punchline; short enough that stacked bubbles don't
// pile up forever.
const CAPTION_LINGER_MS = 3500;

type TrackUI = {
  id: string;
  classId: number;
  className: string;
  // Element-space position + size, driven by the RAF loop.
  left: number;
  top: number;
  scale: number;
  opacity: number;
  // Audio-driven mouth shape, emitted by the lip-sync classifier.
  shape: MouthShape;
  // Lifecycle — set by tap/speak flow, not the RAF.
  caption: string | null;
  thinking: boolean;
  speaking: boolean;
  // Silhouette clip. `maskDataUrl` is null until the first inference tick
  // after lock produces a mask for this track (or if the seg head didn't
  // supply one). When null, the face falls back to unclipped render.
  maskDataUrl: string | null;
  maskLeft: number;
  maskTop: number;
  maskWidth: number;
  maskHeight: number;
  // Face rotation (radians) piped from TrackRefs. Applied via CSS rotate()
  // inside the face transform.
  rotation: number;
};

// Stream an mp3 response body into a MediaSource SourceBuffer and play it
// through the track's analyser so the mouth syncs. Runs at module scope
// (not a hook) because the logic is pure browser plumbing and keeps the
// hot-path useCallback readable. Resolves when the 'ended' event fires or
// the caller supersedes (callGen mismatch).
async function playViaMediaSource(args: {
  ctx: AudioContext;
  track: TrackRefs;
  trackId: string;
  callGen: number;
  respBody: ReadableStream<Uint8Array>;
  mediaSourceCtor: typeof MediaSource;
  ensureAnalyser: () => void;
  scheduleCaptionClear: () => void;
  setSpeaking: (on: boolean) => void;
  tStart: number;
}): Promise<void> {
  const {
    ctx,
    track,
    trackId,
    callGen,
    respBody,
    mediaSourceCtor,
    ensureAnalyser,
    scheduleCaptionClear,
    setSpeaking,
    tStart,
  } = args;

  const mediaSource = new mediaSourceCtor();
  const url = URL.createObjectURL(mediaSource);
  const audioEl = new Audio();
  audioEl.preload = "auto";
  audioEl.crossOrigin = "anonymous";
  audioEl.src = url;

  track.streamingAudio = audioEl;
  track.streamingUrl = url;

  await new Promise<void>((resolve, reject) => {
    const onOpen = () => {
      cleanup();
      resolve();
    };
    const onErr = () => {
      cleanup();
      reject(new Error("MediaSource open failed"));
    };
    const cleanup = () => {
      mediaSource.removeEventListener("sourceopen", onOpen);
      mediaSource.removeEventListener("error", onErr);
    };
    mediaSource.addEventListener("sourceopen", onOpen, { once: true });
    mediaSource.addEventListener("error", onErr, { once: true });
  });

  if (callGen !== track.speakGen) {
    try {
      await respBody.cancel();
    } catch {
      // ignore
    }
    return;
  }

  const sourceBuffer = mediaSource.addSourceBuffer("audio/mpeg");
  sourceBuffer.mode = "sequence";

  // Wire the audio element into the track's analyser so the mouth syncs
  // to THIS stream. One MediaElementAudioSourceNode per element per ctx.
  ensureAnalyser();
  const mediaElSource = ctx.createMediaElementSource(audioEl);
  mediaElSource.connect(track.analyser!);

  let started = false;
  let endedFired = false;
  audioEl.addEventListener("ended", () => {
    endedFired = true;
    if (track.streamingAudio === audioEl) {
      track.streamingAudio = null;
      if (track.streamingUrl) {
        try {
          URL.revokeObjectURL(track.streamingUrl);
        } catch {
          // ignore
        }
        track.streamingUrl = null;
      }
      setSpeaking(false);
      scheduleCaptionClear();
    }
  });

  const reader = respBody.getReader();

  const appendChunk = (chunk: Uint8Array) =>
    new Promise<void>((resolve, reject) => {
      const onUpdate = () => {
        sourceBuffer.removeEventListener("updateend", onUpdate);
        sourceBuffer.removeEventListener("error", onErr);
        resolve();
      };
      const onErr = () => {
        sourceBuffer.removeEventListener("updateend", onUpdate);
        sourceBuffer.removeEventListener("error", onErr);
        reject(new Error("SourceBuffer append failed"));
      };
      sourceBuffer.addEventListener("updateend", onUpdate);
      sourceBuffer.addEventListener("error", onErr);
      try {
        // chunk.buffer can be SharedArrayBuffer per TS lib types; SourceBuffer
        // wants a plain ArrayBuffer. Copy into a fresh ArrayBuffer to satisfy
        // the type and guarantee the append sees a stable buffer.
        const copy = new Uint8Array(chunk.byteLength);
        copy.set(chunk);
        sourceBuffer.appendBuffer(copy.buffer);
      } catch (e) {
        sourceBuffer.removeEventListener("updateend", onUpdate);
        sourceBuffer.removeEventListener("error", onErr);
        reject(e instanceof Error ? e : new Error(String(e)));
      }
    });

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (callGen !== track.speakGen) {
        // Retap while streaming — abandon and let stopTrackAudio clean up.
        try {
          await reader.cancel();
        } catch {
          // ignore
        }
        return;
      }
      if (done) break;
      if (!value || value.byteLength === 0) continue;
      await appendChunk(value);
      // Kick off playback the moment the first chunk is decodable.
      if (!started) {
        started = true;
        // eslint-disable-next-line no-console
        console.log(
          `[stream:${trackId}] first chunk → playing after ${Math.round(performance.now() - tStart)}ms (${value.byteLength}B)`
        );
        setSpeaking(true);
        try {
          await audioEl.play();
        } catch (err) {
          // Autoplay policies should be satisfied (user gesture triggered
          // recording), but swallow + log just in case.
          // eslint-disable-next-line no-console
          console.log(
            `[stream:${trackId}] audioEl.play() rejected: ${err instanceof Error ? err.message : String(err)}`
          );
        }
      }
    }
    // Signal end-of-stream to MediaSource so the audio element can
    // schedule its 'ended' event properly.
    if (mediaSource.readyState === "open") {
      try {
        mediaSource.endOfStream();
      } catch {
        // ignore
      }
    }
    // eslint-disable-next-line no-console
    console.log(
      `[stream:${trackId}] done streaming after ${Math.round(performance.now() - tStart)}ms  ended=${endedFired}`
    );
  } catch (err) {
    // eslint-disable-next-line no-console
    console.log(
      `[stream:${trackId}] ✖ ${err instanceof Error ? err.message : String(err)}`
    );
    throw err;
  }
}

export function Tracker() {
  const videoRef = useRef<HTMLVideoElement>(null);

  const [phase, setPhase] = useState<Phase>("starting");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [rejection, setRejection] = useState<string | null>(null);
  const [yoloReady, setYoloReady] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [yoloStatus, setYoloStatusState] = useState<YoloStatus>(() => getYoloStatus());
  const [lastInferMs, setLastInferMs] = useState<number | null>(null);
  // Which TTS backend rendered the most recent voice line. Shown in the
  // diag panel so you can confirm Fish (vs. OpenAI fallback / caption-only)
  // is actually in play without having to listen carefully.
  const [lastTtsBackend, setLastTtsBackend] = useState<
    "fish" | "openai" | "none" | null
  >(null);
  const [diagOpen, setDiagOpen] = useState(false);
  const [diagError, setDiagError] = useState<string | null>(null);
  const [retryToken, setRetryToken] = useState(0);
  const rejectionTimerRef = useRef<number | null>(null);

  // --- Push-to-talk state (UI only, decoupled from per-track voice) -----
  const [isRecording, setIsRecording] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const [talkFlash, setTalkFlash] = useState(false);
  // "you said: …" toast that briefly echoes the Whisper transcript so the
  // user has a clear signal their voice message landed.
  const [heardText, setHeardText] = useState<string | null>(null);
  const heardClearTimerRef = useRef<number | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<BlobPart[]>([]);
  const micStreamRef = useRef<MediaStream | null>(null);
  const recordedBlobRef = useRef<Blob | null>(null);
  // Browser-side Web Speech API. Runs in parallel with MediaRecorder so the
  // transcript is usually already final by the time the user releases the
  // talk button — saves the entire server STT roundtrip (~700–1300ms).
  // Falls back to server STT silently when SpeechRecognition isn't
  // available (Firefox, some embedded webviews) or yields nothing.
  const speechRecognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const speechTranscriptPartsRef = useRef<string[]>([]);
  const speechFinishedRef = useRef<Promise<void> | null>(null);
  const resolveSpeechFinishedRef = useRef<(() => void) | null>(null);
  const talkAnalyserRef = useRef<AnalyserNode | null>(null);
  const talkFreqDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const talkSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const talkLevelRafRef = useRef<number | null>(null);
  const talkFlashTimerRef = useRef<number | null>(null);

  // Adaptive waveform. `bandLevelsRef` holds a smoothed 0..1 per-bar envelope
  // the readLevel RAF writes each frame; `barRefs` lets us push heights
  // straight to the DOM at 60fps without re-rendering the component tree.
  // Layout is symmetric — outer bars read higher freq bands than inner bars,
  // so speech lights up the center first, which reads as natural.
  const bandLevelsRef = useRef<Float32Array>(new Float32Array(WAVE_BARS));
  const barRefs = useRef<(HTMLSpanElement | null)[]>([]);

  // Detections from the continuous loop. Feeds the breathing boxes in the
  // ready phase AND tap resolution without an extra inference.
  const [detections, setDetections] = useState<Detection[]>([]);
  const detectionsTsRef = useRef(0);
  const detectionsRef = useRef<Detection[]>([]);

  // Visible tap-frame while we wait on the first inference after a tap.
  const [tapFrame, setTapFrame] = useState<
    (ViewportBox & { gen: number }) | null
  >(null);

  // The locked faces. Mutable mirror lives in tracksRef (written every RAF);
  // React state is the rendering snapshot, updated once per RAF.
  const tracksRef = useRef<TrackRefs[]>([]);
  const [tracksUI, setTracksUI] = useState<TrackUI[]>([]);
  const nextTrackIdRef = useRef(1);

  const yoloReadyRef = useRef(false);
  const inferenceInFlightRef = useRef(false);
  const lastInferenceAtRef = useRef(0);
  const inferenceCountRef = useRef(0);

  const yoloCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const cropCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const fpsLastSampleRef = useRef(0);
  const fpsFrameCountRef = useRef(0);

  // Single shared AudioContext for all tracks. Each track hangs its own
  // AnalyserNode off it; sources play concurrently (chaos is the feature).
  const audioCtxRef = useRef<AudioContext | null>(null);

  // Global tap generation — any tap that triggers async work captures this;
  // if it changes before the async resolves, the result is dropped.
  const generationRef = useRef(0);

  // --- Camera setup ------------------------------------------------------
  useEffect(() => {
    let stream: MediaStream | null = null;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        const v = videoRef.current;
        if (!v) return;
        v.srcObject = stream;
        await v.play();
        // eslint-disable-next-line no-console
        console.log(`[tracker] camera ready: ${v.videoWidth}x${v.videoHeight}`);
        setCameraReady(true);
      } catch (e) {
        // eslint-disable-next-line no-console
        console.log("[tracker] camera failed:", e);
        setPhase("error");
        setErrorMsg(e instanceof Error ? e.message : "camera unavailable");
      }
    })();
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (rejectionTimerRef.current != null) {
        clearTimeout(rejectionTimerRef.current);
        rejectionTimerRef.current = null;
      }
      // Stop every track's audio source + disconnect analysers.
      for (const t of tracksRef.current) {
        if (t.source) {
          try {
            t.source.stop();
          } catch {
            // already stopped
          }
          t.source = null;
        }
        if (t.analyser) {
          try {
            t.analyser.disconnect();
          } catch {
            // already disconnected
          }
          t.analyser = null;
        }
        if (t.gain) {
          try {
            t.gain.disconnect();
          } catch {
            // already disconnected
          }
          t.gain = null;
        }
        if (t.captionClearTimer != null) {
          clearTimeout(t.captionClearTimer);
          t.captionClearTimer = null;
        }
      }
      tracksRef.current = [];
      // Push-to-talk cleanup.
      if (recorderRef.current && recorderRef.current.state === "recording") {
        try {
          recorderRef.current.stop();
        } catch {
          // ignore
        }
      }
      recorderRef.current = null;
      if (talkLevelRafRef.current != null) {
        cancelAnimationFrame(talkLevelRafRef.current);
        talkLevelRafRef.current = null;
      }
      if (talkFlashTimerRef.current != null) {
        clearTimeout(talkFlashTimerRef.current);
        talkFlashTimerRef.current = null;
      }
      if (heardClearTimerRef.current != null) {
        clearTimeout(heardClearTimerRef.current);
        heardClearTimerRef.current = null;
      }
      if (talkSourceRef.current) {
        try {
          talkSourceRef.current.disconnect();
        } catch {
          // already disconnected
        }
        talkSourceRef.current = null;
      }
      micStreamRef.current?.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
      audioCtxRef.current?.close().catch(() => {});
    };
  }, []);

  // --- Phase transitions ------------------------------------------------
  useEffect(() => {
    if (phase === "error") return;
    if (cameraReady && yoloReady) {
      setPhase(tracksUI.length > 0 ? "locked" : "ready");
    }
  }, [cameraReady, yoloReady, tracksUI.length, phase]);

  // --- YOLO status subscription + warm-up --------------------------------
  useEffect(() => {
    const unsub = subscribeYoloStatus((s) => {
      setYoloStatusState(s);
      if (s.stage === "error" && s.error) {
        // eslint-disable-next-line no-console
        console.log("[tracker] yolo status error:", s.error);
      }
    });
    return unsub;
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // eslint-disable-next-line no-console
        console.log("[tracker] initYolo start");
        await initYolo();
        if (cancelled) return;
        yoloReadyRef.current = true;
        setYoloReady(true);
        // eslint-disable-next-line no-console
        console.log("[tracker] yolo ready");
      } catch (e) {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : "detector failed";
        setErrorMsg(`detector: ${msg}`);
        // eslint-disable-next-line no-console
        console.log("[tracker] initYolo failed:", e);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [retryToken]);

  const handleRetryYolo = useCallback(() => {
    setErrorMsg(null);
    setYoloReady(false);
    yoloReadyRef.current = false;
    resetYolo();
    setRetryToken((t) => t + 1);
  }, []);

  // Nuke every track. Used by the × button and Esc key — the "demo reset"
  // the app was missing. Stops all audio, clears all timers, goes back to
  // the ready phase (via the phase useEffect watching tracksUI.length).
  const clearAllTracks = useCallback(() => {
    for (const t of tracksRef.current) {
      if (t.source) {
        try {
          t.source.stop();
        } catch {
          // already stopped
        }
        t.source = null;
      }
      if (t.analyser) {
        try {
          t.analyser.disconnect();
        } catch {
          // already disconnected
        }
        t.analyser = null;
      }
      if (t.gain) {
        try {
          t.gain.disconnect();
        } catch {
          // already disconnected
        }
        t.gain = null;
      }
      if (t.captionClearTimer != null) {
        clearTimeout(t.captionClearTimer);
        t.captionClearTimer = null;
      }
    }
    tracksRef.current = [];
    setTracksUI([]);
    // Bump generation so any in-flight tap's generateLine/TTS drops silent.
    generationRef.current++;
  }, []);

  // Esc key wipes the scene — handy for live demos where you want to reset
  // between "oh look" moments without reloading the page.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && tracksRef.current.length > 0) {
        clearAllTracks();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clearAllTracks]);

  // --- Audio plumbing ----------------------------------------------------
  const ensureAudioCtx = useCallback(() => {
    let ctx = audioCtxRef.current;
    if (!ctx) {
      const Ctor =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext })
          .webkitAudioContext;
      if (!Ctor) return null;
      ctx = new Ctor();
      audioCtxRef.current = ctx;
    }
    if (ctx.state === "suspended") {
      ctx.resume().catch(() => {});
    }
    return ctx;
  }, []);

  // --- Get-or-make YOLO scratch canvas -----------------------------------
  const ensureYoloCanvas = useCallback((): HTMLCanvasElement => {
    let c = yoloCanvasRef.current;
    if (!c) {
      c = document.createElement("canvas");
      yoloCanvasRef.current = c;
    }
    return c;
  }, []);

  // --- Rejection toast ---------------------------------------------------
  const showRejection = useCallback((msg: string, ms = 2400) => {
    if (rejectionTimerRef.current != null) {
      clearTimeout(rejectionTimerRef.current);
    }
    setRejection(msg);
    rejectionTimerRef.current = window.setTimeout(() => {
      setRejection(null);
      setTapFrame(null);
      rejectionTimerRef.current = null;
    }, ms);
  }, []);

  // --- Background rich description -------------------------------------
  //
  // YOLO gives us "cup"; this fetches a juicy sentence or two (chewed straw,
  // dust film, dent, etc.) that converseWithObject uses to write funnier
  // replies without paying vision latency on the hot path. Fire-and-forget:
  // a stale response can't overwrite a fresher one thanks to descriptionGen.
  const refreshTrackDescription = useCallback(
    (trackId: string, cropDataUrl: string) => {
      const track = tracksRef.current.find((t) => t.id === trackId);
      if (!track || !cropDataUrl) return;
      const gen = ++track.descriptionGen;
      const t0 = performance.now();
      // eslint-disable-next-line no-console
      console.log(
        `[describe:${trackId}] → describeObject  class="${track.className}"  crop=${Math.round(cropDataUrl.length / 1024)}KB  gen=${gen}`
      );
      describeObject(cropDataUrl, track.className)
        .then(({ description }) => {
          // Track may have been evicted while we waited; bail if so.
          const current = tracksRef.current.find((t) => t.id === trackId);
          if (!current) return;
          // A fresher describe kicked off while this one was in flight.
          if (gen !== current.descriptionGen) {
            // eslint-disable-next-line no-console
            console.log(
              `[describe:${trackId}] ← superseded (gen ${gen} vs ${current.descriptionGen})`
            );
            return;
          }
          current.description = description || null;
          // eslint-disable-next-line no-console
          console.log(
            `[describe:${trackId}] ← ${Math.round(performance.now() - t0)}ms  "${(description || "").slice(0, 100)}"`
          );
        })
        .catch((err) => {
          // Non-fatal — converseWithObject just falls back to the bare class.
          // eslint-disable-next-line no-console
          console.log(
            `[describe:${trackId}] ✖ ${err instanceof Error ? err.message : String(err)}`
          );
        });
    },
    []
  );

  // Stop whatever a track is currently saying — both the
  // AudioBufferSource (lock-time line) and the streaming HTMLAudioElement
  // (conversation reply). Called from retap, eviction, etc.
  const stopTrackAudio = useCallback((track: TrackRefs) => {
    if (track.source) {
      try {
        track.source.stop();
      } catch {
        // already stopped
      }
      track.source = null;
    }
    if (track.streamingAudio) {
      try {
        track.streamingAudio.pause();
      } catch {
        // ignore
      }
      track.streamingAudio.src = "";
      track.streamingAudio = null;
    }
    if (track.streamingUrl) {
      try {
        URL.revokeObjectURL(track.streamingUrl);
      } catch {
        // ignore
      }
      track.streamingUrl = null;
    }
  }, []);

  // --- Streaming TTS playback -------------------------------------------
  //
  // The conversation hot path. We used to synthesize the whole mp3
  // server-side, base64-encode it, pass it through a server action, fetch
  // the data URL, and only then decode + play — burning ~600–1500ms of
  // dead air. This helper fetches `/api/tts/stream` instead, feeds the
  // bytes into a MediaSource SourceBuffer, and starts playback the moment
  // the first chunk lands. Audio element → MediaElementSource → track.
  // analyser → gain → destination keeps lip-sync intact.
  //
  // MediaSource isn't everywhere (notably older iOS), so we fall back to
  // buffering the whole blob and going through the existing
  // AudioBufferSource path — same code that speakOnTrack uses.
  const playStreamingReply = useCallback(
    async (
      trackId: string,
      callGen: number,
      text: string,
      voiceId: string | null
    ): Promise<void> => {
      const ctx = ensureAudioCtx();
      const track = tracksRef.current.find((t) => t.id === trackId);
      if (!ctx || !track) return;

      // Reset lip-sync state for this new reply. Peak calibrates to the
      // incoming stream's actual loudness; without this, a previous loud
      // line leaves peak high and the quieter reply shows as closed mouth.
      track.lipSync = createLipSyncState();

      const ensureAnalyser = () => {
        if (track.analyser) return;
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 1024;
        analyser.smoothingTimeConstant = 0.4;
        const gain = ctx.createGain();
        gain.gain.value = track.opacity;
        analyser.connect(gain);
        gain.connect(ctx.destination);
        track.analyser = analyser;
        track.gain = gain;
        track.freqData = new Uint8Array(
          new ArrayBuffer(analyser.frequencyBinCount)
        );
        track.timeData = new Uint8Array(new ArrayBuffer(analyser.fftSize));
      };

      const scheduleCaptionClear = () => {
        if (track.captionClearTimer != null) {
          clearTimeout(track.captionClearTimer);
        }
        track.captionClearTimer = window.setTimeout(() => {
          track.captionClearTimer = null;
          setTracksUI((prev) =>
            prev.map((t) =>
              t.id === trackId ? { ...t, caption: null } : t
            )
          );
        }, CAPTION_LINGER_MS);
      };

      const t0 = performance.now();
      // eslint-disable-next-line no-console
      console.log(
        `[stream:${trackId}] → /api/tts/stream text=${text.length}ch voice=${voiceId ?? "default"}`
      );

      const resp = await fetch("/api/tts/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voiceId: voiceId ?? "" }),
      });
      if (callGen !== track.speakGen) {
        // Retap while we were awaiting headers — drop.
        try {
          await resp.body?.cancel();
        } catch {
          // ignore
        }
        return;
      }
      if (!resp.ok || !resp.body) {
        const err = await resp.text().catch(() => "");
        throw new Error(`tts stream ${resp.status}: ${err.slice(0, 120)}`);
      }
      const backend = resp.headers.get("X-Tts-Backend") ?? "stream";
      setLastTtsBackend(
        backend === "fish" ? "fish" : backend === "openai" ? "openai" : "none"
      );
      // eslint-disable-next-line no-console
      console.log(
        `[stream:${trackId}] headers in ${Math.round(performance.now() - t0)}ms backend=${backend}`
      );

      const mediaSourceCtor: typeof MediaSource | null =
        typeof window !== "undefined" &&
        typeof window.MediaSource !== "undefined" &&
        window.MediaSource.isTypeSupported?.("audio/mpeg")
          ? window.MediaSource
          : null;

      if (mediaSourceCtor) {
        await playViaMediaSource({
          ctx,
          track,
          trackId,
          callGen,
          respBody: resp.body,
          mediaSourceCtor,
          ensureAnalyser,
          scheduleCaptionClear,
          setSpeaking: (on: boolean) =>
            setTracksUI((prev) =>
              prev.map((t) =>
                t.id === trackId ? { ...t, speaking: on } : t
              )
            ),
          tStart: t0,
        });
      } else {
        // Fallback: buffer the whole mp3, decode, play via AudioBufferSource.
        // Loses the streaming latency win but keeps the app functional on
        // browsers without MediaSource (older iOS, etc.).
        // eslint-disable-next-line no-console
        console.log(
          `[stream:${trackId}] MediaSource unavailable — buffering full mp3 as fallback`
        );
        const buf = await resp.arrayBuffer();
        if (callGen !== track.speakGen) return;
        const audioBuf = await ctx.decodeAudioData(buf);
        if (callGen !== track.speakGen) return;
        ensureAnalyser();
        const source = ctx.createBufferSource();
        source.buffer = audioBuf;
        source.connect(track.analyser!);
        source.onended = () => {
          if (track.source === source) {
            track.source = null;
            setTracksUI((prev) =>
              prev.map((t) =>
                t.id === trackId ? { ...t, speaking: false } : t
              )
            );
            scheduleCaptionClear();
          }
        };
        track.source = source;
        setTracksUI((prev) =>
          prev.map((t) => (t.id === trackId ? { ...t, speaking: true } : t))
        );
        source.start();
      }
    },
    [ensureAudioCtx]
  );

  // --- Per-track speak ---------------------------------------------------
  //
  // Each track owns its own audio source + analyser, so three objects can
  // talk over each other at the same time and their mouths each sync to
  // their own line. The trackId closes over which track owns the result.
  const speakOnTrack = useCallback(
    async (trackId: string, cropDataUrl: string) => {
      const ctx = ensureAudioCtx();
      const track = tracksRef.current.find((t) => t.id === trackId);
      if (!ctx || !track) return;

      // Bump this track's speak gen so an in-flight call from a previous
      // tap on the SAME track drops its result when it resolves late.
      const callGen = ++track.speakGen;

      // Retap cancels any pending caption auto-dismiss — the new line wants
      // its full linger window.
      if (track.captionClearTimer != null) {
        clearTimeout(track.captionClearTimer);
        track.captionClearTimer = null;
      }

      // Stop whatever this track was already saying (buffer source OR
      // streaming element); other tracks keep going — concurrent voices
      // are a feature.
      stopTrackAudio(track);
      // UI: thinking=true, speaking=false, caption cleared.
      setTracksUI((prev) =>
        prev.map((t) =>
          t.id === trackId
            ? { ...t, thinking: true, speaking: false, caption: null }
            : t
        )
      );

      try {
        // eslint-disable-next-line no-console
        console.log(
          `[speak:${trackId}] → generateLine  class="${track.className}"  crop=${Math.round(cropDataUrl.length / 1024)}KB  voice=${track.voiceId ?? "(picking)"}  persona=${track.description ? "cached" : "(new)"}`
        );
        const t0 = performance.now();
        const {
          line,
          voiceId: chosenVoiceId,
          description: chosenDescription,
          audioDataUrl,
          backend,
        } = await generateLine(cropDataUrl, track.voiceId, track.description);
        if (callGen !== track.speakGen) {
          // eslint-disable-next-line no-console
          console.log(`[speak:${trackId}] ← superseded (speakGen mismatch)`);
          return;
        }
        setLastTtsBackend(backend);
        // Pin the voice on first return — every subsequent speak/talk on
        // this track passes this id back so the same Fish voice speaks the
        // whole session. Once set, we never overwrite it.
        if (!track.voiceId && chosenVoiceId) {
          track.voiceId = chosenVoiceId;
        }
        // Pin the persona card from the first tap. Don't clobber a richer
        // description with a later-tap re-describe; the background describe
        // pass handles ongoing refresh (and uses descriptionGen to race-guard).
        if (!track.description && chosenDescription) {
          track.description = chosenDescription;
        }
        // Record what the object said into its own memory so a later
        // conversation turn can call back to it.
        track.history = [
          ...track.history,
          { role: "assistant" as const, content: line },
        ].slice(-16);
        // eslint-disable-next-line no-console
        console.log(
          `[speak:${trackId}] ← ${Math.round(performance.now() - t0)}ms  backend=${backend}  voice=${track.voiceId ?? "default"}  line="${line}"`
        );

        setTracksUI((prev) =>
          prev.map((t) =>
            t.id === trackId ? { ...t, caption: line, thinking: false } : t
          )
        );

        // Caption-only mode when TTS key is missing.
        if (!audioDataUrl) return;

        const resp = await fetch(audioDataUrl);
        const buf = await resp.arrayBuffer();
        const audioBuf = await ctx.decodeAudioData(buf);
        if (callGen !== track.speakGen) return;

        // Lazy-init this track's analyser + gain on first speak.
        // Chain: source → analyser → gain → destination. Gain is driven by
        // the RAF loop from t.opacity so audio fades with the face.
        if (!track.analyser) {
          const analyser = ctx.createAnalyser();
          analyser.fftSize = 1024;
          analyser.smoothingTimeConstant = 0.4;
          const gain = ctx.createGain();
          gain.gain.value = track.opacity;
          analyser.connect(gain);
          gain.connect(ctx.destination);
          track.analyser = analyser;
          track.gain = gain;
          track.freqData = new Uint8Array(
            new ArrayBuffer(analyser.frequencyBinCount)
          );
          track.timeData = new Uint8Array(new ArrayBuffer(analyser.fftSize));
        }

        // Reset lip-sync envelope + adaptive peak for this new line so
        // openness normalizes against THIS utterance's level, not the
        // previous one's (Fish.audio and OpenAI fallback can differ 2–3×).
        track.lipSync = createLipSyncState();

        const source = ctx.createBufferSource();
        source.buffer = audioBuf;
        source.connect(track.analyser);
        source.onended = () => {
          // Only clear if still the current source — guards against a
          // newer tap's source already having replaced this one.
          if (track.source === source) {
            track.source = null;
            setTracksUI((prev) =>
              prev.map((t) =>
                t.id === trackId ? { ...t, speaking: false } : t
              )
            );
            // Schedule the caption to fade off-screen so stacked bubbles
            // don't linger indefinitely. A retap will cancel this.
            if (track.captionClearTimer != null) {
              clearTimeout(track.captionClearTimer);
            }
            track.captionClearTimer = window.setTimeout(() => {
              track.captionClearTimer = null;
              setTracksUI((prev) =>
                prev.map((t) =>
                  t.id === trackId ? { ...t, caption: null } : t
                )
              );
            }, CAPTION_LINGER_MS);
          }
        };
        track.source = source;
        setTracksUI((prev) =>
          prev.map((t) => (t.id === trackId ? { ...t, speaking: true } : t))
        );
        source.start();
      } catch (e) {
        if (callGen !== track.speakGen) return;
        const msg = e instanceof Error ? e.message : "line failed";
        setErrorMsg(msg);
        setDiagError(msg);
        setTracksUI((prev) =>
          prev.map((t) =>
            t.id === trackId
              ? { ...t, thinking: false, speaking: false }
              : t
          )
        );
        showRejection(
          /zhipu|glm|api key|api_key|401|403/i.test(msg)
            ? "voice model unconfigured — check .env.local"
            : `couldn't speak: ${msg.slice(0, 80)}`,
          3200
        );
        // eslint-disable-next-line no-console
        console.log("[tracker] speak failed:", e);
      } finally {
        if (callGen === track.speakGen) {
          setTracksUI((prev) =>
            prev.map((t) => (t.id === trackId ? { ...t, thinking: false } : t))
          );
        }
      }
    },
    [ensureAudioCtx, showRejection, stopTrackAudio]
  );

  // --- Tracking + lip-sync RAF ------------------------------------------
  //
  // Single loop does three things per frame:
  //   1) optionally launches a YOLO inference (rate-limited)
  //   2) advances each track (match, EMA, opacity)
  //   3) samples each track's analyser → mouth shape
  //   4) writes position/opacity into tracksUI state once per frame
  useEffect(() => {
    const tick = (now: number) => {
      rafRef.current = requestAnimationFrame(tick);
      const v = videoRef.current;
      if (!v || !v.videoWidth) return;

      // (1) Inference rate-limit gate.
      const minInterval = 1000 / MAX_INFERENCE_FPS;
      const canLaunch =
        yoloReadyRef.current &&
        !inferenceInFlightRef.current &&
        now - lastInferenceAtRef.current >= minInterval;

      if (canLaunch) {
        inferenceInFlightRef.current = true;
        lastInferenceAtRef.current = now;
        const inferStart = performance.now();
        detect(v, ensureYoloCanvas(), {
          confThreshold: CONTINUOUS_CONF,
          iouThreshold: 0.45,
          maxDetections: CONTINUOUS_MAX_DET,
          classFilter: (id) => !EXCLUDED_CLASS_IDS.has(id),
        })
          .then((dets) => {
            inferenceInFlightRef.current = false;
            const inferMs = Math.round(performance.now() - inferStart);
            setLastInferMs(inferMs);
            detectionsRef.current = dets;
            detectionsTsRef.current = performance.now();
            setDetections(dets);

            inferenceCountRef.current++;
            if (
              inferenceCountRef.current <= 3 ||
              inferenceCountRef.current % 30 === 0
            ) {
              // eslint-disable-next-line no-console
              console.log(
                `[tracker] inference #${inferenceCountRef.current}: ${dets.length} dets in ${inferMs}ms`,
                dets
                  .slice(0, 3)
                  .map((d) => `${d.className}@${(d.score * 100).toFixed(0)}%`)
                  .join(", ")
              );
            }

            advanceAllTracks(dets);

            fpsFrameCountRef.current++;
            if (now - fpsLastSampleRef.current > 500) {
              setFps(
                Math.round(
                  (fpsFrameCountRef.current * 1000) /
                    (now - fpsLastSampleRef.current)
                )
              );
              fpsFrameCountRef.current = 0;
              fpsLastSampleRef.current = now;
            }
            setDiagError((prev) => (prev ? null : prev));
          })
          .catch((err: unknown) => {
            inferenceInFlightRef.current = false;
            const msg = err instanceof Error ? err.message : "inference failed";
            // eslint-disable-next-line no-console
            console.log("[tracker] inference error:", err);
            setDiagError(msg);
          });
      }

      // (2) Per-track opacity. Fade in while present, fade out once the
      // object is "lost" (LOST_AFTER_MISSES). Same threshold the matcher
      // uses to decide a snap-on-return, so the face fades out and snaps
      // back in cleanly at the new location instead of gliding through
      // wrong space.
      for (const t of tracksRef.current) {
        const target = t.missedFrames >= LOST_AFTER_MISSES ? 0 : 1;
        t.opacity = lerp(t.opacity, target, OPACITY_EMA_ALPHA);
        // Voice rides with the face — disembodied audio when the face has
        // faded out feels haunted. Direct .value assignment is fine at 60Hz
        // for the small per-frame steps the lerp produces.
        if (t.gain) t.gain.gain.value = t.opacity;
      }

      // (3) Per-track mouth-shape classification. Gate on either audio
      // path — `t.source` covers the lock-time AudioBufferSource line and
      // `t.streamingAudio` covers the conversation reply's MediaSource
      // stream. Without the streaming gate, replies play audibly but the
      // mouth stays shut. `classifyShapeSmooth` keeps an envelope +
      // adaptive peak in `t.lipSync` so openness is normalized to this
      // utterance's actual level (quiet Fish voices still open the mouth)
      // and frame-to-frame flicker is filtered out.
      for (const t of tracksRef.current) {
        let shape: MouthShape = "X";
        const audible = t.source != null || t.streamingAudio != null;
        if (audible && t.analyser && t.freqData && t.timeData) {
          t.analyser.getByteTimeDomainData(t.timeData);
          t.analyser.getByteFrequencyData(t.freqData);
          shape = classifyShapeSmooth(t.lipSync, t.timeData, t.freqData);
        } else {
          // Not playing — decay the envelope so the next line starts clean.
          t.lipSync.envelope = 0;
          t.lipSync.prevShape = "X";
          t.lipSync.heldFrames = 0;
        }
        t.shape = shape;
      }

      // (4) Compute element-space transform for each track, write to UI.
      const rect = v.getBoundingClientRect();
      const srcToElAvg =
        (rect.width / v.videoWidth + rect.height / v.videoHeight) / 2;

      setTracksUI((prev) => {
        if (prev.length !== tracksRef.current.length) return prev;
        const renderNow = performance.now();
        let changed = false;
        const next = prev.map((ui) => {
          const t = tracksRef.current.find((x) => x.id === ui.id);
          if (!t) return ui;
          // Velocity-extrapolate between YOLO inferences so 60fps render
          // actually looks 60fps, not a 5–10fps stepper. Cap the window so
          // a lost target can't drift off-screen; the miss-path decay on
          // t.vx/t.vy is the safety belt here.
          const sinceUpdate = Math.min(
            EXTRAP_MAX_MS,
            renderNow - t.lastUpdatedAt
          );
          // Glide through the first few misses too — losing one inference
          // frame is common and freezing on it reads as jank. Velocity has
          // already decayed by VELOCITY_DECAY_PER_MISS each miss, so the
          // glide naturally tapers as confidence drops.
          const renderBox: Box =
            t.missedFrames <= EXTRAP_MISS_LIMIT
              ? makeBox(
                  t.smoothedBox.cx + t.vx * sinceUpdate,
                  t.smoothedBox.cy + t.vy * sinceUpdate,
                  t.smoothedBox.w,
                  t.smoothedBox.h
                )
              : t.smoothedBox;
          const facePoint = applyAnchor(t.anchor, renderBox);
          const el = sourceToElementPoint(facePoint, v);
          if (!el) return ui;
          const left = el.clientX - rect.left;
          const top = el.clientY - rect.top;
          const minSide = Math.min(renderBox.w, renderBox.h);
          const targetPx = minSide * FACE_BBOX_FRACTION * srcToElAvg;
          const scale = Math.max(
            FACE_SCALE_MIN,
            Math.min(FACE_SCALE_MAX, targetPx / FACE_NATIVE_PX)
          );
          const opacity = t.opacity;
          const shape = t.shape;
          const rotation = t.rotation;

          // Project the silhouette clip into element space, offset by how
          // far the tracker thinks the object has moved since the mask was
          // captured. This keeps the mask glued to the object during the
          // inter-inference glide instead of lagging at inference rate.
          let maskDataUrl: string | null = null;
          let maskLeft = 0;
          let maskTop = 0;
          let maskWidth = 0;
          let maskHeight = 0;
          if (t.maskDataUrl && t.maskSrcBox && t.maskAnchor) {
            const dx = renderBox.cx - t.maskAnchor.cx;
            const dy = renderBox.cy - t.maskAnchor.cy;
            const projected = sourceBoxToElement(
              {
                x1: t.maskSrcBox.x1 + dx,
                y1: t.maskSrcBox.y1 + dy,
                x2: t.maskSrcBox.x2 + dx,
                y2: t.maskSrcBox.y2 + dy,
              },
              v
            );
            if (projected) {
              maskDataUrl = t.maskDataUrl;
              maskLeft = projected.left;
              maskTop = projected.top;
              maskWidth = projected.width;
              maskHeight = projected.height;
            }
          }

          if (
            ui.left === left &&
            ui.top === top &&
            ui.scale === scale &&
            ui.opacity === opacity &&
            ui.shape === shape &&
            ui.maskDataUrl === maskDataUrl &&
            ui.maskLeft === maskLeft &&
            ui.maskTop === maskTop &&
            ui.maskWidth === maskWidth &&
            ui.maskHeight === maskHeight &&
            ui.rotation === rotation
          ) {
            return ui;
          }
          changed = true;
          return {
            ...ui,
            left,
            top,
            scale,
            opacity,
            shape,
            maskDataUrl,
            maskLeft,
            maskTop,
            maskWidth,
            maskHeight,
            rotation,
          };
        });
        return changed ? next : prev;
      });
    };

    // Match each track to its best same-class detection. Greedy by track
    // order — earlier tracks claim detections first. Detections claimed by
    // one track can't be reused by a later track (prevents two same-class
    // tracks snapping onto the same box during a crowded frame).
    const advanceAllTracks = (dets: readonly Detection[]) => {
      if (tracksRef.current.length === 0) return;
      const claimed = new Set<number>();
      for (const t of tracksRef.current) {
        const candidates: Detection[] = [];
        for (let i = 0; i < dets.length; i++) {
          if (!claimed.has(i)) candidates.push(dets[i]);
        }

        let match: Detection | null = matchTarget(
          candidates,
          { ...t.smoothedBox, classId: t.classId },
          IDENTITY_IOU_MIN
        );

        // Widen to same-class closest-center after a few misses — the
        // object may have moved across the frame between inferences.
        if (!match && t.missedFrames >= WIDEN_MATCH_AFTER_MISSES) {
          let nearest: Detection | null = null;
          let nearestD = Infinity;
          for (const d of candidates) {
            if (d.classId !== t.classId) continue;
            const dx = d.cx - t.smoothedBox.cx;
            const dy = d.cy - t.smoothedBox.cy;
            const dist = Math.hypot(dx, dy);
            if (dist < nearestD) {
              nearestD = dist;
              nearest = d;
            }
          }
          match = nearest;
        }

        if (match) {
          // Record the claim so other tracks can't take the same box.
          const idx = dets.indexOf(match);
          if (idx >= 0) claimed.add(idx);

          const wasLost = t.missedFrames >= LOST_AFTER_MISSES;
          const prevBox = t.smoothedBox;
          const now = performance.now();

          // Hold-last-good: if the match's dimensions jumped dramatically
          // vs. the previous smoothed box, this is almost certainly a
          // neighbor, a sibling, or a misfire rather than the real object
          // continuing. Keep identity, decay velocity, don't update pose.
          const ratioW = Math.max(match.w, prevBox.w) /
            Math.max(1, Math.min(match.w, prevBox.w));
          const ratioH = Math.max(match.h, prevBox.h) /
            Math.max(1, Math.min(match.h, prevBox.h));
          const suspect =
            !wasLost && (ratioW > SUSPECT_SIZE_RATIO || ratioH > SUSPECT_SIZE_RATIO);

          if (suspect) {
            t.missedFrames = 0;
            t.vx *= VELOCITY_DECAY_PER_MISS;
            t.vy *= VELOCITY_DECAY_PER_MISS;
          } else {
            t.missedFrames = 0;

            // Prefer the seg mask centroid — it's the "middle of the actual
            // pixels", more stable than bbox center and unbiased by
            // appendages (handles, spouts, stands).
            const obsCx = match.maskCentroid?.x ?? match.cx;
            const obsCy = match.maskCentroid?.y ?? match.cy;
            const observation: Box = makeBox(obsCx, obsCy, match.w, match.h);

            if (wasLost) {
              seedBoxEMA(t.boxEma, observation);
              t.smoothedBox = observation;
              t.vx = 0;
              t.vy = 0;
            } else {
              const prevCx = prevBox.cx;
              const prevCy = prevBox.cy;
              t.smoothedBox = smoothBox(t.boxEma, observation);
              // Velocity EMA — raw delta / dt is spiky, so we blend into the
              // running estimate at VELOCITY_EMA.
              const dt = now - t.lastUpdatedAt;
              if (dt > 10 && dt < 500) {
                const rawVx = (t.smoothedBox.cx - prevCx) / dt;
                const rawVy = (t.smoothedBox.cy - prevCy) / dt;
                t.vx = t.vx * (1 - VELOCITY_EMA) + rawVx * VELOCITY_EMA;
                t.vy = t.vy * (1 - VELOCITY_EMA) + rawVy * VELOCITY_EMA;
              }
            }
            t.lastUpdatedAt = now;

            // Refresh the silhouette clip now that smoothedBox is current.
            // Skipped in the `suspect` branch above because that match is
            // likely a neighbor, not our object — we'd rather keep the
            // last-good silhouette than paint the face onto the wrong shape.
            // maskAnchor = smoothedBox center at capture; the render RAF
            // offsets the silhouette by (renderBox.cx - maskAnchor.cx) so
            // the clip glides with the face between inferences instead of
            // jumping at 3–30 hz.
            if (match.mask) {
              if (!t.maskCanvas) t.maskCanvas = document.createElement("canvas");
              const url = renderMaskToDataUrl(t.maskCanvas, match.mask);
              if (url) {
                t.maskDataUrl = url;
                t.maskSrcBox = {
                  x1: match.x1,
                  y1: match.y1,
                  x2: match.x2,
                  y2: match.y2,
                };
                t.maskAnchor = {
                  cx: t.smoothedBox.cx,
                  cy: t.smoothedBox.cy,
                };
              }
            }

            // Orientation — rotate the face along the object's long axis
            // when the silhouette is comfortably elongated. Below the
            // ratio threshold, target 0 so round objects (cups, balls)
            // stay upright instead of drifting with PCA noise. Principal
            // axis is 180° ambiguous — wrap delta to ±π/2 before the EMA
            // so we don't spin the long way around when the axis flips.
            const axisRatio = match.axisRatio ?? 0;
            const rawAngle = match.principalAngle ?? 0;
            const targetAngle =
              axisRatio >= ORIENTATION_MIN_RATIO ? rawAngle : 0;
            let delta = targetAngle - t.rotation;
            while (delta > Math.PI / 2) delta -= Math.PI;
            while (delta < -Math.PI / 2) delta += Math.PI;
            t.rotation += delta * ROTATION_EMA_ALPHA;
          }
        } else {
          t.missedFrames++;
          // Decay velocity during misses so the face doesn't drift away
          // from the last known position while we wait for a re-acquire.
          t.vx *= VELOCITY_DECAY_PER_MISS;
          t.vy *= VELOCITY_DECAY_PER_MISS;
          // Face stays pinned at last known pose, audio keeps talking.
          // "Forever" tracking per product brief.
        }
      }
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [ensureYoloCanvas]);

  // --- Capture a source-space rectangle as a jpeg data URL --------------
  const captureBoxFrame = useCallback(
    (box: { x1: number; y1: number; x2: number; y2: number }): string | null => {
      const v = videoRef.current;
      if (!v || !v.videoWidth) return null;
      const pw = (box.x2 - box.x1) * 0.05;
      const ph = (box.y2 - box.y1) * 0.05;
      const cropSx = Math.max(0, box.x1 - pw);
      const cropSy = Math.max(0, box.y1 - ph);
      const cropW = Math.min(v.videoWidth - cropSx, box.x2 - box.x1 + 2 * pw);
      const cropH = Math.min(v.videoHeight - cropSy, box.y2 - box.y1 + 2 * ph);
      if (cropW < 8 || cropH < 8) return null;
      const longSide = Math.max(cropW, cropH);
      const s = Math.min(1, 512 / longSide);
      const outW = Math.max(1, Math.round(cropW * s));
      const outH = Math.max(1, Math.round(cropH * s));
      let canvas = cropCanvasRef.current;
      if (!canvas) {
        canvas = document.createElement("canvas");
        cropCanvasRef.current = canvas;
      }
      canvas.width = outW;
      canvas.height = outH;
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;
      ctx.drawImage(v, cropSx, cropSy, cropW, cropH, 0, 0, outW, outH);
      return canvas.toDataURL("image/jpeg", 0.82);
    },
    []
  );

  // Pick the track the mic should talk to. Most-recent-tap wins; returns
  // null when the scene has no faces. No UI for focus-switching yet; the
  // LRU-by-tap heuristic matches "the thing you just tapped is the thing
  // you want to speak to."
  const pickTalkTarget = useCallback((): TrackRefs | null => {
    if (tracksRef.current.length === 0) return null;
    let best = tracksRef.current[0];
    for (const t of tracksRef.current) {
      if (t.lastTapAt > best.lastTapAt) best = t;
    }
    return best;
  }, []);

  // Voice-in → voice-out conversation on a specific track. Mirrors the
  // structure of speakOnTrack (per-track speakGen guard, per-track analyser
  // so lips sync) but routes through the converseWithObject server action
  // instead of the one-shot generateLine.
  const sendTalkToTrack = useCallback(
    async (trackId: string, blob: Blob, clientTranscript?: string) => {
      const ctx = ensureAudioCtx();
      const track = tracksRef.current.find((t) => t.id === trackId);
      if (!ctx || !track) return;

      const callGen = ++track.speakGen;

      // Retap cancels any pending caption auto-dismiss — a fresh reply
      // deserves its full linger.
      if (track.captionClearTimer != null) {
        clearTimeout(track.captionClearTimer);
        track.captionClearTimer = null;
      }

      // Stop whatever this track was saying (buffer source OR streaming
      // audio element) so the fresh reply owns the airwaves.
      stopTrackAudio(track);

      setTracksUI((prev) =>
        prev.map((t) =>
          t.id === trackId
            ? { ...t, thinking: true, speaking: false, caption: null }
            : t
        )
      );

      try {
        const formData = new FormData();
        // Match the file extension to the recorder's MIME type so the STT
        // backend picks the right decoder.
        const filename =
          blob.type.includes("mp4")
            ? "talk.mp4"
            : blob.type.includes("ogg")
              ? "talk.ogg"
              : "talk.webm";
        formData.append("audio", blob, filename);
        formData.append("className", track.className);
        if (track.voiceId) formData.append("voiceId", track.voiceId);
        // Rich visual notes hydrated in the background by describeObject.
        // This is what lets converseWithObject stay text-only on the hot
        // path (no vision call) while still getting funnier-than-classname
        // context — the chewed straw, dust, dent, etc.
        if (track.description) formData.append("description", track.description);
        // Browser-side Web Speech API transcript. When present, the server
        // skips its STT call entirely (~700–1300ms saved). Empty falls back
        // to server STT against the audio blob.
        const trimmedTranscript = (clientTranscript ?? "").trim();
        if (trimmedTranscript) {
          formData.append("transcript", trimmedTranscript);
        }
        // Full conversation so far — object's prior lines + user turns.
        // The server re-caps to 16; we cap here so the payload stays small.
        formData.append(
          "history",
          JSON.stringify(track.history.slice(-16))
        );

        // eslint-disable-next-line no-console
        console.log(
          `[talk:${trackId}] → converseWithObject  class="${track.className}"  audio=${Math.round(blob.size / 1024)}KB (${blob.type || "?"})  voice=${track.voiceId ?? "default"}  history=${track.history.length}  desc=${track.description ? track.description.length + "ch" : "none"}  client-stt=${trimmedTranscript ? trimmedTranscript.length + "ch" : "no"}`
        );
        const t0 = performance.now();
        const { transcript, reply, voiceId: replyVoiceId } =
          await converseWithObject(formData);
        if (callGen !== track.speakGen) {
          // eslint-disable-next-line no-console
          console.log(`[talk:${trackId}] ← superseded (speakGen mismatch)`);
          return;
        }
        // Commit the exchange to this track's memory so future turns see
        // the full thread. Cap to the same 16 turns the server enforces.
        const nextHistory = [...track.history];
        if (transcript)
          nextHistory.push({ role: "user" as const, content: transcript });
        if (reply)
          nextHistory.push({ role: "assistant" as const, content: reply });
        track.history = nextHistory.slice(-16);
        // eslint-disable-next-line no-console
        console.log(
          `[talk:${trackId}] ← ${Math.round(performance.now() - t0)}ms  heard="${transcript}"  reply="${reply}"  history=${track.history.length}`
        );

        // Fire-and-forget: refresh the description off a fresh crop so the
        // NEXT conversation turn sees the current visual state (user moved
        // closer, opened a drawer, put hoodies on the chair, etc.). Runs
        // in the background — doesn't block audio playback below.
        const freshCrop = captureBoxFrame(track.smoothedBox);
        if (freshCrop) refreshTrackDescription(trackId, freshCrop);

        // Echo what STT heard so the user has an instant signal that
        // their voice message landed — even before the reply starts playing.
        if (transcript) {
          setHeardText(transcript);
          if (heardClearTimerRef.current != null) {
            clearTimeout(heardClearTimerRef.current);
          }
          heardClearTimerRef.current = window.setTimeout(() => {
            setHeardText(null);
            heardClearTimerRef.current = null;
          }, 3600);
        }

        setTracksUI((prev) =>
          prev.map((t) =>
            t.id === trackId ? { ...t, caption: reply, thinking: false } : t
          )
        );

        if (!reply) return;

        // Kick off the streaming TTS — bytes start flowing into the audio
        // element as Fish generates them. No base64 round-trip, no "wait
        // for full mp3" gap between LLM done and mouth moving.
        await playStreamingReply(trackId, callGen, reply, replyVoiceId ?? track.voiceId);
      } catch (e) {
        if (callGen !== track.speakGen) return;
        const msg = e instanceof Error ? e.message : "talk failed";
        setErrorMsg(msg);
        setDiagError(msg);
        setTracksUI((prev) =>
          prev.map((t) =>
            t.id === trackId
              ? { ...t, thinking: false, speaking: false }
              : t
          )
        );
        showRejection(
          /zhipu|glm|whisper|openai|api key|api_key|401|403/i.test(msg)
            ? "voice backend unconfigured — check .env.local"
            : `couldn't reply: ${msg.slice(0, 80)}`,
          3200
        );
        // eslint-disable-next-line no-console
        console.log("[tracker] talk failed:", e);
      } finally {
        if (callGen === track.speakGen) {
          setTracksUI((prev) =>
            prev.map((t) => (t.id === trackId ? { ...t, thinking: false } : t))
          );
        }
      }
    },
    [
      captureBoxFrame,
      ensureAudioCtx,
      playStreamingReply,
      refreshTrackDescription,
      showRejection,
      stopTrackAudio,
    ]
  );

  // Hit-test a tap against existing tracks' smoothed boxes. Returns the
  // innermost hit (smallest box containing the point) so nested tracks
  // resolve intuitively.
  const findTrackAtPoint = useCallback(
    (srcX: number, srcY: number): TrackRefs | null => {
      let best: TrackRefs | null = null;
      let bestArea = Infinity;
      for (const t of tracksRef.current) {
        const b = t.smoothedBox;
        if (srcX < b.x1 || srcX > b.x2 || srcY < b.y1 || srcY > b.y2) continue;
        const area = b.w * b.h;
        if (area < bestArea) {
          bestArea = area;
          best = t;
        }
      }
      return best;
    },
    []
  );

  // --- Tap handler -------------------------------------------------------
  const handleTap = useCallback(
    async (e: React.PointerEvent) => {
      const v = videoRef.current;
      if (!v) return;

      const gen = ++generationRef.current;
      const rectNow = v.getBoundingClientRect();
      const tapElX = e.clientX - rectNow.left;
      const tapElY = e.clientY - rectNow.top;

      // Instant fallback frame — replaced once we know the box.
      const elMin = Math.min(rectNow.width, rectNow.height);
      const fallbackEl = elMin * TAP_FRAME_FRACTION;
      setTapFrame({
        left: tapElX - fallbackEl / 2,
        top: tapElY - fallbackEl / 2,
        width: fallbackEl,
        height: fallbackEl,
        gen,
      });

      // Unlock audio inside the pointer gesture.
      ensureAudioCtx();

      if (phase === "error") {
        showRejection(errorMsg ?? "camera not ready");
        return;
      }
      if (!v.videoWidth) {
        showRejection("waiting for the camera");
        return;
      }
      if (!yoloReadyRef.current) {
        showRejection("detector still loading");
        return;
      }

      const srcTap = elementPointToSource(e.clientX, e.clientY, v);
      if (!srcTap) {
        showRejection("couldn't grab the frame");
        return;
      }

      // (a) If the tap lands inside an existing track, retrigger that
      //     track's voice line with a fresh crop. This is the "tap a face
      //     again for a new opinion" interaction.
      const existing = findTrackAtPoint(srcTap.x, srcTap.y);
      if (existing) {
        setTapFrame(null);
        const dataUrl = captureBoxFrame(existing.smoothedBox);
        if (!dataUrl) {
          showRejection("couldn't grab the frame");
          return;
        }
        existing.lastTapAt = performance.now();
        void speakOnTrack(existing.id, dataUrl);
        return;
      }

      // (b) Otherwise resolve a detection under the tap and add a new face.
      let tapDets: Detection[] = [];
      const cacheAge = performance.now() - detectionsTsRef.current;
      if (detectionsRef.current.length && cacheAge < TAP_CACHE_MAX_AGE_MS) {
        tapDets = detectionsRef.current;
      } else {
        try {
          tapDets = await detect(v, ensureYoloCanvas(), {
            confThreshold: TAP_CONF,
            maxDetections: CONTINUOUS_MAX_DET,
            classFilter: (id) => !EXCLUDED_CLASS_IDS.has(id),
          });
        } catch {
          tapDets = [];
        }
        if (gen !== generationRef.current) return;
      }

      let tapped = pickTappedDetection(tapDets, srcTap.x, srcTap.y);

      if (!tapped) {
        try {
          const fresh = await detect(v, ensureYoloCanvas(), {
            confThreshold: TAP_CONF,
            classFilter: (id) => !EXCLUDED_CLASS_IDS.has(id),
          });
          if (gen !== generationRef.current) return;
          tapped = pickTappedDetection(fresh, srcTap.x, srcTap.y);
          if (fresh.length) {
            detectionsRef.current = fresh;
            detectionsTsRef.current = performance.now();
            setDetections(fresh);
          }
        } catch {
          // fall through
        }
      }

      if (!tapped) {
        showRejection("nothing I recognize there");
        return;
      }

      // Snap the visible tap frame to the detected box.
      {
        const vp = sourceBoxToElement(tapped, v);
        if (vp) {
          setTapFrame({
            left: vp.left,
            top: vp.top,
            width: Math.max(24, vp.width),
            height: Math.max(24, vp.height),
            gen,
          });
        }
      }

      const dataUrl = captureBoxFrame(tapped);
      if (!dataUrl) {
        showRejection("couldn't grab the frame");
        return;
      }

      setErrorMsg(null);
      setRejection(null);
      if (rejectionTimerRef.current != null) {
        clearTimeout(rejectionTimerRef.current);
        rejectionTimerRef.current = null;
      }

      // Build the new track. Anchor is box center. When the seg head
      // supplies a mask centroid, use it as the box center — on asymmetric
      // objects (mug with handle, lamp with stand) this lands the face on
      // the body rather than slightly toward the appendage.
      const lockCx = tapped.maskCentroid?.x ?? (tapped.x1 + tapped.x2) / 2;
      const lockCy = tapped.maskCentroid?.y ?? (tapped.y1 + tapped.y2) / 2;
      const lockBox: Box = makeBox(
        lockCx,
        lockCy,
        tapped.x2 - tapped.x1,
        tapped.y2 - tapped.y1
      );
      // Split EMA — position responsive, size slow (kills bbox-edge breathing).
      const boxEma = newBoxEMA(BOX_POS_ALPHA, BOX_SIZE_ALPHA);
      seedBoxEMA(boxEma, lockBox);
      const newId = `t${nextTrackIdRef.current++}`;
      const nowTs = performance.now();
      const newTrack: TrackRefs = {
        id: newId,
        classId: tapped.classId,
        className: tapped.className,
        anchor: { rx: 0, ry: 0 },
        boxEma,
        smoothedBox: lockBox,
        missedFrames: 0,
        opacity: 0,
        lastTapAt: nowTs,
        vx: 0,
        vy: 0,
        lastUpdatedAt: nowTs,
        speakGen: 0,
        analyser: null,
        gain: null,
        freqData: null,
        timeData: null,
        source: null,
        shape: "X",
        lipSync: createLipSyncState(),
        captionClearTimer: null,
        streamingAudio: null,
        streamingUrl: null,
        voiceId: null,
        history: [],
        description: null,
        descriptionGen: 0,
        maskCanvas: null,
        maskDataUrl: null,
        maskSrcBox: null,
        maskAnchor: null,
        rotation: 0,
      };

      // LRU eviction when the slots are full.
      if (tracksRef.current.length >= MAX_FACES) {
        let oldest = tracksRef.current[0];
        for (const t of tracksRef.current) {
          if (t.lastTapAt < oldest.lastTapAt) oldest = t;
        }
        if (oldest.source) {
          try {
            oldest.source.stop();
          } catch {
            // already stopped
          }
        }
        if (oldest.streamingAudio) {
          try {
            oldest.streamingAudio.pause();
          } catch {
            // ignore
          }
          oldest.streamingAudio.src = "";
          oldest.streamingAudio = null;
        }
        if (oldest.streamingUrl) {
          try {
            URL.revokeObjectURL(oldest.streamingUrl);
          } catch {
            // ignore
          }
          oldest.streamingUrl = null;
        }
        if (oldest.analyser) {
          try {
            oldest.analyser.disconnect();
          } catch {
            // already disconnected
          }
        }
        if (oldest.gain) {
          try {
            oldest.gain.disconnect();
          } catch {
            // already disconnected
          }
        }
        if (oldest.captionClearTimer != null) {
          clearTimeout(oldest.captionClearTimer);
          oldest.captionClearTimer = null;
        }
        tracksRef.current = tracksRef.current.filter((t) => t.id !== oldest.id);
        setTracksUI((prev) => prev.filter((t) => t.id !== oldest.id));
      }
      tracksRef.current = [...tracksRef.current, newTrack];

      // Seed UI entry — RAF will fill position/scale/opacity next frame.
      setTracksUI((prev) => [
        ...prev,
        {
          id: newId,
          classId: tapped!.classId,
          className: tapped!.className,
          left: 0,
          top: 0,
          scale: 1,
          opacity: 0,
          shape: "X",
          caption: null,
          thinking: false,
          speaking: false,
          maskDataUrl: null,
          maskLeft: 0,
          maskTop: 0,
          maskWidth: 0,
          maskHeight: 0,
          rotation: 0,
        },
      ]);

      setTapFrame(null);
      void speakOnTrack(newId, dataUrl);
      // Hydrate rich visual context in the background while the first line
      // is being generated, so the FIRST conversation turn already has
      // something funnier than the bare classname to work with.
      refreshTrackDescription(newId, dataUrl);
    },
    [
      captureBoxFrame,
      ensureAudioCtx,
      ensureYoloCanvas,
      errorMsg,
      findTrackAtPoint,
      phase,
      refreshTrackDescription,
      showRejection,
      speakOnTrack,
    ]
  );

  // --- Push-to-talk helpers (same behavior as before) -------------------
  const stopTalkLevelLoop = useCallback(() => {
    if (talkLevelRafRef.current != null) {
      cancelAnimationFrame(talkLevelRafRef.current);
      talkLevelRafRef.current = null;
    }
    const bands = bandLevelsRef.current;
    for (let i = 0; i < bands.length; i++) {
      bands[i] = 0;
      const bar = barRefs.current[i];
      if (bar) bar.style.transform = "scaleY(1)";
    }
  }, []);

  const openMicStream = useCallback(async (): Promise<MediaStream | null> => {
    if (micStreamRef.current) return micStreamRef.current;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      micStreamRef.current = stream;
      return stream;
    } catch (e) {
      const msg = e instanceof Error ? e.message : "mic unavailable";
      // eslint-disable-next-line no-console
      console.log("[tracker] mic denied:", e);
      setMicError(msg);
      return null;
    }
  }, []);

  const startRecording = useCallback(async () => {
    if (recorderRef.current && recorderRef.current.state === "recording") return;
    // eslint-disable-next-line no-console
    console.log("[mic] ▶ start recording");
    const ctx = ensureAudioCtx();
    const stream = await openMicStream();
    if (!stream) {
      // eslint-disable-next-line no-console
      console.log("[mic] ✖ no mic stream");
      return;
    }
    if (ctx && !talkAnalyserRef.current) {
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.55;
      talkAnalyserRef.current = analyser;
      talkFreqDataRef.current = new Uint8Array(
        new ArrayBuffer(analyser.frequencyBinCount)
      );
    }
    if (ctx && talkAnalyserRef.current && !talkSourceRef.current) {
      try {
        talkSourceRef.current = ctx.createMediaStreamSource(stream);
        talkSourceRef.current.connect(talkAnalyserRef.current);
      } catch {
        // already consumed
      }
    }
    const mime =
      ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", ""].find(
        (t) => !t || MediaRecorder.isTypeSupported(t)
      ) ?? "";
    const mr = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    recordedChunksRef.current = [];
    mr.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) recordedChunksRef.current.push(e.data);
    };
    mr.onstop = () => {
      const blob = new Blob(recordedChunksRef.current, {
        type: mr.mimeType || "audio/webm",
      });
      recordedBlobRef.current = blob;
      // eslint-disable-next-line no-console
      console.log(
        `[tracker] captured ${Math.round(blob.size / 1024)} KB of audio (${blob.type})`
      );

      // The voice-in, voice-out loop — send the blob to whichever face is
      // currently in focus (most-recently-tapped). If nothing is locked
      // yet or the clip is too short to be real speech, bail with a toast.
      const target = pickTalkTarget();
      if (!target) {
        showRejection("tap something first — then talk to it");
        return;
      }
      if (blob.size < 1024) {
        showRejection("too short — hold the button longer");
        return;
      }

      // Visual "sent" flash on the button so the release feels decisive.
      setTalkFlash(true);
      if (talkFlashTimerRef.current != null) clearTimeout(talkFlashTimerRef.current);
      talkFlashTimerRef.current = window.setTimeout(() => {
        setTalkFlash(false);
        talkFlashTimerRef.current = null;
      }, 650);

      // Wait briefly for the in-flight SpeechRecognition to settle so we
      // can pass its transcript through and skip the server STT roundtrip
      // entirely. Cap at 500ms — if SR is slower than that something is
      // off and the server fallback will still produce a transcript.
      void (async () => {
        const finished = speechFinishedRef.current;
        if (finished) {
          await Promise.race([
            finished,
            new Promise<void>((r) => window.setTimeout(r, 500)),
          ]);
        }
        const transcript = speechTranscriptPartsRef.current
          .join(" ")
          .replace(/\s+/g, " ")
          .trim();
        speechTranscriptPartsRef.current = [];
        // eslint-disable-next-line no-console
        console.log(
          `[sr] → handing off transcript (${transcript ? transcript.length + "ch" : "empty"}) to converseWithObject`
        );
        await sendTalkToTrack(target.id, blob, transcript);
      })();
    };

    // Kick off Web Speech API in parallel. By the time the user releases
    // the talk button the transcript is usually already final, so we send
    // it with the request and the server skips its STT call entirely.
    const SR = getSpeechRecognitionCtor();
    if (SR) {
      try {
        if (speechRecognitionRef.current) {
          try {
            speechRecognitionRef.current.abort();
          } catch {
            // ignore
          }
          speechRecognitionRef.current = null;
        }
        speechTranscriptPartsRef.current = [];
        speechFinishedRef.current = new Promise<void>((resolve) => {
          resolveSpeechFinishedRef.current = resolve;
        });
        const sr = new SR();
        sr.lang = "en-US";
        sr.continuous = true;
        sr.interimResults = false;
        sr.maxAlternatives = 1;
        sr.onresult = (e) => {
          for (let i = e.resultIndex; i < e.results.length; i++) {
            const r = e.results[i];
            if (r.isFinal && r[0]?.transcript) {
              speechTranscriptPartsRef.current.push(r[0].transcript);
            }
          }
        };
        sr.onerror = (ev) => {
          // eslint-disable-next-line no-console
          console.log(
            `[sr] ✖ ${ev.error}${ev.message ? ` (${ev.message})` : ""}`
          );
        };
        sr.onend = () => {
          // eslint-disable-next-line no-console
          console.log(
            `[sr] ◼ ended  parts=${speechTranscriptPartsRef.current.length}`
          );
          resolveSpeechFinishedRef.current?.();
          resolveSpeechFinishedRef.current = null;
          speechRecognitionRef.current = null;
        };
        sr.start();
        speechRecognitionRef.current = sr;
        // eslint-disable-next-line no-console
        console.log("[sr] ▶ started (browser-side transcription)");
      } catch (err) {
        // eslint-disable-next-line no-console
        console.log(
          `[sr] ✖ start failed: ${err instanceof Error ? err.message : String(err)} — falling back to server STT`
        );
        speechRecognitionRef.current = null;
        speechFinishedRef.current = null;
        resolveSpeechFinishedRef.current = null;
      }
    } else {
      // No Web Speech in this browser — leave the refs unset so onstop
      // takes the server-STT path.
      speechFinishedRef.current = null;
      speechTranscriptPartsRef.current = [];
    }

    recorderRef.current = mr;
    mr.start(100);
    setIsRecording(true);
    setMicError(null);

    if (typeof navigator.vibrate === "function") {
      try {
        navigator.vibrate(8);
      } catch {
        // ignore
      }
    }

    const readLevel = () => {
      const an = talkAnalyserRef.current;
      const buf = talkFreqDataRef.current;
      if (!an || !buf) {
        talkLevelRafRef.current = requestAnimationFrame(readLevel);
        return;
      }
      an.getByteFrequencyData(buf);

      // Per-bar band energies → DOM scaleY. Distance from center maps to a
      // band slice so the center bars read low-mids (where speech lives) and
      // outer bars pick up higher freqs. Each bar keeps its own EMA to kill
      // jitter without smearing transients.
      const bands = bandLevelsRef.current;
      const half = (WAVE_BARS - 1) / 2;
      for (let i = 0; i < WAVE_BARS; i++) {
        const d = Math.abs(i - half) / half;
        const start = 2 + Math.floor(d * 36);
        const stop = Math.min(buf.length, start + 5);
        let bs = 0;
        for (let k = start; k < stop; k++) bs += buf[k];
        const bavg = bs / Math.max(1, stop - start) / 255;
        const raw = Math.max(0, Math.min(1, (bavg - 0.04) * 2.4));
        const prev = bands[i];
        const nextLvl = prev + (raw - prev) * 0.45;
        bands[i] = nextLvl;
        const bar = barRefs.current[i];
        if (bar) bar.style.transform = `scaleY(${1 + nextLvl * 6})`;
      }

      talkLevelRafRef.current = requestAnimationFrame(readLevel);
    };
    if (talkLevelRafRef.current == null) {
      talkLevelRafRef.current = requestAnimationFrame(readLevel);
    }
  }, [ensureAudioCtx, openMicStream, pickTalkTarget, sendTalkToTrack, showRejection]);

  const stopRecording = useCallback(() => {
    // eslint-disable-next-line no-console
    console.log("[mic] ◼ stop recording");
    const mr = recorderRef.current;
    if (mr && mr.state === "recording") {
      try {
        mr.stop();
      } catch {
        // ignore
      }
    }
    recorderRef.current = null;
    // Tell SpeechRecognition to flush its buffer and emit a final result.
    // The `onend` handler resolves `speechFinishedRef` which the recorder
    // onstop is already awaiting (with a 500ms cap) before sending the turn.
    if (speechRecognitionRef.current) {
      try {
        speechRecognitionRef.current.stop();
      } catch {
        // ignore — already stopped
      }
    }
    setIsRecording(false);
    stopTalkLevelLoop();
  }, [stopTalkLevelLoop]);

  // --- Derived UI --------------------------------------------------------
  const anyThinking = tracksUI.some((t) => t.thinking);
  const anySpeaking = tracksUI.some((t) => t.speaking);

  // Which face the mic is aimed at — the most-recently-tapped one. Drives
  // the button label so the user sees "talk to the cup" before they press.
  const talkTargetClass: string | null = useMemo(() => {
    if (tracksUI.length === 0) return null;
    let best: TrackRefs | null = null;
    for (const t of tracksRef.current) {
      if (!best || t.lastTapAt > best.lastTapAt) best = t;
    }
    return best?.className ?? null;
    // Invalidate on track add/remove. lastTapAt churn within tracksRef
    // is intentionally not a dep — it'd trigger every tap without changing
    // the target, since the last-tap-wins order is already stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tracksUI]);

  const dotClass =
    phase === "error"
      ? "bg-rose-400"
      : phase === "starting"
        ? "bg-amber-300"
        : phase === "ready"
          ? "bg-fuchsia-400"
          : anyThinking
            ? "bg-amber-300"
            : anySpeaking
              ? "bg-emerald-400"
              : "bg-sky-400";

  const statusText =
    phase === "starting"
      ? !cameraReady
        ? "warming up camera"
        : !yoloReady
          ? "loading detector"
          : "warming up"
      : phase === "error"
        ? (errorMsg ?? "camera error")
        : phase === "ready"
          ? (errorMsg ?? "tap anything")
          : anyThinking
            ? "thinking"
            : anySpeaking
              ? "speaking"
              : `${tracksUI.length} face${tracksUI.length === 1 ? "" : "s"} · ${fps} fps`;

  const breathingBoxes = useMemo(() => {
    const v = videoRef.current;
    if (!v || phase !== "ready") return [];
    return detections
      .map((d) => {
        if (EXCLUDED_CLASS_IDS.has(d.classId)) return null;
        const vp = sourceBoxToElement(d, v);
        if (!vp) return null;
        if (vp.width < 8 || vp.height < 8) return null;
        return {
          id: `${d.classId}-${d.cx.toFixed(0)}-${d.cy.toFixed(0)}-${d.w.toFixed(0)}`,
          className: COCO_CLASSES[d.classId] ?? "?",
          score: d.score,
          ...vp,
        };
      })
      .filter((b): b is NonNullable<typeof b> => b !== null);
  }, [detections, phase]);

  return (
    <div className="fixed inset-0 overflow-hidden touch-none select-none bg-gradient-to-br from-[#1a0f2e] via-[#2a1540] to-[#3d1a4d]">
      <video
        ref={videoRef}
        playsInline
        muted
        className="absolute inset-0 h-full w-full object-cover"
        onPointerDown={handleTap}
      />

      {/* Face overlays — one per track. Positioned by RAF via state. */}
      {tracksUI.map((t) => {
        // Bubble scale: tied to the face, clamped so tiny objects don't
        // produce unreadable 8px text and huge objects don't drown the frame.
        const bubbleScale = Math.max(0.6, Math.min(1.3, t.scale * 0.85));
        // Gap between the face's visual top and the bubble's tail, in CSS px.
        const faceHalfH = (FACE_VOICE_HEIGHT / 2) * t.scale;
        const BUBBLE_GAP = 16;
        const bubbleTop = t.top - faceHalfH - BUBBLE_GAP;
        // Max width scales with face width so a mug gets a pill and a sofa
        // gets a paragraph. Native units; the outer scale transform handles
        // the rest.
        const bubbleMaxWidth = Math.max(140, FACE_VOICE_WIDTH * 1.1);
        // Once the first inference after lock lands, `maskDataUrl` is set
        // and the face renders INSIDE the object's silhouette (clipped +
        // blend-moded into the surface). Until then, fall back to the
        // unclipped float so the face appears instantly on tap instead of
        // waiting for YOLO.
        const clipped = t.maskDataUrl && t.maskWidth > 0 && t.maskHeight > 0;
        return (
          <div key={t.id}>
            {clipped ? (
              <div
                className="pointer-events-none absolute will-change-transform"
                style={{
                  left: t.maskLeft,
                  top: t.maskTop,
                  width: t.maskWidth,
                  height: t.maskHeight,
                  WebkitMaskImage: `url(${t.maskDataUrl})`,
                  WebkitMaskSize: "100% 100%",
                  WebkitMaskRepeat: "no-repeat",
                  maskImage: `url(${t.maskDataUrl})`,
                  maskSize: "100% 100%",
                  maskRepeat: "no-repeat",
                  mixBlendMode: FACE_BLEND_MODE,
                  // drop-shadow projects OUTSIDE the mask silhouette — mixed
                  // with the video through `hard-light` below, it reads as a
                  // soft contact darkening around the face, anchoring it to
                  // the surface instead of floating over it.
                  filter: "drop-shadow(0 3px 8px rgba(0,0,0,0.35))",
                  opacity: t.opacity,
                }}
              >
                {/* Inner: face positioned at its anchor in element coords,
                    expressed relative to the mask wrapper's origin. Rotation
                    (driven by the object's PCA principal angle) goes between
                    centering and scale so the pivot stays at the anchor. */}
                <div
                  style={{
                    position: "absolute",
                    left: t.left - t.maskLeft,
                    top: t.top - t.maskTop,
                    transformOrigin: "0 0",
                    transform: `translate(-50%, -50%) rotate(${t.rotation}rad) scale(${t.scale})`,
                  }}
                >
                  <FaceVoice shape={t.shape} />
                </div>
              </div>
            ) : (
              <div
                className="pointer-events-none absolute left-0 top-0 will-change-transform"
                style={{
                  transformOrigin: "0 0",
                  transform: `translate(${t.left}px, ${t.top}px) translate(-50%, -50%) rotate(${t.rotation}rad) scale(${t.scale})`,
                  opacity: t.opacity,
                }}
              >
                <FaceVoice shape={t.shape} />
              </div>
            )}
            <div
              className="pointer-events-none absolute left-0 top-0 will-change-transform"
              style={{
                transformOrigin: "50% 100%",
                transform: `translate(${t.left}px, ${bubbleTop}px) translate(-50%, -100%) scale(${bubbleScale})`,
                opacity: t.opacity,
              }}
            >
              <SpeechBubble
                caption={t.caption}
                thinking={t.thinking}
                speaking={t.speaking}
                maxWidth={bubbleMaxWidth}
              />
            </div>
          </div>
        );
      })}

      {/* Breathing boxes (ready phase only). */}
      {breathingBoxes.map((b) => (
        <div
          key={b.id}
          className="pointer-events-none absolute rounded-[18px]"
          style={{
            left: b.left,
            top: b.top,
            width: b.width,
            height: b.height,
            boxShadow:
              "0 0 0 2px rgba(255,137,190,0.55), inset 0 0 0 1px rgba(255,255,255,0.35), 0 0 24px rgba(255,137,190,0.22)",
            animation: "breathing-box 2.6s ease-in-out infinite",
          }}
        >
          <span className="serif-italic absolute -top-[26px] left-1 flex items-center gap-1 rounded-full bg-[color:var(--ink)]/75 px-2.5 py-0.5 text-[11px] font-medium text-white ring-1 ring-white/20 backdrop-blur-md">
            <span>{b.className}</span>
            <span className="tabular-nums text-white/60">
              {Math.round(b.score * 100)}%
            </span>
          </span>
        </div>
      ))}

      {/* Tap frame while a new face is resolving. */}
      {tapFrame && (
        <div
          key={tapFrame.gen}
          className="pointer-events-none absolute"
          style={{
            left: tapFrame.left,
            top: tapFrame.top,
            width: tapFrame.width,
            height: tapFrame.height,
          }}
        >
          <div
            className="absolute inset-0 rounded-[28px]"
            style={{
              boxShadow:
                "0 0 0 2px rgba(255,137,190,0.9), 0 0 0 6px rgba(255,255,255,0.18), 0 0 40px rgba(255,137,190,0.35)",
              animation: "tap-frame 480ms cubic-bezier(0.16,1,0.3,1) both",
            }}
          />
          {(
            [
              ["top-0 left-0", "border-t-2 border-l-2 rounded-tl-[22px]"],
              ["top-0 right-0", "border-t-2 border-r-2 rounded-tr-[22px]"],
              ["bottom-0 left-0", "border-b-2 border-l-2 rounded-bl-[22px]"],
              ["bottom-0 right-0", "border-b-2 border-r-2 rounded-br-[22px]"],
            ] as const
          ).map(([pos, edges], i) => (
            <span
              key={i}
              className={`absolute ${pos} h-5 w-5 ${edges} border-white/95`}
            />
          ))}
        </div>
      )}

      {/* Top wordmark + status pill + diag toggle. */}
      <div className="absolute inset-x-0 top-0 flex items-center justify-between px-5 pt-[max(env(safe-area-inset-top),18px)]">
        <div className="pointer-events-none flex items-center gap-2 rounded-full bg-white/15 px-3.5 py-1.5 shadow-[0_8px_24px_-12px_rgba(0,0,0,0.6)] ring-1 ring-white/25 backdrop-blur-xl">
          <span className="h-1.5 w-1.5 rounded-full bg-[#ff89be] shadow-[0_0_0_3px_rgba(255,137,190,0.28)]" />
          <span className="serif-italic text-[17px] font-medium leading-none text-white/95">
            mirror
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div
            className={
              "pointer-events-none flex items-center gap-2 rounded-full px-3.5 py-1.5 shadow-[0_8px_24px_-12px_rgba(0,0,0,0.6)] ring-1 backdrop-blur-xl transition " +
              (phase === "error"
                ? "bg-rose-500/30 ring-rose-200/40"
                : "bg-white/15 ring-white/25")
            }
          >
            <span className={"h-1.5 w-1.5 rounded-full " + dotClass} />
            <span className="text-[11.5px] font-medium tabular-nums tracking-wide text-white/90">
              {statusText}
            </span>
          </div>
          <button
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              setDiagOpen((o) => !o);
            }}
            className="grid h-7 w-7 place-items-center rounded-full bg-white/15 text-[11px] font-semibold text-white/90 ring-1 ring-white/25 backdrop-blur-xl transition hover:bg-white/25"
            aria-label="toggle diagnostics"
          >
            i
          </button>
          {tracksUI.length > 0 && (
            <button
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                clearAllTracks();
              }}
              className="grid h-7 w-7 place-items-center rounded-full bg-white/15 text-white/90 ring-1 ring-white/25 backdrop-blur-xl transition hover:bg-rose-500/40 hover:ring-rose-200/50"
              aria-label="clear all faces"
              title="clear all (Esc)"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.6"
                strokeLinecap="round"
              >
                <path d="M6 6l12 12M18 6L6 18" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* YOLO loading overlay. */}
      {!yoloReady && yoloStatus.stage !== "error" && (
        <div
          className="pointer-events-none absolute inset-x-0 top-[72px] flex justify-center px-4"
          style={{ animation: "fade-in 220ms ease-out both" }}
        >
          <div className="flex min-w-[260px] max-w-sm flex-col gap-2 rounded-2xl bg-black/40 px-4 py-3 ring-1 ring-white/15 backdrop-blur-xl">
            <div className="flex items-center justify-between gap-3">
              <span className="serif-italic text-[12px] font-medium text-white/95">
                {yoloStatus.stage === "downloading"
                  ? "downloading detector"
                  : yoloStatus.stage === "compiling"
                    ? "warming up detector"
                    : "loading detector"}
              </span>
              <span className="tabular-nums text-[11px] text-white/60">
                {yoloStatus.bytesTotal > 0
                  ? `${Math.round((yoloStatus.bytesLoaded / 1024 / 1024) * 10) / 10}/${Math.round((yoloStatus.bytesTotal / 1024 / 1024) * 10) / 10} MB`
                  : yoloStatus.bytesLoaded > 0
                    ? `${Math.round((yoloStatus.bytesLoaded / 1024 / 1024) * 10) / 10} MB`
                    : ""}
              </span>
            </div>
            <div className="h-1 overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-[color:var(--accent)] transition-[width]"
                style={{
                  width:
                    yoloStatus.progress >= 0
                      ? `${Math.max(4, Math.min(100, yoloStatus.progress * 100))}%`
                      : "30%",
                  animation:
                    yoloStatus.progress < 0 || yoloStatus.stage === "compiling"
                      ? "soft-pulse 1.2s ease-in-out infinite"
                      : undefined,
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* YOLO load error banner. */}
      {yoloStatus.stage === "error" && !yoloReady && (
        <div
          className="absolute inset-x-0 top-[72px] flex justify-center px-4"
          style={{ animation: "bubble-in 320ms cubic-bezier(0.16,1,0.3,1) both" }}
        >
          <div className="flex max-w-md flex-col gap-2 rounded-2xl bg-rose-500/25 px-4 py-3 ring-1 ring-rose-200/40 backdrop-blur-xl">
            <span className="serif-italic text-[12px] font-medium text-white/95">
              detector failed: {yoloStatus.error ?? "unknown"}
            </span>
            <button
              onClick={handleRetryYolo}
              className="self-start rounded-full bg-white/90 px-3 py-1 text-[11px] font-semibold text-[color:var(--ink)] ring-1 ring-white/80 transition hover:bg-white"
            >
              retry
            </button>
          </div>
        </div>
      )}

      {/* Diagnostic panel. */}
      {diagOpen && (
        <div
          className="absolute right-4 top-[72px] w-[260px] rounded-2xl bg-black/70 p-3 text-[11px] text-white/90 ring-1 ring-white/20 backdrop-blur-xl"
          style={{ animation: "fade-in 160ms ease-out both" }}
          onPointerDown={(e) => e.stopPropagation()}
        >
          <div className="mb-2 flex items-center justify-between">
            <span className="serif-italic text-[13px] font-medium">diagnostics</span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setDiagOpen(false);
              }}
              className="text-white/60 hover:text-white"
              aria-label="close diagnostics"
            >
              ×
            </button>
          </div>
          <dl className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-1 tabular-nums">
            <dt className="text-white/50">phase</dt>
            <dd>{phase}</dd>
            <dt className="text-white/50">camera</dt>
            <dd>{cameraReady ? "ok" : "…"}</dd>
            <dt className="text-white/50">detector</dt>
            <dd>
              {yoloStatus.stage}
              {yoloStatus.backend ? ` · ${yoloStatus.backend}` : ""}
            </dd>
            <dt className="text-white/50">model</dt>
            <dd className="truncate">{yoloStatus.modelUrl ?? "—"}</dd>
            <dt className="text-white/50">fps</dt>
            <dd>{fps}</dd>
            <dt className="text-white/50">infer</dt>
            <dd>{lastInferMs != null ? `${lastInferMs} ms` : "—"}</dd>
            <dt className="text-white/50">detections</dt>
            <dd>{detections.length}</dd>
            <dt className="text-white/50">faces</dt>
            <dd>
              {tracksUI.length}/{MAX_FACES}
              {tracksUI.length > 0 &&
                ` · ${tracksUI.map((t) => t.className).join(", ")}`}
            </dd>
            <dt className="text-white/50">tts</dt>
            <dd
              className={
                lastTtsBackend === "fish"
                  ? "text-emerald-200"
                  : lastTtsBackend === "openai"
                    ? "text-amber-200"
                    : lastTtsBackend === "none"
                      ? "text-rose-200"
                      : ""
              }
            >
              {lastTtsBackend ?? "—"}
            </dd>
            {diagError && (
              <>
                <dt className="text-white/50">err</dt>
                <dd className="text-rose-200">{diagError}</dd>
              </>
            )}
          </dl>
          {yoloStatus.stage === "error" && (
            <button
              onClick={handleRetryYolo}
              className="mt-2 w-full rounded-full bg-white/90 px-3 py-1 text-[11px] font-semibold text-[color:var(--ink)] hover:bg-white"
            >
              retry detector
            </button>
          )}
        </div>
      )}

      {/* Ready-state centered hint. */}
      {phase === "ready" && !errorMsg && (
        <div
          className="pointer-events-none absolute inset-0 grid place-items-center"
          style={{ animation: "fade-in 500ms ease-out both" }}
        >
          <div className="flex flex-col items-center gap-3">
            <span className="bubble-btn grid h-16 w-16 place-items-center rounded-full bg-white/15 ring-1 ring-white/30 backdrop-blur-xl">
              <span className="h-3 w-3 rounded-full bg-white/90" />
            </span>
            <span className="serif-italic rounded-full bg-black/25 px-4 py-1.5 text-[13px] font-medium text-white/90 ring-1 ring-white/15 backdrop-blur-xl">
              tap anything — up to {MAX_FACES} at once
            </span>
          </div>
        </div>
      )}

      {rejection && !anyThinking && (
        <div
          className="pointer-events-none absolute inset-x-0 top-20 flex justify-center px-6"
          style={{ animation: "bubble-in 360ms cubic-bezier(0.16,1,0.3,1) both" }}
        >
          <div className="rounded-full bg-white/90 px-4 py-2 shadow-[0_12px_28px_-14px_rgba(0,0,0,0.5)] ring-1 ring-white/80 backdrop-blur-xl">
            <span className="serif-italic text-[13.5px] font-medium text-[color:var(--ink)]">
              {rejection}
            </span>
          </div>
        </div>
      )}

      {/* "You said …" transcript echo — instant signal that Whisper heard
          the voice message. Rendered just below the status pill so it
          overlaps the thinking + reply caption without fighting either. */}
      {heardText && (
        <div
          className="pointer-events-none absolute inset-x-0 top-[132px] flex justify-center px-6"
          style={{ animation: "bubble-in 320ms cubic-bezier(0.16,1,0.3,1) both" }}
        >
          <div className="flex max-w-[min(92vw,34rem)] flex-col items-center gap-0.5 rounded-2xl bg-black/55 px-4 py-2 ring-1 ring-white/15 backdrop-blur-xl">
            <span className="text-[10px] uppercase tracking-[0.18em] text-white/55">
              you said
            </span>
            <span className="serif-italic text-center text-[13px] font-medium leading-snug text-white/95 break-words">
              {heardText}
            </span>
          </div>
        </div>
      )}

      {anyThinking && (
        <div
          className="pointer-events-none absolute inset-x-0 top-20 flex justify-center"
          style={{ animation: "fade-in 220ms ease-out both" }}
        >
          <div className="flex items-center gap-2 rounded-full bg-white/15 px-4 py-2 ring-1 ring-white/25 backdrop-blur-xl">
            <span className="inline-block h-1.5 w-1.5 animate-ping rounded-full bg-[#ff89be]" />
            <span className="serif-italic text-[13px] font-medium text-white/95">
              thinking
            </span>
          </div>
        </div>
      )}

      {/* Hold-to-talk button (unchanged). */}
      <div
        className="absolute inset-x-0 bottom-0 flex flex-col items-center gap-2 px-5 pb-[max(env(safe-area-inset-bottom),22px)] pt-3"
        style={{
          opacity: phase === "starting" || phase === "error" ? 0.35 : 1,
          transition: "opacity 220ms ease",
          pointerEvents: phase === "starting" || phase === "error" ? "none" : "auto",
        }}
      >
        <div className="relative grid h-[104px] w-[104px] place-items-center">
          {/* Ambient halo — steady when live, breathing when idle. */}
          <span
            aria-hidden
            className="pointer-events-none absolute -inset-3 rounded-full"
            style={{
              background:
                "radial-gradient(circle, rgba(255,137,190,0.38) 0%, rgba(255,137,190,0) 72%)",
              opacity: isRecording ? 1 : 0.55,
              transition: "opacity 220ms ease",
              animation: isRecording ? undefined : "soft-pulse 2.4s ease-in-out infinite",
            }}
          />
          {/* Thin outer accent ring — adds crispness to the silhouette. */}
          <span
            aria-hidden
            className="pointer-events-none absolute rounded-full"
            style={{
              inset: 6,
              boxShadow: isRecording
                ? "0 0 0 1px rgba(255,255,255,0.45), 0 0 0 6px rgba(255,137,190,0.18)"
                : "0 0 0 1px rgba(255,255,255,0.55), 0 0 0 6px rgba(255,137,190,0.10)",
              transition: "box-shadow 240ms ease",
            }}
          />
          <button
            type="button"
            aria-label={isRecording ? "recording — release to send" : "hold to speak"}
            onPointerDown={(e) => {
              e.preventDefault();
              e.stopPropagation();
              try {
                (e.currentTarget as Element).setPointerCapture(e.pointerId);
              } catch {
                // old browsers or bad pointer id — fine to skip
              }
              void startRecording();
            }}
            onPointerUp={(e) => {
              e.preventDefault();
              e.stopPropagation();
              stopRecording();
            }}
            onPointerCancel={() => stopRecording()}
            onPointerLeave={() => {
              if (isRecording) stopRecording();
            }}
            onContextMenu={(e) => e.preventDefault()}
            className="relative grid h-[84px] w-[84px] place-items-center overflow-hidden rounded-full backdrop-blur-xl transition-[transform,box-shadow] duration-200 ease-out"
            style={{
              background: isRecording
                ? "linear-gradient(155deg, #ffb4d1 0%, #ff6aa8 55%, #ec4899 100%)"
                : "linear-gradient(155deg, rgba(255,255,255,0.96) 0%, rgba(255,220,234,0.92) 55%, rgba(255,182,213,0.92) 100%)",
              boxShadow: isRecording
                ? "0 22px 50px -18px rgba(236,72,153,0.8), 0 2px 6px rgba(0,0,0,0.12), inset 0 1px 0 rgba(255,255,255,0.55), inset 0 -14px 22px -14px rgba(236,72,153,0.55)"
                : "0 14px 36px -16px rgba(236,72,153,0.35), 0 1px 3px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.9), inset 0 -12px 22px -14px rgba(236,72,153,0.22)",
              transform: isRecording
                ? "scale(1.06)"
                : talkFlash
                  ? "scale(1.04)"
                  : "scale(1)",
              WebkitTouchCallout: "none",
              WebkitUserSelect: "none",
              userSelect: "none",
              touchAction: "none",
            }}
          >
            {/* Top-edge sheen — soft glossy highlight, not a full glare. */}
            <span
              aria-hidden
              className="pointer-events-none absolute inset-0 rounded-full"
              style={{
                background:
                  "radial-gradient(120% 55% at 50% 0%, rgba(255,255,255,0.6) 0%, rgba(255,255,255,0) 58%)",
                opacity: isRecording ? 0.5 : 0.9,
                transition: "opacity 220ms ease",
              }}
            />
            {/* Mic glyph — crossfades out when the waveform takes over. */}
            <svg
              width="30"
              height="30"
              viewBox="0 0 24 24"
              fill="none"
              stroke="var(--ink)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="relative"
              style={{
                opacity: isRecording ? 0 : 1,
                transform: `scale(${isRecording ? 0.85 : 1})`,
                transition:
                  "opacity 180ms ease, transform 220ms cubic-bezier(0.22, 1, 0.36, 1)",
              }}
            >
              <path d="M12 2.5c1.7 0 3 1.35 3 3v6c0 1.65-1.3 3-3 3s-3-1.35-3-3v-6c0-1.65 1.3-3 3-3z" />
              <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0" />
              <path d="M12 17.5V21.5" />
              <path d="M8.5 21.5h7" />
            </svg>

            {/* Adaptive waveform, rendered INSIDE the button. Bars are written
                with DOM scaleY at 60fps so they dance with the voice — React
                doesn't re-render per frame. */}
            <div
              aria-hidden
              className="pointer-events-none absolute inset-0 grid place-items-center"
              style={{
                opacity: isRecording ? 1 : 0,
                transform: `scale(${isRecording ? 1 : 0.9})`,
                transition:
                  "opacity 180ms ease, transform 220ms cubic-bezier(0.22, 1, 0.36, 1)",
              }}
            >
              <div className="flex h-[46px] items-center gap-[3px]">
                {Array.from({ length: WAVE_BARS }).map((_, i) => (
                  <span
                    key={i}
                    ref={(el) => {
                      barRefs.current[i] = el;
                    }}
                    className="block w-[2px] rounded-full"
                    style={{
                      height: 6,
                      background:
                        "linear-gradient(to top, rgba(255,255,255,0.95), rgba(255,255,255,0.8))",
                      transform: "scaleY(1)",
                      transformOrigin: "50% 50%",
                      willChange: "transform",
                      transition: "transform 80ms cubic-bezier(0.22, 1, 0.36, 1)",
                      boxShadow:
                        "0 0 6px rgba(255,255,255,0.55), 0 0 14px rgba(255,137,190,0.35)",
                    }}
                  />
                ))}
              </div>
            </div>
            {talkFlash && (
              <span
                className="pointer-events-none absolute inset-0 grid place-items-center rounded-full"
                style={{ animation: "fade-in 180ms ease-out both" }}
              >
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="#fff"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M5 12l5 5L20 7" />
                </svg>
              </span>
            )}
          </button>
        </div>
        <span
          className={
            "serif-italic rounded-full px-3 py-1 text-[12px] font-medium tracking-wide transition-colors " +
            (isRecording
              ? "bg-[color:var(--accent)] text-white shadow-[0_8px_24px_-12px_rgba(236,72,153,0.7)]"
              : micError
                ? "bg-rose-500/25 text-white ring-1 ring-rose-200/40"
                : "bg-black/30 text-white/90 ring-1 ring-white/15 backdrop-blur-md")
          }
        >
          {isRecording
            ? "listening…"
            : micError
              ? "tap to enable mic"
              : talkFlash
                ? "got it"
                : talkTargetClass
                  ? `hold to talk to the ${talkTargetClass}`
                  : "tap an object first"}
        </span>
      </div>
    </div>
  );
}
