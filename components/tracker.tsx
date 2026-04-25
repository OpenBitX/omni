import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  converseWithObject,
  describeObject,
  generateLine,
  groupLine,
} from "@/lib/api-client";
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
import { CapturePopup, type CaptureItem } from "@/components/capture-popup";
import {
  addSessionCard,
  appendCardHistory,
  cardDisplayName,
  clearSessionCards,
  setCardImageStatus,
  setCardGeneratedImage,
  setCardLanguages,
  setCardTeachMode,
  useSessionCards,
  type SessionCard,
} from "@/lib/session-cards";
import { type AppLang } from "@/lib/lang-detect";
import { readOnboardingPrefs, type Lens } from "@/lib/onboarding";
import { OnboardingOverlay } from "@/components/onboarding-overlay";
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
} from "@/lib/yolo-ws";
import {
  applyAnchor,
  iou,
  makeBox,
  matchTarget,
  newBoxEMA,
  seedBoxEMA,
  smoothBox,
  type Anchor,
  type Box,
  type BoxEMA,
} from "@/lib/iou";
// STT happens server-side. All the client does is record audio with
// MediaRecorder, POST the blob to /api/converse, and render the
// `transcript` field that comes back. No browser-side models, no
// @huggingface/transformers, no onnxruntime-web.

// === Pipeline tuning knobs ===============================================

// Inference rate cap. YOLOv8n on mobile CPU-WASM sits around 3–8 FPS; on
// desktop WebGPU it hits 30+. The cap keeps us from burning battery for
// motion we can already handle with EMA between inferences.
const MAX_INFERENCE_FPS = 30;

// IoU gate for "this new box is the same instance I was tracking". Tuned
// for the backend-WS path where inference runs at ~10-15 fps — at that rate
// a moving object can clear a 0.3 IoU gate in one gap. 0.15 keeps identity
// through realistic motion while still rejecting a sibling object on the
// other side of the frame. Same-class nearest-center fallback widens this
// further after the first miss (see WIDEN_MATCH_AFTER_MISSES).
const IDENTITY_IOU_MIN = 0.15;

// EMA alphas. Position alpha is damped so YOLO box wiggle on stationary
// objects doesn't pass through as face wobble — we want the face to feel
// "engrained" on the product, not jittering around inside the box. Size
// is slower again — bbox edges breathe more than centers do. Opacity
// fades fast so reacquisition and disappearance both feel instant.
const BOX_POS_ALPHA = 0.4;
const BOX_SIZE_ALPHA = 0.2;
const OPACITY_EMA_ALPHA = 0.4;

// "Lost" threshold. One concept, two consequences: (a) the face fades
// out — the object has actually left the frame, not just blinked for one
// inference; (b) the next match SNAPS the smoothed pose instead of EMA
// sliding — avoids a visible glide across the screen as the face fades
// back in at the new location. These MUST be the same number, otherwise
// there's a window where we fade out without snap-on-return (or vice
// versa) and the face visibly slides through the wrong path.
//
// At backend-WS inference rates (~10-15 fps), 4 misses = ~270-400 ms which
// is way too eager — borderline-confidence detections drop out for 2-3
// frames routinely and we'd flash the face every time. 18 misses at
// 15 fps ≈ 1.2 s of hold before declaring the object truly gone, which
// reads as "the face is still there, waiting" instead of "it flashed".
const LOST_AFTER_MISSES = 18;

// Widen matching to same-class + closest-center on the first miss. At
// slower backend inference rates a moving object often leaves the previous
// IoU footprint between frames even when the same-class detection is still
// clearly present — widening eagerly keeps identity intact.
const WIDEN_MATCH_AFTER_MISSES = 1;

// Hold-last-good: reject position updates whose dimensions jumped this much
// vs. the previous smoothed box. Prevents the face from snapping onto a
// transient-but-larger neighbor of the same class during a noisy frame.
// The track keeps its identity and carries on — next inference usually
// gives a clean match.
const SUSPECT_SIZE_RATIO = 1.75;

// Velocity extrapolation. Between YOLO inferences we glide the face at
// 60 fps using per-track velocity, so the face stays pasted on a moving
// object instead of stuttering at the inference rate.
//
// EXTRAP_MAX_MS covers a full inference gap with headroom. At backend-WS
// rates (~10-15 fps, up to ~300 ms between inferences when the model is
// bigger) we need a larger window than the old 220 ms so the face doesn't
// freeze visibly *before* the next detection arrives. Cap at 500 ms — any
// longer and the face would slide through empty space if the object
// vanished. EXTRAP_MISS_LIMIT = 1 means: extrapolate through a single
// missed frame (common case: one brief dip in detection score), then
// freeze in place instead of drifting on stale velocity — this is what
// makes the face look "still until the next frame" the way we want.
const EXTRAP_MAX_MS = 220;
const EXTRAP_MISS_LIMIT = 1;
const VELOCITY_EMA = 0.5;
const VELOCITY_DECAY_PER_MISS = 0.6;

// Velocity deadzone. Below this magnitude (source-pixels / ms) the object
// is considered stationary and velocity is zeroed. Without this, YOLO box
// jitter on a still object produces small non-zero vx/vy which the RAF
// loop then extrapolates every frame, making the face micro-wander inside
// the box. 0.02 px/ms = 20 px/sec is well under intentional motion but
// well above noise on a stationary detection.
const VELOCITY_DEADZONE = 0.02;


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

// Velocity-driven tilt. Rotates the face slightly in the direction the
// box is moving so motion reads as intentional lean rather than rigid
// glide. Target tilt = (vx in box-widths/sec) × gain, clamped to MAX_DEG.
// Smoothed each render with TILT_EMA so it can't snap on a single noisy
// inference; bubble stays upright (only the face transform rotates).
const TILT_GAIN_DEG = 4;
const TILT_MAX_DEG = 10;
const TILT_EMA = 0.12;

// Orientation EMA alpha — smoothed per-inference update of pole offset,
// axis ratio, and principal angle. Lower than BOX_POS_ALPHA on purpose:
// mask PCA and distance-transform output jitter more than bbox centers
// because they're computed over many pixels whose inclusion/exclusion
// flips at the boundary. A heavy EMA keeps the face planted.
const ORIENT_EMA_ALPHA = 0.25;

// Face rotation from the mask's principal axis. Folded to ±π/4 so a
// vertical-long object doesn't sit sideways: PCA on a vertical banana
// returns ~+π/2 which we fold to 0 (upright). Only applied when the
// object is clearly elongated (ORIENT_MIN_RATIO) — round objects have a
// meaningless principal axis.
const ORIENT_MIN_RATIO = 1.35;
const ORIENT_MAX_DEG = 30;

// Face size reduction when the silhouette is elongated. At axisRatio=1
// (round) we use FACE_BBOX_FRACTION straight; at higher ratios we shrink
// toward the *minor* axis so the face fits the narrow dimension instead
// of poking out through the neck of a bottle or the sides of a banana.
// Shrink factor = 1 when ratio ≤ ORIENT_MIN_RATIO, tapering to about 0.55
// at ratio=3 (strongly elongated). Derived from 1/sqrt(ratio/MIN) which
// matches the geometry of rotated rectangles in axis-aligned bboxes.
const ORIENT_SIZE_MIN_FACTOR = 0.55;

// Voice gain is modulated by how much of the box is still in frame (slides
// off → gets quieter) and by box size relative to the viewport (fills frame
// → ULTRA loud). Reference size at which the size-boost equals 1.0, as a
// fraction of min(videoWidth, videoHeight). Exponent >1 makes the growth
// super-linear so "it's RIGHT THERE" actually feels louder. Capped so we
// don't blow the speakers.
const VOICE_SIZE_REF_FRAC = 0.22;
const VOICE_SIZE_EXP = 1.6;
const VOICE_GAIN_MAX = 4.0;

// Voice persistence when tracking drops out. Tracking briefly loses the
// object all the time (half-second occlusions, a single missed frame) — if
// the voice followed opacity directly it'd cut in and out. Hold full audio
// for VOICE_PERSIST_MS after loss, then fade with VOICE_FADE_ALPHA. Face
// still fades on its own schedule; only the audio lingers.
const VOICE_PERSIST_MS = 2000;
const VOICE_FADE_ALPHA = 0.04;

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

// A tap that resolves to a detection matching an existing track (same class,
// IoU over this threshold) is treated as a retap of that track, not a second
// face on the same item. Guards against the user tapping a different region
// of the same object after the smoothed box has drifted off their finger.
const DUPLICATE_IOU_MIN = 0.35;

// Adaptive waveform bar count. Symmetric layout around center; bumping this
// widens the strip but does not retune the bar-to-band mapping (see readLevel).
// Kept small so the strip fits cleanly inside the 64px mic button.
const WAVE_BARS = 11;

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

// Shortest signed delta to `target` along an axis (180°-ambiguous) angle
// space. Used for EMA-smoothing the principal axis: naive `target - current`
// wraps wrong near ±π/2, where e.g. +85° and −85° describe nearly the
// same axis but their raw difference is 170°. Fold to [-π/2, π/2].
function shortestAxisDelta(current: number, target: number): number {
  let d = target - current;
  while (d > Math.PI / 2) d -= Math.PI;
  while (d < -Math.PI / 2) d += Math.PI;
  return d;
}

// Fold a principal-axis angle into a face-rotation value in degrees,
// clamped to ±ORIENT_MAX_DEG. The principal axis is 180°-ambiguous, so a
// PCA return of ~+π/2 (vertical-long object) should not rotate the face
// sideways — we flip to the perpendicular (angle - π/2) when |angle| > π/4,
// bringing vertical objects back toward upright. Net behavior:
//   - horizontal-long (~0°)      → 0° (upright, no tilt)
//   - vertical-long (~±π/2)     → 0° (upright, no tilt)
//   - tilted ~30°                → ~30° (face leans with object)
//   - tilted ~60°                → ~-30° (face leans opposite-and-back to upright side)
// The ±30° clamp then prevents any remaining visual weirdness.
function foldOrientationDeg(angleRad: number): number {
  let a = angleRad;
  while (a > Math.PI / 2) a -= Math.PI;
  while (a < -Math.PI / 2) a += Math.PI;
  if (a > Math.PI / 4) a -= Math.PI / 2;
  else if (a < -Math.PI / 4) a += Math.PI / 2;
  const deg = (a * 180) / Math.PI;
  return Math.max(-ORIENT_MAX_DEG, Math.min(ORIENT_MAX_DEG, deg));
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
  // Audio-only fade. Stays at 1 while present AND through a grace period
  // (VOICE_PERSIST_MS) after loss, then lerps down. Decoupled from opacity
  // so a half-second dropout doesn't punch a hole in the voice.
  audioLevel: number;
  lostSinceMs: number | null;
  lastTapAt: number;
  // Velocity for inter-inference glide (source pixels / ms), plus timestamp
  // of the last observation that moved the smoothed box. Extrapolation at
  // render time uses `now - lastUpdatedAt`, capped by EXTRAP_MAX_MS.
  vx: number;
  vy: number;
  lastUpdatedAt: number;
  // Smoothed lean angle (deg), driven from vx each render. Stored on refs
  // so EMA persists across frames without re-reading the previous UI value.
  tiltDeg: number;
  // Pole offset from the bbox center, in source pixels. The pole (point of
  // inaccessibility — inside pixel farthest from any edge) is the ideal
  // place to plant the face: fitting the widest possible face inside the
  // silhouette without touching boundaries. We store it as an offset from
  // bbox center so it glides correctly during inter-inference velocity
  // extrapolation. EMA-smoothed each observation so PCA/DT noise can't
  // yank the face around.
  poleOffsetX: number;
  poleOffsetY: number;
  // Smoothed principal-axis angle (radians) and major/minor axis ratio.
  // Driven off the seg mask's PCA; folded + clamped at render time so the
  // face never rotates into a sideways posture that would clip it out of a
  // narrow silhouette. Ratio is used to shrink face size for elongated
  // objects (bottle, banana) so the face fits the *minor* axis, not the
  // bbox min — that's the "disappears at edges" failure mode.
  orientAngle: number;
  orientRatio: number;
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
  // Fish.audio reference_id chosen by GLM on the first `generateLine` for
  // this track. Once set, it's reused for every subsequent line and reply
  // so the object's voice stays consistent for the whole session on it.
  // null until the first generateLine returns (or catalog empty → stays null).
  voiceId: string | null;
  // Conversation memory for this object. Populated in order: first line
  // from generateLine → user utterance (server STT transcript) → assistant
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
  // Lens snapshotted at this track's lock time. Changing the pill mid-
  // session does NOT rewrite already-locked tracks — a mug locked in
  // history mode keeps speaking history until it's evicted.
  mode: Lens;
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
  // Small lean (deg) in the direction the box is moving — UX hint only.
  tilt: number;
  // Lifecycle — set by tap/speak flow, not the RAF.
  caption: string | null;
  thinking: boolean;
  speaking: boolean;
};

// Map a speak-path error message to a short, user-friendly toast. Keeps
// the branching out of the already-busy speakOnTrack try/catch and gives
// us a single place to tweak user-facing wording.
function toastForSpeakError(msg: string): string {
  if (/zhipu|glm|api key|api_key|401|403/i.test(msg)) {
    return "voice model unconfigured — check .env.local";
  }
  if (/timed out|timeout/i.test(msg)) {
    return "voice model took too long — tap again to retry";
  }
  if (/decode|audio context|play\(\)/i.test(msg)) {
    return "couldn't play audio on this device — caption only";
  }
  if (/tts stream|no TTS backend|cartesia/i.test(msg)) {
    return "TTS backend failed — caption only";
  }
  return `couldn't speak: ${msg.slice(0, 80)}`;
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
  // Which TTS backend rendered the most recent voice line. Now always
  // `cartesia` on success or `none` on failure — kept as a single piece
  // of state so the diag panel still shows whether audio ran at all.
  const [lastTtsBackend, setLastTtsBackend] = useState<
    "cartesia" | "none" | null
  >(null);
  const [diagOpen, setDiagOpen] = useState(false);
  const [diagError, setDiagError] = useState<string | null>(null);
  const [retryToken, setRetryToken] = useState(0);
  const rejectionTimerRef = useRef<number | null>(null);

  // Session UI language. Drives:
  //   - prompt language (opening line, persona, conversation reply)
  //   - which voices the server is allowed to pick from (en-voices vs zh-voices)
  //   - SpeechRecognition.lang (browser Web Speech) + server STT `language`
  // Mirrored into a ref so async handlers (tap → describe → generateLine →
  // TTS) always read the current value without going stale on re-renders.
  // Spoken / learn language pair. The user speaks whatever `spokenLang`
  // says; `learnLang` is always the opposite (zh ↔ en). The LEARN toggle
  // that used to flip `learnLang` independently is gone — one SPEAK
  // control now drives both, and downstream code reads refs that are
  // kept in sync here.
  //
  // After the global output-language flip (default reply language =
  // spokenLang), all downstream reads that used to consume `langRef.current`
  // (TTS lang, groupLine lang, generateLine lang arg, converse lang field)
  // have been redirected to `spokenLangRef.current`. learnLangRef stays
  // for the teach-mode branches that still need the target language.
  const [spokenLang, setSpokenLang] = useState<AppLang>("zh");
  const learnLang: AppLang = spokenLang === "zh" ? "en" : "zh";
  const spokenLangRef = useRef<AppLang>("zh");
  const learnLangRef = useRef<AppLang>("en");
  useEffect(() => {
    spokenLangRef.current = spokenLang;
    learnLangRef.current = learnLang;
  }, [spokenLang, learnLang]);

  // Active lens. The LENS picker + language/history variants were removed
  // from the UI — the main route now always runs the "play" lens. Each
  // track still snapshots `mode` at lock time (via `TrackRefs.mode`) for
  // downstream prompt routing, so we keep the ref shape intact.
  const mode: Lens = "play";
  const modeRef = useRef<Lens>("play");

  // Lens onboarding overlay. Shown when:
  //   (a) URL has `?onboarding=1` (landing's CTAs add it on every click —
  //       the user asked to start fresh), OR
  //   (b) no prior onboarding exists (first-ever visit, no `completedAt`).
  // Initial lens for step-2 highlight comes from the same URL lens param
  // that seeds `mode` above. Finishing clears both URL params without a
  // navigation (history.replaceState) so back-forward stays clean.
  const [showLensOnboarding, setShowLensOnboarding] = useState(false);
  const [overlayLens, setOverlayLens] = useState<Lens | null>(null);
  useEffect(() => {
    let lensParam: Lens | null = null;
    let forced = false;
    try {
      const url = new URL(window.location.href);
      const qs = url.searchParams.get("lens");
      if (qs === "play" || qs === "language" || qs === "history") {
        lensParam = qs;
      }
      forced = url.searchParams.get("onboarding") === "1";
    } catch {
      // ignore
    }
    const prefs = readOnboardingPrefs();
    const needed = forced || !prefs || !prefs.completedAt;
    if (needed) {
      setOverlayLens(lensParam);
      setShowLensOnboarding(true);
    }
  }, []);
  const handleOnboardingFinished = useCallback(
    (completed: { lens: Lens; spokenLang: "en" | "zh" }) => {
      setShowLensOnboarding(false);
      // Lens is hardcoded to "play" now — onboarding's lens pick no longer
      // wires anywhere. We still sync the spoken language so the UI
      // matches the user's onboarding answer without a second flip.
      setSpokenLang(completed.spokenLang);
      // Suppress the legacy "tap anything" bubble — the lens overlay IS
      // the onboarding now; two consecutive overlays feels like paperwork.
      try {
        localStorage.setItem("tracker:onboarded:v1", "1");
      } catch {
        // ignore
      }
      // Clean the URL — no refresh, just strips `?onboarding=1&lens=…`.
      try {
        const url = new URL(window.location.href);
        url.searchParams.delete("onboarding");
        url.searchParams.delete("lens");
        const clean = url.pathname + (url.search ? `?${url.searchParams.toString()}` : "");
        window.history.replaceState({}, "", clean);
      } catch {
        // ignore
      }
    },
    []
  );

  // Subscribe to the card store so the "collect to gallery" button can
  // reflect per-track generatedImageStatus.
  const sessionCards = useSessionCards();

  // Capture popup state. `pendingCapture` is set the instant a YOLO box
  // is tapped so the popup appears at the same moment the face spawns —
  // not 3–5s later when the VLM finishes. It carries just the crop +
  // class name; the popup renders an "preparing…" disabled-collect state
  // until the matching SessionCard lands in the store, at which point
  // the popup upgrades seamlessly (same trackId, no remount).
  type PendingCapture = {
    trackId: string;
    className: string;
    imageDataUrl: string;
    createdAt: number;
  };
  const [pendingCapture, setPendingCapture] = useState<PendingCapture | null>(
    null
  );
  // Dismissal is keyed by trackId so pending→card upgrades don't revive
  // a dismissed popup.
  const [dismissedTrackIds, setDismissedTrackIds] = useState<
    ReadonlySet<string>
  >(() => new Set());

  // --- Push-to-talk state (UI only, decoupled from per-track voice) -----
  const [isRecording, setIsRecording] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const [talkFlash, setTalkFlash] = useState(false);
  // "you said: …" toast that briefly echoes the server STT transcript so
  // the user has a clear signal their voice message landed.
  const [heardText, setHeardText] = useState<string | null>(null);
  const heardClearTimerRef = useRef<number | null>(null);
  // First-run onboarding card. Starts false so SSR markup matches; a client
  // effect flips it on when localStorage says the user hasn't dismissed it.
  const [showOnboarding, setShowOnboarding] = useState(false);
  // Transient "+ add another" cue. Flashes a hint banner for a couple of
  // seconds after the user taps the explicit add-another button, so the
  // tap-to-add affordance is discoverable without a permanent overlay.
  const [addHintActive, setAddHintActive] = useState(false);
  const addHintTimerRef = useRef<number | null>(null);
  // Group chat: when ≥2 tracks are locked, a scheduler picks whoever has
  // been quiet longest and has them say the next line. Off by default —
  // demos often want a single object first. Tick timer + in-flight guard
  // sit on refs so the scheduler doesn't retrigger on every render.
  const [groupChatEnabled, setGroupChatEnabled] = useState(true);
  const groupHistoryRef = useRef<
    {
      speaker: string;
      trackId: string | null;
      line: string;
      role: "assistant" | "user";
      ts: number;
      // className of the peer this turn was addressed AT (model-emitted,
      // server-validated against the roster). null when musing aloud or
      // talking to the human. The scheduler uses this to bias who speaks
      // next — when A addresses B, B should answer, not whoever's been
      // quiet longest. Without this, banter feels like parallel monologues.
      addressing?: string | null;
    }[]
  >([]);
  const groupLastSpokeAtRef = useRef<Record<string, number>>({});
  const groupTimerRef = useRef<number | null>(null);
  const groupInFlightRef = useRef(false);
  // Invalidation token. External events that change what the NEXT turn
  // should look like (mic press, track add/remove, group toggle, clear)
  // bump this. In-flight prepare / in-flight run check the captured gen
  // on resume and drop if mismatched. runGroupTurn itself does NOT bump —
  // it only reads — so a clean turn isn't self-invalidated.
  const groupGenRef = useRef(0);
  // Single-slot pre-cache for the next scheduled group turn. While the
  // current speaker is playing, we kick off groupLine + full TTS download
  // in the background and stash the resulting Blob here. The next turn
  // pops this, decodes via AudioBufferSource, and plays with near-zero
  // click-to-sound latency. Null when nothing is staged.
  const preparedTurnRef = useRef<{
    trackId: string;
    speakerName: string;
    line: string;
    voiceId: string | null;
    audioBlob: Blob | null; // null if TTS failed + caption-only
    emotion: string | null;
    speed: string | null;
    addressing: string | null; // peer className this line is directed at
    mode: "chat" | "followup";
    gen: number;
    speakerGen: number;
  } | null>(null);
  const preparingRef = useRef(false);
  // Boot-phase elapsed-time counter, shown in the loading overlay so the
  // user can see "hey, this takes a sec" rather than staring at a static
  // spinner. Ticks every 100ms while starting; frozen once we hit ready.
  const bootStartedAtRef = useRef<number>(
    typeof performance !== "undefined" ? performance.now() : 0
  );
  const [bootElapsedMs, setBootElapsedMs] = useState(0);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<BlobPart[]>([]);
  const micStreamRef = useRef<MediaStream | null>(null);
  const recordedBlobRef = useRef<Blob | null>(null);
  // Per-press correlation id + timestamps so we can emit one clean
  // press-to-first-sound summary per turn that's easy to scan in the
  // dev console.
  const turnCounterRef = useRef<number>(0);
  const turnIdRef = useRef<string>("0");
  const turnPressAtRef = useRef<number>(0);
  // Per-tap correlation id (distinct from turn ids above which tag
  // voice-in conversation presses). Every tap mints a new `tN` tag that
  // threads through the speak → VLM → TTS → first-sound logs so one
  // tap's end-to-end timing can be grepped out of interleaved output.
  const tapCounterRef = useRef<number>(0);
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
  // Mirror of tracksUI for read-from-callback paths (mic gate). Refs avoid
  // re-creating useCallbacks on every render-driven UI flip.
  const tracksUIRef = useRef<TrackUI[]>([]);
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

  // --- First-run onboarding ---------------------------------------------
  // Checked once on mount. localStorage key is versioned so we can re-show
  // it cheaply if the flow changes materially.
  useEffect(() => {
    try {
      if (localStorage.getItem("tracker:onboarded:v1") !== "1") {
        setShowOnboarding(true);
      }
    } catch {
      setShowOnboarding(true);
    }
  }, []);

  const dismissOnboarding = useCallback(() => {
    setShowOnboarding(false);
    try {
      localStorage.setItem("tracker:onboarded:v1", "1");
    } catch {
      // private mode / storage disabled — fine, they'll see it again next visit.
    }
  }, []);

  // Toggles picker mode. While active, the breathing detection boxes
  // re-appear (even in the locked phase) so the user can tap another object
  // to lock. Auto-cancels after 12s if they don't tap — generous because
  // the user may be aiming the camera at something new.
  const triggerAddHint = useCallback(() => {
    if (addHintTimerRef.current != null) {
      clearTimeout(addHintTimerRef.current);
      addHintTimerRef.current = null;
    }
    setAddHintActive((was) => {
      if (was) return false;
      addHintTimerRef.current = window.setTimeout(() => {
        setAddHintActive(false);
        addHintTimerRef.current = null;
      }, 12000);
      return true;
    });
  }, []);

  const cancelAddHint = useCallback(() => {
    if (addHintTimerRef.current != null) {
      clearTimeout(addHintTimerRef.current);
      addHintTimerRef.current = null;
    }
    setAddHintActive(false);
  }, []);

  useEffect(() => {
    return () => {
      if (addHintTimerRef.current != null) {
        clearTimeout(addHintTimerRef.current);
        addHintTimerRef.current = null;
      }
    };
  }, []);

  // Tick the boot-elapsed counter while we're still getting ready. Stops
  // the moment the app is interactive so we don't burn a 10Hz interval
  // for nothing.
  useEffect(() => {
    const ready = cameraReady && yoloReady;
    if (ready) {
      setBootElapsedMs(performance.now() - bootStartedAtRef.current);
      return;
    }
    const id = window.setInterval(() => {
      setBootElapsedMs(performance.now() - bootStartedAtRef.current);
    }, 100);
    return () => window.clearInterval(id);
  }, [cameraReady, yoloReady]);

  // --- Camera setup ------------------------------------------------------
  useEffect(() => {
    let stream: MediaStream | null = null;
    (async () => {
      try {
        // Browsers only expose navigator.mediaDevices in a secure context
        // (HTTPS or localhost). Hitting the dev server over http://<LAN-IP>
        // silently drops the API. Surface a useful message instead of the
        // cryptic "Cannot read properties of undefined" stack.
        if (typeof window !== "undefined" && !window.isSecureContext) {
          const host = window.location.host;
          throw new Error(
            `Insecure origin (${window.location.protocol}//${host}). ` +
            `Camera/mic need HTTPS over LAN — visit https://${host} and accept the cert, ` +
            `or add http://${host} to chrome://flags/#unsafely-treat-insecure-origin-as-secure.`
          );
        }
        if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
          throw new Error(
            "navigator.mediaDevices is unavailable in this browser/context. " +
            "If this is a LAN demo, ensure the page is loaded over HTTPS."
          );
        }
        // Ask for camera + mic up front so both browser permission prompts
        // appear on page load. The mic stream is stashed into micStreamRef
        // and reused by the talk button later; if the user denies audio we
        // still try for video-only and fall back to per-press mic access.
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: {
              facingMode: { ideal: "environment" },
              width: { ideal: 1280 },
              height: { ideal: 720 },
            },
            audio: {
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
            },
          });
        } catch (bothErr) {
          // Audio denied (or device missing) — fall back to video-only so
          // the tracker still works caption-first.
          // eslint-disable-next-line no-console
          console.log("[tracker] camera+mic failed, retrying video only:", bothErr);
          stream = await navigator.mediaDevices.getUserMedia({
            video: {
              facingMode: { ideal: "environment" },
              width: { ideal: 1280 },
              height: { ideal: 720 },
            },
            audio: false,
          });
        }
        const audioTracks = stream.getAudioTracks();
        if (audioTracks.length > 0) {
          const micStream = new MediaStream(audioTracks);
          micStreamRef.current = micStream;
        }
        const v = videoRef.current;
        if (!v) return;
        const videoOnly = new MediaStream(stream.getVideoTracks());
        v.srcObject = videoOnly;
        // ``v.play()`` can reject with ``AbortError`` / "interrupted by a
        // new load request" when React StrictMode remounts, Vite HMR
        // tears the effect down mid-promise, or the tab backgrounds before
        // the first frame. We must NOT `return` here without ever calling
        // ``setCameraReady(true)`` — a paused <video> often keeps
        // ``videoWidth === 0``, so the inference RAF bails early and YOLO
        // boxes never appear (looks like "detector disappeared").
        // Retry a few frames; only surface a *persistent* play failure.
        const isBenignPlayFailure = (playErr: unknown) => {
          const name = (playErr as { name?: string } | null)?.name ?? "";
          const msg =
            playErr instanceof Error ? playErr.message : String(playErr);
          return (
            name === "AbortError" || /interrupted by a new load/i.test(msg)
          );
        };
        let playOk = false;
        for (let attempt = 0; attempt < 6 && !playOk; attempt++) {
          try {
            await v.play();
            playOk = true;
          } catch (playErr) {
            if (!isBenignPlayFailure(playErr)) throw playErr;
            // eslint-disable-next-line no-console
            if (attempt === 0) {
              console.log(
                `[tracker] camera play() interrupted — retrying (${playErr instanceof Error ? playErr.message : String(playErr)})`
              );
            }
            await new Promise<void>((r) => requestAnimationFrame(() => r()));
          }
        }
        if (!playOk) {
          throw new Error("camera video would not start (play blocked)");
        }
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

  // --- Foreground/background recovery ------------------------------------
  // Mobile Safari suspends AudioContexts when the tab is backgrounded and
  // pauses the <video> element. Without this, returning to the tab leaves
  // the camera frozen (so YOLO inferences stall) and the next tap plays
  // with a silent analyser (so the mouth stays closed). We don't try to
  // reset track velocity/lastUpdatedAt — EXTRAP_MAX_MS already clamps any
  // post-resume delta, and tracks naturally reacquire via the matcher.
  useEffect(() => {
    if (typeof document === "undefined") return;
    const onVisibilityChange = () => {
      if (document.visibilityState !== "visible") return;
      const ctx = audioCtxRef.current;
      if (ctx && ctx.state === "suspended") {
        ctx.resume().catch((e) => {
          // eslint-disable-next-line no-console
          console.log(
            "[tracker] audio resume on foreground failed:",
            e instanceof Error ? e.message : e
          );
        });
      }
      const v = videoRef.current;
      if (v && v.srcObject && v.paused) {
        v.play().catch((e) => {
          // eslint-disable-next-line no-console
          console.log(
            "[tracker] video play on foreground failed:",
            e instanceof Error ? e.message : e
          );
        });
      }
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    // pageshow fires on back/forward cache restore, which visibilitychange
    // doesn't cover on iOS Safari.
    window.addEventListener("pageshow", onVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", onVisibilityChange);
      window.removeEventListener("pageshow", onVisibilityChange);
    };
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
        // STT is server-side (see /api/transcribe in the backend). Nothing
        // to warm up in the browser — the first mic press pays only a
        // single OpenAI transcription round-trip (~300-600ms on short
        // clips), no local model load.
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
      // Bump speakGen so any in-flight speakOnTrack/playStreamingReply
      // bails at its next guard instead of racing with analyser teardown.
      t.speakGen++;
      stopTrackAudioRef.current?.(t);
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
    // Full-reset also wipes the session gallery — the × button handles
    // per-card removal; this is the "clean slate" demo hard-reset.
    clearSessionCards();
    setDismissedTrackIds(new Set());
    setPendingCapture(null);
    // Bump generation so any in-flight tap's generateLine/TTS drops silent.
    generationRef.current++;
    // Invalidate any staged group turn + in-flight prepare; scene is gone.
    groupGenRef.current++;
    preparedTurnRef.current = null;
  }, []);

  // stopTrackAudio is defined AFTER clearAllTracks (there's a forward ref
  // via describeObject). Park a lazy reference so clearAllTracks can call
  // through without a circular useCallback dep.
  const stopTrackAudioRef = useRef<((t: TrackRefs) => void) | null>(null);

  // Esc key wipes the scene — handy for live demos where you want to reset
  // between "oh look" moments without reloading the page.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (addHintTimerRef.current != null || addHintActive) {
        cancelAddHint();
        return;
      }
      if (tracksRef.current.length > 0) {
        clearAllTracks();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clearAllTracks, cancelAddHint, addHintActive]);

  // --- Audio plumbing ----------------------------------------------------
  const ensureAudioCtx = useCallback(() => {
    let ctx = audioCtxRef.current;
    if (!ctx) {
      const Ctor =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext })
          .webkitAudioContext;
      if (!Ctor) return null;
      try {
        ctx = new Ctor();
      } catch (e) {
        // eslint-disable-next-line no-console
        console.log("[audio] AudioContext construction failed:", e);
        return null;
      }
      audioCtxRef.current = ctx;
    }
    if (ctx.state === "suspended") {
      // Fire-and-forget; retried every subsequent tap. Logged so a silent
      // suspension doesn't vanish into the void.
      ctx.resume().catch((e) => {
        // eslint-disable-next-line no-console
        console.log(
          `[audio] resume failed (state=${ctx!.state}):`,
          e instanceof Error ? e.message : e
        );
      });
    }
    return ctx;
  }, []);

  // Block briefly until the AudioContext is actually `running`. On mobile
  // Safari the resume() triggered in the tap handler can still be pending
  // when we reach the speak path; without this, the analyser feeds silence
  // and the mouth stays closed for the first ~200ms of audio.
  const waitForAudioRunning = useCallback(
    async (ctx: AudioContext, timeoutMs = 800): Promise<boolean> => {
      const isRunning = () => (ctx.state as string) === "running";
      if (isRunning()) return true;
      try {
        await ctx.resume();
      } catch {
        // fall through to poll
      }
      const start = performance.now();
      while (performance.now() - start < timeoutMs) {
        if (isRunning()) return true;
        await new Promise((r) => setTimeout(r, 20));
      }
      return isRunning();
    },
    []
  );

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
    (trackId: string, cropDataUrl: string, tapTag?: string) => {
      const track = tracksRef.current.find((t) => t.id === trackId);
      if (!track || !cropDataUrl) return;
      const gen = ++track.descriptionGen;
      const t0 = performance.now();
      const tag = tapTag ? ` ${tapTag}` : "";
      // eslint-disable-next-line no-console
      console.log(
        `[describe:${trackId}${tag}] → describeObject  class="${track.className}"  crop=${Math.round(cropDataUrl.length / 1024)}KB  gen=${gen}`
      );
      // Describe in the learn-target language so the downstream persona
      // card + retap lines stay consistent with what the user wants to
      // hear (e.g. learnLang=en → object description is in English even
      // when the user speaks Chinese).
      describeObject(cropDataUrl, track.className, spokenLangRef.current, tapTag)
        .then(({ description }) => {
          // Track may have been evicted while we waited; bail if so.
          const current = tracksRef.current.find((t) => t.id === trackId);
          if (!current) return;
          // A fresher describe kicked off while this one was in flight.
          if (gen !== current.descriptionGen) {
            // eslint-disable-next-line no-console
            console.log(
              `[describe:${trackId}${tag}] ← superseded (gen ${gen} vs ${current.descriptionGen})`
            );
            return;
          }
          current.description = description || null;
          // eslint-disable-next-line no-console
          console.log(
            `[describe:${trackId}${tag}] ← ${Math.round(performance.now() - t0)}ms  "${(description || "").slice(0, 100)}"`
          );
        })
        .catch((err) => {
          // Non-fatal — converseWithObject just falls back to the bare class.
          // eslint-disable-next-line no-console
          console.log(
            `[describe:${trackId}${tag}] ✖ ${err instanceof Error ? err.message : String(err)}`
          );
        });
    },
    []
  );

  // Stop whatever a track is currently saying. One code path — the
  // AudioBufferSource — because all TTS (first-tap, retap, converse,
  // group) decodes the full mp3 once and plays through the same node.
  // Called from retap, eviction, mic press, etc.
  const stopTrackAudio = useCallback((track: TrackRefs) => {
    const ctx = audioCtxRef.current;
    // Brief gain ramp to 0 kills the click that otherwise fires when a
    // hard source.stop() lands mid-waveform. ~8ms is below the perceptual
    // threshold but above the single-frame click window. The RAF loop
    // restores gain the next time audio plays.
    if (track.gain && ctx) {
      try {
        const now = ctx.currentTime;
        track.gain.gain.cancelScheduledValues(now);
        track.gain.gain.setTargetAtTime(0, now, 0.004);
      } catch {
        // ignore — scheduling failures are not fatal
      }
    }
    if (track.source) {
      try {
        track.source.onended = null;
      } catch {
        // ignore
      }
      try {
        track.source.stop();
      } catch {
        // already stopped
      }
      try {
        track.source.disconnect();
      } catch {
        // already disconnected
      }
      track.source = null;
    }
  }, []);

  // Park a forward reference to stopTrackAudio so clearAllTracks (defined
  // earlier in the file) can call it without needing to be reordered.
  useEffect(() => {
    stopTrackAudioRef.current = stopTrackAudio;
    return () => {
      stopTrackAudioRef.current = null;
    };
  }, [stopTrackAudio]);

  // --- Streaming TTS playback -------------------------------------------
  //
  // Unified TTS playback for every path (first-tap, retap, converse,
  // group). Fetches the mp3 from `/api/tts/stream` (or reuses a body
  // already opened by `/api/speak`), buffers to an ArrayBuffer, decodes
  // once, and plays through an AudioBufferSource wired to this track's
  // analyser + gain. One code path = one bug surface.
  //
  // This used to have a second path (MediaSource + SourceBuffer + an
  // HTMLAudioElement) for sub-second TTFB streaming. It was fragile —
  // retap/converse/group turns after the first kept dropping audio
  // because of sourceopen races, orphaned MediaElementSource nodes, and
  // piecemeal teardown between retaps. Deleted in favour of this one.
  // We pay ~300–500ms of extra latency per turn and get audio that
  // actually plays every single time.
  const playStreamingReply = useCallback(
    async (
      trackId: string,
      callGen: number,
      text: string,
      voiceId: string | null,
      turnId?: string,
      onFirstSound?: () => void,
      emotion?: string | null,
      speed?: string | null,
      // Optional pre-opened response body + backend label. When the
      // caller already has an audio stream in hand (e.g. from the
      // `/api/speak` combined endpoint that folds VLM + TTS into one
      // response), skip the redundant `/api/tts/stream` POST.
      preopened?: { respBody: ReadableStream<Uint8Array>; backend: string }
    ): Promise<void> => {
      const ctx = ensureAudioCtx();
      const track = tracksRef.current.find((t) => t.id === trackId);
      if (!ctx || !track) return;
      const tid = turnId ?? "?";

      // Reset lip-sync state for this new reply. Peak calibrates to the
      // incoming stream's actual loudness; without this, a previous loud
      // line leaves peak high and the quieter reply shows as closed mouth.
      track.lipSync = createLipSyncState();

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

      // Wait for the ctx to be `running` before we start the network leg.
      // On mobile, issuing the POST first and THEN resuming adds a race
      // where the stream arrives while ctx is still suspended.
      const running = await waitForAudioRunning(ctx);
      if (!running) {
        throw new Error(`audio context not running (state=${ctx.state})`);
      }
      if (callGen !== track.speakGen) return;

      let respBody: ReadableStream<Uint8Array>;
      let backend: string;
      if (preopened) {
        respBody = preopened.respBody;
        backend = preopened.backend;
        // eslint-disable-next-line no-console
        console.log(
          `[tts #${tid}] ◀ reusing preopened stream backend=${backend}`
        );
      } else {
        // eslint-disable-next-line no-console
        console.log(
          `[tts #${tid}] → /api/tts/stream text=${text.length}ch voice=${voiceId ?? "default"}`
        );
        const resp = await fetch("/api/tts/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text,
            voiceId: voiceId ?? "",
            turnId: tid,
            lang: spokenLangRef.current,
            emotion: emotion ? [emotion] : [],
            speed: speed ?? null,
          }),
        });
        if (callGen !== track.speakGen) {
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
        respBody = resp.body;
        backend = resp.headers.get("X-Tts-Backend") ?? "stream";
        // eslint-disable-next-line no-console
        console.log(
          `[tts #${tid}] ◀ headers in ${Math.round(performance.now() - t0)}ms backend=${backend}`
        );
      }
      setLastTtsBackend(backend === "cartesia" ? "cartesia" : "none");

      // Buffer the full mp3, decode once, then hand to the AudioBufferSource
      // pipeline. This is the same pipeline `playPreparedTurn` uses for
      // group turns — unifying the two kept the "works first time, silent
      // after" retap/converse bugs from coming back.
      const buf = await new Response(respBody).arrayBuffer();
      if (callGen !== track.speakGen) return;
      // eslint-disable-next-line no-console
      console.log(
        `[tts #${tid}] ◀ body buffered ${Math.round(buf.byteLength / 1024)}KB in ${Math.round(performance.now() - t0)}ms`
      );

      let audioBuf: AudioBuffer;
      try {
        audioBuf = await ctx.decodeAudioData(buf);
      } catch (e) {
        throw new Error(
          `decodeAudioData failed: ${e instanceof Error ? e.message : String(e)}`
        );
      }
      if (callGen !== track.speakGen) return;

      // Lazy analyser + gain. If they already exist, restore gain (it
      // may have been ramped to 0 by stopTrackAudio during this same
      // turn's interrupt of a previous one). The RAF loop then drives
      // gain from opacity * visibility * size-boost per frame.
      if (!track.analyser) {
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 1024;
        analyser.smoothingTimeConstant = 0.4;
        const gain = ctx.createGain();
        gain.gain.value = Math.max(0, track.opacity || 1);
        analyser.connect(gain);
        gain.connect(ctx.destination);
        track.analyser = analyser;
        track.gain = gain;
        track.freqData = new Uint8Array(
          new ArrayBuffer(analyser.frequencyBinCount)
        );
        track.timeData = new Uint8Array(new ArrayBuffer(analyser.fftSize));
      } else if (track.gain) {
        try {
          const now = ctx.currentTime;
          track.gain.gain.cancelScheduledValues(now);
          track.gain.gain.setValueAtTime(
            Math.max(0, track.opacity || 1),
            now
          );
        } catch {
          // scheduling failures are not fatal
        }
      }
      if (!track.analyser) {
        throw new Error("analyser unavailable (AudioContext state?)");
      }

      const source = ctx.createBufferSource();
      source.buffer = audioBuf;
      source.connect(track.analyser);

      // Resolve when this utterance finishes so callers that `await` the
      // play can sequence follow-ups (e.g. the group turn scheduler
      // triggering the next prepare after the current turn ends).
      const ended = new Promise<void>((resolve) => {
        source.onended = () => {
          if (track.source === source) {
            try {
              source.disconnect();
            } catch {
              // ignore
            }
            track.source = null;
            setTracksUI((prev) =>
              prev.map((t) =>
                t.id === trackId ? { ...t, speaking: false } : t
              )
            );
            scheduleCaptionClear();
          }
          resolve();
        };
      });

      try {
        source.start();
      } catch (e) {
        try {
          source.disconnect();
        } catch {
          // ignore
        }
        throw new Error(
          `source.start failed: ${e instanceof Error ? e.message : String(e)}`
        );
      }
      // Publish speaking state only after start() succeeds, so an error
      // path can't leave speaking=true wedged.
      track.source = source;
      setTracksUI((prev) =>
        prev.map((t) => (t.id === trackId ? { ...t, speaking: true } : t))
      );
      try {
        onFirstSound?.();
      } catch (cbErr) {
        // eslint-disable-next-line no-console
        console.log(
          `[tts #${tid}] onFirstSound threw: ${cbErr instanceof Error ? cbErr.message : String(cbErr)}`
        );
      }
      await ended;
    },
    [ensureAudioCtx, waitForAudioRunning]
  );

  // --- Per-track speak ---------------------------------------------------
  //
  // Each track owns its own audio source + analyser, so three objects can
  // talk over each other at the same time and their mouths each sync to
  // their own line. The trackId closes over which track owns the result.
  const speakOnTrack = useCallback(
    async (
      trackId: string,
      cropDataUrl: string,
      tapCtx?: { tapId: string; pressedAt: number }
    ) => {
      const ctx = ensureAudioCtx();
      const track = tracksRef.current.find((t) => t.id === trackId);
      if (!ctx || !track) return;

      // Correlation tag for this tap's end-to-end logs. `#tN` distinguishes
      // from conversation turn ids (`#N`). If no tapCtx (e.g. a retap path
      // that didn't mint one), fall back to a synthesized id so the logs
      // still carry something greppable.
      const tapId =
        tapCtx?.tapId ?? `t${++tapCounterRef.current}-auto`;
      const tapTag = `#${tapId}`;
      const pressedAt = tapCtx?.pressedAt ?? performance.now();
      const isRetap = !!track.voiceId && !!track.description;

      // Bump this track's speak gen so an in-flight call from a previous
      // tap on the SAME track drops its result when it resolves late.
      const callGen = ++track.speakGen;

      // Retap cancels any pending caption auto-dismiss — the new line wants
      // its full linger window.
      if (track.captionClearTimer != null) {
        clearTimeout(track.captionClearTimer);
        track.captionClearTimer = null;
      }

      // Stop whatever this track was already saying. Other tracks keep
      // going — concurrent voices are a feature.
      stopTrackAudio(track);
      // UI: thinking=true, speaking=false, caption cleared.
      setTracksUI((prev) =>
        prev.map((t) =>
          t.id === trackId
            ? { ...t, thinking: true, speaking: false, caption: null }
            : t
        )
      );

      // How long it took from physical tap to actually reaching the
      // generateLine call. Includes detect→lock, crop capture, and any
      // React state flushes. Usually <100ms on desktop; can balloon on
      // mobile if an inference is contending.
      const pressToSpeakMs = Math.round(performance.now() - pressedAt);

      // Captured once generateLine returns so the catch block below can
      // still surface the caption if the TTS streaming leg fails —
      // otherwise a stream error after we have a valid line would leave
      // the user with a toast and no bubble.
      let capturedLine: string | null = null;

      try {
        // eslint-disable-next-line no-console
        console.log(
          `[speak:${trackId} ${tapTag}] → ${isRetap ? "generateLine (retap, text-only)" : "/api/speak (bundled VLM + TTS in one response)"}  class="${track.className}"  crop=${Math.round(cropDataUrl.length / 1024)}KB  voice=${track.voiceId ?? "(picking)"}  persona=${track.description ? "cached" : "(new)"}  history=${track.history.length}  press→speak=${pressToSpeakMs}ms`
        );
        const t0 = performance.now();
        // Hard cap on the whole line-generation step — a hung vision call
        // must not strand the UI on "thinking=true" forever.
        // Covers VLM + Cartesia TTFB end-to-end on /api/speak. Long
        // enough to ride out a slow Cartesia start instead of aborting
        // a request that was about to succeed (the main cause of "no
        // audio at all" on first tap).
        const GENERATE_LINE_TIMEOUT_MS = 120_000;

        // First-tap: use /api/speak to fold VLM + Cartesia into one
        // response. Server fires TTS the instant `line` is parsed from the
        // VLM and returns audio/mpeg with metadata in the X-Speak-Meta
        // header, so caption + audio surface together (saves ~600ms of
        // client-side round-trip dead air). Retap stays on the old
        // server-action path — it's already text-only and sub-1s.
        let line: string;
        let chosenVoiceId: string | null;
        let chosenDescription: string | null;
        // VLM-emitted short name for this specific object. Hoisted so
        // both branches (first-tap bundled + retap text-only) can stash
        // it onto the new SessionCard below.
        let chosenName: string | null = null;
        let preopenedForTts:
          | { respBody: ReadableStream<Uint8Array>; backend: string }
          | undefined;
        let generateLineMs: number;

        if (!isRetap) {
          const speakCtrl = new AbortController();
          const speakTimer = setTimeout(
            () =>
              speakCtrl.abort(
                new Error(
                  `/api/speak timed out after ${Math.round(
                    GENERATE_LINE_TIMEOUT_MS / 1000
                  )}s`
                )
              ),
            GENERATE_LINE_TIMEOUT_MS
          );
          let speakResp: Response;
          try {
            speakResp = await fetch("/api/speak", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                imageDataUrl: cropDataUrl,
                voiceId: track.voiceId,
                description: track.description,
                history: track.history.slice(-32),
                lang: spokenLangRef.current,
                spokenLang: spokenLangRef.current,
                learnLang: learnLangRef.current,
                mode: track.mode,
                turnId: tapTag.replace(/^#/, "").slice(0, 16),
              }),
              signal: speakCtrl.signal,
            });
          } finally {
            clearTimeout(speakTimer);
          }
          if (callGen !== track.speakGen) {
            try {
              await speakResp.body?.cancel();
            } catch {
              // ignore
            }
            // eslint-disable-next-line no-console
            console.log(
              `[speak:${trackId} ${tapTag}] ← superseded (speakGen mismatch)`
            );
            return;
          }
          if (!speakResp.ok && speakResp.status !== 204) {
            const errBody = await speakResp.text().catch(() => "");
            throw new Error(
              `/api/speak ${speakResp.status}: ${errBody.slice(0, 160)}`
            );
          }
          const metaB64 = speakResp.headers.get("X-Speak-Meta") ?? "";
          if (!metaB64) {
            try {
              await speakResp.body?.cancel();
            } catch {
              // ignore
            }
            throw new Error("/api/speak missing X-Speak-Meta header");
          }
          let meta: {
            line?: string;
            voiceId?: string | null;
            description?: string | null;
            name?: string | null;
          };
          try {
            // Decode as UTF-8 — the server base64-encodes UTF-8 bytes, and
            // `atob` alone returns a Latin-1 binary string that mangles
            // non-ASCII (Chinese chars become mojibake like "ä¸­").
            let metaJson: string;
            if (typeof atob === "function") {
              const bin = atob(metaB64);
              const bytes = new Uint8Array(bin.length);
              for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
              metaJson = new TextDecoder("utf-8").decode(bytes);
            } else {
              metaJson = Buffer.from(metaB64, "base64").toString("utf-8");
            }
            meta = JSON.parse(metaJson);
          } catch (parseErr) {
            try {
              await speakResp.body?.cancel();
            } catch {
              // ignore
            }
            throw new Error(
              `/api/speak meta parse failed: ${parseErr instanceof Error ? parseErr.message : String(parseErr)}`
            );
          }
          if (typeof meta.line !== "string" || !meta.line.trim()) {
            try {
              await speakResp.body?.cancel();
            } catch {
              // ignore
            }
            throw new Error("/api/speak meta has no line");
          }
          line = meta.line;
          chosenVoiceId =
            typeof meta.voiceId === "string" && meta.voiceId.trim()
              ? meta.voiceId.trim()
              : null;
          chosenDescription =
            typeof meta.description === "string" && meta.description.trim()
              ? meta.description.trim()
              : null;
          chosenName =
            typeof meta.name === "string" && meta.name.trim()
              ? meta.name.trim()
              : null;
          // 204 = no TTS backend configured. Caption-only degraded mode;
          // leave preopenedForTts undefined so we don't try to stream.
          if (speakResp.status === 200 && speakResp.body) {
            preopenedForTts = {
              respBody: speakResp.body,
              backend:
                speakResp.headers.get("X-Tts-Backend") ?? "stream",
            };
          }
          generateLineMs = Math.round(performance.now() - t0);
        } else {
          // Retap path: text-only generateLine, then /api/tts/stream via
          // playStreamingReply (unchanged).
          const result = await Promise.race([
            // Signature extended with the spoken/learn language pair.
            // When both are provided the retap prompt biases output
            // toward `learnLang` so the user gets practice input.
            generateLine(
              cropDataUrl,
              track.voiceId,
              track.description,
              track.history.slice(-32),
              spokenLangRef.current,
              tapTag,
              spokenLangRef.current,
              learnLangRef.current,
              track.mode
            ),
            new Promise<never>((_, reject) =>
              setTimeout(
                () =>
                  reject(
                    new Error(
                      `generateLine timed out after ${Math.round(
                        GENERATE_LINE_TIMEOUT_MS / 1000
                      )}s`
                    )
                  ),
                GENERATE_LINE_TIMEOUT_MS
              )
            ),
          ]);
          if (callGen !== track.speakGen) {
            // eslint-disable-next-line no-console
            console.log(
              `[speak:${trackId} ${tapTag}] ← superseded (speakGen mismatch)`
            );
            return;
          }
          generateLineMs = Math.round(performance.now() - t0);
          line = result.line;
          chosenVoiceId = result.voiceId;
          chosenDescription = result.description;
          chosenName = result.name ?? null;
        }
        capturedLine = line;
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
        // Session gallery: on the first tap of each track (the /api/speak
        // branch runs only when !isRetap) save a card capturing everything
        // the VLM just decided — crop, persona, voice, and opening line.
        // The card lives in a sessionStorage-backed store so the /gallery
        // route (a separate page) can read it without a shared parent.
        // Guarded by voiceId + description + line so a degraded no-backend
        // run doesn't fill the gallery with half-built cards.
        if (
          !isRetap &&
          chosenVoiceId &&
          chosenDescription &&
          typeof line === "string" &&
          line.trim()
        ) {
          const newCard: SessionCard = {
            id: `card-${trackId}-${Date.now()}`,
            trackId,
            createdAt: Date.now(),
            className: track.className,
            objectName: chosenName ?? undefined,
            description: chosenDescription,
            voiceId: chosenVoiceId,
            line,
            imageDataUrl: cropDataUrl,
            spokenLang: spokenLangRef.current,
            learnLang: learnLangRef.current,
            mode: track.mode,
          };
          addSessionCard(newCard);
          // Seed the card's persisted history with the opening line so
          // the Runware prompt builder has context even if no conversation
          // happens before the user hits collect.
          appendCardHistory({ trackId }, [
            { role: "assistant", content: line },
          ]);
        }
        // Record what the object said into its own memory so a later
        // conversation turn can call back to it.
        track.history = [
          ...track.history,
          { role: "assistant" as const, content: line },
        ].slice(-32);
        // Also seed the shared group transcript so the next group turn sees
        // the opening line as context — otherwise peers don't know what was
        // just said and the first group line ignores the opener. Sets
        // lastSpokeAt so the scheduler's silence clock starts ticking now.
        {
          const nowTs = performance.now();
          groupHistoryRef.current.push({
            speaker: track.className,
            trackId,
            line,
            role: "assistant",
            ts: nowTs,
          });
          groupHistoryRef.current = groupHistoryRef.current.slice(-48);
          groupLastSpokeAtRef.current[trackId] = nowTs;
        }
        // eslint-disable-next-line no-console
        console.log(
          `[speak:${trackId} ${tapTag}] ← generateLine=${generateLineMs}ms  voice=${track.voiceId ?? "default"}  line="${line}"`
        );

        // Show caption + stop spinner as soon as we have the line. The
        // buffered TTS path adds ~300–500ms between caption and first
        // audible sample — the trade for a reliable playback pipeline
        // that doesn't silently drop on retap/converse/group.
        setTracksUI((prev) =>
          prev.map((t) =>
            t.id === trackId ? { ...t, caption: line, thinking: false } : t
          )
        );

        // Hand off to the unified playback path — fetch (or reuse the
        // /api/speak body), decode once, play via AudioBufferSource.
        const tStream = performance.now();
        let firstSoundMs = 0;
        // First-tap path passes the already-open audio body from
        // /api/speak so we skip the redundant /api/tts/stream POST.
        if (preopenedForTts) {
          await playStreamingReply(
            trackId,
            callGen,
            line,
            track.voiceId,
            tapTag.replace(/^#/, ""),
            () => {
              if (firstSoundMs === 0) {
                firstSoundMs = Math.round(performance.now() - pressedAt);
              }
            },
            undefined,
            undefined,
            preopenedForTts
          );
        } else {
          await playStreamingReply(
            trackId,
            callGen,
            line,
            track.voiceId,
            tapTag.replace(/^#/, ""),
            () => {
              if (firstSoundMs === 0) {
                firstSoundMs = Math.round(performance.now() - pressedAt);
              }
            }
          );
        }
        if (callGen !== track.speakGen) return;
        const streamMs = Math.round(performance.now() - tStream);
        // End-of-tap summary — ties every phase from tap press to first
        // audible sample. Layout:
        //   press→speak   — tap bookkeeping before generateLine ran
        //   generateLine  — server action round trip (VLM or text-only LLM)
        //   tts-stream    — from generateLine return to playback ended
        //   first-sound   — press → first audible PCM sample (the only
        //                   number the user actually feels)
        // eslint-disable-next-line no-console
        console.log(
          `[tap ${tapTag}] ◀ FIRST SOUND in ${firstSoundMs || streamMs}ms  ━  kind=${isRetap ? "retap" : "first-tap"}  ▸ press→speak=${pressToSpeakMs}ms  ▸ generateLine=${generateLineMs}ms  ▸ tts-stream=${streamMs}ms`
        );
      } catch (e) {
        if (callGen !== track.speakGen) return;
        const msg = e instanceof Error ? e.message : "line failed";
        setErrorMsg(msg);
        setDiagError(msg);
        // If we got as far as a line but the TTS stream failed, still
        // surface the caption so the user sees what the object would've
        // said (caption-only degradation). Otherwise just stop the
        // spinner — generateLine itself failed and there's nothing to show.
        setTracksUI((prev) =>
          prev.map((t) =>
            t.id === trackId
              ? {
                  ...t,
                  thinking: false,
                  speaking: false,
                  caption: capturedLine ?? t.caption,
                }
              : t
          )
        );
        showRejection(toastForSpeakError(msg), 3200);
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
    [
      ensureAudioCtx,
      playStreamingReply,
      showRejection,
      stopTrackAudio,
      waitForAudioRunning,
    ]
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
        const lost = t.missedFrames >= LOST_AFTER_MISSES;
        const target = lost ? 0 : 1;
        t.opacity = lerp(t.opacity, target, OPACITY_EMA_ALPHA);
        // Audio-only fade: hold full volume through brief dropouts so a
        // half-second tracking blip doesn't pop the voice. Only after
        // VOICE_PERSIST_MS of continuous loss does the voice start fading.
        if (lost) {
          if (t.lostSinceMs == null) t.lostSinceMs = now;
          if (now - t.lostSinceMs >= VOICE_PERSIST_MS) {
            t.audioLevel = lerp(t.audioLevel, 0, VOICE_FADE_ALPHA);
          }
        } else {
          t.lostSinceMs = null;
          t.audioLevel = 1;
        }
        // Gain = audio-level × visible-area fraction (slides off frame →
        // quieter) × super-linear size boost (fills frame → ULTRA loud).
        // Capped at VOICE_GAIN_MAX. NaN/∞ guards everywhere — a malformed
        // frame (videoWidth=0, zero box) would otherwise assign NaN to
        // gain.value, which silences the voice and, in some browsers,
        // produces a click when the RAF recovers.
        if (t.gain) {
          const b = t.smoothedBox;
          const vw = v.videoWidth || 0;
          const vh = v.videoHeight || 0;
          const visW = Math.max(0, Math.min(b.x2, vw) - Math.max(b.x1, 0));
          const visH = Math.max(0, Math.min(b.y2, vh) - Math.max(b.y1, 0));
          const area = b.w * b.h;
          const visibleFrac =
            area > 0 && Number.isFinite(area) ? (visW * visH) / area : 0;
          const refPx = VOICE_SIZE_REF_FRAC * Math.min(vw || 1, vh || 1);
          const sizeFrac =
            refPx > 0 ? Math.min(b.w, b.h) / refPx : 1;
          const safeSizeFrac = Number.isFinite(sizeFrac) ? sizeFrac : 0.05;
          const sizeBoost = Math.pow(
            Math.max(safeSizeFrac, 0.05),
            VOICE_SIZE_EXP
          );
          const raw = t.audioLevel * visibleFrac * sizeBoost;
          const g = Number.isFinite(raw) ? raw : 0;
          const clamped = Math.max(0, Math.min(g, VOICE_GAIN_MAX));
          // Only assign if the previous schedule isn't a mid-ramp (set
          // by stopTrackAudio). A direct `.value =` would cancel the ramp
          // and cause the click we're trying to avoid. We detect a mid-
          // ramp by seeing if the source is null (stopTrackAudio runs
          // when source is being torn down) — if source is active, we're
          // the ones driving the gain.
          if (t.source != null) {
            t.gain.gain.value = clamped;
          }
        }
      }

      // (3) Per-track mouth-shape classification. One gate — `t.source`
      // is the AudioBufferSource driving every line (first-tap, retap,
      // converse, group). `classifyShapeSmooth` keeps an envelope +
      // adaptive peak in `t.lipSync` so openness is normalized to this
      // utterance's actual level (quiet voices still open the mouth)
      // and frame-to-frame flicker is filtered out.
      for (const t of tracksRef.current) {
        let shape: MouthShape = "X";
        const audible = t.source != null;
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
          const anchorPoint = applyAnchor(t.anchor, renderBox);
          // Pole offset plants the face at the max-inset interior point of
          // the silhouette — the point that can fit the biggest face
          // without touching any edge. This is the core fix for the
          // "face clipped at edges" problem: centroid-based placement
          // lands near boundaries on crescent/asymmetric shapes; the
          // pole does not, by construction.
          const facePoint = {
            x: anchorPoint.x + t.poleOffsetX,
            y: anchorPoint.y + t.poleOffsetY,
          };
          const el = sourceToElementPoint(facePoint, v);
          if (!el) return ui;
          const left = el.clientX - rect.left;
          const top = el.clientY - rect.top;
          const minSide = Math.min(renderBox.w, renderBox.h);
          // Shrink face for elongated silhouettes so it fits the MINOR
          // axis, not the bbox min. orientRatio = 1 → no shrink; ratio = 2
          // → ~0.71 of bbox min; ratio = 3 → ~0.58. Clamp to
          // ORIENT_SIZE_MIN_FACTOR so very elongated objects still get a
          // readable face. Below ORIENT_MIN_RATIO (round-ish objects), no
          // shrink since axis ratio is meaningless there.
          const r = Math.max(1, t.orientRatio);
          const sizeFactor =
            r <= ORIENT_MIN_RATIO
              ? 1
              : Math.max(ORIENT_SIZE_MIN_FACTOR, Math.sqrt(ORIENT_MIN_RATIO / r));
          const targetPx = minSide * FACE_BBOX_FRACTION * sizeFactor * srcToElAvg;
          const scale = Math.max(
            FACE_SCALE_MIN,
            Math.min(FACE_SCALE_MAX, targetPx / FACE_NATIVE_PX)
          );
          const opacity = t.opacity;
          const shape = t.shape;

          // Lean toward direction of motion. vx is source-px/ms; convert to
          // box-widths/sec, scale to degrees, clamp, then EMA so a single
          // jumpy inference can't whip the head around. Decays to 0 once
          // velocity dies.
          const boxWidthsPerSec = renderBox.w > 0 ? (t.vx * 1000) / renderBox.w : 0;
          const targetTilt = Math.max(
            -TILT_MAX_DEG,
            Math.min(TILT_MAX_DEG, boxWidthsPerSec * TILT_GAIN_DEG)
          );
          t.tiltDeg = t.tiltDeg + TILT_EMA * (targetTilt - t.tiltDeg);
          // Compose motion lean + silhouette orientation. Orientation only
          // contributes when the silhouette is meaningfully elongated;
          // foldOrientationDeg keeps vertical objects upright (the failure
          // mode the old pole-anchor version suffered).
          const orientDeg =
            t.orientRatio >= ORIENT_MIN_RATIO ? foldOrientationDeg(t.orientAngle) : 0;
          const tilt = Math.round((t.tiltDeg + orientDeg) * 10) / 10;

          if (
            ui.left === left &&
            ui.top === top &&
            ui.scale === scale &&
            ui.opacity === opacity &&
            ui.shape === shape &&
            ui.tilt === tilt
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
            tilt,
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

        // Identity is pinned from the VLM at lock time — don't let YOLO
        // class flips drop the track. Match by geometry (IoU + center).
        let match: Detection | null = matchTarget(
          candidates,
          { ...t.smoothedBox, classId: t.classId },
          IDENTITY_IOU_MIN,
          true
        );

        // Widen to closest-center after a few misses — the object may have
        // moved across the frame between inferences.
        if (!match && t.missedFrames >= WIDEN_MATCH_AFTER_MISSES) {
          let nearest: Detection | null = null;
          let nearestD = Infinity;
          for (const d of candidates) {
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

            // Prefer the mask centroid — "middle of the actual pixels",
            // more stable than bbox center and unbiased by appendages
            // (handles, spouts, stands). The pole offset is tracked
            // separately below so the face can sit at the max-inset
            // interior point while the smoothed box still follows the
            // mean pixel position.
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
                const nextVx = t.vx * (1 - VELOCITY_EMA) + rawVx * VELOCITY_EMA;
                const nextVy = t.vy * (1 - VELOCITY_EMA) + rawVy * VELOCITY_EMA;
                // Deadzone: if both components are below noise floor, treat
                // the object as stationary. Keeps the face perfectly still
                // on still objects instead of micro-drifting between frames.
                const mag = Math.hypot(nextVx, nextVy);
                if (mag < VELOCITY_DEADZONE) {
                  t.vx = 0;
                  t.vy = 0;
                } else {
                  t.vx = nextVx;
                  t.vy = nextVy;
                }
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

            // Face anchor is {rx:0, ry:0} — render adds `poleOffset` on
            // top, so the face sits at the pole (max-inset interior point)
            // rather than the bbox/centroid center. This is the cure for
            // "face disappears at the edges": the pole guarantees the face
            // is planted in the widest part of the silhouette by
            // construction (it's the point with the largest inscribed
            // circle). An earlier attempt rotated the face by principal
            // axis with no fold and landed sideways on vertical objects —
            // the fold in the render path (±30°, with a perpendicular flip
            // for > 45° PCA returns) is what makes that robust now.
            if (match.maskPole) {
              const rawOffX = match.maskPole.x - t.smoothedBox.cx;
              const rawOffY = match.maskPole.y - t.smoothedBox.cy;
              // Clamp offset to stay inside the bbox — guards against a
              // noisy mask putting the pole outside the detection.
              const halfW = t.smoothedBox.w * 0.5;
              const halfH = t.smoothedBox.h * 0.5;
              const cOffX = Math.max(-halfW, Math.min(halfW, rawOffX));
              const cOffY = Math.max(-halfH, Math.min(halfH, rawOffY));
              if (wasLost) {
                t.poleOffsetX = cOffX;
                t.poleOffsetY = cOffY;
              } else {
                const a = ORIENT_EMA_ALPHA;
                t.poleOffsetX = t.poleOffsetX * (1 - a) + cOffX * a;
                t.poleOffsetY = t.poleOffsetY * (1 - a) + cOffY * a;
              }
            }
            if (typeof match.axisRatio === "number") {
              if (wasLost) {
                t.orientRatio = match.axisRatio;
              } else {
                const a = ORIENT_EMA_ALPHA;
                t.orientRatio = t.orientRatio * (1 - a) + match.axisRatio * a;
              }
            }
            if (typeof match.principalAngle === "number") {
              // Shortest-path EMA on angles — raw blend wraps wrong near
              // ±π/2 (principal axis is 180° ambiguous so the target lies
              // on whichever is closer).
              const target = shortestAxisDelta(t.orientAngle, match.principalAngle);
              if (wasLost) {
                t.orientAngle = match.principalAngle;
              } else {
                const a = ORIENT_EMA_ALPHA;
                t.orientAngle = t.orientAngle + target * a;
              }
            }
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
    async (
      trackId: string,
      blob: Blob,
      clientTranscript?: string,
      turnId?: string
    ) => {
      const ctx = ensureAudioCtx();
      const track = tracksRef.current.find((t) => t.id === trackId);
      if (!ctx || !track) return;
      // Tag every log line for this turn with the press-incremented id so
      // the dev console can be filtered to one user interaction at a time.
      const tid = turnId ?? "?";

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
        // Prefer the VLM's specific name from the card store; fall back
        // to the YOLO class only when the card hasn't landed yet. The
        // server uses this as the subject label in prompts — richer
        // names keep the persona on-topic across a session.
        {
          const card = sessionCards.find((c) => c.trackId === track.id);
          const subject =
            card?.objectName?.trim() || card?.className || track.className;
          formData.append("className", subject);
        }
        formData.append("turnId", tid);
        formData.append("lang", spokenLangRef.current);
        formData.append("spokenLang", spokenLangRef.current);
        formData.append("learnLang", learnLangRef.current);
        formData.append("mode", track.mode);
        if (track.voiceId) formData.append("voiceId", track.voiceId);
        // Rich visual notes hydrated in the background by describeObject.
        // This is what lets converseWithObject stay text-only on the hot
        // path (no vision call) while still getting funnier-than-classname
        // context — the chewed straw, dust, dent, etc.
        if (track.description) formData.append("description", track.description);
        // Optional client-side transcript hint. Empty is the normal path
        // — the server transcribes the audio blob itself (OpenAI STT) and
        // returns the final `transcript` field in the converse response.
        // If the UI ever re-enables a browser-side fast path (e.g. Web
        // Speech API), it can be forwarded here to skip the server STT
        // round-trip.
        const trimmedTranscript = (clientTranscript ?? "").trim();
        if (trimmedTranscript) {
          formData.append("transcript", trimmedTranscript);
          setHeardText(trimmedTranscript);
          if (heardClearTimerRef.current != null) {
            clearTimeout(heardClearTimerRef.current);
            heardClearTimerRef.current = null;
          }
        }
        // Full conversation so far — object's prior lines + user turns.
        // The server re-caps to 16; we cap here so the payload stays small.
        formData.append(
          "history",
          JSON.stringify(track.history.slice(-32))
        );

        const t0 = performance.now();
        const converseResp = (await converseWithObject(formData)) as {
          transcript: string;
          reply: string;
          voiceId: string | null;
          emotion?: string | null;
          speed?: string | null;
          teachMode?: boolean;
        };
        const {
          transcript,
          reply,
          voiceId: replyVoiceId,
          emotion: replyEmotion,
          speed: replySpeed,
          teachMode: replyTeachMode,
        } = converseResp;
        const converseMs = Math.round(performance.now() - t0);
        if (callGen !== track.speakGen) {
          // eslint-disable-next-line no-console
          console.log(`[converse #${tid}] ✗ superseded (speakGen mismatch)`);
          return;
        }
        // spokenLang is locked to zh (product target audience); no
        // per-utterance refinement. We still use detectLangFromText when
        // debugging but don't mutate state from it.
        // Persist the turn into the gallery card's history so the Runware
        // prompt builder has real conversation context, and mirror the
        // current language pair onto the card.
        appendCardHistory({ trackId }, [
          ...(transcript
            ? [{ role: "user" as const, content: transcript }]
            : []),
          ...(reply ? [{ role: "assistant" as const, content: reply }] : []),
        ]);
        setCardLanguages(trackId, spokenLangRef.current, learnLangRef.current);
        if (typeof replyTeachMode === "boolean") {
          setCardTeachMode(trackId, replyTeachMode);
        }
        // Commit the exchange to this track's memory so future turns see
        // the full thread. Cap to the same 16 turns the server enforces.
        const nextHistory = [...track.history];
        if (transcript)
          nextHistory.push({ role: "user" as const, content: transcript });
        if (reply)
          nextHistory.push({ role: "assistant" as const, content: reply });
        track.history = nextHistory.slice(-32);
        // Mirror the turn into the shared group transcript so other tracks
        // in the circle can react to what the user just said + this track's
        // reply on their next scheduled turn.
        const nowTs = performance.now();
        if (transcript) {
          groupHistoryRef.current.push({
            speaker: "you",
            trackId: null,
            line: transcript,
            role: "user",
            ts: nowTs,
          });
        }
        if (reply) {
          groupHistoryRef.current.push({
            speaker: track.className,
            trackId,
            line: reply,
            role: "assistant",
            ts: nowTs,
          });
          groupLastSpokeAtRef.current[trackId] = nowTs;
        }
        groupHistoryRef.current = groupHistoryRef.current.slice(-48);
        // eslint-disable-next-line no-console
        console.log(
          `[converse #${tid}] ◀ client-roundtrip=${converseMs}ms  heard="${transcript.slice(0, 60)}${transcript.length > 60 ? "…" : ""}"  reply="${reply.slice(0, 60)}${reply.length > 60 ? "…" : ""}"`
        );

        // Fire-and-forget: refresh the description off a fresh crop so the
        // NEXT conversation turn sees the current visual state (user moved
        // closer, opened a drawer, put hoodies on the chair, etc.). Runs
        // in the background — doesn't block audio playback below.
        const freshCrop = captureBoxFrame(track.smoothedBox);
        if (freshCrop) refreshTrackDescription(trackId, freshCrop, `#${tid}`);

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
        const ttsT0 = performance.now();
        await playStreamingReply(
          trackId,
          callGen,
          reply,
          replyVoiceId ?? track.voiceId,
          tid,
          () => {
            // First audio chunk has been queued — user hears sound now.
            const firstSoundAt = performance.now();
            const ttsTtfb = Math.round(firstSoundAt - ttsT0);
            const press = turnPressAtRef.current;
            const totalMs = press ? Math.round(firstSoundAt - press) : -1;
            // eslint-disable-next-line no-console
            console.log(
              `[turn #${tid}] ◀ FIRST SOUND in ${totalMs}ms  ━  converse=${converseMs}ms ▸ tts-ttfb=${ttsTtfb}ms`
            );
          },
          replyEmotion,
          replySpeed
        );
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

  // --- Collect to gallery ------------------------------------------------
  //
  // Fire-and-forget: trigger the Runware pipeline on the parallel
  // /api/runware/generate endpoint and update card status in the store.
  // We don't await the result in the caller — the gallery page reads the
  // "pending → done" transition off the card store and renders a shimmer
  // until the image URL arrives, so the user can keep chatting here.
  const handleCollect = useCallback(
    async (trackId: string) => {
      const card = sessionCards.find((c) => c.trackId === trackId);
      if (!card) {
        showRejection("not captured yet — tap again", 2000);
        return;
      }
      if (card.generatedImageStatus === "pending") return;
      if (card.generatedImageStatus === "done") return;
      setCardImageStatus(card.id, "pending");

      // Server budget is 35s; give the client 45s to cover transport +
      // cold-start + backoff before we stop waiting and mark the card
      // failed. Without this ceiling a stalled provider could leave the
      // card in "pending" forever (shimmer, no retry).
      const ac = new AbortController();
      const timeoutMs = 45_000;
      const timer = setTimeout(() => ac.abort(), timeoutMs);
      try {
        const resp = await fetch("/api/runware/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: ac.signal,
          body: JSON.stringify({
            cardId: card.id,
            className: card.className,
            description: card.description,
            history: card.history ?? [],
            spokenLang: spokenLangRef.current,
            learnLang: learnLangRef.current,
            // Let the VLM see the actual crop so it can write a prompt with
            // details you couldn't guess from className alone.
            imageDataUrl: card.imageDataUrl,
          }),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({ error: "unknown" }));
          setCardImageStatus(card.id, "failed", {
            error: String(err.error ?? "unknown"),
          });
          return;
        }
        const { imageUrl, prompt } = (await resp.json()) as {
          imageUrl: string;
          prompt: string;
          promptSource?: "vlm" | "heuristic";
        };
        setCardGeneratedImage(card.id, imageUrl, prompt);
      } catch (e) {
        const aborted =
          (e instanceof Error && e.name === "AbortError") ||
          (e as { name?: string } | null)?.name === "AbortError";
        setCardImageStatus(card.id, "failed", {
          error: aborted
            ? "timed out"
            : e instanceof Error
              ? e.message
              : String(e),
        });
      } finally {
        clearTimeout(timer);
      }
    },
    [sessionCards, showRejection]
  );

  // --- Group chat -------------------------------------------------------
  //
  // Two-stage pipeline:
  //   1. `prepareGroupTurn` (background) — runs groupLine + full TTS fetch
  //      while the CURRENT speaker is talking, stashes the result Blob in
  //      preparedTurnRef. Callable multiple times; cheap no-op if a turn
  //      is already staged or in flight.
  //   2. `runGroupTurn` (foreground) — when the scheduler's timer fires,
  //      consumes preparedTurnRef if present (decode → AudioBufferSource,
  //      near-zero click-to-sound) or falls back to inline generate +
  //      streaming TTS for the very first turn of a session.
  //
  // Invalidation: external events (mic press, track add/evict, group-off,
  // clear) bump groupGenRef and null preparedTurnRef. Anything in flight
  // compares its captured gen on resume and drops silently if mismatched.

  // Shared speaker-pick logic. Two-layer:
  //   1. If the LAST turn was addressed AT a specific peer (and that peer
  //      is still locked and isn't the speaker of the last turn), that
  //      peer wins ~70% of the time — "A insulted me, I answer." The 30%
  //      escape hatch lets a third track cut in occasionally so it doesn't
  //      become a deterministic tennis match.
  //   2. Otherwise (or 30% of the time under #1): quietest track wins;
  //      ties broken by older lastTapAt so a freshly-tapped object doesn't
  //      steal the floor from a track that's barely spoken.
  // Resolve the user-facing label for a track — prefers the VLM's
  // objectName (stored on the session card) over the YOLO class. Used
  // everywhere a name flows into an LLM prompt or a user-facing string
  // so the model never echoes a raw detector class like "cup".
  const nameForTrack = useCallback(
    (t: TrackRefs): string => {
      const card = sessionCards.find((c) => c.trackId === t.id);
      return card?.objectName?.trim() || card?.className || t.className;
    },
    [sessionCards]
  );

  const pickGroupSpeaker = useCallback((): TrackRefs | null => {
    const tracks = tracksRef.current;
    if (tracks.length < 1) return null;

    // Layer 1: reactive addressing.
    const lastTurn = groupHistoryRef.current[groupHistoryRef.current.length - 1];
    if (
      lastTurn &&
      lastTurn.role === "assistant" &&
      lastTurn.addressing &&
      tracks.length >= 2 &&
      Math.random() < 0.7
    ) {
      const target = lastTurn.addressing.toLowerCase();
      const addressed = tracks.find(
        (t) =>
          t.className.toLowerCase() === target &&
          t.id !== lastTurn.trackId
      );
      if (addressed) return addressed;
    }

    // Layer 2: quietest-wins round-robin.
    let pick: TrackRefs | null = null;
    let oldestSpokeAt = Infinity;
    for (const t of tracks) {
      const last = groupLastSpokeAtRef.current[t.id] ?? 0;
      if (last < oldestSpokeAt) {
        oldestSpokeAt = last;
        pick = t;
      } else if (
        last === oldestSpokeAt &&
        pick &&
        t.lastTapAt < pick.lastTapAt
      ) {
        pick = t;
      }
    }
    return pick;
  }, []);

  // Background: generate the next group line + TTS blob and stash. The
  // prospective history includes any already-prepared-but-not-played turn
  // so the model sees the true upcoming context, not the state as of the
  // last played line. We never bump groupGenRef here — only read.
  const prepareGroupTurn = useCallback(async () => {
    if (preparingRef.current) return;
    if (preparedTurnRef.current) return;
    const tracks = tracksRef.current;
    if (tracks.length < 1) return;
    // Prospective pick: if a turn is staged (shouldn't happen given the
    // guard above, defensive) skip it from selection for "who speaks next".
    const pick = pickGroupSpeaker();
    if (!pick) return;
    const mode: "chat" | "followup" =
      tracks.length === 1 ? "followup" : "chat";
    const peers = tracks
      .filter((t) => t.id !== pick.id)
      .map((t) => ({ name: nameForTrack(t), description: t.description ?? null }));
    const recentTurns = groupHistoryRef.current.slice(-16).map((t) => ({
      speaker: t.speaker,
      line: t.line,
      role: t.role,
    }));
    preparingRef.current = true;
    const gen = groupGenRef.current;
    const speakerGen = pick.speakGen;
    try {
      // eslint-disable-next-line no-console
      console.log(
        `[group:prep] ▶ speaker=${pick.className} (${pick.id}) mode=${mode} peers=${peers.length} history=${recentTurns.length}`
      );
      const { line, emotion, speed, addressing } = await groupLine({
        speaker: {
          name: nameForTrack(pick),
          description: pick.description ?? null,
        },
        peers,
        recentTurns,
        mode,
        lang: spokenLangRef.current,
      });
      if (gen !== groupGenRef.current) {
        // eslint-disable-next-line no-console
        console.log(`[group:prep] ← stale after groupLine`);
        return;
      }
      // Fetch the full TTS audio as a Blob so playback is click-to-sound
      // fast when the scheduler fires the prepared turn.
      let audioBlob: Blob | null = null;
      try {
        const resp = await fetch("/api/tts/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: line,
            voiceId: pick.voiceId ?? "",
            turnId: `gp${gen}`,
            lang: spokenLangRef.current,
            emotion: emotion ? [emotion] : [],
            speed: speed ?? null,
          }),
        });
        if (gen !== groupGenRef.current) {
          try {
            await resp.body?.cancel();
          } catch {
            // ignore
          }
          // eslint-disable-next-line no-console
          console.log(`[group:prep] ← stale after tts headers`);
          return;
        }
        if (resp.ok && resp.body) {
          audioBlob = await resp.blob();
        } else {
          // eslint-disable-next-line no-console
          console.log(
            `[group:prep] tts non-ok ${resp.status} — caption-only fallback`
          );
        }
      } catch (ttsErr) {
        // eslint-disable-next-line no-console
        console.log(
          `[group:prep] tts fetch fail — caption-only: ${ttsErr instanceof Error ? ttsErr.message : String(ttsErr)}`
        );
      }
      if (gen !== groupGenRef.current) return;
      preparedTurnRef.current = {
        trackId: pick.id,
        speakerName: nameForTrack(pick),
        line,
        voiceId: pick.voiceId,
        audioBlob,
        emotion,
        speed,
        addressing,
        mode,
        gen,
        speakerGen,
      };
      // eslint-disable-next-line no-console
      console.log(
        `[group:prep] ✓ ready speaker=${pick.className} (${pick.id}) addressing=${addressing ?? "-"} audio=${audioBlob ? `${Math.round(audioBlob.size / 1024)}KB` : "null"} line="${line}"`
      );
    } catch (err) {
      // eslint-disable-next-line no-console
      console.log(
        `[group:prep] ✖ ${err instanceof Error ? err.message : String(err)}`
      );
    } finally {
      preparingRef.current = false;
    }
  }, [pickGroupSpeaker]);

  // Decode the prepared Blob and play via AudioBufferSource. Returns a
  // promise that resolves when playback ends. Mirrors the lock-time buffer
  // path in speakOnTrack so lip-sync + gain + caption-clear all work.
  const playPreparedTurn = useCallback(
    async (prepared: NonNullable<typeof preparedTurnRef.current>) => {
      const ctx = ensureAudioCtx();
      const track = tracksRef.current.find((t) => t.id === prepared.trackId);
      if (!ctx || !track) throw new Error("prepared track not found");

      const callGen = ++track.speakGen;
      if (track.captionClearTimer != null) {
        clearTimeout(track.captionClearTimer);
        track.captionClearTimer = null;
      }
      stopTrackAudio(track);
      track.lipSync = createLipSyncState();

      // Lazy analyser + restore gain (same pattern as speakOnTrack).
      if (!track.analyser) {
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 1024;
        analyser.smoothingTimeConstant = 0.4;
        const gain = ctx.createGain();
        gain.gain.value = Math.max(0, track.opacity || 1);
        analyser.connect(gain);
        gain.connect(ctx.destination);
        track.analyser = analyser;
        track.gain = gain;
        track.freqData = new Uint8Array(
          new ArrayBuffer(analyser.frequencyBinCount)
        );
        track.timeData = new Uint8Array(new ArrayBuffer(analyser.fftSize));
      } else if (track.gain) {
        try {
          const now = ctx.currentTime;
          track.gain.gain.cancelScheduledValues(now);
          track.gain.gain.setValueAtTime(
            Math.max(0, track.opacity || 1),
            now
          );
        } catch {
          // ignore
        }
      }

      const running = await waitForAudioRunning(ctx);
      if (!running) throw new Error(`audio ctx not running (${ctx.state})`);
      if (callGen !== track.speakGen) return;

      if (!prepared.audioBlob) {
        // Caption-only fallback — no audio, just let the UI linger on the
        // caption and then clear it.
        if (track.captionClearTimer != null) {
          clearTimeout(track.captionClearTimer);
        }
        track.captionClearTimer = window.setTimeout(() => {
          track.captionClearTimer = null;
          setTracksUI((prev) =>
            prev.map((t) =>
              t.id === prepared.trackId ? { ...t, caption: null } : t
            )
          );
        }, CAPTION_LINGER_MS);
        return;
      }

      const arrayBuf = await prepared.audioBlob.arrayBuffer();
      if (callGen !== track.speakGen) return;
      let audioBuf: AudioBuffer;
      try {
        audioBuf = await ctx.decodeAudioData(arrayBuf);
      } catch (e) {
        throw new Error(
          `decodeAudioData failed: ${e instanceof Error ? e.message : String(e)}`
        );
      }
      if (callGen !== track.speakGen) return;
      if (!track.analyser) throw new Error("analyser vanished");

      const source = ctx.createBufferSource();
      source.buffer = audioBuf;
      source.connect(track.analyser);

      const ended = new Promise<void>((resolve) => {
        source.onended = () => {
          if (track.source === source) {
            try {
              source.disconnect();
            } catch {
              // ignore
            }
            track.source = null;
            setTracksUI((prev) =>
              prev.map((t) =>
                t.id === prepared.trackId ? { ...t, speaking: false } : t
              )
            );
            if (track.captionClearTimer != null) {
              clearTimeout(track.captionClearTimer);
            }
            track.captionClearTimer = window.setTimeout(() => {
              track.captionClearTimer = null;
              setTracksUI((prev) =>
                prev.map((t) =>
                  t.id === prepared.trackId ? { ...t, caption: null } : t
                )
              );
            }, CAPTION_LINGER_MS);
          }
          resolve();
        };
      });

      try {
        source.start();
      } catch (e) {
        try {
          source.disconnect();
        } catch {
          // ignore
        }
        throw new Error(
          `source.start failed: ${e instanceof Error ? e.message : String(e)}`
        );
      }
      track.source = source;
      setTracksUI((prev) =>
        prev.map((t) =>
          t.id === prepared.trackId ? { ...t, speaking: true } : t
        )
      );
      await ended;
    },
    [ensureAudioCtx, stopTrackAudio, waitForAudioRunning]
  );

  const runGroupTurn = useCallback(async () => {
    if (groupInFlightRef.current) return;
    const tracks = tracksRef.current;
    if (tracks.length < 1) return;
    groupInFlightRef.current = true;
    // Capture the staged turn BEFORE bumping gen so its `gen` still matches.
    // Bumping gen here cancels any in-flight prepare that was racing with
    // us — we don't want a stale prepare result landing in preparedTurnRef
    // after our own turn lands in groupHistoryRef and the next scheduler
    // tick pops it without context from the line we just spoke.
    const preparedSnapshot = preparedTurnRef.current;
    preparedTurnRef.current = null;
    const turnGen = ++groupGenRef.current;
    try {
      // Fast path: consume the staged turn. Only valid if the speaker
      // still exists and its speakGen matches what we captured at prep.
      const prepared = preparedSnapshot;
      const speakerStillValid =
        prepared != null &&
        prepared.gen === turnGen - 1 &&
        tracksRef.current.some(
          (t) => t.id === prepared.trackId && t.speakGen === prepared.speakerGen
        );
      if (prepared && speakerStillValid) {
        const ts = performance.now();
        groupHistoryRef.current.push({
          speaker: prepared.speakerName,
          trackId: prepared.trackId,
          line: prepared.line,
          role: "assistant",
          ts,
          addressing: prepared.addressing,
        });
        groupHistoryRef.current = groupHistoryRef.current.slice(-48);
        groupLastSpokeAtRef.current[prepared.trackId] = ts;
        const speakerTrack = tracksRef.current.find(
          (t) => t.id === prepared.trackId
        );
        if (speakerTrack) {
          speakerTrack.history = [
            ...speakerTrack.history,
            { role: "assistant" as const, content: prepared.line },
          ].slice(-32);
        }
        setTracksUI((prev) =>
          prev.map((t) =>
            t.id === prepared.trackId
              ? { ...t, caption: prepared.line, thinking: false }
              : t
          )
        );
        // eslint-disable-next-line no-console
        console.log(
          `[group] ▶ play(cached) speaker=${prepared.speakerName} (${prepared.trackId}) line="${prepared.line}"`
        );
        // Kick off the NEXT prep now so it runs while this one plays.
        void prepareGroupTurn();
        try {
          await playPreparedTurn(prepared);
        } catch (err) {
          // eslint-disable-next-line no-console
          console.log(
            `[group] ✖ cached playback ${err instanceof Error ? err.message : String(err)}`
          );
          setTracksUI((prev) =>
            prev.map((t) =>
              t.id === prepared.trackId
                ? { ...t, thinking: false, speaking: false }
                : t
            )
          );
        }
        return;
      }

      // Cold path: no prep ready (first turn of a session, or invalidated).
      // Generate + stream inline, same path converseWithObject uses.
      const mode: "chat" | "followup" =
        tracks.length === 1 ? "followup" : "chat";
      const pick = pickGroupSpeaker();
      if (!pick) return;
      const peers = tracks
        .filter((t) => t.id !== pick.id)
        .map((t) => ({
          name: nameForTrack(t),
          description: t.description ?? null,
        }));
      const recentTurns = groupHistoryRef.current.slice(-16).map((t) => ({
        speaker: t.speaker,
        line: t.line,
        role: t.role,
      }));
      const speaker = pick;
      const callGen = ++speaker.speakGen;
      // eslint-disable-next-line no-console
      console.log(
        `[group] ▶ turn(cold) speaker=${speaker.className} (${speaker.id}) mode=${mode} peers=${peers.length} history=${recentTurns.length}`
      );
      setTracksUI((prev) =>
        prev.map((t) =>
          t.id === speaker.id
            ? { ...t, thinking: true, speaking: false, caption: null }
            : t
        )
      );
      const { line, emotion, speed, addressing } = await groupLine({
        speaker: {
          name: nameForTrack(speaker),
          description: speaker.description ?? null,
        },
        peers,
        recentTurns,
        mode,
        lang: spokenLangRef.current,
      });
      if (turnGen !== groupGenRef.current || callGen !== speaker.speakGen) {
        // eslint-disable-next-line no-console
        console.log(`[group] ← superseded (cold)`);
        return;
      }
      const ts = performance.now();
      groupHistoryRef.current.push({
        speaker: speaker.className,
        trackId: speaker.id,
        line,
        role: "assistant",
        ts,
        addressing,
      });
      groupHistoryRef.current = groupHistoryRef.current.slice(-48);
      speaker.history = [
        ...speaker.history,
        { role: "assistant" as const, content: line },
      ].slice(-32);
      groupLastSpokeAtRef.current[speaker.id] = ts;
      setTracksUI((prev) =>
        prev.map((t) =>
          t.id === speaker.id ? { ...t, caption: line, thinking: false } : t
        )
      );
      // Kick off next prep in parallel with this cold playback.
      void prepareGroupTurn();
      await playStreamingReply(
        speaker.id,
        callGen,
        line,
        speaker.voiceId,
        `g${turnGen}`,
        undefined,
        emotion,
        speed
      );
    } catch (err) {
      // eslint-disable-next-line no-console
      console.log(
        `[group] ✖ ${err instanceof Error ? err.message : String(err)}`
      );
      setTracksUI((prev) =>
        prev.map((t) => ({ ...t, thinking: false, speaking: false }))
      );
    } finally {
      groupInFlightRef.current = false;
    }
  }, [pickGroupSpeaker, playPreparedTurn, playStreamingReply, prepareGroupTurn]);

  // Scheduler: whenever the scene is idle (≥1 track, nothing thinking or
  // speaking, user isn't holding the mic, no picker up) wait a beat and
  // fire the next turn. Behavior splits on track count:
  //   • 1 track — "follow-up" mode. Wait a random 12–20s after the last
  //     utterance before firing, so the human has room to reply. If they
  //     do reply (converseWithObject pushes into groupHistoryRef), the
  //     silence clock resets naturally.
  //   • ≥2 tracks — banter mode. Short 0.9–1.8s between lines; ~15% chance
  //     of a 2.5–5s natural break so it doesn't read as a shouting match.
  //     After a user line, snap back faster (500ms) so they feel heard.
  // NOTE: the deps list deliberately uses `tracksUI.length`, NOT `tracksUI`.
  // The RAF render loop rewrites the tracksUI array every ~16ms (positions,
  // shapes, etc.); depending on the full array would tear this effect down
  // every frame and clear the scheduled timer before it could ever fire.
  // That was the bug where only one track ever replied.
  const tracksLen = tracksUI.length;
  const anyThinkingSched = tracksUI.some((t) => t.thinking);
  const anySpeakingSched = tracksUI.some((t) => t.speaking);
  useEffect(() => {
    if (!groupChatEnabled) return;
    if (tracksLen < 1) return;
    if (anyThinkingSched || anySpeakingSched) return;
    if (groupInFlightRef.current) return;
    if (isRecording) return;
    if (addHintActive) return;
    if (groupTimerRef.current != null) return;

    const history = groupHistoryRef.current;
    const lastTurn = history[history.length - 1];
    const now = performance.now();
    const silenceMs = lastTurn ? now - lastTurn.ts : Infinity;

    let delay: number;
    if (tracksLen === 1) {
      // Solo path — random 12–20s silence threshold then a follow-up.
      const minWait = 12_000 + Math.random() * 8_000;
      delay = Math.max(400, minWait - silenceMs);
    } else if (lastTurn?.role === "user") {
      // Respond to the human quickly so they feel heard.
      delay = 500;
    } else {
      // Multi-track banter with occasional breather. 15% of the time take
      // a 2.5–5s pause, otherwise 0.9–1.8s snap back.
      const takeBreak = Math.random() < 0.15;
      delay = takeBreak
        ? 2500 + Math.random() * 2500
        : 900 + Math.random() * 900;
    }

    // Kick off prep NOW (while waiting) so by the time the timer fires,
    // the next turn's script + TTS are ready. The mode is decided inside
    // prepare based on current track count; the delay-only role of the
    // scheduler is timing. No-op if something is already staged or
    // preparing.
    void prepareGroupTurn();

    groupTimerRef.current = window.setTimeout(() => {
      groupTimerRef.current = null;
      void runGroupTurn();
    }, delay);
    return () => {
      if (groupTimerRef.current != null) {
        clearTimeout(groupTimerRef.current);
        groupTimerRef.current = null;
      }
    };
  }, [
    groupChatEnabled,
    tracksLen,
    anyThinkingSched,
    anySpeakingSched,
    isRecording,
    addHintActive,
    prepareGroupTurn,
    runGroupTurn,
  ]);

  // When group chat is toggled off OR the scene clears, stop any in-flight
  // scheduled turn so a flipped-off toggle kills the loop immediately.
  useEffect(() => {
    if (groupChatEnabled && tracksLen >= 1) return;
    if (groupTimerRef.current != null) {
      clearTimeout(groupTimerRef.current);
      groupTimerRef.current = null;
    }
    // Bumping the gen tag makes any in-flight groupLine/prepare drop its
    // result when it resolves — no stale reply plays after the toggle flip.
    // Also drop the staged turn so a re-enable starts fresh.
    groupGenRef.current++;
    preparedTurnRef.current = null;
  }, [groupChatEnabled, tracksLen]);

  // Overlap prep with ANY ongoing playback — not just group turns. When a
  // converse reply starts playing (or an opening line), this kicks off the
  // next group turn's script + TTS in the background, so by the time the
  // current audio ends, the next line is already cached. Without this, the
  // gap between "object-answers-user" and "peer-picks-up-the-thread" is a
  // full groupLine + TTS roundtrip (~800ms of dead air), which kills the
  // alive-committee feel. Prep itself is idempotent — cheap no-op if
  // something is already staged or in flight.
  useEffect(() => {
    if (!groupChatEnabled) return;
    if (tracksLen < 1) return;
    if (!anySpeakingSched) return;
    if (isRecording) return;
    // Fire-and-forget. prepareGroupTurn's own guards handle the no-op case
    // when a prepare is already in flight or a turn is already staged.
    void prepareGroupTurn();
  }, [
    groupChatEnabled,
    tracksLen,
    anySpeakingSched,
    isRecording,
    prepareGroupTurn,
  ]);

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
      // Per-tap correlation id. Threaded through every downstream log
      // (speak, describe, server-side generateLine) so one tap's full
      // press-to-first-sound timeline can be grepped out of the console.
      tapCounterRef.current += 1;
      const tapId = `t${tapCounterRef.current}`;
      const tapTag = `#${tapId}`;
      const pressedAt = performance.now();
      const rectNow = v.getBoundingClientRect();
      const tapElX = e.clientX - rectNow.left;
      const tapElY = e.clientY - rectNow.top;
      // eslint-disable-next-line no-console
      console.log(
        `[tap ${tapTag}] ▶ press  at=(${Math.round(tapElX)},${Math.round(tapElY)}) videoSize=${v.videoWidth}x${v.videoHeight}`
      );

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
        // eslint-disable-next-line no-console
        console.log(
          `[tap ${tapTag}] ▸ hit existing track ${existing.id} (class="${existing.className}")  capture=${Math.round(performance.now() - pressedAt)}ms`
        );
        void speakOnTrack(existing.id, dataUrl, { tapId, pressedAt });
        return;
      }

      // (b) Otherwise resolve a detection under the tap and add a new face.
      let tapDets: Detection[] = [];
      const detectStart = performance.now();
      const cacheAge = performance.now() - detectionsTsRef.current;
      const usedCache = detectionsRef.current.length && cacheAge < TAP_CACHE_MAX_AGE_MS;
      if (usedCache) {
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
      const detectMs = usedCache
        ? Math.round(cacheAge)
        : Math.round(performance.now() - detectStart);
      const detectEndAt = performance.now();
      // eslint-disable-next-line no-console
      console.log(
        `[tap ${tapTag}] ▸ detect ${usedCache ? `cache-hit age=${detectMs}ms` : `one-shot ${detectMs}ms`}  dets=${tapDets.length}`
      );

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

      // Duplicate guard: if the resolved detection matches an existing track
      // (same class, high IoU), retap that track instead of minting a second
      // face on the same item. The findTrackAtPoint check above only catches
      // taps inside the smoothed box; this covers the drift case where the
      // box has shrunk or lagged off the physical object.
      {
        const tappedBox = makeBox(
          (tapped.x1 + tapped.x2) / 2,
          (tapped.y1 + tapped.y2) / 2,
          tapped.x2 - tapped.x1,
          tapped.y2 - tapped.y1
        );
        const duplicate = tracksRef.current.find(
          (t) =>
            t.classId === tapped!.classId &&
            iou(t.smoothedBox, tappedBox) >= DUPLICATE_IOU_MIN
        );
        if (duplicate) {
          setTapFrame(null);
          const dataUrl = captureBoxFrame(duplicate.smoothedBox);
          if (!dataUrl) {
            showRejection("couldn't grab the frame");
            return;
          }
          duplicate.lastTapAt = performance.now();
          // eslint-disable-next-line no-console
          console.log(
            `[tap ${tapTag}] ▸ duplicate-of ${duplicate.id} (IoU≥${DUPLICATE_IOU_MIN})  capture=${Math.round(performance.now() - pressedAt)}ms`
          );
          void speakOnTrack(duplicate.id, dataUrl, { tapId, pressedAt });
          return;
        }
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
      // Seed the capture popup NOW so it appears the same instant the
      // face does — not 3–5s later when the VLM returns. The popup's
      // collect button stays disabled ("preparing…") until the matching
      // SessionCard lands in the store. Resetting dismissedTrackIds for
      // this id lets the user retap a previously-dismissed object.
      setPendingCapture({
        trackId: newId,
        className: tapped.className,
        imageDataUrl: dataUrl,
        createdAt: Date.now(),
      });
      setDismissedTrackIds((prev) => {
        if (!prev.has(newId)) return prev;
        const next = new Set(prev);
        next.delete(newId);
        return next;
      });
      const newTrack: TrackRefs = {
        id: newId,
        classId: tapped.classId,
        className: tapped.className,
        anchor: { rx: 0, ry: 0 },
        boxEma,
        smoothedBox: lockBox,
        missedFrames: 0,
        opacity: 0,
        audioLevel: 1,
        lostSinceMs: null,
        lastTapAt: nowTs,
        vx: 0,
        vy: 0,
        lastUpdatedAt: nowTs,
        tiltDeg: 0,
        poleOffsetX: tapped.maskPole ? tapped.maskPole.x - lockCx : 0,
        poleOffsetY: tapped.maskPole ? tapped.maskPole.y - lockCy : 0,
        orientAngle: tapped.principalAngle ?? 0,
        orientRatio: tapped.axisRatio ?? 1,
        speakGen: 0,
        analyser: null,
        gain: null,
        freqData: null,
        timeData: null,
        source: null,
        shape: "X",
        lipSync: createLipSyncState(),
        captionClearTimer: null,
        voiceId: null,
        history: [],
        description: null,
        descriptionGen: 0,
        maskCanvas: null,
        maskDataUrl: null,
        maskSrcBox: null,
        maskAnchor: null,
        mode: modeRef.current,
      };

      // A new peer changes who the prepared turn would have addressed —
      // drop the pre-cache so the next prepare sees the expanded roster.
      groupGenRef.current++;
      preparedTurnRef.current = null;
      // LRU eviction when the slots are full. Bump speakGen first so any
      // in-flight generateLine/stream on the evictee bails at its guard.
      // Route through stopTrackAudio for the same audio teardown the
      // retap path uses — one code path, one bug surface.
      if (tracksRef.current.length >= MAX_FACES) {
        let oldest = tracksRef.current[0];
        for (const t of tracksRef.current) {
          if (t.lastTapAt < oldest.lastTapAt) oldest = t;
        }
        oldest.speakGen++;
        stopTrackAudio(oldest);
        if (oldest.analyser) {
          try {
            oldest.analyser.disconnect();
          } catch {
            // already disconnected
          }
          oldest.analyser = null;
        }
        if (oldest.gain) {
          try {
            oldest.gain.disconnect();
          } catch {
            // already disconnected
          }
          oldest.gain = null;
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
          tilt: 0,
          caption: null,
          thinking: false,
          speaking: false,
        },
      ]);

      setTapFrame(null);
      if (addHintTimerRef.current != null) {
        clearTimeout(addHintTimerRef.current);
        addHintTimerRef.current = null;
      }
      setAddHintActive(false);
      const lockEndAt = performance.now();
      const pressToDetect = Math.round(detectEndAt - detectMs - pressedAt);
      const detectToLock = Math.round(lockEndAt - detectEndAt);
      const totalMs = Math.round(lockEndAt - pressedAt);
      // eslint-disable-next-line no-console
      console.log(
        `[tap ${tapTag}] ◀ LOCKED track=${newId} class="${tapped!.className}"  ━  press→detect=${pressToDetect}ms ▸ detect=${detectMs}ms${usedCache ? " (cache)" : ""} ▸ detect→lock=${detectToLock}ms ▸ total=${totalMs}ms`
      );
      void speakOnTrack(newId, dataUrl, { tapId, pressedAt });
      // Hydrate rich visual context in the background while the first line
      // is being generated, so the FIRST conversation turn already has
      // something funnier than the bare classname to work with.
      refreshTrackDescription(newId, dataUrl, tapTag);
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
    // Pressing talk while a face is mid-reply interrupts it: bump speakGen
    // (any in-flight generateLine / streaming TTS bails at its next guard)
    // and ramp+stop its audio. The user wanted to cut in without sitting
    // through the rest of the line.
    // Snapshot UI state BEFORE bumping speakGen — the in-flight speakOnTrack
    // / sendTalkToTrack `finally` blocks won't clear `thinking` once their
    // captured callGen no longer matches, so we have to do it ourselves.
    const wasBusy = tracksUIRef.current.some((t) => t.thinking || t.speaking);
    let stoppedAudio = 0;
    for (const t of tracksRef.current) {
      if (t.source) stoppedAudio++;
      t.speakGen++;
      stopTrackAudio(t);
      if (t.captionClearTimer != null) {
        clearTimeout(t.captionClearTimer);
        t.captionClearTimer = null;
      }
    }
    // Drop any staged group turn + in-flight prepare — it was scripted
    // without knowledge of what the user is about to say. The NEXT prepare
    // (scheduled after the user's reply lands) sees the updated transcript.
    groupGenRef.current++;
    preparedTurnRef.current = null;
    if (groupTimerRef.current != null) {
      clearTimeout(groupTimerRef.current);
      groupTimerRef.current = null;
    }
    if (wasBusy) {
      setTracksUI((prev) =>
        prev.map((t) =>
          t.thinking || t.speaking
            ? { ...t, speaking: false, thinking: false }
            : t
        )
      );
      // eslint-disable-next-line no-console
      console.log(`[mic] ⤴ interrupted (audio-stopped=${stoppedAudio})`);
    }
    // Bump per-press correlation id so all logs for this turn share a tag.
    turnCounterRef.current += 1;
    turnIdRef.current = String(turnCounterRef.current);
    turnPressAtRef.current = performance.now();
    const turnId = turnIdRef.current;
    // eslint-disable-next-line no-console
    console.log(`[turn #${turnId}] ▶ press`);
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
        `[mic #${turnIdRef.current}] ◼ captured ${Math.round(blob.size / 1024)}KB ${blob.type}`
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

      // STT runs server-side now (see backend services/stt.py). The
      // frontend just hands the raw audio blob to /api/converse — the
      // backend transcribes with OpenAI's speech-to-text model and the
      // final `transcript` field of the response is what we surface as
      // "you said". No on-device Whisper, no transformers.js, no ONNX
      // runtime shipped to the browser.
      const turnId = turnIdRef.current;
      void (async () => {
        // eslint-disable-next-line no-console
        console.log(
          `[stt #${turnId}] → handoff  blob=${blob.size}B type=${blob.type || "?"}`
        );
        await sendTalkToTrack(target.id, blob, "", turnId);
      })();
    };

    // Clear any stale "you said" bubble from the previous turn so the
    // in-record "listening…" placeholder (rendered by the JSX below) takes
    // over cleanly.
    if (heardClearTimerRef.current != null) {
      clearTimeout(heardClearTimerRef.current);
      heardClearTimerRef.current = null;
    }
    setHeardText(null);

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
  }, [ensureAudioCtx, openMicStream, pickTalkTarget, sendTalkToTrack, showRejection, stopTrackAudio]);

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
    setIsRecording(false);
    stopTalkLevelLoop();
  }, [stopTalkLevelLoop]);

  // --- Derived UI --------------------------------------------------------
  const anyThinking = tracksUI.some((t) => t.thinking);
  const anySpeaking = tracksUI.some((t) => t.speaking);
  // Keep the ref in sync for callback paths (startRecording snapshots it
  // to decide whether an interrupt cleanup is needed).
  tracksUIRef.current = tracksUI;
  // Drives the mic button's aria-label only — pressing talk while busy
  // now interrupts the current voice instead of being blocked.
  const micBusy = anyThinking || anySpeaking;

  // Which face the mic is aimed at — the most-recently-tapped one. Drives
  // the button label so the user sees "talk to the mug" before they press.
  // Prefers the VLM's objectName (from the card store) over the YOLO class
  // so the label matches everything else in the UI; falls back to a neutral
  // "it" when no card has landed yet.
  const talkTargetClass: string | null = useMemo(() => {
    if (tracksUI.length === 0) return null;
    let best: TrackRefs | null = null;
    for (const t of tracksRef.current) {
      if (!best || t.lastTapAt > best.lastTapAt) best = t;
    }
    if (!best) return null;
    const card = sessionCards.find((c) => c.trackId === best!.id);
    if (card) return cardDisplayName(card);
    return learnLang === "zh" ? "它" : "it";
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tracksUI, sessionCards, learnLang]);

  // Unified popup item: only renders when there's an active fresh-tap
  // `pendingCapture`. On page load / reload we intentionally show
  // nothing, even if sessionCards are hydrated from storage — the
  // gallery page is the home for historical cards, not a popup on /.
  const activeCaptureItem: CaptureItem | null = useMemo(() => {
    const placeholderName = learnLang === "zh" ? "识别中…" : "looking…";
    if (!pendingCapture) return null;
    if (dismissedTrackIds.has(pendingCapture.trackId)) return null;
    const card = sessionCards.find(
      (c) => c.trackId === pendingCapture.trackId
    );
    if (card) {
      return {
        trackId: card.trackId,
        // Never show the YOLO class — use the VLM's name, falling
        // back to a neutral language-appropriate placeholder.
        className: cardDisplayName(card),
        imageDataUrl: card.imageDataUrl,
        cardId: card.id,
        status: (card.generatedImageStatus ?? "idle") as CaptureItem["status"],
      };
    }
    return {
      trackId: pendingCapture.trackId,
      // While the VLM is in flight we intentionally don't surface the
      // YOLO class — it would be "cup" / "chair", defeating the whole
      // point of VLM-named cards. Show a language-appropriate
      // placeholder and let the popup upgrade when the name arrives.
      className: placeholderName,
      imageDataUrl: pendingCapture.imageDataUrl,
      cardId: null,
      status: "preparing",
    };
  }, [sessionCards, dismissedTrackIds, pendingCapture, learnLang]);

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
    // Live YOLO boxes: show whenever the camera view is valid — including
    // after the first tap (`locked`). The old rule only drew boxes in
    // `ready` or during `addHintActive`, which hid *all* proposal boxes
    // as soon as one object had a face — users read that as "detector
    // / tracking broke" even though inference was still running.
    if (!v || phase === "error") return [];
    const visible: Detection[] = [];
    for (const d of detections) {
      if (EXCLUDED_CLASS_IDS.has(d.classId)) continue;
      visible.push(d);
    }
    // Nested-box suppression: if a smaller detection sits inside this one,
    // hide the outer box — only the front-most (smallest enclosing) shows.
    // A box B is "inside" A when B's center lies within A and B is smaller.
    const hidden = new Set<Detection>();
    for (let i = 0; i < visible.length; i++) {
      const a = visible[i];
      const aArea = (a.x2 - a.x1) * (a.y2 - a.y1);
      for (let j = 0; j < visible.length; j++) {
        if (i === j) continue;
        const b = visible[j];
        const bArea = (b.x2 - b.x1) * (b.y2 - b.y1);
        if (bArea >= aArea) continue;
        if (b.cx < a.x1 || b.cx > a.x2 || b.cy < a.y1 || b.cy > a.y2) continue;
        hidden.add(a);
        break;
      }
    }
    return visible
      .map((d) => {
        if (hidden.has(d)) return null;
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

      {/* Face + bubble overlays — one group per track. Both live inside
          the same positioned wrapper anchored at (t.left, t.top), so they
          can never drift apart: one transform moves both. Bubble's tail
          tip lands at the anchor; the face hangs directly below it. */}
      {tracksUI.map((t) => {
        const bubbleScale = Math.max(0.6, Math.min(1.3, t.scale * 0.85));
        const faceHalfH = (FACE_VOICE_HEIGHT / 2) * t.scale;
        const FACE_GAP = 8;
        const faceOffsetY = faceHalfH + FACE_GAP;
        const bubbleMaxWidth = "80vw";
        return (
          <div
            key={t.id}
            className="pointer-events-none absolute left-0 top-0 will-change-transform"
            style={{
              transform: `translate(${t.left}px, ${t.top}px)`,
              opacity: t.opacity,
            }}
          >
            {/* Face: centered on the anchor (0, 0). Outer div uses negative
                left/top to plant the element's center exactly at (0,0);
                inner div scales+rotates around its own 50%/50% origin.
                Combining `scale()` with `translate(-50%,-50%)` under
                `transformOrigin: "0 0"` misplaces the visual center by
                ((S-1)*W/2, (S-1)*H/2) — that was the off-anchor drift. */}
            <div
              className="absolute"
              style={{
                left: `${-FACE_VOICE_WIDTH / 2}px`,
                top: `${-FACE_VOICE_HEIGHT / 2}px`,
                width: `${FACE_VOICE_WIDTH}px`,
                height: `${FACE_VOICE_HEIGHT}px`,
              }}
            >
              <div
                style={{
                  width: "100%",
                  height: "100%",
                  transformOrigin: "50% 50%",
                  transform: `scale(${t.scale}) rotate(${t.tilt}deg)`,
                  filter: "drop-shadow(0 3px 8px rgba(0,0,0,0.35))",
                }}
              >
                <FaceVoice shape={t.shape} />
              </div>
            </div>
            {/* Bubble: tail tip sits directly above the face.
                `width: max-content` is load-bearing: this wrapper is an
                abspos child of a parent whose in-flow content is zero (the
                face is also abspos), so its containing block collapses to 0
                and shrink-to-fit falls back to min-content — which is one
                character wide given the inner <p>'s `break-words`. */}
            <div
              className="absolute left-0 top-0"
              style={{
                width: "max-content",
                maxWidth: bubbleMaxWidth,
                transformOrigin: "50% 100%",
                transform: `translate(-50%, calc(-100% - ${faceOffsetY}px)) scale(${bubbleScale})`,
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

      {/* Live YOLO proposal boxes (non-person COCO classes). */}
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
            <span>{learnLang === "zh" ? "点我" : "tap me"}</span>
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
      <div className="absolute inset-x-0 top-0 flex items-start justify-between gap-2 px-3 pt-[max(env(safe-area-inset-top),18px)] sm:px-5">
        <div className="pointer-events-none flex shrink-0 items-center gap-2 rounded-full bg-white/15 px-3 py-1.5 shadow-[0_8px_24px_-12px_rgba(0,0,0,0.6)] ring-1 ring-white/25 backdrop-blur-xl sm:px-3.5">
          <span className="h-1.5 w-1.5 rounded-full bg-[#ff89be] shadow-[0_0_0_3px_rgba(255,137,190,0.28)]" />
          <span className="serif-italic text-[15px] font-medium leading-none text-white/95 sm:text-[17px]">
            omni
          </span>
        </div>
        <div className="flex flex-1 flex-wrap items-center justify-end gap-1.5 sm:gap-2">
          <div
            className={
              "pointer-events-none flex items-center gap-2 rounded-full px-2.5 py-1.5 shadow-[0_8px_24px_-12px_rgba(0,0,0,0.6)] ring-1 backdrop-blur-xl transition sm:px-3.5 " +
              (phase === "error"
                ? "bg-rose-500/30 ring-rose-200/40"
                : "bg-white/15 ring-white/25")
            }
          >
            <span className={"h-1.5 w-1.5 rounded-full " + dotClass} />
            <span className="hidden text-[11.5px] font-medium tabular-nums tracking-wide text-white/90 sm:inline">
              {statusText}
            </span>
          </div>
          {/* Spoken-language toggle — 中 / EN. Sets the user's native
              tongue; paired with `learn` so the user can see both at once.
              STT routing reads `spokenLang`, so flipping this re-routes
              the mic to the selected locale on the next utterance. */}
          <div
            role="group"
            aria-label="spoken language"
            className="flex items-center gap-1 rounded-full bg-white/15 p-0.5 pl-1.5 ring-1 ring-white/25 shadow-[0_8px_24px_-12px_rgba(0,0,0,0.6)] backdrop-blur-xl sm:pl-2"
          >
            <span
              aria-hidden
              className="hidden select-none text-[9px] font-semibold uppercase tracking-[0.15em] text-white/55 sm:inline"
            >
              speak
            </span>
            <button
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                setSpokenLang("zh");
              }}
              aria-pressed={spokenLang === "zh"}
              className={
                "grid h-6 min-w-[28px] place-items-center rounded-full px-2 text-[11px] font-semibold transition " +
                (spokenLang === "zh"
                  ? "bg-white/80 text-[#c23a7a] shadow-sm"
                  : "text-white/80 hover:text-white")
              }
            >
              中
            </button>
            <button
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                setSpokenLang("en");
              }}
              aria-pressed={spokenLang === "en"}
              className={
                "grid h-6 min-w-[28px] place-items-center rounded-full px-2 text-[10.5px] font-semibold tracking-wide transition " +
                (spokenLang === "en"
                  ? "bg-white/80 text-[#c23a7a] shadow-sm"
                  : "text-white/80 hover:text-white")
              }
            >
              EN
            </button>
          </div>
          <Link
            to="/gallery"
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => e.stopPropagation()}
            className="btn-frost h-7 px-3 text-[10.5px] font-semibold uppercase tracking-wider"
            aria-label="open gallery"
          >
            gallery
          </Link>
          {tracksUI.length >= 2 && (
            <button
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                setGroupChatEnabled((v) => !v);
              }}
              aria-pressed={groupChatEnabled}
              title={
                groupChatEnabled
                  ? "group chat on — pause the circle"
                  : "group chat off — let them riff"
              }
              className={
                "h-7 px-3 text-[11px] font-semibold " +
                (groupChatEnabled ? "btn-frost-accent" : "btn-frost")
              }
            >
              <span aria-hidden>{groupChatEnabled ? "\u25CF" : "\u25B6"}</span>
              <span>chat</span>
            </button>
          )}
          {tracksUI.length > 0 && (
            <button
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                if (tracksUI.length >= MAX_FACES) return;
                triggerAddHint();
              }}
              disabled={tracksUI.length >= MAX_FACES || !yoloReady}
              aria-label={
                tracksUI.length >= MAX_FACES
                  ? `all ${MAX_FACES} slots full`
                  : "add another face"
              }
              title={
                tracksUI.length >= MAX_FACES
                  ? `${MAX_FACES}/${MAX_FACES} — clear one first`
                  : `add another (${tracksUI.length}/${MAX_FACES})`
              }
              className={
                "h-7 px-3 text-[11px] font-semibold " +
                (tracksUI.length >= MAX_FACES
                  ? "btn-frost cursor-not-allowed opacity-50"
                  : addHintActive
                    ? "btn-frost-accent"
                    : "btn-frost")
              }
            >
              <svg
                width="11"
                height="11"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="3"
                strokeLinecap="round"
              >
                <path d="M12 5v14M5 12h14" />
              </svg>
              <span className="tabular-nums">
                {tracksUI.length}/{MAX_FACES}
              </span>
            </button>
          )}
          {tracksUI.length > 0 && (
            <button
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                clearAllTracks();
              }}
              className="btn-frost btn-frost-danger h-7 w-7"
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

      {/* Unified boot overlay — camera + detector progress in one clean card.
          Three steps (camera → detector download → warming up) with tick
          indicators, a live progress bar, and an elapsed-time readout so the
          wait feels purposeful instead of silent. */}
      {(!yoloReady || !cameraReady) && yoloStatus.stage !== "error" && (() => {
        type BootStep = {
          id: "camera" | "download" | "warmup";
          label: string;
          state: "pending" | "active" | "done";
        };
        const downloading = yoloStatus.stage === "downloading";
        const compiling = yoloStatus.stage === "compiling";
        const downloadDone = downloading
          ? yoloStatus.bytesTotal > 0 &&
            yoloStatus.bytesLoaded >= yoloStatus.bytesTotal
          : compiling || yoloStatus.stage === "ready";
        const steps: BootStep[] = [
          {
            id: "camera",
            label: "camera",
            state: cameraReady ? "done" : "active",
          },
          {
            id: "download",
            label: "detector",
            state: !cameraReady
              ? "pending"
              : downloadDone
                ? "done"
                : "active",
          },
          {
            id: "warmup",
            label: "warming up",
            state: !cameraReady || !downloadDone
              ? "pending"
              : yoloStatus.stage === "ready"
                ? "done"
                : "active",
          },
        ];
        const activeStep = steps.find((s) => s.state === "active");
        const headline = !cameraReady
          ? "warming up camera"
          : downloading
            ? "downloading detector"
            : compiling
              ? "warming up detector"
              : "getting ready";
        const progressPct =
          !cameraReady
            ? 15
            : downloading
              ? yoloStatus.progress >= 0
                ? Math.max(6, Math.min(100, yoloStatus.progress * 100))
                : 30
              : compiling
                ? 92
                : 100;
        const indeterminate =
          compiling ||
          (downloading && yoloStatus.progress < 0) ||
          (!cameraReady && !downloading);
        const bytesText =
          downloading && yoloStatus.bytesTotal > 0
            ? `${(yoloStatus.bytesLoaded / 1024 / 1024).toFixed(1)} / ${(yoloStatus.bytesTotal / 1024 / 1024).toFixed(1)} MB`
            : downloading && yoloStatus.bytesLoaded > 0
              ? `${(yoloStatus.bytesLoaded / 1024 / 1024).toFixed(1)} MB`
              : "";
        const elapsedSec = bootElapsedMs / 1000;
        const elapsedText =
          elapsedSec < 10
            ? `${elapsedSec.toFixed(1)}s`
            : `${Math.round(elapsedSec)}s`;
        return (
          <div
            className="pointer-events-none absolute inset-x-0 top-[72px] flex justify-center px-4"
            style={{ animation: "fade-in 220ms ease-out both" }}
          >
            <div className="flex w-[min(22rem,92vw)] flex-col gap-3 rounded-[22px] bg-black/45 px-4 py-3.5 ring-1 ring-white/15 backdrop-blur-xl shadow-[0_16px_40px_-18px_rgba(0,0,0,0.7)]">
              <div className="flex items-center justify-between gap-3">
                <span className="serif-italic text-[13px] font-medium text-white/95">
                  {headline}
                </span>
                <span className="tabular-nums text-[10.5px] font-medium text-white/55">
                  {elapsedText}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                {steps.map((s, i) => (
                  <React.Fragment key={s.id}>
                    <div className="flex flex-1 flex-col items-center gap-1">
                      <span
                        className={
                          "grid h-[18px] w-[18px] place-items-center rounded-full ring-1 transition " +
                          (s.state === "done"
                            ? "bg-[color:var(--accent)] ring-white/60 text-white"
                            : s.state === "active"
                              ? "bg-white/90 ring-white text-[color:var(--ink)]"
                              : "bg-white/10 ring-white/25 text-white/50")
                        }
                        style={
                          s.state === "active"
                            ? { animation: "soft-pulse 1.4s ease-in-out infinite" }
                            : undefined
                        }
                      >
                        {s.state === "done" ? (
                          <svg
                            width="10"
                            height="10"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="3.4"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <path d="M5 12l5 5L20 7" />
                          </svg>
                        ) : (
                          <span className="h-1.5 w-1.5 rounded-full bg-current" />
                        )}
                      </span>
                      <span
                        className={
                          "text-[10px] font-medium tracking-wide transition " +
                          (s.state === "pending"
                            ? "text-white/45"
                            : "text-white/90")
                        }
                      >
                        {s.label}
                      </span>
                    </div>
                    {i < steps.length - 1 && (
                      <span
                        className={
                          "mt-[-16px] h-px flex-1 rounded-full transition " +
                          (steps[i].state === "done"
                            ? "bg-[color:var(--accent)]/75"
                            : "bg-white/15")
                        }
                      />
                    )}
                  </React.Fragment>
                ))}
              </div>
              <div className="flex flex-col gap-1.5">
                <div className="h-1 overflow-hidden rounded-full bg-white/10">
                  <div
                    className="h-full rounded-full bg-[color:var(--accent)] transition-[width] duration-200"
                    style={{
                      width: `${progressPct}%`,
                      animation: indeterminate
                        ? "soft-pulse 1.2s ease-in-out infinite"
                        : undefined,
                    }}
                  />
                </div>
                {(activeStep || bytesText) && (
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-[10.5px] text-white/55">
                      {activeStep?.id === "camera"
                        ? "asking for camera access…"
                        : downloading
                          ? "streaming model weights"
                          : compiling
                            ? "compiling kernels (one-time)"
                            : "almost there"}
                    </span>
                    <span className="tabular-nums text-[10.5px] text-white/55">
                      {bytesText}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      })()}

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
                lastTtsBackend === "cartesia"
                  ? "text-emerald-200"
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

      {/* First-run onboarding card. Shown once, then dismissed via
          localStorage. Replaces the old persistent "tap anything" hint. */}
      {phase === "ready" && !errorMsg && showOnboarding && (
        <div
          className="absolute inset-0 z-20 grid place-items-center px-6"
          style={{ animation: "fade-in 280ms ease-out both" }}
        >
          <div
            className="absolute inset-0 bg-black/30 backdrop-blur-[2px]"
            onClick={dismissOnboarding}
          />
          <div
            className="relative flex max-w-[20rem] flex-col items-center gap-4 rounded-[28px] bg-white/95 px-7 pt-7 pb-5 text-center shadow-[0_30px_80px_-20px_rgba(236,72,153,0.5)] ring-1 ring-white/80 backdrop-blur-xl"
            style={{ animation: "bubble-in 460ms cubic-bezier(0.16,1,0.3,1) both" }}
          >
            <span className="bubble-btn grid h-16 w-16 place-items-center rounded-full bg-gradient-to-br from-pink-200 to-rose-300 ring-1 ring-white">
              <span className="h-3 w-3 rounded-full bg-white" />
            </span>
            <div className="flex flex-col gap-1">
              <span className="serif-italic text-[20px] font-semibold leading-tight text-[color:var(--ink)]">
                hi — tap anything
              </span>
              <span className="text-[13px] leading-snug text-[color:var(--ink)]/70">
                give it a face, hear it talk. up to {MAX_FACES} at once.
              </span>
            </div>
            <button
              onClick={dismissOnboarding}
              className="mt-1 rounded-full bg-gradient-to-r from-pink-400 to-rose-400 px-6 py-2 text-[13px] font-semibold text-white shadow-[0_10px_24px_-10px_rgba(236,72,153,0.7)] ring-1 ring-white/40 transition hover:brightness-105 active:scale-[0.97]"
            >
              ok, got it
            </button>
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

      {addHintActive && !rejection && (
        <div
          className="pointer-events-none absolute inset-x-0 top-20 flex justify-center px-6"
          style={{ animation: "bubble-in 360ms cubic-bezier(0.16,1,0.3,1) both" }}
        >
          <div className="flex items-center gap-2 rounded-full bg-white/90 px-4 py-2 shadow-[0_12px_28px_-14px_rgba(0,0,0,0.5)] ring-1 ring-white/80 backdrop-blur-xl">
            <span
              className="h-2 w-2 rounded-full bg-[#ff89be]"
              style={{ animation: "soft-pulse 1.1s ease-in-out infinite" }}
            />
            <span className="serif-italic text-[13.5px] font-medium text-[color:var(--ink)]">
              tap any object to add a face
            </span>
          </div>
        </div>
      )}

      {/* Live speech feedback. While the mic is hot we show a soft
          "listening…" placeholder; once the server STT round-trip lands,
          `heardText` flips to the final transcript (set in
          sendTalkToTrack after /api/converse returns). */}
      {(isRecording || heardText) && (
        <div
          className="pointer-events-none absolute inset-x-0 top-[132px] flex justify-center px-6"
          style={{ animation: "bubble-in 320ms cubic-bezier(0.16,1,0.3,1) both" }}
        >
          <div className="flex max-w-[min(92vw,34rem)] flex-col items-center gap-0.5 rounded-2xl bg-black/55 px-4 py-2 ring-1 ring-white/15 backdrop-blur-xl">
            <span className="text-[10px] uppercase tracking-[0.18em] text-white/55">
              {isRecording && !heardText ? "listening" : "you said"}
            </span>
            <span className="serif-italic text-center text-[13px] font-medium leading-snug text-white/95 break-words">
              {heardText ?? "…"}
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

      {/* Capture popup — appears the instant the user taps a YOLO box
          (same moment the face spawns). Shows the crop + name in a
          "preparing" state while the VLM is in flight, then upgrades
          to an enabled "collect to gallery" button once the card lands.
          Face/voice chat above it is unchanged — this popup is a
          separate, intentional "collect" affordance. */}
      <CapturePopup
        item={activeCaptureItem}
        onCollect={(trackId) => {
          void handleCollect(trackId);
        }}
        onDismiss={(trackId) => {
          setDismissedTrackIds((prev) => {
            if (prev.has(trackId)) return prev;
            const next = new Set(prev);
            next.add(trackId);
            return next;
          });
        }}
      />

      {/* Hold-to-talk button (unchanged). */}
      <div
        className="absolute inset-x-0 bottom-0 flex flex-col items-center gap-2 px-5 pb-[max(env(safe-area-inset-bottom),22px)] pt-3"
        style={{
          opacity: phase === "starting" || phase === "error" ? 0.35 : 1,
          transition: "opacity 220ms ease",
          pointerEvents: phase === "starting" || phase === "error" ? "none" : "auto",
        }}
      >
        <div className="relative grid h-[76px] w-[76px] place-items-center">
          {/* Ambient halo — steady when live, breathing when idle. */}
          <span
            aria-hidden
            className="pointer-events-none absolute -inset-2 rounded-full"
            style={{
              background:
                "radial-gradient(circle, rgba(255,137,190,0.32) 0%, rgba(255,137,190,0) 72%)",
              opacity: isRecording ? 1 : 0.5,
              transition: "opacity 220ms ease",
              animation: isRecording ? undefined : "soft-pulse 2.4s ease-in-out infinite",
            }}
          />
          {/* Thin outer accent ring — adds crispness to the silhouette. */}
          <span
            aria-hidden
            className="pointer-events-none absolute rounded-full"
            style={{
              inset: 4,
              boxShadow: isRecording
                ? "0 0 0 1px rgba(255,255,255,0.5), 0 0 0 4px rgba(255,137,190,0.16)"
                : "0 0 0 1px rgba(255,255,255,0.6), 0 0 0 4px rgba(255,137,190,0.08)",
              transition: "box-shadow 240ms ease",
            }}
          />
          <button
            type="button"
            aria-label={
              isRecording
                ? "recording — release to send"
                : micBusy
                  ? "tap to interrupt and speak"
                  : "hold to speak"
            }
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
            className="relative grid h-[60px] w-[60px] place-items-center overflow-hidden rounded-full backdrop-blur-xl transition-[transform,box-shadow,opacity,filter] duration-200 ease-out"
            style={{
              background: isRecording
                ? "linear-gradient(155deg, #ffb4d1 0%, #ff6aa8 55%, #ec4899 100%)"
                : "linear-gradient(155deg, rgba(255,255,255,0.97) 0%, rgba(255,228,238,0.94) 55%, rgba(255,196,220,0.92) 100%)",
              boxShadow: isRecording
                ? "0 14px 32px -14px rgba(236,72,153,0.7), 0 1px 3px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.55), inset 0 -10px 16px -12px rgba(236,72,153,0.5)"
                : "0 10px 24px -14px rgba(236,72,153,0.32), 0 1px 2px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.9), inset 0 -8px 14px -10px rgba(236,72,153,0.18)",
              transform: isRecording
                ? "scale(1.06)"
                : talkFlash
                  ? "scale(1.04)"
                  : "scale(1)",
              opacity: 1,
              filter: undefined,
              cursor: undefined,
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
              width="22"
              height="22"
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
              <div className="flex h-[32px] items-center gap-[2px]">
                {Array.from({ length: WAVE_BARS }).map((_, i) => (
                  <span
                    key={i}
                    ref={(el) => {
                      barRefs.current[i] = el;
                    }}
                    className="block w-[2px] rounded-full"
                    style={{
                      height: 5,
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
                  width="24"
                  height="24"
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
      </div>

      {/* Lens onboarding overlay — mounts above the camera feed on
          first-ever visit or when the landing sent us back in with
          `?onboarding=1`. Self-unmounts after its own fade-out finishes. */}
      {showLensOnboarding && (
        <OnboardingOverlay
          initialLens={overlayLens}
          onFinished={handleOnboardingFinished}
        />
      )}
    </div>
  );
}
