"use client";

import { useEffect, useRef } from "react";

// "Talking face" built on the OpenBitX/face_voice asset set:
//   - `eyes.{webm,mp4}` looping video of real eyes (natural blinks/darts
//     already baked in)
//   - 9 PNG mouth shapes (`shape-A` … `shape-X`, 500×250 each with transparent
//     alpha) swapped per audio frame
//
// The shape letter is computed upstream from the TTS AnalyserNode — see the
// `classifyShape` helper in tracker.tsx — so this component stays dumb and
// just renders the right image over the right eyes video.

export type MouthShape = "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "X";

// Native dimensions. These are the pixel size of the component when
// rendered at CSS scale=1. Tracker's FACE_NATIVE_PX must match the larger
// axis so its box-size → face-size math stays correct.
export const FACE_VOICE_WIDTH = 280;
export const FACE_VOICE_HEIGHT = 160;

// Preload URLs so the browser starts fetching the PNG atlas as soon as the
// module evaluates. Swapping shapes at 30 fps should never block on network.
const SHAPES: readonly MouthShape[] = ["A", "B", "C", "D", "E", "F", "G", "H", "X"];
const SHAPE_URL = (s: MouthShape) => `/facevoice/shape-${s}.png`;

type FaceVoiceProps = {
  shape: MouthShape;
};

// Intermediate canvas resolution for the luma key. The source video is
// 1280×220 but displayed at ~62% of 280 CSS px, so 512 wide keeps retina
// crisp and caps per-frame pixel work at ~45k — cheap even on mid-tier
// Android. Height preserves the 1280:220 source aspect.
const EYES_CANVAS_WIDTH = 512;
const EYES_CANVAS_HEIGHT = 88;

export function FaceVoice({ shape }: FaceVoiceProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Preload every shape once — single cached HEAD request per file, then the
  // subsequent <img src> swaps are instant.
  useEffect(() => {
    const preloaded: HTMLImageElement[] = [];
    for (const s of SHAPES) {
      const im = new Image();
      im.src = SHAPE_URL(s);
      preloaded.push(im);
    }
    return () => {
      // Let GC collect; no cleanup needed beyond dropping the refs.
      preloaded.length = 0;
    };
  }, []);

  // Autoplay the eyes video and pipe each decoded frame through a canvas
  // that luma-keys black → transparent (alpha = max(r,g,b)). Both shipped
  // source files are yuv420p (no real alpha channel), and `mix-blend-mode:
  // screen` on a raw <video> breaks on mobile Safari and Android Chrome —
  // hardware-accelerated video lives on its own compositor layer and ignores
  // the blend, so a black rectangle ends up painted under the mouth. By
  // rasterising into a 2D canvas we emit true RGBA that every browser honours.
  useEffect(() => {
    const v = videoRef.current;
    const canvas = canvasRef.current;
    if (!v || !canvas) return;
    v.muted = true;
    v.loop = true;
    v.playsInline = true;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    canvas.width = EYES_CANVAS_WIDTH;
    canvas.height = EYES_CANVAS_HEIGHT;

    let cancelled = false;
    let rafId: number | null = null;
    let vfcId: number | null = null;
    const videoAny = v as unknown as {
      requestVideoFrameCallback?: (cb: () => void) => number;
      cancelVideoFrameCallback?: (id: number) => void;
    };

    const tryPlay = (reload = false) => {
      if (cancelled || !v) return;
      if (reload) {
        try {
          v.load();
        } catch {}
      }
      const p = v.play();
      if (p && typeof p.catch === "function") p.catch(() => {});
    };

    const drawFrame = () => {
      if (cancelled) return;
      if (v.readyState >= 2 && v.videoWidth > 0) {
        ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
        const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const d = img.data;
        // Brightness gain before luma-keying. The key sets alpha =
        // max(r,g,b), so mid-grey pixels (lids, iris shadow) end up half
        // transparent and visually vanish against anything non-white.
        // Boosting RGB lifts both the whites AND the resulting alpha in
        // one pass — eyes look brighter and more opaque simultaneously.
        const GAIN = 1.45;
        for (let i = 0; i < d.length; i += 4) {
          let r = d[i] * GAIN;
          let g = d[i + 1] * GAIN;
          let b = d[i + 2] * GAIN;
          if (r > 255) r = 255;
          if (g > 255) g = 255;
          if (b > 255) b = 255;
          d[i] = r;
          d[i + 1] = g;
          d[i + 2] = b;
          const m = r > g ? r : g;
          d[i + 3] = m > b ? m : b;
        }
        ctx.putImageData(img, 0, 0);
      }
      scheduleNext();
    };

    const scheduleNext = () => {
      if (cancelled) return;
      if (typeof videoAny.requestVideoFrameCallback === "function") {
        vfcId = videoAny.requestVideoFrameCallback(drawFrame);
      } else {
        rafId = requestAnimationFrame(drawFrame);
      }
    };

    tryPlay();
    const onLoaded = () => tryPlay();
    v.addEventListener("loadedmetadata", onLoaded);
    v.addEventListener("loadeddata", onLoaded);
    v.addEventListener("canplay", onLoaded);
    v.addEventListener("error", () => tryPlay(true));
    scheduleNext();

    const gestureRetry = () => {
      if (v.paused || v.currentTime === 0) tryPlay(v.readyState < 2);
    };
    window.addEventListener("pointerdown", gestureRetry, { passive: true });
    window.addEventListener("touchstart", gestureRetry, { passive: true });
    const onVisibility = () => {
      if (!document.hidden && (v.paused || v.currentTime === 0)) tryPlay(true);
    };
    document.addEventListener("visibilitychange", onVisibility);

    // Watchdog: if the video still hasn't moved past t=0 after 600ms,
    // force a full reload + replay. Retry a couple of times before giving up.
    let attempts = 0;
    const watchdog = window.setInterval(() => {
      if (cancelled || !v) return;
      const stuck = v.paused || v.currentTime === 0 || v.readyState < 2;
      if (!stuck) {
        window.clearInterval(watchdog);
        return;
      }
      attempts += 1;
      tryPlay(true);
      if (attempts >= 4) window.clearInterval(watchdog);
    }, 600);

    return () => {
      cancelled = true;
      window.clearInterval(watchdog);
      if (rafId !== null) cancelAnimationFrame(rafId);
      if (vfcId !== null && typeof videoAny.cancelVideoFrameCallback === "function") {
        videoAny.cancelVideoFrameCallback(vfcId);
      }
      v.removeEventListener("loadedmetadata", onLoaded);
      v.removeEventListener("loadeddata", onLoaded);
      v.removeEventListener("canplay", onLoaded);
      window.removeEventListener("pointerdown", gestureRetry);
      window.removeEventListener("touchstart", gestureRetry);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return (
    <div
      className="relative overflow-visible"
      style={{
        width: FACE_VOICE_WIDTH,
        height: FACE_VOICE_HEIGHT,
        filter: "drop-shadow(0 6px 10px rgba(0,0,0,0.55))",
      }}
    >
      {/* Hidden source video — kept in the DOM at 1×1 with opacity 0 so
          mobile Safari keeps it decoding. All visible pixels come from the
          canvas below, which carries the luma-keyed RGBA frames. */}
      <video
        ref={videoRef}
        muted
        loop
        playsInline
        autoPlay
        preload="auto"
        aria-hidden
        style={{
          position: "absolute",
          width: 1,
          height: 1,
          opacity: 0,
          pointerEvents: "none",
        }}
      >
        <source src="/facevoice/eyes.webm" type="video/webm" />
        <source src="/facevoice/eyes.mp4" type="video/mp4" />
      </video>
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute"
        style={{
          top: "10%",
          left: "50%",
          width: "62%",
          transform: "translateX(-50%)",
        }}
      />

      {/* Mouth layer. PNG has transparent alpha channel — no blend needed. */}
      <img
        src={SHAPE_URL(shape)}
        alt=""
        draggable={false}
        className="pointer-events-none absolute select-none"
        style={{
          bottom: "8%",
          left: "50%",
          width: "52%",
          transform: "translateX(-50%)",
        }}
      />
    </div>
  );
}

// Port of OpenBitX/face_voice's lip-sync heuristic. Given time-domain +
// frequency-domain buffers from an AnalyserNode (fftSize=1024), returns the
// 9-way mouth-shape classification the PNG atlas expects.
//
// Thresholds are their originals — kept for callers that don't hold state.
// Prefer `classifyShapeSmooth` for real playback; it adds an envelope
// follower, adaptive peak-normalized openness, and shape hysteresis so the
// mouth doesn't flicker on noise and stays in sync even when different TTS
// backends deliver very different output levels.
export function classifyShape(
  timeBuf: Uint8Array<ArrayBuffer>,
  freqBuf: Uint8Array<ArrayBuffer>
): { shape: MouthShape; rms: number; centroid: number; midEnergy: number } {
  const { rms, centroid, midEnergy } = extractFeatures(timeBuf, freqBuf);

  let shape: MouthShape;
  if (rms < 0.02) shape = "X";
  else if (rms < 0.06) shape = "A";
  else if (centroid > 0.55) shape = rms > 0.18 ? "C" : "B";
  else if (centroid < 0.25) shape = rms > 0.2 ? "D" : "F";
  else if (midEnergy > 0.5) shape = "E";
  else if (rms > 0.25) shape = "D";
  else shape = "C";

  return { shape, rms, centroid, midEnergy };
}

function extractFeatures(
  timeBuf: Uint8Array<ArrayBuffer>,
  freqBuf: Uint8Array<ArrayBuffer>
): {
  rms: number;
  centroid: number;
  midEnergy: number;
  highEnergy: number;
  zcr: number;
} {
  // RMS + zero-crossing rate over normalized time-domain samples. ZCR
  // spikes during fricatives (F/V/S/SH) — used below to separate the
  // labiodental "G" shape from plain bright vowels.
  let s = 0;
  let zc = 0;
  let prev = timeBuf[0] - 128;
  for (let i = 0; i < timeBuf.length; i++) {
    const v = (timeBuf[i] - 128) / 128;
    s += v * v;
    const cur = timeBuf[i] - 128;
    if ((cur >= 0) !== (prev >= 0)) zc++;
    prev = cur;
  }
  const rms = Math.sqrt(s / timeBuf.length);
  const zcr = zc / timeBuf.length;

  // Spectral centroid + mid-band + high-band energy fractions.
  let total = 0;
  let weighted = 0;
  let mids = 0;
  let highs = 0;
  const loMid = freqBuf.length * 0.2;
  const hiMid = freqBuf.length * 0.5;
  for (let i = 0; i < freqBuf.length; i++) {
    const m = freqBuf[i];
    total += m;
    weighted += m * i;
    if (i > loMid && i < hiMid) mids += m;
    if (i >= hiMid) highs += m;
  }
  const centroid = total > 0 ? weighted / total / freqBuf.length : 0;
  const midEnergy = total > 0 ? mids / total : 0;
  const highEnergy = total > 0 ? highs / total : 0;
  return { rms, centroid, midEnergy, highEnergy, zcr };
}

// Per-track state held across RAF frames. One instance per talking face.
// Keeps an envelope follower + adaptive peak so openness is normalized to
// whatever this TTS backend's actual output level is, plus hysteresis on
// the chosen shape so single-frame spikes don't flick the mouth.
export type LipSyncState = {
  envelope: number;    // Asymmetric-follower RMS envelope (fast attack / slow release)
  centroid: number;    // EMA-smoothed spectral centroid
  midEnergy: number;   // EMA-smoothed mid-band energy fraction
  highEnergy: number;  // EMA-smoothed high-band fraction — fricative detector
  zcr: number;         // EMA-smoothed zero-crossing rate — fricative confirm
  peak: number;        // Rolling peak of envelope — used to normalize openness
  prevShape: MouthShape;
  heldFrames: number;  // Frames we've been on prevShape
};

export function createLipSyncState(): LipSyncState {
  return {
    envelope: 0,
    centroid: 0,
    midEnergy: 0,
    highEnergy: 0,
    zcr: 0,
    peak: 0,
    prevShape: "X",
    heldFrames: 0,
  };
}

// Tuned for an RAF loop at ~60 fps with AnalyserNode.smoothingTimeConstant=0.4.
// Attack/release are per-frame blend factors (not time-constants).
const ENV_ATTACK = 0.55;         // fast — mouth opens promptly on syllable onsets
const ENV_RELEASE = 0.15;        // slow — don't slam shut between syllables
const SPECTRAL_ALPHA = 0.3;      // EMA for centroid / mid
const PEAK_ATTACK = 0.6;         // peak follows envelope up quickly
const PEAK_DECAY = 0.0015;       // and drifts down slowly (~1s to halve)
const PEAK_FLOOR = 0.04;         // never normalize against a silence-level peak
const SILENCE_ENV = 0.012;       // below this → closed mouth (X)
const MIN_HOLD_FRAMES = 2;       // hysteresis — require 2 frames before reclassifying an open shape

// Stateful, smoothed lip-sync classifier. Maintains an envelope follower
// whose openness is divided by a rolling peak, so quiet TTS still opens
// the mouth and loud TTS doesn't sit on the widest shape the whole line.
// Closed→open transitions are immediate (responsiveness); open→other-open
// transitions wait MIN_HOLD_FRAMES (stability).
export function classifyShapeSmooth(
  state: LipSyncState,
  timeBuf: Uint8Array<ArrayBuffer>,
  freqBuf: Uint8Array<ArrayBuffer>
): MouthShape {
  const { rms, centroid, midEnergy, highEnergy, zcr } = extractFeatures(
    timeBuf,
    freqBuf
  );

  // Asymmetric envelope follower on instantaneous RMS.
  const a = rms > state.envelope ? ENV_ATTACK : ENV_RELEASE;
  state.envelope = state.envelope + a * (rms - state.envelope);
  state.centroid = state.centroid + SPECTRAL_ALPHA * (centroid - state.centroid);
  state.midEnergy = state.midEnergy + SPECTRAL_ALPHA * (midEnergy - state.midEnergy);
  state.highEnergy = state.highEnergy + SPECTRAL_ALPHA * (highEnergy - state.highEnergy);
  state.zcr = state.zcr + SPECTRAL_ALPHA * (zcr - state.zcr);

  // Adaptive peak. Fast up, slow down, clamped to a floor so we don't
  // amplify DC/background noise into a wide-open mouth during silence.
  if (state.envelope > state.peak) {
    state.peak = state.peak + PEAK_ATTACK * (state.envelope - state.peak);
  } else {
    state.peak = Math.max(PEAK_FLOOR, state.peak - PEAK_DECAY);
  }

  // Normalized openness in [0, 1].
  const openness = Math.min(1, state.envelope / Math.max(state.peak, PEAK_FLOOR));

  // Fricative/labiodental detector: quiet-ish, high-frequency-dominant,
  // high zero-crossing rate. F/V/TH-like sounds. Maps to the `G` atlas
  // (upper teeth on lower lip). Gate before the plain centroid branches
  // so we don't steal every bright vowel.
  const isFricative =
    openness < 0.55 &&
    state.highEnergy > 0.32 &&
    state.centroid > 0.45 &&
    state.zcr > 0.12;

  let next: MouthShape;
  if (state.envelope < SILENCE_ENV) {
    next = "X";
  } else if (openness < 0.2) {
    next = "A";
  } else if (isFricative) {
    // F/V labiodental — teeth-on-lip posture.
    next = "G";
  } else if (state.centroid > 0.5) {
    // Bright / front vowels — wide but not tall.
    next = openness > 0.65 ? "C" : "B";
  } else if (state.centroid < 0.28) {
    // Dark / back vowels — rounded, tall.
    next = openness > 0.7 ? "D" : "F";
  } else if (state.midEnergy > 0.48) {
    // Mid-frequency, rounded vowel — "oh"-like.
    next = "E";
  } else if (openness > 0.35 && openness < 0.7) {
    // Mid-centroid, moderate openness, not clearly E — L-lateral /
    // half-open transitional vowels. Maps to the `H` atlas.
    next = "H";
  } else {
    next = openness > 0.75 ? "D" : "C";
  }

  // Hysteresis. Closing or opening-from-closed is always immediate (feels
  // sluggish otherwise). Lateral moves between open shapes require a few
  // frames of agreement so a single jittery spectrum doesn't flap the
  // mouth shape at 60 Hz.
  const prev = state.prevShape;
  const instantTransition =
    next === "X" || prev === "X" || next === prev;
  if (instantTransition) {
    state.prevShape = next;
    state.heldFrames = next === prev ? state.heldFrames + 1 : 0;
    return next;
  }
  state.heldFrames += 1;
  if (state.heldFrames < MIN_HOLD_FRAMES) {
    return prev;
  }
  state.prevShape = next;
  state.heldFrames = 0;
  return next;
}
