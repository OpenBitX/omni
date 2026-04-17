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

export function FaceVoice({ shape }: FaceVoiceProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

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

  // Autoplay the eyes video. Muted + playsInline so iOS Safari is happy.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    const p = v.play();
    if (p && typeof p.catch === "function") {
      // Autoplay can be blocked until a user gesture; we retry on the next
      // pointer event in practice the tap that locks the face already
      // clears the policy.
      p.catch(() => {});
    }
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
      {/* Eyes layer. VP9+alpha webm has real transparency; the mp4 fallback
          is matted on black so we screen-blend to key it out. Both paths
          land visually identical. */}
      <video
        ref={videoRef}
        muted
        loop
        playsInline
        autoPlay
        preload="auto"
        className="absolute"
        style={{
          top: "10%",
          left: "50%",
          width: "62%",
          transform: "translateX(-50%)",
          mixBlendMode: "screen",
        }}
      >
        <source src="/facevoice/eyes.webm" type="video/webm" />
        <source src="/facevoice/eyes.mp4" type="video/mp4" />
      </video>

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
// Thresholds are their originals — good enough for a demo, and tuneable in
// isolation without touching the rendering layer.
export function classifyShape(
  timeBuf: Uint8Array<ArrayBuffer>,
  freqBuf: Uint8Array<ArrayBuffer>
): { shape: MouthShape; rms: number; centroid: number; midEnergy: number } {
  // RMS over normalized time-domain samples.
  let s = 0;
  for (let i = 0; i < timeBuf.length; i++) {
    const v = (timeBuf[i] - 128) / 128;
    s += v * v;
  }
  const rms = Math.sqrt(s / timeBuf.length);

  // Spectral centroid + mid-band energy fraction.
  let total = 0;
  let weighted = 0;
  let mids = 0;
  const loMid = freqBuf.length * 0.2;
  const hiMid = freqBuf.length * 0.5;
  for (let i = 0; i < freqBuf.length; i++) {
    const m = freqBuf[i];
    total += m;
    weighted += m * i;
    if (i > loMid && i < hiMid) mids += m;
  }
  const centroid = total > 0 ? weighted / total / freqBuf.length : 0;
  const midEnergy = total > 0 ? mids / total : 0;

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
