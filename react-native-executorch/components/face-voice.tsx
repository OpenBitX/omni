import { useEffect, useMemo } from "react";
import { Image, StyleSheet, View } from "react-native";
import { VideoView, useVideoPlayer } from "expo-video";

// "Talking face" renderer — RN port of the browser component. Same 9-way
// mouth atlas (shape-A … shape-X, from OpenBitX/face_voice), same
// FACE_VOICE_WIDTH/HEIGHT contract so Tracker's box→face math is unchanged.
//
// classifyShapeSmooth is kept verbatim from the browser port — it's pure
// number crunching over two Uint8Array buffers, and the same heuristic
// reads correctly off react-native-audio-api's AnalyserNode output.

export type MouthShape = "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "X";

export const FACE_VOICE_WIDTH = 280;
export const FACE_VOICE_HEIGHT = 160;

// Static map — RN's bundler requires static `require()` paths. Preloaded by
// the bundler so shape swaps never touch the network.
const SHAPE_SOURCES: Record<MouthShape, number> = {
  A: require("@/assets/facevoice/shape-A.png"),
  B: require("@/assets/facevoice/shape-B.png"),
  C: require("@/assets/facevoice/shape-C.png"),
  D: require("@/assets/facevoice/shape-D.png"),
  E: require("@/assets/facevoice/shape-E.png"),
  F: require("@/assets/facevoice/shape-F.png"),
  G: require("@/assets/facevoice/shape-G.png"),
  H: require("@/assets/facevoice/shape-H.png"),
  X: require("@/assets/facevoice/shape-X.png"),
};

const EYES_SOURCE = require("@/assets/facevoice/eyes.mp4");

type FaceVoiceProps = {
  shape: MouthShape;
};

export function FaceVoice({ shape }: FaceVoiceProps) {
  const player = useVideoPlayer(EYES_SOURCE, (p) => {
    p.loop = true;
    p.muted = true;
    p.play();
  });

  useEffect(() => {
    if (!player.playing) {
      try {
        player.play();
      } catch {
        /* hot reload races */
      }
    }
  }, [player]);

  const mouthSource = useMemo(() => SHAPE_SOURCES[shape] ?? SHAPE_SOURCES.X, [shape]);

  return (
    <View style={styles.root} pointerEvents="none">
      <VideoView
        player={player}
        style={styles.eyes}
        contentFit="cover"
        nativeControls={false}
      />
      <Image source={mouthSource} style={styles.mouth} resizeMode="contain" />
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    width: FACE_VOICE_WIDTH,
    height: FACE_VOICE_HEIGHT,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.55,
    shadowRadius: 10,
    elevation: 8,
  },
  eyes: {
    position: "absolute",
    top: "10%",
    left: "19%",
    width: "62%",
    height: "55%",
    backgroundColor: "transparent",
  },
  mouth: {
    position: "absolute",
    bottom: "8%",
    left: "24%",
    width: "52%",
    height: "40%",
  },
});

// === Lip-sync classifier (ported verbatim from browser) ==================

export function classifyShape(
  timeBuf: Uint8Array,
  freqBuf: Uint8Array
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
  timeBuf: Uint8Array,
  freqBuf: Uint8Array
): {
  rms: number;
  centroid: number;
  midEnergy: number;
  highEnergy: number;
  zcr: number;
} {
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

export type LipSyncState = {
  envelope: number;
  centroid: number;
  midEnergy: number;
  highEnergy: number;
  zcr: number;
  peak: number;
  prevShape: MouthShape;
  heldFrames: number;
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

const ENV_ATTACK = 0.55;
const ENV_RELEASE = 0.15;
const SPECTRAL_ALPHA = 0.3;
const PEAK_ATTACK = 0.6;
const PEAK_DECAY = 0.0015;
const PEAK_FLOOR = 0.04;
const SILENCE_ENV = 0.012;
const MIN_HOLD_FRAMES = 2;

export function classifyShapeSmooth(
  state: LipSyncState,
  timeBuf: Uint8Array,
  freqBuf: Uint8Array
): MouthShape {
  const { rms, centroid, midEnergy, highEnergy, zcr } = extractFeatures(
    timeBuf,
    freqBuf
  );

  const a = rms > state.envelope ? ENV_ATTACK : ENV_RELEASE;
  state.envelope = state.envelope + a * (rms - state.envelope);
  state.centroid = state.centroid + SPECTRAL_ALPHA * (centroid - state.centroid);
  state.midEnergy = state.midEnergy + SPECTRAL_ALPHA * (midEnergy - state.midEnergy);
  state.highEnergy = state.highEnergy + SPECTRAL_ALPHA * (highEnergy - state.highEnergy);
  state.zcr = state.zcr + SPECTRAL_ALPHA * (zcr - state.zcr);

  if (state.envelope > state.peak) {
    state.peak = state.peak + PEAK_ATTACK * (state.envelope - state.peak);
  } else {
    state.peak = Math.max(PEAK_FLOOR, state.peak - PEAK_DECAY);
  }

  const openness = Math.min(1, state.envelope / Math.max(state.peak, PEAK_FLOOR));

  const isFricative =
    openness < 0.55 &&
    state.highEnergy > 0.32 &&
    state.centroid > 0.45 &&
    state.zcr > 0.12;

  let next: MouthShape;
  if (state.envelope < SILENCE_ENV) next = "X";
  else if (openness < 0.2) next = "A";
  else if (isFricative) next = "G";
  else if (state.centroid > 0.5) next = openness > 0.65 ? "C" : "B";
  else if (state.centroid < 0.28) next = openness > 0.7 ? "D" : "F";
  else if (state.midEnergy > 0.48) next = "E";
  else if (openness > 0.35 && openness < 0.7) next = "H";
  else next = openness > 0.75 ? "D" : "C";

  const prev = state.prevShape;
  const instantTransition = next === "X" || prev === "X" || next === prev;
  if (instantTransition) {
    state.prevShape = next;
    state.heldFrames = next === prev ? state.heldFrames + 1 : 0;
    return next;
  }
  state.heldFrames += 1;
  if (state.heldFrames < MIN_HOLD_FRAMES) return prev;
  state.prevShape = next;
  state.heldFrames = 0;
  return next;
}
