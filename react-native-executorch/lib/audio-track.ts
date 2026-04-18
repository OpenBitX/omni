// Per-track audio graph + decoded playback for the streaming TTS endpoint.
// Mirrors the browser's source → analyser → gain → destination wiring,
// using react-native-audio-api which implements the Web Audio API natively
// on iOS + Android.
//
// Each locked track owns one of these. The tracker hot-path reads the
// analyser each RAF via classifyShapeSmooth to drive the mouth, and writes
// gain.value = opacity * voiceBoost so voices fade with their face.
//
// Streaming note: react-native-audio-api's decodeAudioData requires a
// complete buffer, so we fetch the TTS body to completion before starting
// playback. Expect ~400–900 ms TTFB on Fish — noticeable but not fatal.
// Upgrade to chunked append (if the lib exposes it) once we see the
// streaming-seconds figure in telemetry.

import {
  AudioContext,
  type AudioBufferSourceNode,
  type AnalyserNode,
  type GainNode,
} from "react-native-audio-api";

import { API_BASE } from "@/lib/api";

export type TrackAudio = {
  ctx: AudioContext;
  gain: GainNode;
  analyser: AnalyserNode;
  timeBuf: Uint8Array;
  freqBuf: Uint8Array;
  current: AudioBufferSourceNode | null;
  disposed: boolean;
};

// Shared context — iOS complains if you spin up many AudioContexts and
// react-native-audio-api's AnalyserNode is per-source anyway. One context,
// N analyser+gain pairs works cleanly.
let sharedCtx: AudioContext | null = null;
function getCtx(): AudioContext {
  if (!sharedCtx) sharedCtx = new AudioContext();
  return sharedCtx;
}

export function createTrackAudio(): TrackAudio {
  const ctx = getCtx();
  const gain = ctx.createGain();
  gain.gain.value = 1;
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.4;
  // source → analyser → gain → destination
  analyser.connect(gain);
  gain.connect(ctx.destination);
  return {
    ctx,
    gain,
    analyser,
    timeBuf: new Uint8Array(analyser.fftSize),
    freqBuf: new Uint8Array(analyser.frequencyBinCount),
    current: null,
    disposed: false,
  };
}

export function disposeTrackAudio(t: TrackAudio) {
  if (t.disposed) return;
  t.disposed = true;
  try {
    t.current?.stop();
  } catch {}
  try {
    t.current?.disconnect();
  } catch {}
  try {
    t.analyser.disconnect();
  } catch {}
  try {
    t.gain.disconnect();
  } catch {}
  t.current = null;
}

// Read the latest frame of analyser data into the track's reusable buffers.
// Caller passes timeBuf/freqBuf into classifyShapeSmooth.
export function readAnalyser(t: TrackAudio): {
  timeBuf: Uint8Array;
  freqBuf: Uint8Array;
} {
  if (!t.disposed) {
    // @ts-expect-error — RNAA's signature matches Web Audio exactly
    t.analyser.getByteTimeDomainData(t.timeBuf);
    // @ts-expect-error — see above
    t.analyser.getByteFrequencyData(t.freqBuf);
  }
  return { timeBuf: t.timeBuf, freqBuf: t.freqBuf };
}

export type SpeakResult = { promise: Promise<void>; cancel: () => void };

// Stop any in-flight playback and start a new one with the given line.
// Returns a promise that resolves when the audio finishes (or aborts early).
export function speakLine(
  t: TrackAudio,
  opts: { text: string; voiceId: string; turnId?: string }
): SpeakResult {
  let cancelled = false;
  try {
    t.current?.stop();
    t.current?.disconnect();
  } catch {}
  t.current = null;

  const abort = new AbortController();
  const cancel = () => {
    cancelled = true;
    abort.abort();
    try {
      t.current?.stop();
    } catch {}
    t.current = null;
  };

  const promise = (async () => {
    const res = await fetch(`${API_BASE}/api/tts/stream`, {
      method: "POST",
      signal: abort.signal,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        text: opts.text,
        voiceId: opts.voiceId,
        turnId: opts.turnId,
      }),
    });
    if (cancelled) return;
    if (!res.ok || !res.body) {
      throw new Error(`tts ${res.status}`);
    }
    const bytes = await res.arrayBuffer();
    if (cancelled || t.disposed) return;
    // @ts-expect-error — RNAA decodeAudioData takes an ArrayBuffer
    const audioBuf = await t.ctx.decodeAudioData(bytes);
    if (cancelled || t.disposed) return;

    const src = t.ctx.createBufferSource();
    src.buffer = audioBuf;
    src.connect(t.analyser);
    t.current = src;

    return new Promise<void>((resolve) => {
      src.onended = () => {
        if (t.current === src) t.current = null;
        resolve();
      };
      src.start();
    });
  })();

  return { promise, cancel };
}

export function setGain(t: TrackAudio, v: number) {
  if (t.disposed) return;
  const clamped = Math.max(0, Math.min(VOICE_GAIN_HARDCAP, v));
  t.gain.gain.value = clamped;
}

const VOICE_GAIN_HARDCAP = 6;
