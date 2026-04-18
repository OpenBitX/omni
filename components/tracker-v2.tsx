"use client";

// Tracker V2 — SAM2-tiny backend-driven segmentation/tracking.
//
// Deliberately parallel to `components/tracker.tsx`: shares NO runtime
// state, NO refs, NO loops. The two can be cross-tested side by side by
// keeping one tab on `/` and another on `/v2`. What IS shared:
//   - `app/actions.ts` server actions (generateLine, converseWithObject)
//   - `/api/tts/stream` route
//   - `components/face-voice.tsx` (FaceVoice + classifyShapeSmooth)
//   - `lib/iou.ts` anchor + EMA helpers (pure)
//
// Everything else — camera, WebSocket protocol, tracking loop, audio
// plumbing — is its own simpler implementation, because V2 is
// single-track and the multi-track complexity in V1 isn't needed yet.
//
// V1 scope:
//   - single locked object at a time
//   - tap to lock, Esc / stop button to release
//   - FaceVoice rides on the SAM2-reported bbox via the same anchor math
//     used by V1, smoothed with BoxEMA so ~15 fps server updates still
//     feel glassy
//   - generateLine bundled-vision first tap → TTS via /api/tts/stream
//     (buffered, not MediaSource — simpler; we can add streaming later
//     if the click-to-sound latency isn't acceptable)
//   - mic button → converseWithObject
//
// Backend: `npm run server:v2` on port 8001. See server-v2/README.md.

import { useCallback, useEffect, useRef, useState } from "react";
import { converseWithObject } from "@/app/actions";
import {
  FACE_VOICE_HEIGHT,
  FACE_VOICE_WIDTH,
  FaceVoice,
  classifyShapeSmooth,
  createLipSyncState,
  type LipSyncState,
  type MouthShape,
} from "@/components/face-voice";
import {
  anchorFromPoint,
  applyAnchor,
  makeBox,
  newBoxEMA,
  seedBoxEMA,
  smoothBox,
  type Anchor,
  type Box,
  type BoxEMA,
} from "@/lib/iou";

// ---------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------

const WS_URL = "ws://localhost:8001/sam2/ws";
const HTTP_URL = "http://localhost:8001";
const SEND_WIDTH = 640; // match server's MAX_LONG_EDGE default
const JPEG_QUALITY = 0.72;
const STALL_MS = 6000;
const RECONNECT_MIN_MS = 500;
const RECONNECT_MAX_MS = 8000;

// Split-alpha EMA, same philosophy as V1 (lib/iou.ts defaults): position
// moves fast so the face doesn't lag, size moves slow so the face
// doesn't "breathe" with bbox jitter.
const BOX_POS_ALPHA = 0.6;
const BOX_SIZE_ALPHA = 0.2;

// FaceVoice is FACE_VOICE_WIDTH × FACE_VOICE_HEIGHT CSS px at scale=1.
// Scale the face so its width ≈ `FACE_BBOX_FRACTION × min(bboxW, bboxH)`
// in display pixels, clamped to reasonable mins/maxes.
const FACE_BBOX_FRACTION = 0.9;
const FACE_SCALE_MIN = 0.35;
const FACE_SCALE_MAX = 1.6;

const CAPTION_LINGER_MS = 6500;
const ERROR_LINGER_MS = 4500;

// ---------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------

type Phase = "idle" | "opening" | "live" | "error";

type ServerEvent =
  | { type: "initialized"; trackId: string; bbox: [number, number, number, number]; score: number }
  | { type: "track"; bbox: [number, number, number, number] | null; score: number; stale?: boolean }
  | { type: "lost"; reason: string }
  | { type: "error"; message: string };

type ActiveTrack = {
  id: string;
  // Monotonic client-side generation number, bumped on each tap. Used
  // to filter out stale `track` events still in flight from the previous
  // session — without this, an old-object bbox corrupts the new track's
  // EMA for one frame, producing a visible "face jump to old object".
  gen: number;
  anchor: Anchor;
  ema: BoxEMA;
  // Latest smoothed box in normalized [0,1] coords.
  smoothed: Box;
  // Last raw bbox from the server, for debugging / visible score.
  lastScore: number;
  // Persona pinned after the first generateLine — reused for retaps +
  // converseWithObject so we skip the bundled vision call on subsequent
  // turns (same optimization V1 does).
  voiceId: string | null;
  description: string | null;
  // UI state mirrored to React; refs keep the hot-path writes off render.
  caption: string | null;
  speaking: boolean;
  thinking: boolean;
  opacity: number; // 0..1 — 0 while waiting for first bbox, 1 when locked
  history: { role: "user" | "assistant"; content: string }[];
};

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

function boxFromNormalizedBbox(b: [number, number, number, number]): Box {
  return makeBox(b[0], b[1], b[2], b[3]);
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

// Guard against malformed bboxes — NaN, infinities, or degenerate sizes
// that would produce garbage face positions. Server-side SAM2 should
// only ever emit sane values, but a bad decode or a future server bug
// shouldn't leak into the RAF math. Cheap isFinite + range check.
function validBbox(b: [number, number, number, number] | null | undefined): b is [number, number, number, number] {
  if (!b) return false;
  const [cx, cy, w, h] = b;
  return (
    Number.isFinite(cx) &&
    Number.isFinite(cy) &&
    Number.isFinite(w) &&
    Number.isFinite(h) &&
    w > 0.005 &&
    h > 0.005 &&
    w <= 1.05 &&
    h <= 1.05 &&
    cx >= -0.05 &&
    cx <= 1.05 &&
    cy >= -0.05 &&
    cy <= 1.05
  );
}

// Capture the current video frame and return a JPEG blob sized to
// SEND_WIDTH × (aspect-scaled height). Shared for both WS frames and
// the one-off crop we hand to generateLine on lock.
function captureFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  mirrorX: boolean,
  quality = JPEG_QUALITY,
): Promise<Blob | null> {
  return new Promise((resolve) => {
    if (!video.videoWidth) {
      resolve(null);
      return;
    }
    const aspect = video.videoHeight / video.videoWidth;
    canvas.width = SEND_WIDTH;
    canvas.height = Math.round(SEND_WIDTH * aspect);
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      resolve(null);
      return;
    }
    try {
      ctx.save();
      if (mirrorX) {
        ctx.scale(-1, 1);
        ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
      } else {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }
      ctx.restore();
    } catch {
      resolve(null);
      return;
    }
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", quality);
  });
}

async function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("FileReader"));
    reader.readAsDataURL(blob);
  });
}

// ---------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------

export function TrackerV2() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const sendCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inFlightRef = useRef(false);
  const wantConnectedRef = useRef(false);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const watchdogRef = useRef<number | null>(null);
  const lastFrameAtRef = useRef(0);
  const mirrorXRef = useRef(false); // front camera → mirror preview

  // Mutable "hot path" track state — updated every WS response, read
  // every animation frame. We mirror the subset we want to render into
  // `trackUI` state so React only re-renders when visual state changes,
  // not per-frame bbox jitter.
  const trackRef = useRef<ActiveTrack | null>(null);
  const speakGenRef = useRef(0); // cancel stale generateLine / converseWithObject
  // Bumped synchronously on each tap. Every outgoing WS frame + every
  // incoming WS event captures this so stale responses from the previous
  // track can't write into the new track's EMA / caption / audio pipeline.
  const trackGenRef = useRef(0);
  // In-flight /api/speak fetch (bundled VLM + TTS). Aborted on release /
  // retap so we don't burn LLM/TTS credits on replies the user won't hear
  // and don't risk the response landing after a new track is locked.
  const speakAbortRef = useRef<AbortController | null>(null);

  // Face display — driven by RAF from trackRef + anchor.
  const faceWrapRef = useRef<HTMLDivElement | null>(null);
  const mouthShapeRef = useRef<MouthShape>("X");
  const [mouthShape, setMouthShape] = useState<MouthShape>("X");

  // Audio: one shared graph for V2 since we only have one track.
  // Persistent `<audio>` + `MediaElementAudioSourceNode` wired once into
  // the analyser — matches the OpenBitX/face_voice reference: swap
  // `audio.src` to a blob URL per reply, `audio.play()`, analyser feeds
  // the mouth-shape classifier. `createMediaElementSource` can only be
  // called once per element per context, so these refs stay alive for
  // the whole session.
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const audioElRef = useRef<HTMLAudioElement | null>(null);
  const mediaElSourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const currentBlobUrlRef = useRef<string | null>(null);
  const freqDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const timeDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const lipSyncRef = useRef<LipSyncState>(createLipSyncState());

  // Mic recording state
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recorderChunksRef = useRef<Blob[]>([]);
  const [listening, setListening] = useState(false);

  // React state — only what drives visible UI.
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [caption, setCaption] = useState<string | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [serverOnline, setServerOnline] = useState<boolean | null>(null);
  const [trackLocked, setTrackLocked] = useState(false);
  const [lastScore, setLastScore] = useState(0);

  const captionClearTimerRef = useRef<number | null>(null);

  // -----------------------------------------------------------------
  // Server health poll — tells the user up front whether the backend
  // is reachable. Same vibe as V1's YOLO load HUD.
  // -----------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const r = await fetch(`${HTTP_URL}/health`, { cache: "no-store" });
        if (cancelled) return;
        setServerOnline(r.ok);
      } catch {
        if (!cancelled) setServerOnline(false);
      }
    };
    check();
    const id = window.setInterval(check, 4000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  // -----------------------------------------------------------------
  // Audio graph
  // -----------------------------------------------------------------
  const ensureAudioGraph = useCallback((): AudioContext | null => {
    if (typeof window === "undefined") return null;
    const Ctor: typeof AudioContext | undefined =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!Ctor) return null;
    const ctx = audioCtxRef.current ?? new Ctor();
    audioCtxRef.current = ctx;
    if (!analyserRef.current || !gainRef.current) {
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.4;
      const gain = ctx.createGain();
      gain.gain.value = 1;
      analyser.connect(gain);
      gain.connect(ctx.destination);
      analyserRef.current = analyser;
      gainRef.current = gain;
      freqDataRef.current = new Uint8Array(
        new ArrayBuffer(analyser.frequencyBinCount),
      );
      timeDataRef.current = new Uint8Array(new ArrayBuffer(analyser.fftSize));
    }
    if (!audioElRef.current) {
      const audioEl = new Audio();
      audioEl.preload = "auto";
      audioEl.crossOrigin = "anonymous";
      audioElRef.current = audioEl;
    }
    if (!mediaElSourceRef.current && audioElRef.current && analyserRef.current) {
      try {
        const src = ctx.createMediaElementSource(audioElRef.current);
        src.connect(analyserRef.current);
        mediaElSourceRef.current = src;
      } catch (e) {
        // eslint-disable-next-line no-console
        console.log(
          `[v2] createMediaElementSource failed: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    }
    return ctx;
  }, []);

  const stopCurrentAudio = useCallback(() => {
    const el = audioElRef.current;
    if (el) {
      try {
        el.pause();
        el.removeAttribute("src");
        el.load();
      } catch {
        // ignore
      }
    }
    const url = currentBlobUrlRef.current;
    if (url) {
      try {
        URL.revokeObjectURL(url);
      } catch {
        // ignore
      }
      currentBlobUrlRef.current = null;
    }
    setSpeaking(false);
  }, []);

  // -----------------------------------------------------------------
  // Camera
  // -----------------------------------------------------------------
  const ensureCamera = useCallback(async () => {
    if (streamRef.current) return streamRef.current;
    // Prefer rear camera for laptops with multiple feeds + mobile; fall
    // back to user-facing if the environment constraint is rejected.
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: 1280, height: 720 },
        audio: false,
      });
      mirrorXRef.current = false;
    } catch {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 1280, height: 720 },
        audio: false,
      });
      mirrorXRef.current = true;
    }
    // If the camera goes away mid-session (user yanks permission, USB
    // webcam unplugged, OS privacy overlay), the video frames stop but
    // the WS keeps sending ghost lockstep frames from the last decoded
    // pixels. Surface it as an error and drop back to idle.
    stream.getVideoTracks().forEach((t) => {
      t.addEventListener("ended", () => {
        if (streamRef.current === stream) {
          setError("camera disconnected");
          disconnectRef.current?.();
        }
      });
      t.addEventListener("mute", () => {
        if (streamRef.current === stream) {
          setError("camera muted by the OS");
        }
      });
    });
    streamRef.current = stream;
    const v = videoRef.current;
    if (v) {
      v.srcObject = stream;
      try {
        await v.play();
      } catch {
        // autoplay blocked until user gesture — fine, we connect on button press
      }
    }
    return stream;
  }, []);

  // `disconnect` is declared below — the camera `ended` listener above
  // needs a stable reference, so we late-bind via a ref.
  const disconnectRef = useRef<(() => void) | null>(null);

  // -----------------------------------------------------------------
  // WebSocket lockstep loop
  // -----------------------------------------------------------------
  const sendFrame = useCallback(async () => {
    const v = videoRef.current;
    const c = sendCanvasRef.current;
    const ws = wsRef.current;
    if (!v || !c || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (inFlightRef.current) return;
    if (!v.videoWidth) {
      window.setTimeout(() => void sendFrame(), 100);
      return;
    }
    const blob = await captureFrame(v, c, mirrorXRef.current);
    if (!blob) return;
    const buf = await blob.arrayBuffer();
    const ws2 = wsRef.current;
    if (!ws2 || ws2.readyState !== WebSocket.OPEN) return;
    try {
      inFlightRef.current = true;
      ws2.send(buf);
    } catch {
      inFlightRef.current = false;
    }
  }, []);

  // Drop the current track and tell the server to release state. Caller
  // is responsible for any downstream UI reset that isn't track-local.
  const resetTrackServer = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: "reset" }));
      } catch {
        // ignore
      }
    }
  }, []);

  const abortInFlightSpeak = useCallback(() => {
    const ctrl = speakAbortRef.current;
    if (ctrl) {
      try {
        ctrl.abort();
      } catch {
        // ignore
      }
      speakAbortRef.current = null;
    }
  }, []);

  const clearCaptionSoon = useCallback(() => {
    if (captionClearTimerRef.current != null) {
      clearTimeout(captionClearTimerRef.current);
    }
    captionClearTimerRef.current = window.setTimeout(() => {
      setCaption(null);
      captionClearTimerRef.current = null;
    }, CAPTION_LINGER_MS);
  }, []);

  const releaseTrack = useCallback(() => {
    speakGenRef.current++;
    trackGenRef.current++;
    abortInFlightSpeak();
    stopCurrentAudio();
    resetTrackServer();
    trackRef.current = null;
    setTrackLocked(false);
    setCaption(null);
    setSpeaking(false);
    setThinking(false);
    setLastScore(0);
    mouthShapeRef.current = "X";
    setMouthShape("X");
    inFlightRef.current = false;
  }, [abortInFlightSpeak, resetTrackServer, stopCurrentAudio]);

  // -----------------------------------------------------------------
  // Speak pipeline — streams mp3 bytes from either `/api/speak` (first
  // tap, bundled VLM + TTS) or `/api/tts/stream` (mic reply) into a
  // MediaSource SourceBuffer, so playback starts on the first chunk.
  // Mirrors the V1 tracker's `playViaMediaSource` minus the per-track
  // plumbing (V2 is single-track). Falls back to a buffered decode on
  // browsers without MSE audio/mpeg (older iOS).
  // -----------------------------------------------------------------
  const playStream = useCallback(
    async (
      respBody: ReadableStream<Uint8Array>,
      callGen: number,
    ): Promise<void> => {
      const ctx = ensureAudioGraph();
      if (!ctx) throw new Error("audio unavailable");
      if (ctx.state === "suspended") {
        try {
          await ctx.resume();
        } catch {
          // user gesture required; we were called from a click so this should resolve
        }
      }
      const audioEl = audioElRef.current;
      if (!audioEl) throw new Error("audio element missing");

      // Drain the streamed response to a single blob. OpenBitX/face_voice
      // pattern: `audio.src = URL.createObjectURL(blob)`, wait for
      // loadedmetadata, then play(). The persistent audio element's
      // MediaElementSource is already wired into the analyser so lip-sync
      // works on every reply without per-call graph setup.
      const blob = await new Response(respBody).blob();
      if (callGen !== speakGenRef.current) return;

      stopCurrentAudio();
      lipSyncRef.current = createLipSyncState();

      const url = URL.createObjectURL(blob);
      currentBlobUrlRef.current = url;
      audioEl.src = url;

      try {
        await new Promise<void>((resolve, reject) => {
          const onMeta = () => {
            cleanup();
            resolve();
          };
          const onErr = () => {
            cleanup();
            reject(new Error("audio decode failed"));
          };
          const cleanup = () => {
            audioEl.removeEventListener("loadedmetadata", onMeta);
            audioEl.removeEventListener("error", onErr);
          };
          audioEl.addEventListener("loadedmetadata", onMeta, { once: true });
          audioEl.addEventListener("error", onErr, { once: true });
        });
      } catch (err) {
        if (currentBlobUrlRef.current === url) {
          try {
            URL.revokeObjectURL(url);
          } catch {
            // ignore
          }
          currentBlobUrlRef.current = null;
        }
        throw err;
      }

      if (callGen !== speakGenRef.current) return;

      const onEnded = () => {
        audioEl.removeEventListener("ended", onEnded);
        if (currentBlobUrlRef.current === url) {
          try {
            URL.revokeObjectURL(url);
          } catch {
            // ignore
          }
          currentBlobUrlRef.current = null;
        }
        setSpeaking(false);
        mouthShapeRef.current = "X";
        setMouthShape("X");
      };
      audioEl.addEventListener("ended", onEnded);

      setSpeaking(true);
      try {
        await audioEl.play();
      } catch (err) {
        onEnded();
        throw new Error(
          `audio.play rejected: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    },
    [ensureAudioGraph, stopCurrentAudio],
  );

  // Thin wrapper for the mic-reply path — fetches `/api/tts/stream` with
  // the reply text + pinned voice, then hands the response body to
  // `playStream`. First-tap skips this and streams /api/speak directly.
  const playTtsText = useCallback(
    async (
      text: string,
      voiceId: string | null,
      callGen: number,
    ): Promise<void> => {
      const resp = await fetch("/api/tts/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          voiceId: voiceId ?? "",
          turnId: `v2-${callGen}`,
          lang: "en",
        }),
      });
      if (callGen !== speakGenRef.current) {
        try {
          await resp.body?.cancel();
        } catch {
          // ignore
        }
        return;
      }
      if (!resp.ok || !resp.body) {
        const err = await resp.text().catch(() => "");
        throw new Error(`tts ${resp.status}: ${err.slice(0, 120)}`);
      }
      await playStream(resp.body, callGen);
    },
    [playStream],
  );

  // -----------------------------------------------------------------
  // WS handlers
  // -----------------------------------------------------------------
  const handleServerEvent = useCallback(
    (payload: ServerEvent) => {
      // Capture the gen at the moment the event arrives. If the user
      // retaps between this event's enqueue and its handler running, the
      // track ref now points at a DIFFERENT session; applying this
      // event's bbox/id would corrupt the new track. Dropping stale
      // events is the only reliable fix — order-of-arrival guarantees
      // don't help because taps are local.
      const track = trackRef.current;
      const gen = trackGenRef.current;
      if (!track || track.gen !== gen) return;

      if (payload.type === "initialized") {
        if (!validBbox(payload.bbox)) {
          setError("server returned invalid initial bbox");
          releaseTrack();
          return;
        }
        // `trackRef` was pre-seeded on tap with a provisional anchor
        // computed against an estimated box; now that the server has
        // given us the real bbox, re-anchor against it so the face
        // actually sits where the user tapped.
        const b = boxFromNormalizedBbox(payload.bbox);
        seedBoxEMA(track.ema, b);
        track.smoothed = b;
        track.id = payload.trackId;
        track.lastScore = payload.score;
        track.opacity = 1;
        setTrackLocked(true);
        setLastScore(payload.score);
        return;
      }
      if (payload.type === "track") {
        track.lastScore = payload.score;
        setLastScore(payload.score);
        if (payload.bbox && !payload.stale && validBbox(payload.bbox)) {
          const b = boxFromNormalizedBbox(payload.bbox);
          track.smoothed = smoothBox(track.ema, b);
          track.opacity = 1;
        } else {
          // Stale / missing / malformed frame — fade slightly but don't drop.
          track.opacity = Math.max(0.5, track.opacity - 0.1);
        }
        return;
      }
      if (payload.type === "lost") {
        releaseTrack();
        return;
      }
      if (payload.type === "error") {
        // Benign "no active track" errors race with our own release — we
        // send a frame, release happens, server replies with an error.
        // Swallow these; they're not user-facing.
        if (/no active track/i.test(payload.message)) return;
        setError(payload.message);
        return;
      }
    },
    [releaseTrack],
  );

  const teardownWS = useCallback(() => {
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) {
      ws.onopen = null;
      ws.onerror = null;
      ws.onclose = null;
      ws.onmessage = null;
      try {
        ws.close();
      } catch {
        // already closed
      }
    }
    inFlightRef.current = false;
  }, []);

  const clearReconnect = useCallback(() => {
    if (reconnectTimerRef.current != null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const clearWatchdog = useCallback(() => {
    if (watchdogRef.current != null) {
      window.clearInterval(watchdogRef.current);
      watchdogRef.current = null;
    }
  }, []);

  const openSocket = useCallback(() => {
    teardownWS();
    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      reconnectAttemptsRef.current = 0;
      setError(null);
      lastFrameAtRef.current = performance.now();
      setPhase("live");
      // Don't send frames until the user taps and we're PENDING_INIT —
      // the backend returns an error for track-mode frames when IDLE.
    };

    ws.onerror = () => {
      // close follows
    };

    ws.onclose = () => {
      clearWatchdog();
      inFlightRef.current = false;
      // Any locked track is effectively dead because the server holds
      // SAM2 session state in memory keyed to this WS. If we let the
      // UI keep showing "locked" while we reconnect, the next frame the
      // client sends would hit the server's IDLE state and return an
      // error. Force a fresh start instead.
      if (trackRef.current != null) {
        releaseTrack();
      }
      if (!wantConnectedRef.current) {
        setPhase("idle");
        return;
      }
      // Exponential backoff reconnect.
      const attempt = ++reconnectAttemptsRef.current;
      const delay = Math.min(
        RECONNECT_MAX_MS,
        RECONNECT_MIN_MS * Math.pow(2, attempt - 1),
      );
      setPhase("opening");
      setError(`server unreachable — retry in ${Math.round(delay / 1000)}s (#${attempt})`);
      reconnectTimerRef.current = window.setTimeout(() => {
        openSocket();
      }, delay);
    };

    ws.onmessage = (ev) => {
      lastFrameAtRef.current = performance.now();
      if (typeof ev.data !== "string") {
        // No binary responses from the SAM2 server — ignore.
        return;
      }
      let payload: ServerEvent | null = null;
      try {
        payload = JSON.parse(ev.data) as ServerEvent;
      } catch {
        return;
      }
      // Every server response (initialized / track / lost / error)
      // completes one lockstep round. Release the in-flight flag and
      // let the next frame go. `lost`/`error` clear the track but we
      // still want to keep streaming so a re-tap can re-init.
      inFlightRef.current = false;
      handleServerEvent(payload);
      if (trackRef.current != null) {
        requestAnimationFrame(() => void sendFrame());
      }
    };

    clearWatchdog();
    watchdogRef.current = window.setInterval(() => {
      if (!wantConnectedRef.current) return;
      const ws2 = wsRef.current;
      if (!ws2 || ws2.readyState !== WebSocket.OPEN) return;
      if (trackRef.current == null) return; // idle between taps is fine
      if (performance.now() - lastFrameAtRef.current > STALL_MS) {
        try {
          ws2.close();
        } catch {
          // ignore
        }
      }
    }, 1500);
  }, [teardownWS, clearWatchdog, handleServerEvent, sendFrame, releaseTrack]);

  // -----------------------------------------------------------------
  // Lifecycle: connect / disconnect
  // -----------------------------------------------------------------
  const connect = useCallback(async () => {
    setError(null);
    wantConnectedRef.current = true;
    setPhase("opening");
    try {
      await ensureCamera();
    } catch (e) {
      wantConnectedRef.current = false;
      setPhase("error");
      setError(e instanceof Error ? e.message : "camera blocked");
      return;
    }
    openSocket();
  }, [ensureCamera, openSocket]);

  const disconnect = useCallback(() => {
    wantConnectedRef.current = false;
    clearReconnect();
    clearWatchdog();
    releaseTrack();
    teardownWS();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    const v = videoRef.current;
    if (v) v.srcObject = null;
    setPhase("idle");
  }, [clearReconnect, clearWatchdog, releaseTrack, teardownWS]);

  useEffect(() => {
    disconnectRef.current = disconnect;
  }, [disconnect]);

  useEffect(() => {
    return () => {
      wantConnectedRef.current = false;
      clearReconnect();
      clearWatchdog();
      teardownWS();
      abortInFlightSpeak();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      // MediaRecorder: if the user unmounts mid-recording, flush the
      // chunks, stop the recorder, and kill the mic stream. Without this
      // the mic stays "in use" per the browser's privacy indicator.
      const rec = recorderRef.current;
      if (rec && rec.state !== "inactive") {
        try {
          rec.stop();
        } catch {
          // ignore
        }
      }
      recorderRef.current = null;
      recorderChunksRef.current = [];
      if (captionClearTimerRef.current != null) {
        clearTimeout(captionClearTimerRef.current);
      }
      if (audioCtxRef.current) {
        try {
          audioCtxRef.current.close();
        } catch {
          // ignore
        }
      }
    };
  }, [abortInFlightSpeak, clearReconnect, clearWatchdog, teardownWS]);

  // Tab becomes visible again: the browser may have paused the <video>
  // element. Poke it. Also — if our lockstep was mid-roundtrip when the
  // tab went hidden and the WS died silently, the STALL_MS watchdog
  // catches it on its next interval tick, which fires on visibility
  // return anyway. So we only need to nudge the video.
  useEffect(() => {
    const onVis = () => {
      if (document.visibilityState !== "visible") return;
      const v = videoRef.current;
      if (v && v.srcObject && v.paused) {
        v.play().catch(() => {
          // user gesture required; nothing we can do until next interaction
        });
      }
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, []);

  // Errors are informative but shouldn't linger forever. Auto-clear
  // after ERROR_LINGER_MS unless replaced by a newer error. Reconnect
  // messages (which count down) are re-set on every retry so they'll
  // refresh themselves; this timeout only bites steady-state errors.
  useEffect(() => {
    if (!error) return;
    const id = window.setTimeout(() => setError(null), ERROR_LINGER_MS);
    return () => window.clearTimeout(id);
  }, [error]);

  // -----------------------------------------------------------------
  // Tap → init track + run generateLine in parallel
  // -----------------------------------------------------------------
  const onStageTap = useCallback(
    async (e: React.MouseEvent<HTMLDivElement>) => {
      const stage = stageRef.current;
      const ws = wsRef.current;
      const v = videoRef.current;
      const c = sendCanvasRef.current;
      if (!stage || !ws || !v || !c) return;
      if (ws.readyState !== WebSocket.OPEN) return;
      if (phase !== "live") return;

      const rect = stage.getBoundingClientRect();
      const xNorm = clamp((e.clientX - rect.left) / rect.width, 0, 1);
      const yNorm = clamp((e.clientY - rect.top) / rect.height, 0, 1);

      // Cancel anything in flight for the previous track.
      speakGenRef.current++;
      const nextTrackGen = ++trackGenRef.current;
      const callGen = speakGenRef.current;
      abortInFlightSpeak();
      stopCurrentAudio();
      // Retap mid-lockstep: the old `track` response might still be on
      // the wire or we might be mid-frame-send. Clear the lockstep flag
      // so the next frame (the one the server pairs with our new init)
      // actually goes out.
      inFlightRef.current = false;

      // Provisional anchor: assume the tap is dead-center of a box that
      // spans ~40% of the frame. When `initialized` returns, we re-seed
      // the EMA to the real bbox but keep this anchor — it's where the
      // user clicked relative to the frame, which is close enough to
      // where they clicked relative to the object.
      const provisionalBox: Box = makeBox(xNorm, yNorm, 0.4, 0.4);
      const anchor = anchorFromPoint({ x: xNorm, y: yNorm }, provisionalBox);
      const ema = newBoxEMA(BOX_POS_ALPHA, BOX_SIZE_ALPHA);
      seedBoxEMA(ema, provisionalBox);
      trackRef.current = {
        id: "pending",
        gen: nextTrackGen,
        anchor,
        ema,
        smoothed: provisionalBox,
        lastScore: 0,
        voiceId: null,
        description: null,
        caption: null,
        speaking: false,
        thinking: true,
        opacity: 0.7,
        history: [],
      };
      setThinking(true);
      setCaption(null);
      setTrackLocked(false);
      setError(null);

      // 1) Tell the server to drop any lingering session state from the
      //    previous tap (TRACKING → IDLE), then kick init. The first
      //    subsequent JPEG is the frame SAM2 will run init against.
      try {
        ws.send(JSON.stringify({ type: "reset" }));
        ws.send(JSON.stringify({ type: "init", x: xNorm, y: yNorm }));
      } catch {
        setError("failed to send init");
        trackRef.current = null;
        setThinking(false);
        return;
      }
      void sendFrame();

      // 2) In parallel, capture a stills-quality JPEG and hand it to
      //    generateLine. The bundled vision call returns description +
      //    voiceId + line. We don't wait for SAM2 — the two calls race
      //    and we wire results up as they land.
      const stillBlob = await captureFrame(v, c, mirrorXRef.current, 0.88);
      if (!stillBlob) {
        setError("frame capture failed");
        return;
      }
      let dataUrl: string;
      try {
        dataUrl = await blobToDataUrl(stillBlob);
      } catch {
        setError("frame encode failed");
        return;
      }

      // Wake the audio context on the tap gesture so MediaSource has a
      // running graph to stream into by the time /api/speak's first byte
      // lands. Without this, mobile Safari blocks play() even though the
      // tap initiated the whole flow.
      const ctx = ensureAudioGraph();
      if (ctx && ctx.state === "suspended") {
        try {
          await ctx.resume();
        } catch {
          // ignore — playStream will re-check
        }
      }

      // Fresh AbortController — retaps abort prior in-flight fetches so
      // we don't pay LLM/TTS cost for a reply the user will never hear.
      abortInFlightSpeak();
      const abortCtrl = new AbortController();
      speakAbortRef.current = abortCtrl;

      try {
        // /api/speak folds the bundled VLM + TTS into one streaming
        // response: metadata (line, voiceId, description) arrives on the
        // X-Speak-Meta header the instant the VLM returns, and the mp3
        // body starts flowing as soon as Cartesia emits its first chunk.
        // Saves the ~600ms client-side round-trip we'd pay with the old
        // generateLine → /api/tts/stream split.
        const resp = await fetch("/api/speak", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            imageDataUrl: dataUrl,
            voiceId: null,
            description: null,
            history: [],
            lang: "en",
            turnId: `v2-${callGen}`,
          }),
          signal: abortCtrl.signal,
        });
        if (callGen !== speakGenRef.current) {
          try {
            await resp.body?.cancel();
          } catch {
            // ignore
          }
          return;
        }
        if (!resp.ok && resp.status !== 204) {
          const errBody = await resp.text().catch(() => "");
          throw new Error(
            `/api/speak ${resp.status}: ${errBody.slice(0, 160)}`,
          );
        }
        const metaB64 = resp.headers.get("X-Speak-Meta") ?? "";
        if (!metaB64) {
          try {
            await resp.body?.cancel();
          } catch {
            // ignore
          }
          throw new Error("/api/speak missing X-Speak-Meta header");
        }
        let meta: {
          line?: string;
          voiceId?: string | null;
          description?: string | null;
        };
        try {
          // UTF-8 safe base64 decode — the server encodes UTF-8 bytes and
          // plain atob would mangle non-ASCII.
          const bin = atob(metaB64);
          const bytes = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
          const metaJson = new TextDecoder("utf-8").decode(bytes);
          meta = JSON.parse(metaJson);
        } catch (parseErr) {
          try {
            await resp.body?.cancel();
          } catch {
            // ignore
          }
          throw new Error(
            `/api/speak meta parse failed: ${parseErr instanceof Error ? parseErr.message : String(parseErr)}`,
          );
        }
        if (typeof meta.line !== "string" || !meta.line.trim()) {
          try {
            await resp.body?.cancel();
          } catch {
            // ignore
          }
          throw new Error("/api/speak meta has no line");
        }
        const line = meta.line;
        const voiceId =
          typeof meta.voiceId === "string" && meta.voiceId.trim()
            ? meta.voiceId.trim()
            : null;
        const description =
          typeof meta.description === "string" && meta.description.trim()
            ? meta.description.trim()
            : null;

        const track = trackRef.current;
        if (track) {
          track.voiceId = voiceId;
          track.description = description;
          track.history = [{ role: "assistant", content: line }];
        }
        setCaption(line);
        clearCaptionSoon();
        setThinking(false);

        // status=200 with body → TTS stream available; status=204 means
        // the server ran the VLM but has no TTS backend configured, so
        // caption-only degraded mode.
        if (resp.status === 200 && resp.body) {
          try {
            await playStream(resp.body, callGen);
          } catch (err) {
            // eslint-disable-next-line no-console
            console.log("[v2] tts stream failed:", err);
          }
        }
      } catch (err) {
        if (callGen !== speakGenRef.current) return;
        // AbortError is expected on retap — don't surface it as an error.
        if (err instanceof DOMException && err.name === "AbortError") return;
        const msg = err instanceof Error ? err.message : "generate failed";
        setError(msg);
        setThinking(false);
      } finally {
        if (speakAbortRef.current === abortCtrl) {
          speakAbortRef.current = null;
        }
      }
    },
    [
      phase,
      sendFrame,
      stopCurrentAudio,
      abortInFlightSpeak,
      playStream,
      clearCaptionSoon,
      ensureAudioGraph,
    ],
  );

  // -----------------------------------------------------------------
  // Mic → converseWithObject
  // -----------------------------------------------------------------
  const startRecording = useCallback(async () => {
    if (listening) return;
    const track = trackRef.current;
    if (!track || !trackLocked) {
      setError("tap something first");
      return;
    }
    // Press-to-talk while the object is mid-reply should interrupt — the
    // user pressed talk because they want to be heard NOW, not after the
    // current line finishes. Also aborts any in-flight /api/speak so we
    // don't pay for a line that'll be overwritten by the mic reply.
    abortInFlightSpeak();
    stopCurrentAudio();
    // Re-use the existing camera stream's peers — MediaRecorder needs
    // its own audio-only stream.
    let micStream: MediaStream;
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    } catch (e) {
      setError(e instanceof Error ? e.message : "mic blocked");
      return;
    }
    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/mp4")
        ? "audio/mp4"
        : "";
    const recorder = new MediaRecorder(micStream, mimeType ? { mimeType } : undefined);
    recorderRef.current = recorder;
    recorderChunksRef.current = [];
    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) recorderChunksRef.current.push(e.data);
    };
    recorder.onstop = async () => {
      micStream.getTracks().forEach((t) => t.stop());
      const chunks = recorderChunksRef.current;
      recorderChunksRef.current = [];
      const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
      recorderRef.current = null;
      setListening(false);
      await sendTalk(blob);
    };
    recorder.start();
    setListening(true);
    setError(null);
  }, [abortInFlightSpeak, listening, stopCurrentAudio, trackLocked]);

  const stopRecording = useCallback(() => {
    const rec = recorderRef.current;
    if (!rec) return;
    if (rec.state !== "inactive") {
      try {
        rec.stop();
      } catch {
        // already stopped
      }
    }
  }, []);

  const sendTalk = useCallback(
    async (blob: Blob) => {
      const track = trackRef.current;
      if (!track) return;
      const callGen = ++speakGenRef.current;
      stopCurrentAudio();
      setThinking(true);
      setCaption(null);
      try {
        const fd = new FormData();
        const filename = blob.type.includes("mp4")
          ? "talk.mp4"
          : blob.type.includes("ogg")
            ? "talk.ogg"
            : "talk.webm";
        fd.append("audio", blob, filename);
        // SAM2 doesn't give us a className. Pass the persona's own
        // description text as a stand-in subject — the server prompt
        // leans on description anyway when it's present.
        fd.append("className", track.description?.slice(0, 60) ?? "object");
        fd.append("turnId", `v2-${callGen}`);
        fd.append("lang", "en");
        if (track.voiceId) fd.append("voiceId", track.voiceId);
        if (track.description) fd.append("description", track.description);
        fd.append("history", JSON.stringify(track.history.slice(-32)));

        const { transcript, reply, voiceId } = await converseWithObject(fd);
        if (callGen !== speakGenRef.current) return;

        const next = [...track.history];
        if (transcript) next.push({ role: "user", content: transcript });
        if (reply) next.push({ role: "assistant", content: reply });
        track.history = next.slice(-32);
        if (voiceId) track.voiceId = voiceId;

        setThinking(false);
        setCaption(reply || null);
        clearCaptionSoon();
        if (reply) {
          await playTtsText(reply, voiceId ?? track.voiceId, callGen);
        }
      } catch (err) {
        if (callGen !== speakGenRef.current) return;
        const msg = err instanceof Error ? err.message : "reply failed";
        setError(msg);
        setThinking(false);
      }
    },
    [stopCurrentAudio, playTtsText, clearCaptionSoon],
  );

  // -----------------------------------------------------------------
  // Esc to release track
  // -----------------------------------------------------------------
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && trackRef.current) {
        releaseTrack();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [releaseTrack]);

  // -----------------------------------------------------------------
  // RAF: position the face on the smoothed bbox + pick a mouth shape.
  // -----------------------------------------------------------------
  useEffect(() => {
    let raf = 0;
    const step = () => {
      const stage = stageRef.current;
      const wrap = faceWrapRef.current;
      const track = trackRef.current;
      if (stage && wrap) {
        if (!track) {
          wrap.style.opacity = "0";
          wrap.style.transform = "translate(-9999px, -9999px)";
        } else {
          const rect = stage.getBoundingClientRect();
          const b = track.smoothed;
          const center = applyAnchor(track.anchor, b);
          const boxPxMin = Math.min(b.w * rect.width, b.h * rect.height);
          const scale = clamp(
            (boxPxMin * FACE_BBOX_FRACTION) / FACE_VOICE_WIDTH,
            FACE_SCALE_MIN,
            FACE_SCALE_MAX,
          );
          const px = center.x * rect.width;
          const py = center.y * rect.height;
          wrap.style.opacity = String(track.opacity);
          wrap.style.transform = `translate(${px - (FACE_VOICE_WIDTH * scale) / 2}px, ${py - (FACE_VOICE_HEIGHT * scale) / 2}px) scale(${scale})`;
        }
      }

      // Mouth shape: if audio is flowing, classify; otherwise rest.
      const analyser = analyserRef.current;
      const freq = freqDataRef.current;
      const time = timeDataRef.current;
      const el = audioElRef.current;
      const hasAudio = !!el && !el.paused && !el.ended;
      let next: MouthShape = "X";
      if (analyser && freq && time && hasAudio) {
        analyser.getByteTimeDomainData(time);
        analyser.getByteFrequencyData(freq);
        next = classifyShapeSmooth(lipSyncRef.current, time, freq);
      }
      if (next !== mouthShapeRef.current) {
        mouthShapeRef.current = next;
        setMouthShape(next);
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, []);

  // -----------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------
  const isConnecting = phase === "opening";
  const dotColor =
    phase === "live"
      ? "bg-emerald-400"
      : isConnecting
        ? "bg-amber-400"
        : phase === "error"
          ? "bg-rose-400"
          : "bg-fuchsia-300";
  const statusLabel =
    phase === "live"
      ? trackLocked
        ? `tracking · ${lastScore.toFixed(2)}`
        : "tap an object"
      : phase === "opening"
        ? "connecting…"
        : phase === "error"
          ? "oops"
          : "ready";

  return (
    <main className="relative flex min-h-[100svh] flex-col overflow-hidden">
      <div
        className="blob"
        style={{
          width: 380,
          height: 380,
          top: -120,
          left: -120,
          background: "radial-gradient(circle, #ffc4de 0%, #ffd9bd 70%)",
        }}
      />
      <div
        className="blob"
        style={{
          width: 460,
          height: 460,
          top: "38%",
          right: -180,
          background: "radial-gradient(circle, #cfd9ff 0%, #e4d1ff 70%)",
          animationDelay: "-6s",
        }}
      />

      <div className="relative z-10 mx-auto flex w-full max-w-[880px] flex-1 flex-col">
        <header className="flex items-center justify-between px-6 pt-7 sm:px-8 sm:pt-9">
          <div className="flex items-baseline gap-2">
            <span className="h-2 w-2 rounded-full bg-[color:var(--accent)] shadow-[0_0_0_4px_rgba(236,72,153,0.18)]" />
            <span className="serif-italic text-[28px] font-semibold leading-none text-[color:var(--ink)] sm:text-[32px]">
              tracker v2
            </span>
            <span className="rounded-full bg-white/70 px-2 py-0.5 text-[10px] font-medium tracking-wide text-[color:var(--ink-muted)] ring-1 ring-white/70 backdrop-blur">
              sam2
            </span>
          </div>
          <div className="flex items-center gap-2 rounded-full bg-white/70 px-3 py-1.5 shadow-[0_2px_10px_-4px_rgba(42,21,64,0.15)] ring-1 ring-white/80 backdrop-blur-md">
            <span className={"h-1.5 w-1.5 rounded-full " + dotColor} />
            <span className="text-[11px] font-medium tabular-nums tracking-wide text-[color:var(--ink-soft)]">
              {statusLabel}
            </span>
          </div>
        </header>

        <section className="flex flex-1 items-center justify-center px-4 py-6 sm:px-8 sm:py-8">
          <div className="relative aspect-square w-full max-w-[min(80vh,660px)]">
            <div
              aria-hidden
              className="absolute -inset-6 rounded-[48px] opacity-70 blur-2xl"
              style={{
                background:
                  "conic-gradient(from 140deg, #ffd1e8, #e5d4ff, #d6e6ff, #ffe5d0, #ffd1e8)",
              }}
            />
            <div
              ref={stageRef}
              onClick={onStageTap}
              className="relative h-full w-full overflow-hidden rounded-[32px] bg-white/60 shadow-[0_40px_80px_-30px_rgba(42,21,64,0.25),0_0_0_1px_rgba(255,255,255,0.9)_inset] backdrop-blur"
              style={{ cursor: phase === "live" ? "crosshair" : "default" }}
            >
              <video
                ref={videoRef}
                playsInline
                muted
                className={
                  "h-full w-full object-cover transition-opacity duration-500 " +
                  (phase === "live" ? "opacity-100" : "opacity-0") +
                  (mirrorXRef.current ? " -scale-x-100" : "")
                }
              />

              {/* Face overlay */}
              <div
                ref={faceWrapRef}
                className="pointer-events-none absolute left-0 top-0 transition-opacity duration-200"
                style={{
                  width: FACE_VOICE_WIDTH,
                  height: FACE_VOICE_HEIGHT,
                  transformOrigin: "0 0",
                  opacity: 0,
                  transform: "translate(-9999px,-9999px)",
                }}
              >
                <FaceVoice shape={mouthShape} />
              </div>

              {phase !== "live" && (
                <div className="absolute inset-0 grid place-items-center">
                  {phase === "idle" || phase === "error" ? (
                    <button
                      onClick={connect}
                      disabled={serverOnline === false}
                      className="group flex flex-col items-center gap-5 disabled:cursor-not-allowed disabled:opacity-60"
                      title={
                        serverOnline === false
                          ? "sam2 backend not reachable on :8001 — start it with `pnpm run server:v2`"
                          : undefined
                      }
                    >
                      <span className="bubble-btn grid h-[104px] w-[104px] place-items-center rounded-full bg-gradient-to-br from-[#ff89be] via-[#ec4899] to-[#c026d3] text-white transition duration-300 group-hover:scale-[1.06] group-active:scale-95 group-disabled:scale-100">
                        <svg width="30" height="30" viewBox="0 0 24 24" fill="currentColor" className="translate-x-[2px]">
                          <polygon points="7 4 20 12 7 20 7 4" />
                        </svg>
                      </span>
                      <span className="serif-italic text-[18px] font-medium text-[color:var(--ink-soft)]">
                        {serverOnline === false
                          ? "backend offline"
                          : phase === "error"
                            ? "try once more"
                            : "begin"}
                      </span>
                    </button>
                  ) : (
                    <div className="flex items-center gap-2.5 rounded-full bg-white/80 px-4 py-2 text-[13px] font-medium text-[color:var(--ink-soft)] shadow-sm backdrop-blur">
                      <span className="inline-block h-1.5 w-1.5 animate-ping rounded-full bg-[color:var(--accent)]" />
                      connecting
                    </div>
                  )}
                </div>
              )}

              {phase === "live" && (
                <>
                  <div className="pointer-events-none absolute left-4 top-4 flex items-center gap-2">
                    <span className="rounded-full bg-white/75 px-3 py-1 text-[11px] font-medium tracking-wide text-[color:var(--ink-soft)] backdrop-blur-md">
                      {thinking ? "thinking…" : trackLocked ? "locked" : "tap to lock"}
                    </span>
                    {serverOnline === false && (
                      <span className="rounded-full bg-rose-500/85 px-3 py-1 text-[11px] font-medium tracking-wide text-white backdrop-blur-md">
                        backend offline
                      </span>
                    )}
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      disconnect();
                    }}
                    className="absolute right-4 top-4 rounded-full bg-white/75 px-3.5 py-1 text-[11px] font-medium tracking-wide text-[color:var(--ink-soft)] backdrop-blur-md transition hover:bg-white hover:text-[color:var(--ink)] active:scale-95"
                  >
                    stop
                  </button>

                  {caption && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setCaption(null);
                      }}
                      className="absolute inset-x-6 top-14 mx-auto max-w-[82%] rounded-[20px] bg-white/92 px-5 py-3.5 text-left shadow-[0_16px_40px_-18px_rgba(42,21,64,0.35)] ring-1 ring-white/80 backdrop-blur-md transition hover:bg-white"
                      style={{ animation: "bubble-in 420ms cubic-bezier(0.16,1,0.3,1) both" }}
                    >
                      <span className="serif-italic text-balance text-[16px] leading-[1.35] text-[color:var(--ink)] sm:text-[17px]">
                        &ldquo;{caption}&rdquo;
                      </span>
                    </button>
                  )}

                  {trackLocked && (
                    <div className="absolute inset-x-0 bottom-5 flex justify-center">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (listening) stopRecording();
                          else void startRecording();
                        }}
                        disabled={thinking && !listening}
                        className={
                          "bubble-btn rounded-full px-6 py-2.5 text-[13px] font-semibold tracking-wide transition active:scale-95 disabled:opacity-50 " +
                          (listening
                            ? "bg-gradient-to-br from-[#ff7b7b] via-[#ec4899] to-[#c026d3] text-white"
                            : "bg-gradient-to-br from-[#ff89be] via-[#ec4899] to-[#c026d3] text-white hover:scale-[1.04]")
                        }
                      >
                        {listening ? "stop & send" : speaking ? "…" : "talk"}
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>

            {error && (
              <div
                className="absolute inset-x-4 -bottom-4 text-center"
                style={{ animation: "fade-in 240ms ease-out both" }}
              >
                <span className="inline-block rounded-full bg-white/90 px-4 py-1.5 text-[11px] font-medium text-rose-500 shadow-sm backdrop-blur">
                  {error}
                </span>
              </div>
            )}
          </div>
        </section>
      </div>

      <canvas ref={sendCanvasRef} className="hidden" />
    </main>
  );
}
