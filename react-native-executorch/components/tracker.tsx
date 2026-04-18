import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Dimensions,
  LayoutChangeEvent,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
  type CameraRuntimeError,
} from "react-native-vision-camera";
import { Gesture, GestureDetector } from "react-native-gesture-handler";
import { scheduleOnRN } from "react-native-worklets-core";
import {
  useInstanceSegmentation,
  YOLO26N_SEG,
  type ObjectDetectionInput,
} from "react-native-executorch";
import * as FileSystem from "expo-file-system";
import {
  ExpoSpeechRecognitionModule,
  useSpeechRecognitionEvent,
} from "expo-speech-recognition";

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
  EXCLUDED_CLASS_IDS,
  normalizeDetection,
  type Detection,
  type RawSegInstance,
} from "@/lib/detector";
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
import {
  pickTappedDetection,
  sourceBoxToView,
  sourcePointToView,
  viewPointToSource,
} from "@/lib/tap-cache";
import {
  assessObject,
  converseWithObject,
  generateLine,
  type ConverseTurn,
} from "@/lib/api";
import {
  createTrackAudio,
  disposeTrackAudio,
  readAnalyser,
  setGain,
  speakLine,
  type TrackAudio,
} from "@/lib/audio-track";

// === Pipeline tuning knobs — ported verbatim from the browser tracker ====

const MAX_INFERENCE_FPS = 30;

// IoU gate for "this new box is the same instance I was tracking".
const IDENTITY_IOU_MIN = 0.3;

const BOX_POS_ALPHA = 0.7;
const BOX_SIZE_ALPHA = 0.25;
const OPACITY_EMA_ALPHA = 0.4;

const LOST_AFTER_MISSES = 4;
const WIDEN_MATCH_AFTER_MISSES = 3;
const SUSPECT_SIZE_RATIO = 1.75;

const EXTRAP_MAX_MS = 220;
const EXTRAP_MISS_LIMIT = 2;
const VELOCITY_EMA = 0.75;
const VELOCITY_DECAY_PER_MISS = 0.6;

const CONTINUOUS_CONF = 0.15;
const TAP_CONF = 0.08;
const CONTINUOUS_MAX_DET = 25;

const FACE_BBOX_FRACTION = 0.92;
const FACE_NATIVE_PX = FACE_VOICE_WIDTH;
const FACE_SCALE_MIN = 0.25;
const FACE_SCALE_MAX = 3.0;

const TILT_GAIN_DEG = 6;
const TILT_MAX_DEG = 12;
const TILT_EMA = 0.18;

const VOICE_SIZE_REF_FRAC = 0.22;
const VOICE_SIZE_EXP = 1.6;
const VOICE_GAIN_MAX = 4.0;

const VOICE_PERSIST_MS = 2000;
const VOICE_FADE_ALPHA = 0.04;

const MAX_FACES = 3;
const TAP_CACHE_MAX_AGE_MS = 400;

const SEG_INPUT_SIZE: 384 | 512 | 640 = 384;

// === Types ===============================================================

type Phase = "starting" | "ready" | "locked" | "error";

type TrackRefs = {
  id: string;
  classId: number;
  className: string;
  anchor: Anchor;
  boxEma: BoxEMA;
  smoothedBox: Box;
  missedFrames: number;
  opacity: number;
  audioLevel: number;
  lostSinceMs: number | null;
  lastTapAt: number;
  vx: number;
  vy: number;
  lastUpdatedAt: number;
  tiltDeg: number;
  speakGen: number;
  shape: MouthShape;
  lipSync: LipSyncState;
  voiceId: string | null;
  description: string | null;
  history: ConverseTurn[];
  audio: TrackAudio;
  activeSpeak: { cancel: () => void } | null;
  voicePersistUntil: number;
};

type TrackUI = {
  id: string;
  // Render state — driven from the RAF loop, consumed by React.
  left: number;
  top: number;
  size: number;       // displayed face-native px (square-ish)
  opacity: number;
  tiltDeg: number;
  shape: MouthShape;
  caption: string | null;
  thinking: boolean;
  speaking: boolean;
};

type Rejection = { id: number; reason: string };

// === Component ===========================================================

export default function Tracker() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice("back");
  const cameraRef = useRef<Camera>(null);
  const seg = useInstanceSegmentation({ model: YOLO26N_SEG });

  const [phase, setPhase] = useState<Phase>("starting");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [rejection, setRejection] = useState<Rejection | null>(null);
  const [viewSize, setViewSize] = useState({ width: 0, height: 0 });
  const [uiTracks, setUiTracks] = useState<TrackUI[]>([]);
  const [micRecording, setMicRecording] = useState(false);
  const [activeTrackId, setActiveTrackId] = useState<string | null>(null);

  const tracksRef = useRef(new Map<string, TrackRefs>());
  const detectionsRef = useRef<Detection[]>([]);
  const detectionsTsRef = useRef(0);
  const frameSizeRef = useRef({ width: 0, height: 0 });
  const viewSizeRef = useRef({ width: 0, height: 0 });
  const generationRef = useRef(0);
  const nextTrackIdRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const rejectionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastFrameTsRef = useRef(0);

  // --- Permissions --------------------------------------------------------
  useEffect(() => {
    if (!hasPermission) {
      void requestPermission();
    }
  }, [hasPermission, requestPermission]);

  // --- Model readiness ----------------------------------------------------
  useEffect(() => {
    if (seg.error) {
      setErrorMsg(String(seg.error.message ?? seg.error));
      setPhase("error");
      return;
    }
    if (seg.isReady && hasPermission) {
      setPhase((p) => (p === "starting" ? "ready" : p));
    }
  }, [seg.isReady, seg.error, hasPermission]);

  // --- Frame → detection list --------------------------------------------
  const ingestDetections = useCallback(
    (raws: RawSegInstance[], frameWidth: number, frameHeight: number) => {
      frameSizeRef.current = { width: frameWidth, height: frameHeight };
      const dets: Detection[] = [];
      for (const r of raws) {
        const d = normalizeDetection(r, frameWidth, frameHeight);
        if (!d) continue;
        if (EXCLUDED_CLASS_IDS.has(d.classId)) continue;
        dets.push(d);
      }
      detectionsRef.current = dets;
      detectionsTsRef.current = Date.now();
    },
    []
  );

  // --- Vision Camera frame processor -------------------------------------
  const minFrameIntervalMs = 1000 / MAX_INFERENCE_FPS;
  const onCameraFrame = useFrameOutput(
    {
      pixelFormat: "rgb",
      dropFramesWhileBusy: true,
      onFrame: (frame) => {
        "worklet";
        const now = Date.now();
        // Rate-cap so battery doesn't burn on a 60 Hz capture when the model
        // runs at 10 Hz — saves frames without fighting the camera.
        // @ts-expect-error — worklet-side ref access
        if (now - lastFrameTsRef.value < minFrameIntervalMs) return;
        // @ts-expect-error — worklet-side ref access
        lastFrameTsRef.value = now;
        try {
          const out = seg.runOnFrame(frame, false, {
            confidenceThreshold: CONTINUOUS_CONF,
            iouThreshold: 0.45,
            maxInstances: CONTINUOUS_MAX_DET,
            inputSize: SEG_INPUT_SIZE,
            returnMaskAtOriginalResolution: false,
          });
          const width = frame.width;
          const height = frame.height;
          scheduleOnRN(ingestDetections, out as RawSegInstance[], width, height);
        } catch (e) {
          // Worklet errors are swallowed to keep the camera alive.
        } finally {
          frame.dispose?.();
        }
      },
    },
    [seg, ingestDetections, minFrameIntervalMs]
  );

  // --- Track lifecycle helpers -------------------------------------------
  const removeTrack = useCallback((id: string) => {
    const t = tracksRef.current.get(id);
    if (!t) return;
    try {
      t.activeSpeak?.cancel();
    } catch {}
    disposeTrackAudio(t.audio);
    tracksRef.current.delete(id);
  }, []);

  const evictLruIfNeeded = useCallback(() => {
    if (tracksRef.current.size < MAX_FACES) return;
    let oldestId: string | null = null;
    let oldestAt = Infinity;
    tracksRef.current.forEach((t) => {
      if (t.lastTapAt < oldestAt) {
        oldestAt = t.lastTapAt;
        oldestId = t.id;
      }
    });
    if (oldestId) removeTrack(oldestId);
  }, [removeTrack]);

  const clearAllTracks = useCallback(() => {
    generationRef.current++;
    const ids = Array.from(tracksRef.current.keys());
    for (const id of ids) removeTrack(id);
    setUiTracks([]);
    setActiveTrackId(null);
    setPhase("ready");
  }, [removeTrack]);

  // --- Show a rejection toast for a moment -------------------------------
  const showRejection = useCallback((reason: string) => {
    const id = Date.now();
    setRejection({ id, reason });
    if (rejectionTimerRef.current) clearTimeout(rejectionTimerRef.current);
    rejectionTimerRef.current = setTimeout(() => {
      setRejection((cur) => (cur && cur.id === id ? null : cur));
    }, 2600);
  }, []);

  // --- Render loop: match, EMA, extrapolate, render ---------------------
  useEffect(() => {
    let mounted = true;
    const tick = () => {
      if (!mounted) return;
      const now = Date.now();
      const dets = detectionsRef.current;
      const view = viewSizeRef.current;
      const frame = frameSizeRef.current;

      // One-way claim: each detection can be attached to only one track.
      const claimed = new Set<number>();

      // --- Match each existing track to a fresh detection ---------------
      tracksRef.current.forEach((t) => {
        const candidatePool = dets.filter((_, i) => !claimed.has(i));
        const prev = t.smoothedBox;
        const prevWithClass = { ...prev, classId: t.classId };
        let match: Detection | null = null;
        // Primary: IoU + same-class via matchTarget.
        match = matchTarget(
          candidatePool as (Detection & { classId: number })[],
          prevWithClass,
          IDENTITY_IOU_MIN
        ) as Detection | null;

        // Fallback: after a few misses, widen to same-class nearest-center.
        if (!match && t.missedFrames >= WIDEN_MATCH_AFTER_MISSES) {
          let nearest: Detection | null = null;
          let nearestD = Infinity;
          for (const c of candidatePool) {
            if (c.classId !== t.classId) continue;
            const dx = c.cx - prev.cx;
            const dy = c.cy - prev.cy;
            const dist = Math.hypot(dx, dy);
            if (dist < nearestD) {
              nearestD = dist;
              nearest = c;
            }
          }
          match = nearest;
        }

        if (match) {
          const idx = dets.indexOf(match);
          if (idx >= 0) claimed.add(idx);

          const wasLost = t.missedFrames >= LOST_AFTER_MISSES;
          const prevBox = t.smoothedBox;

          const ratioW =
            Math.max(match.w, prevBox.w) /
            Math.max(1, Math.min(match.w, prevBox.w));
          const ratioH =
            Math.max(match.h, prevBox.h) /
            Math.max(1, Math.min(match.h, prevBox.h));
          const suspect =
            !wasLost && (ratioW > SUSPECT_SIZE_RATIO || ratioH > SUSPECT_SIZE_RATIO);

          if (suspect) {
            t.missedFrames = 0;
            t.vx *= VELOCITY_DECAY_PER_MISS;
            t.vy *= VELOCITY_DECAY_PER_MISS;
          } else {
            t.missedFrames = 0;
            const obsCx = match.maskCentroid?.x ?? match.cx;
            const obsCy = match.maskCentroid?.y ?? match.cy;
            const observation = makeBox(obsCx, obsCy, match.w, match.h);

            if (wasLost) {
              seedBoxEMA(t.boxEma, observation);
              t.smoothedBox = observation;
              t.vx = 0;
              t.vy = 0;
            } else {
              const prevCx = prevBox.cx;
              const prevCy = prevBox.cy;
              t.smoothedBox = smoothBox(t.boxEma, observation);
              const dt = now - t.lastUpdatedAt;
              if (dt > 10 && dt < 500) {
                const rawVx = (t.smoothedBox.cx - prevCx) / dt;
                const rawVy = (t.smoothedBox.cy - prevCy) / dt;
                t.vx = t.vx * (1 - VELOCITY_EMA) + rawVx * VELOCITY_EMA;
                t.vy = t.vy * (1 - VELOCITY_EMA) + rawVy * VELOCITY_EMA;
              }
            }
            t.lastUpdatedAt = now;
            t.lostSinceMs = null;
          }
        } else {
          t.missedFrames++;
          t.vx *= VELOCITY_DECAY_PER_MISS;
          t.vy *= VELOCITY_DECAY_PER_MISS;
          if (t.missedFrames === 1) t.lostSinceMs = now;
        }
      });

      // --- Compute render state for each track --------------------------
      const nextUi: TrackUI[] = [];
      tracksRef.current.forEach((t) => {
        // Extrapolate position between inferences so the face doesn't
        // stutter at the inference rate.
        const dtSinceInf = now - t.lastUpdatedAt;
        let renderBox = t.smoothedBox;
        if (
          t.missedFrames <= EXTRAP_MISS_LIMIT &&
          dtSinceInf > 0 &&
          dtSinceInf < EXTRAP_MAX_MS
        ) {
          renderBox = makeBox(
            t.smoothedBox.cx + t.vx * dtSinceInf,
            t.smoothedBox.cy + t.vy * dtSinceInf,
            t.smoothedBox.w,
            t.smoothedBox.h
          );
        }

        // Fade opacity based on miss count — fast approach, same-speed fade.
        const targetOpacity = t.missedFrames >= LOST_AFTER_MISSES ? 0 : 1;
        t.opacity =
          t.opacity + OPACITY_EMA_ALPHA * (targetOpacity - t.opacity);

        // Face anchor point in frame-source coords, then project to view.
        const facePoint = applyAnchor(t.anchor, renderBox);
        if (view.width <= 0 || view.height <= 0 || frame.width <= 0 || frame.height <= 0) {
          return;
        }
        const faceView = sourcePointToView(facePoint, frame, view);
        const boxView = sourceBoxToView(
          {
            x1: renderBox.x1,
            y1: renderBox.y1,
            x2: renderBox.x2,
            y2: renderBox.y2,
          },
          frame,
          view
        );

        // Face size derived from the displayed box size.
        const target = Math.max(
          FACE_BBOX_FRACTION * Math.min(boxView.width, boxView.height),
          1
        );
        const scale = Math.max(
          FACE_SCALE_MIN,
          Math.min(FACE_SCALE_MAX, target / FACE_NATIVE_PX)
        );
        const size = FACE_NATIVE_PX * scale;

        // Tilt — smoothed from velocity so motion reads as a lean, not glide.
        const vxPerW =
          renderBox.w > 0 ? (t.vx * 1000) / renderBox.w : 0;
        const targetTilt = Math.max(
          -TILT_MAX_DEG,
          Math.min(TILT_MAX_DEG, vxPerW * TILT_GAIN_DEG)
        );
        t.tiltDeg = t.tiltDeg + TILT_EMA * (targetTilt - t.tiltDeg);

        // Lip-sync: read analyser + classify mouth shape.
        const { timeBuf, freqBuf } = readAnalyser(t.audio);
        t.shape = classifyShapeSmooth(t.lipSync, timeBuf, freqBuf);

        // Voice gain: opacity with a voice-persist window, plus a size boost
        // so "right there" reads louder than "across the room".
        const sizeFrac =
          Math.min(boxView.width, boxView.height) /
          Math.max(1, Math.min(view.width, view.height));
        const sizeBoost = Math.min(
          VOICE_GAIN_MAX,
          Math.pow(
            Math.max(sizeFrac, 0.001) / VOICE_SIZE_REF_FRAC,
            VOICE_SIZE_EXP
          )
        );
        let audioTarget = t.opacity * sizeBoost;
        if (t.missedFrames >= LOST_AFTER_MISSES) {
          if (now < t.voicePersistUntil) {
            audioTarget = sizeBoost;
          } else {
            audioTarget = 0;
          }
        }
        t.audioLevel =
          t.audioLevel + VOICE_FADE_ALPHA * (audioTarget - t.audioLevel);
        setGain(t.audio, t.audioLevel);

        nextUi.push({
          id: t.id,
          left: faceView.x - size / 2,
          top: faceView.y - size / 2,
          size,
          opacity: Math.max(0, Math.min(1, t.opacity)),
          tiltDeg: t.tiltDeg,
          shape: t.shape,
          caption: null, // populated below where we keep UI state
          thinking: false,
          speaking: false,
        });
      });

      // React is responsible for the final render. We keep caption/thinking
      // state inside the UI map and merge here:
      setUiTracks((prev) => {
        const byId = new Map(prev.map((p) => [p.id, p]));
        return nextUi.map((n) => {
          const p = byId.get(n.id);
          return p
            ? {
                ...n,
                caption: p.caption,
                thinking: p.thinking,
                speaking: p.speaking,
              }
            : n;
        });
      });

      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      mounted = false;
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // --- Capture a snapshot and return a data URL --------------------------
  const captureDataUrl = useCallback(async (): Promise<string | null> => {
    const cam = cameraRef.current;
    if (!cam) return null;
    try {
      const snap = await cam.takeSnapshot({ quality: 70 });
      const b64 = await FileSystem.readAsStringAsync(`file://${snap.path}`, {
        encoding: FileSystem.EncodingType.Base64,
      });
      return `data:image/jpeg;base64,${b64}`;
    } catch (e) {
      return null;
    }
  }, []);

  // --- Pin caption/thinking on a track UI entry --------------------------
  const patchTrackUi = useCallback(
    (id: string, patch: Partial<Pick<TrackUI, "caption" | "thinking" | "speaking">>) => {
      setUiTracks((prev) =>
        prev.map((p) => (p.id === id ? { ...p, ...patch } : p))
      );
    },
    []
  );

  // --- Speak a line on a track, cancelling any in-flight playback -------
  const speakOnTrack = useCallback(
    async (trackId: string, line: string, voiceId: string, gen: number) => {
      const t = tracksRef.current.get(trackId);
      if (!t) return;
      if (gen !== generationRef.current) return;
      t.speakGen++;
      const mySpeakGen = t.speakGen;
      try {
        t.activeSpeak?.cancel();
      } catch {}
      const turnId = `${trackId}-${mySpeakGen}`;
      const handle = speakLine(t.audio, { text: line, voiceId, turnId });
      t.activeSpeak = handle;
      patchTrackUi(trackId, { caption: line, thinking: false, speaking: true });
      try {
        await handle.promise;
      } finally {
        if (t.speakGen === mySpeakGen) {
          t.activeSpeak = null;
          t.voicePersistUntil = Date.now() + VOICE_PERSIST_MS;
          patchTrackUi(trackId, { speaking: false });
        }
      }
    },
    [patchTrackUi]
  );

  // --- Tap handler --------------------------------------------------------
  const onTap = useCallback(
    async (tapX: number, tapY: number) => {
      const view = viewSizeRef.current;
      const frame = frameSizeRef.current;
      if (view.width <= 0 || frame.width <= 0) return;
      const srcTap = viewPointToSource({ x: tapX, y: tapY }, frame, view);
      const gen = ++generationRef.current;

      // (a) If tap lands inside an existing track's render box → retap.
      for (const t of tracksRef.current.values()) {
        const b = t.smoothedBox;
        if (
          srcTap.x >= b.x1 &&
          srcTap.x <= b.x2 &&
          srcTap.y >= b.y1 &&
          srcTap.y <= b.y2
        ) {
          t.lastTapAt = Date.now();
          patchTrackUi(t.id, { caption: null, thinking: true });
          try {
            const res = await generateLine(null, {
              voiceId: t.voiceId ?? undefined,
              description: t.description ?? undefined,
              className: t.className,
              history: t.history,
            });
            if (gen !== generationRef.current) return;
            t.history.push({ role: "object", text: res.line });
            if (res.voiceId && !t.voiceId) t.voiceId = res.voiceId;
            if (res.description && !t.description)
              t.description = res.description;
            await speakOnTrack(t.id, res.line, res.voiceId ?? t.voiceId ?? "", gen);
          } catch (e) {
            patchTrackUi(t.id, { thinking: false });
            showRejection("line failed");
          }
          return;
        }
      }

      // (b) Otherwise resolve a detection under the tap and add a new face.
      const cacheAge = Date.now() - detectionsTsRef.current;
      const tapDets =
        detectionsRef.current.length && cacheAge < TAP_CACHE_MAX_AGE_MS
          ? detectionsRef.current
          : detectionsRef.current;
      const tapped = pickTappedDetection(tapDets, srcTap.x, srcTap.y);
      if (!tapped) {
        showRejection("nothing I recognize there");
        return;
      }

      // Evict LRU if we're at the cap.
      evictLruIfNeeded();

      // Build the new track. Prefer mask centroid as the anchor origin on
      // asymmetric objects.
      const lockCx = tapped.maskCentroid?.x ?? (tapped.x1 + tapped.x2) / 2;
      const lockCy = tapped.maskCentroid?.y ?? (tapped.y1 + tapped.y2) / 2;
      const lockBox = makeBox(
        lockCx,
        lockCy,
        tapped.x2 - tapped.x1,
        tapped.y2 - tapped.y1
      );
      const boxEma = newBoxEMA(BOX_POS_ALPHA, BOX_SIZE_ALPHA);
      seedBoxEMA(boxEma, lockBox);
      const id = `t${nextTrackIdRef.current++}`;
      const nowTs = Date.now();
      const audio = createTrackAudio();
      const t: TrackRefs = {
        id,
        classId: tapped.classId,
        className: tapped.className,
        anchor: { rx: 0, ry: 0 }, // mask-centroid-aware lock → anchor at center
        boxEma,
        smoothedBox: lockBox,
        missedFrames: 0,
        opacity: 0,
        audioLevel: 0,
        lostSinceMs: null,
        lastTapAt: nowTs,
        vx: 0,
        vy: 0,
        lastUpdatedAt: nowTs,
        tiltDeg: 0,
        speakGen: 0,
        shape: "X",
        lipSync: createLipSyncState(),
        voiceId: null,
        description: null,
        history: [],
        audio,
        activeSpeak: null,
        voicePersistUntil: 0,
      };
      tracksRef.current.set(id, t);
      setUiTracks((prev) => [
        ...prev,
        {
          id,
          left: 0,
          top: 0,
          size: 0,
          opacity: 0,
          tiltDeg: 0,
          shape: "X",
          caption: null,
          thinking: true,
          speaking: false,
        },
      ]);
      setActiveTrackId(id);
      if (phase !== "locked") setPhase("locked");

      // Parallel: snapshot + assess + generateLine(first tap, bundled).
      const dataUrlP = captureDataUrl();
      const dataUrl = await dataUrlP;
      if (!dataUrl) {
        showRejection("couldn't grab the frame");
        patchTrackUi(id, { thinking: false });
        return;
      }
      const assessP = assessObject(dataUrl, lockCx, lockCy).catch(() => null);
      const genP = generateLine(dataUrl, {
        className: tapped.className,
        history: [],
      }).catch((e: Error) => {
        throw e;
      });

      try {
        const [assess, gline] = await Promise.all([assessP, genP]);
        if (gen !== generationRef.current) return;
        if (assess && assess.suitable === false) {
          removeTrack(id);
          setUiTracks((prev) => prev.filter((p) => p.id !== id));
          showRejection(assess.reason || "not a good fit");
          return;
        }
        t.voiceId = gline.voiceId;
        t.description = gline.description;
        t.history.push({ role: "object", text: gline.line });
        await speakOnTrack(id, gline.line, gline.voiceId, gen);
      } catch (e) {
        patchTrackUi(id, { thinking: false });
        showRejection("line failed");
      }
    },
    [
      captureDataUrl,
      evictLruIfNeeded,
      patchTrackUi,
      phase,
      removeTrack,
      showRejection,
      speakOnTrack,
    ]
  );

  // --- Voice conversation (mic button) -----------------------------------
  const [pendingTranscript, setPendingTranscript] = useState("");
  const recordingFileRef = useRef<string | null>(null);

  useSpeechRecognitionEvent("result", (event) => {
    const best = event.results?.[0]?.transcript;
    if (typeof best === "string") setPendingTranscript(best);
  });
  useSpeechRecognitionEvent("end", () => setMicRecording(false));

  const startMic = useCallback(async () => {
    if (!activeTrackId) return;
    const perm = await ExpoSpeechRecognitionModule.requestPermissionsAsync();
    if (!perm.granted) {
      showRejection("mic permission denied");
      return;
    }
    const recordingPath = `${FileSystem.cacheDirectory}converse-${Date.now()}.m4a`;
    recordingFileRef.current = recordingPath;
    setPendingTranscript("");
    setMicRecording(true);
    try {
      ExpoSpeechRecognitionModule.start({
        lang: "en-US",
        interimResults: true,
        continuous: false,
        recordingOptions: {
          persist: true,
          outputFilePath: recordingPath,
        },
      });
    } catch (e) {
      setMicRecording(false);
      showRejection("mic start failed");
    }
  }, [activeTrackId, showRejection]);

  const stopMic = useCallback(async () => {
    if (!activeTrackId) return;
    try {
      ExpoSpeechRecognitionModule.stop();
    } catch {}
    setMicRecording(false);
    const t = tracksRef.current.get(activeTrackId);
    if (!t) return;
    const audioUri = recordingFileRef.current;
    const transcript = pendingTranscript.trim();
    if (!audioUri) {
      showRejection("no recording");
      return;
    }
    patchTrackUi(t.id, { thinking: true });
    try {
      if (transcript) t.history.push({ role: "user", text: transcript });
      const res = await converseWithObject({
        audioUri,
        audioMime: "audio/m4a",
        transcript,
        className: t.className,
        description: t.description ?? "",
        voiceId: t.voiceId ?? "",
        history: t.history,
      });
      if (res.transcript && !transcript) {
        t.history.push({ role: "user", text: res.transcript });
      }
      t.history.push({ role: "object", text: res.reply });
      await speakOnTrack(t.id, res.reply, res.voiceId ?? t.voiceId ?? "", generationRef.current);
    } catch (e) {
      patchTrackUi(t.id, { thinking: false });
      showRejection("conversation failed");
    }
  }, [activeTrackId, patchTrackUi, pendingTranscript, showRejection, speakOnTrack]);

  // --- Measure the camera view for source↔view projection ---------------
  const onLayout = useCallback((e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    viewSizeRef.current = { width, height };
    setViewSize({ width, height });
  }, []);

  // --- Gesture ------------------------------------------------------------
  const tapGesture = useMemo(
    () =>
      Gesture.Tap()
        .maxDuration(400)
        .onEnd((e) => {
          // onEnd runs on the gesture thread — hop to JS.
          scheduleOnRN(onTap, e.x, e.y);
        }),
    [onTap]
  );

  // --- Render =============================================================

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.hint}>Camera permission required.</Text>
        <Pressable
          style={styles.button}
          onPress={() => {
            void requestPermission();
          }}
        >
          <Text style={styles.buttonLabel}>Grant permission</Text>
        </Pressable>
      </View>
    );
  }
  if (!device) {
    return (
      <View style={styles.center}>
        <Text style={styles.hint}>No camera available.</Text>
      </View>
    );
  }
  if (phase === "error") {
    return (
      <View style={styles.center}>
        <Text style={styles.hint}>Model failed to load.</Text>
        <Text style={styles.err}>{errorMsg}</Text>
      </View>
    );
  }

  return (
    <View style={styles.root}>
      <GestureDetector gesture={tapGesture}>
        <View style={styles.cameraWrap} onLayout={onLayout}>
          <Camera
            ref={cameraRef}
            style={StyleSheet.absoluteFill}
            device={device}
            isActive
            pixelFormat="rgb"
            frameOutput={onCameraFrame}
            photo
            orientationSource="device"
            onError={(e: CameraRuntimeError) => {
              setErrorMsg(e.message);
              setPhase("error");
            }}
          />

          {/* Overlays */}
          {uiTracks.map((u) => (
            <TrackOverlay key={u.id} ui={u} isActive={u.id === activeTrackId} />
          ))}

          {/* Loading / status */}
          {phase === "starting" && (
            <View style={styles.statusBadge}>
              <ActivityIndicator color="#ff98c0" />
              <Text style={styles.statusText}>
                {!seg.isReady
                  ? `Loading model… ${Math.round(
                      (seg.downloadProgress ?? 0) * 100
                    )}%`
                  : "Starting"}
              </Text>
            </View>
          )}

          {rejection && (
            <View style={styles.toast}>
              <Text style={styles.toastText}>{rejection.reason}</Text>
            </View>
          )}

          {/* Mic + clear controls */}
          <View style={styles.controls}>
            <Pressable
              onPress={clearAllTracks}
              style={[styles.controlButton, styles.clearButton]}
            >
              <Text style={styles.controlLabel}>clear</Text>
            </Pressable>
            <Pressable
              onPressIn={startMic}
              onPressOut={stopMic}
              disabled={!activeTrackId}
              style={[
                styles.controlButton,
                styles.micButton,
                micRecording && styles.micButtonActive,
                !activeTrackId && styles.controlDisabled,
              ]}
            >
              <Text style={styles.controlLabel}>
                {micRecording ? "listening…" : "hold to talk"}
              </Text>
            </Pressable>
          </View>
        </View>
      </GestureDetector>
    </View>
  );
}

// === Per-track overlay ====================================================

function TrackOverlay({ ui, isActive }: { ui: TrackUI; isActive: boolean }) {
  return (
    <View
      pointerEvents="none"
      style={[
        styles.faceWrap,
        {
          left: ui.left,
          top: ui.top,
          width: ui.size,
          height: ui.size * (FACE_VOICE_HEIGHT / FACE_VOICE_WIDTH),
          opacity: ui.opacity,
          transform: [{ rotate: `${ui.tiltDeg}deg` }],
        },
      ]}
    >
      <View style={{ width: "100%", height: "100%" }}>
        <FaceVoice shape={ui.shape} />
      </View>
      {/* Speech bubble hovers above the face. Max width scales with face. */}
      <View
        style={{
          position: "absolute",
          bottom: "100%",
          left: "50%",
          transform: [{ translateX: -Math.max(ui.size, 120) / 2 }, { rotate: `${-ui.tiltDeg}deg` }],
          width: Math.max(ui.size, 120),
          alignItems: "center",
          marginBottom: 10,
        }}
      >
        <SpeechBubble
          caption={ui.caption}
          thinking={ui.thinking}
          speaking={ui.speaking}
          maxWidth={Math.max(ui.size, 160)}
        />
      </View>
    </View>
  );
}

// === Styles ===============================================================

const windowDims = Dimensions.get("window");

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: "#0a0a0a",
  },
  cameraWrap: {
    flex: 1,
    position: "relative",
    overflow: "hidden",
  },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 32,
    backgroundColor: "#0a0a0a",
    gap: 16,
  },
  hint: { color: "#fff", fontSize: 16, textAlign: "center" },
  err: { color: "#ff98c0", fontSize: 13, textAlign: "center" },
  button: {
    backgroundColor: "#ff98c0",
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 999,
  },
  buttonLabel: { color: "#2a1540", fontWeight: "700" },
  faceWrap: {
    position: "absolute",
  },
  statusBadge: {
    position: "absolute",
    top: 40,
    alignSelf: "center",
    flexDirection: "row",
    gap: 10,
    alignItems: "center",
    backgroundColor: "rgba(18,10,28,0.8)",
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 999,
  },
  statusText: { color: "#fff", fontSize: 13 },
  toast: {
    position: "absolute",
    top: 90,
    alignSelf: "center",
    backgroundColor: "rgba(42,21,64,0.92)",
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 18,
    maxWidth: Math.min(windowDims.width - 48, 420),
  },
  toastText: { color: "#fff", fontSize: 13, textAlign: "center" },
  controls: {
    position: "absolute",
    bottom: Platform.OS === "ios" ? 42 : 26,
    left: 0,
    right: 0,
    flexDirection: "row",
    justifyContent: "center",
    gap: 16,
  },
  controlButton: {
    paddingHorizontal: 22,
    paddingVertical: 14,
    borderRadius: 999,
    backgroundColor: "rgba(255,255,255,0.92)",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.35,
    shadowRadius: 12,
    elevation: 6,
  },
  clearButton: {
    backgroundColor: "rgba(20,10,30,0.82)",
  },
  micButton: {
    backgroundColor: "#ff98c0",
    minWidth: 180,
    alignItems: "center",
  },
  micButtonActive: {
    backgroundColor: "#ff5ca0",
  },
  controlDisabled: {
    opacity: 0.5,
  },
  controlLabel: {
    color: "#2a1540",
    fontSize: 14,
    fontWeight: "700",
  },
});

// Export the constants map so future callers can tweak tuning without
// reaching into the component — matches the browser tracker's implicit
// contract that the knobs are part of the module surface.
export const TRACKER_CONSTANTS = {
  MAX_INFERENCE_FPS,
  IDENTITY_IOU_MIN,
  BOX_POS_ALPHA,
  BOX_SIZE_ALPHA,
  LOST_AFTER_MISSES,
  WIDEN_MATCH_AFTER_MISSES,
  SUSPECT_SIZE_RATIO,
  EXTRAP_MAX_MS,
  EXTRAP_MISS_LIMIT,
  VELOCITY_EMA,
  VELOCITY_DECAY_PER_MISS,
  CONTINUOUS_CONF,
  TAP_CONF,
  FACE_BBOX_FRACTION,
  FACE_NATIVE_PX,
  FACE_SCALE_MIN,
  FACE_SCALE_MAX,
  TILT_GAIN_DEG,
  TILT_MAX_DEG,
  TILT_EMA,
  VOICE_SIZE_REF_FRAC,
  VOICE_SIZE_EXP,
  VOICE_GAIN_MAX,
  VOICE_PERSIST_MS,
  VOICE_FADE_ALPHA,
  MAX_FACES,
  TAP_CACHE_MAX_AGE_MS,
  SEG_INPUT_SIZE,
};

// Stop the bundler from whining about unused imports when a minor code path
// is temporarily disabled.
void COCO_CLASSES;
void FACE_VOICE_HEIGHT;
