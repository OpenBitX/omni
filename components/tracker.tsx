"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { generateLine } from "@/app/actions";
import { Face } from "@/components/face";
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

// === Pipeline tuning knobs ===============================================
//
// These are the numbers you'd reach for when the tracker feels wrong.
// Each one controls a named trade-off; tune in isolation.

// Inference rate cap. YOLOv8n on mobile CPU-WASM sits around 3–8 FPS; we
// don't gate beyond that. Desktop WebGPU can hit 30+ — then this cap keeps
// us from burning battery for motion we can already handle with EMA.
const MAX_INFERENCE_FPS = 30;

// IoU gate for "this new box is the same object I was tracking". 0.3 is a
// good default — lower lets the face drift onto a bumping neighbor, higher
// loses the target whenever the box shape wobbles during fast motion.
const IDENTITY_IOU_MIN = 0.3;

// EMA alphas. Per the brief: 0.4 on position+size = snappy but not jittery.
// Face opacity uses a slightly faster alpha so occlusion in/out feels
// decisive rather than ghostly.
const BOX_EMA_ALPHA = 0.4;
const OPACITY_EMA_ALPHA = 0.55;

// Occlusion latch. If the target's class disappears (or drops IoU) for this
// many consecutive inference frames, hide the face + mute audio. A tight
// count feels responsive; too tight and blips from bad frames cause flicker.
const OCCLUSION_MISS_FRAMES = 5;

// Confidence thresholds. Tap uses a looser threshold — the user's intent is
// a strong prior that something tappable lives at that point. DETR sigmoid
// probabilities run lower than YOLOv8 softmax-ish scores, so we stay
// permissive — we'd rather show too many things than miss them.
const CONTINUOUS_CONF = 0.15;
const TAP_CONF = 0.08;
const CONTINUOUS_MAX_DET = 25;

// Face size = FACE_BBOX_FRACTION × min(box.w, box.h). The <Face /> SVG is
// 200 CSS px at scale=1, hence the /200.
const FACE_BBOX_FRACTION = 0.62;
const FACE_SVG_NATIVE_PX = 200;

// Hard clamp on face scale — keeps tiny boxes from birthing a grain-of-sand
// face and giant boxes from drowning the screen when the user walks up.
const FACE_SCALE_MIN = 0.25;
const FACE_SCALE_MAX = 3.0;

// Breathing boxes: don't show them once something is locked — clutters.
// Classes explicitly forbidden from face placement (the app is about things,
// not people). Matches the policy in ASSESS_SYSTEM.
const EXCLUDED_CLASS_IDS = new Set<number>([PERSON_CLASS_ID]);

// Tap-hit fallback: if the tap isn't inside any detected box, we run a
// one-shot lower-threshold detection. Cached continuous detections are
// reused when fresh enough to avoid the extra inference cost.
const TAP_CACHE_MAX_AGE_MS = 400;

// Visible tap-frame fallback dimensions when YOLO detects nothing at all.
const TAP_FRAME_FRACTION = 0.55;

type Phase = "starting" | "ready" | "locked" | "error";

type Target = {
  classId: number;
  className: string;
  // Anchor = face-placement point expressed as ratio of the tracked box.
  // Replayed against each new smoothed box → face stays on the right spot
  // even as the box stretches, shrinks, or shifts.
  anchor: Anchor;
};

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
  const left = (box.x1 - offX) * pxPerSrcX;
  const top = (box.y1 - offY) * pxPerSrcY;
  const width = (box.x2 - box.x1) * pxPerSrcX;
  const height = (box.y2 - box.y1) * pxPerSrcY;
  return { left, top, width, height };
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

// Among multiple detections under the same tap, prefer the box with the
// smallest area — tighter boxes are usually the correct semantic target
// when two containers are nested (cup inside the region of a dining table).
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
  // Fallback: no box strictly contains the tap — nearest-center within a
  // fraction of the video's min dimension so we don't pick the far corner.
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

export function Tracker() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  const [phase, setPhase] = useState<Phase>("starting");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [thinking, setThinking] = useState(false);
  const [caption, setCaption] = useState<string | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [rejection, setRejection] = useState<string | null>(null);
  const [yoloReady, setYoloReady] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [yoloStatus, setYoloStatusState] = useState<YoloStatus>(() => getYoloStatus());
  const [lastInferMs, setLastInferMs] = useState<number | null>(null);
  const [diagOpen, setDiagOpen] = useState(false);
  const [diagError, setDiagError] = useState<string | null>(null);
  const [retryToken, setRetryToken] = useState(0);
  const rejectionTimerRef = useRef<number | null>(null);

  // --- Push-to-talk state -----------------------------------------------
  // Held-down mic recording. `isRecording` drives the visual state;
  // `talkLevel` is a 0..1 envelope from the mic's frequency analyser so the
  // button ring can pulse with the user's voice. `recordedBlobRef` holds the
  // most recent utterance for wiring into the per-object pipeline later.
  const [isRecording, setIsRecording] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const [talkLevel, setTalkLevel] = useState(0);
  const [talkFlash, setTalkFlash] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<BlobPart[]>([]);
  const micStreamRef = useRef<MediaStream | null>(null);
  const recordedBlobRef = useRef<Blob | null>(null);
  const talkAnalyserRef = useRef<AnalyserNode | null>(null);
  const talkFreqDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const talkSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const talkLevelRafRef = useRef<number | null>(null);
  const talkFlashTimerRef = useRef<number | null>(null);

  // Detections the continuous loop has published. Used both to paint
  // breathing boxes (ready state) and to resolve taps without a fresh
  // inference.
  const [detections, setDetections] = useState<Detection[]>([]);
  const detectionsTsRef = useRef(0);
  const detectionsRef = useRef<Detection[]>([]);

  // The tap frame shown during VLM assessment. Mirrors the previous tracker's
  // UX — instant feedback so the user sees what's being analyzed.
  const [tapFrame, setTapFrame] = useState<
    (ViewportBox & { gen: number }) | null
  >(null);

  // Face animation state (blink / eye darts / audio-driven mouth).
  const [mouth, setMouth] = useState(0);
  const [blink, setBlink] = useState(0);
  const [eye, setEye] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Tracking state.
  const targetRef = useRef<Target | null>(null);
  const boxEmaRef = useRef<BoxEMA>(newBoxEMA(BOX_EMA_ALPHA));
  const smoothedBoxRef = useRef<Box | null>(null);
  const missedFramesRef = useRef(0);
  const faceOpacityRef = useRef(0);
  const lockedRef = useRef(false);
  const yoloReadyRef = useRef(false);
  const inferenceInFlightRef = useRef(false);
  const lastInferenceAtRef = useRef(0);
  const inferenceCountRef = useRef(0);

  // Scratch canvases + RAF handle.
  const yoloCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const cropCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const fpsLastSampleRef = useRef(0);
  const fpsFrameCountRef = useRef(0);

  // Audio plumbing — unchanged from the LK tracker.
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const freqDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const mouthTargetRef = useRef(0);
  const mouthSmoothRef = useRef(0);

  // Generation counter guards async work (VLM call, TTS decode) from stale
  // taps. Every tap increments — if it no longer matches by the time a
  // promise resolves, we drop the result on the floor.
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
      if (currentSourceRef.current) {
        try {
          currentSourceRef.current.stop();
        } catch {
          // already stopped
        }
      }
      // Tear down push-to-talk plumbing.
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

  // --- Phase transition: both pieces have to land before we're ready ---
  useEffect(() => {
    if (cameraReady && yoloReady) {
      setPhase((p) => (p === "error" || p === "locked" ? p : "ready"));
    }
  }, [cameraReady, yoloReady]);

  // --- YOLO status subscription + warm-up --------------------------------
  useEffect(() => {
    const unsub = subscribeYoloStatus((s) => {
      setYoloStatusState(s);
      if (s.stage === "error" && s.error) {
        // console.log, not console.error — the error is already surfaced in
        // the dedicated error banner + diag panel; console.error would
        // additionally show Next.js's dev overlay which covers the UI.
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
        // Don't flip the whole app into an error phase — the camera preview
        // can still be useful, and we offer a retry. Just surface the error.
        const msg = e instanceof Error ? e.message : "detector failed";
        setErrorMsg(`detector: ${msg}`);
        // console.log, not console.error — Next dev overlay would obscure UI.
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

  // --- Face animation loop (blink / eye / audio-driven mouth) ------------
  useEffect(() => {
    let raf = 0;
    let nextBlinkAt = performance.now() + 1800;
    let blinkStart = 0;
    let nextEyeMoveAt = performance.now() + 1200;
    let eyeTarget = { x: 0, y: 0 };
    const eyeCurrent = { x: 0, y: 0 };

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);

      if (blinkStart === 0 && now >= nextBlinkAt) {
        blinkStart = now;
        nextBlinkAt = now + 2500 + Math.random() * 3500;
      }
      let b = 0;
      if (blinkStart > 0) {
        const dt = now - blinkStart;
        const dur = 170;
        if (dt < dur) b = Math.sin((dt / dur) * Math.PI);
        else blinkStart = 0;
      }
      setBlink(b);

      if (now >= nextEyeMoveAt) {
        eyeTarget = {
          x: (Math.random() - 0.5) * 1.6,
          y: (Math.random() - 0.5) * 1.1,
        };
        nextEyeMoveAt = now + 1400 + Math.random() * 2200;
      }
      eyeCurrent.x += (eyeTarget.x - eyeCurrent.x) * 0.1;
      eyeCurrent.y += (eyeTarget.y - eyeCurrent.y) * 0.1;
      setEye({ x: eyeCurrent.x, y: eyeCurrent.y });

      // Mouth follows audio envelope — unless the audio source is gone, in
      // which case the mouth goes slack.
      if (
        currentSourceRef.current &&
        analyserRef.current &&
        freqDataRef.current
      ) {
        analyserRef.current.getByteFrequencyData(freqDataRef.current);
        const data = freqDataRef.current;
        const end = Math.min(40, data.length);
        let sum = 0;
        let count = 0;
        for (let i = 2; i < end; i++) {
          sum += data[i];
          count++;
        }
        const avg = count > 0 ? sum / count / 255 : 0;
        mouthTargetRef.current = Math.min(1, Math.max(0, (avg - 0.05) * 3));
      } else {
        mouthTargetRef.current = 0;
      }
      mouthSmoothRef.current +=
        (mouthTargetRef.current - mouthSmoothRef.current) * 0.45;
      setMouth(mouthSmoothRef.current);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
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

  // --- Overlay writer ----------------------------------------------------
  //
  // Positions the SVG face via a single composed CSS transform — translate
  // to the anchor, pull back half (so the face centers on it), scale, fade.
  // Written every RAF even if no new inference arrived, so the EMA'd face
  // position always reflects the latest tick of tracking state.
  const writeOverlay = useCallback(() => {
    const overlay = overlayRef.current;
    const video = videoRef.current;
    if (!overlay || !video) return;
    const box = smoothedBoxRef.current;
    const target = targetRef.current;
    if (!lockedRef.current || !box || !target) {
      overlay.style.opacity = "0";
      return;
    }
    const facePointSrc = applyAnchor(target.anchor, box);
    const el = sourceToElementPoint(facePointSrc, video);
    if (!el) return;
    const rect = video.getBoundingClientRect();
    const fx = el.clientX - rect.left;
    const fy = el.clientY - rect.top;

    // Face size tracks the box's smaller side so the face stays proportional
    // through zooms. Clamp to the min/max so extreme boxes don't break the UI.
    const boxMinSide = Math.min(box.w, box.h);
    const targetFacePxVideo = boxMinSide * FACE_BBOX_FRACTION;
    // Convert the source-space face pixel size into element-space pixels
    // (again the object-cover letterbox).
    const srcToElAvg = ((rect.width / video.videoWidth) +
      (rect.height / video.videoHeight)) / 2;
    const targetFacePxEl = targetFacePxVideo * srcToElAvg;
    const scale = Math.max(
      FACE_SCALE_MIN,
      Math.min(FACE_SCALE_MAX, targetFacePxEl / FACE_SVG_NATIVE_PX)
    );

    overlay.style.opacity = faceOpacityRef.current.toFixed(3);
    overlay.style.transform = `translate(${fx}px, ${fy}px) translate(-50%, -50%) scale(${scale})`;
  }, []);

  // --- YOLO + tracking RAF loop -----------------------------------------
  useEffect(() => {
    const tick = (now: number) => {
      rafRef.current = requestAnimationFrame(tick);
      const v = videoRef.current;
      if (!v || !v.videoWidth) return;

      // Rate-limit inference. ORT queues runs, so back-to-back submits bloat
      // latency without improving framerate.
      const minInterval = 1000 / MAX_INFERENCE_FPS;
      const canLaunch =
        yoloReadyRef.current &&
        !inferenceInFlightRef.current &&
        now - lastInferenceAtRef.current >= minInterval;

      if (canLaunch) {
        inferenceInFlightRef.current = true;
        lastInferenceAtRef.current = now;
        const launchGen = generationRef.current;
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
            // Publish detections in both ready and locked phases — the diag
            // panel benefits even while locked, and during ready the
            // breathing boxes need them.
            setDetections(dets);

            // Loud logging so the user can see the inference pulse in devtools.
            // First three detections printed compactly.
            inferenceCountRef.current++;
            if (inferenceCountRef.current <= 3 || inferenceCountRef.current % 30 === 0) {
              // eslint-disable-next-line no-console
              console.log(
                `[tracker] inference #${inferenceCountRef.current}: ${dets.length} dets in ${inferMs}ms`,
                dets.slice(0, 3).map((d) => `${d.className}@${(d.score * 100).toFixed(0)}%`).join(", ")
              );
            }

            if (lockedRef.current && targetRef.current && launchGen === generationRef.current) {
              advanceTracking(dets);
            }

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
            // Use console.log here — the Next.js dev overlay promotes every
            // console.error to a fullscreen red card that obscures the camera.
            // Real failures also surface in the diag panel so we don't lose
            // signal; it's console.error that creates the false-alarm UX.
            // eslint-disable-next-line no-console
            console.log("[tracker] inference error:", err);
            setDiagError(msg);
          });
      }

      // Opacity EMA runs every RAF (fast fade in/out for occlusion).
      const targetOpacity = lockedRef.current && missedFramesRef.current === 0 ? 1 : 0;
      faceOpacityRef.current = lerp(
        faceOpacityRef.current,
        targetOpacity,
        OPACITY_EMA_ALPHA
      );

      writeOverlay();
    };

    const advanceTracking = (dets: readonly Detection[]) => {
      const target = targetRef.current;
      const prev = smoothedBoxRef.current;
      if (!target || !prev) return;

      const match = matchTarget(
        dets as Detection[],
        { ...prev, classId: target.classId },
        IDENTITY_IOU_MIN
      );

      if (match) {
        const wasOccluded = missedFramesRef.current >= OCCLUSION_MISS_FRAMES;
        missedFramesRef.current = 0;
        if (wasOccluded) {
          // Reacquisition: snap the EMA rather than slowly sliding from the
          // last-seen position. The user notices the ghost-slide more than
          // they notice the jump, and the jump stays invisible behind the
          // opacity fade-in anyway.
          const snap: Box = makeBox(match.cx, match.cy, match.w, match.h);
          seedBoxEMA(boxEmaRef.current, snap);
          smoothedBoxRef.current = snap;
        } else {
          smoothedBoxRef.current = smoothBox(boxEmaRef.current, match);
        }
      } else {
        missedFramesRef.current++;
        // Keep the last smoothed box, let opacity EMA fade the face out.
        if (missedFramesRef.current >= OCCLUSION_MISS_FRAMES) {
          // Silence audio on occlusion. Preserves the "paused, not lost"
          // model — if the object comes back, we fade back in; if the user
          // taps something else, generation flip kills the re-entry.
          const src = currentSourceRef.current;
          if (src) {
            try {
              src.stop();
            } catch {
              // already stopped
            }
            currentSourceRef.current = null;
            setSpeaking(false);
          }
        }
      }
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [ensureYoloCanvas, writeOverlay]);

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
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.4;
      analyser.connect(ctx.destination);
      analyserRef.current = analyser;
      freqDataRef.current = new Uint8Array(
        new ArrayBuffer(analyser.frequencyBinCount)
      );
    }
    if (ctx.state === "suspended") {
      ctx.resume().catch(() => {});
    }
    return ctx;
  }, []);

  const stopCurrentAudio = useCallback(() => {
    const src = currentSourceRef.current;
    if (src) {
      try {
        src.stop();
      } catch {
        // already stopped
      }
      currentSourceRef.current = null;
    }
    setSpeaking(false);
  }, []);

  // --- Hold-to-talk recording -------------------------------------------
  //
  // Pointer-down starts recording, pointer-up/leave/cancel stops. Mic stream
  // is lazy-opened on the first press so we respect the user-gesture rule
  // on iOS Safari. The analyser feeds the button's pulse-ring level meter.
  const stopTalkLevelLoop = useCallback(() => {
    if (talkLevelRafRef.current != null) {
      cancelAnimationFrame(talkLevelRafRef.current);
      talkLevelRafRef.current = null;
    }
    setTalkLevel(0);
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
    // Audio ctx must exist to drive the TTS playback path; also gets us the
    // analyser for the voice-level ring.
    const ctx = ensureAudioCtx();
    const stream = await openMicStream();
    if (!stream) return;

    // Hook the mic into a dedicated analyser for the pulse-ring. We never
    // connect it to destination — it's read-only level sensing.
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
        // Some browsers throw if the stream's already consumed — fine to skip
      }
    }

    // MIME negotiation: prefer opus, fall back to whatever the browser offers.
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
      // Brief "sent" flash on the button so the user sees the handoff.
      setTalkFlash(true);
      if (talkFlashTimerRef.current != null) clearTimeout(talkFlashTimerRef.current);
      talkFlashTimerRef.current = window.setTimeout(() => {
        setTalkFlash(false);
        talkFlashTimerRef.current = null;
      }, 650);
    };
    recorderRef.current = mr;
    mr.start(100);
    setIsRecording(true);
    setMicError(null);

    // Tiny haptic nudge on mobile where supported. Not a UX load-bearer —
    // just a courtesy hint that recording started.
    if (typeof navigator.vibrate === "function") {
      try {
        navigator.vibrate(8);
      } catch {
        // some browsers disable vibrate without interaction — ignore
      }
    }

    // Kick off the level-meter RAF. We read sigmoid-ish loudness from the
    // analyser's byte-frequency data and feed it to talkLevel state; the
    // button pulse ring width tracks it.
    const readLevel = () => {
      const an = talkAnalyserRef.current;
      const buf = talkFreqDataRef.current;
      if (!an || !buf) {
        talkLevelRafRef.current = requestAnimationFrame(readLevel);
        return;
      }
      an.getByteFrequencyData(buf);
      // Speech energy lives mostly in the first ~40 bins (sub-4 kHz).
      let sum = 0;
      const end = Math.min(40, buf.length);
      for (let i = 2; i < end; i++) sum += buf[i];
      const avg = sum / Math.max(1, end - 2) / 255;
      // Shape: push low values down, compress high values so the ring doesn't
      // pin. Threshold at 0.04 to hide ambient noise.
      const shaped = Math.max(0, Math.min(1, (avg - 0.04) * 2.5));
      setTalkLevel((prev) => prev + (shaped - prev) * 0.4);
      talkLevelRafRef.current = requestAnimationFrame(readLevel);
    };
    if (talkLevelRafRef.current == null) {
      talkLevelRafRef.current = requestAnimationFrame(readLevel);
    }
  }, [ensureAudioCtx, openMicStream]);

  const stopRecording = useCallback(() => {
    const mr = recorderRef.current;
    if (mr && mr.state === "recording") {
      try {
        mr.stop();
      } catch {
        // no-op
      }
    }
    recorderRef.current = null;
    setIsRecording(false);
    stopTalkLevelLoop();
  }, [stopTalkLevelLoop]);

  const speak = useCallback(
    async (cropDataUrl: string, callGen: number) => {
      const ctx = ensureAudioCtx();
      if (!ctx || !analyserRef.current) {
        setErrorMsg("audio unavailable");
        return;
      }
      stopCurrentAudio();
      setThinking(true);
      setCaption(null);
      setErrorMsg(null);

      try {
        const { line, audioDataUrl } = await generateLine(cropDataUrl);
        if (callGen !== generationRef.current) return;

        setCaption(line);

        // Audio is optional (OPENAI_API_KEY may be absent on GLM-only deploys).
        // Caption still shows so the user sees the line.
        if (!audioDataUrl) return;

        const resp = await fetch(audioDataUrl);
        const buf = await resp.arrayBuffer();
        const audioBuf = await ctx.decodeAudioData(buf);
        if (callGen !== generationRef.current) return;

        const source = ctx.createBufferSource();
        source.buffer = audioBuf;
        source.connect(analyserRef.current);
        source.onended = () => {
          if (currentSourceRef.current === source) {
            currentSourceRef.current = null;
            setSpeaking(false);
          }
        };
        currentSourceRef.current = source;
        setSpeaking(true);
        source.start();
      } catch (e) {
        if (callGen !== generationRef.current) return;
        const msg = e instanceof Error ? e.message : "line failed";
        setErrorMsg(msg);
        setDiagError(msg);
        // Failure is otherwise invisible once locked — no caption, no audio,
        // just a silent face. Surface it via the rejection toast too, and
        // with a human-friendly hint when the cause is a missing key.
        showRejection(
          /zhipu|glm|api key|api_key|401|403/i.test(msg)
            ? "voice model unconfigured — check .env.local"
            : `couldn't speak: ${msg.slice(0, 80)}`,
          3200
        );
        // eslint-disable-next-line no-console
        console.error("[tracker] speak failed:", e);
      } finally {
        if (callGen === generationRef.current) setThinking(false);
      }
    },
    [ensureAudioCtx, showRejection, stopCurrentAudio]
  );

  // --- Capture a source-space rectangle as a jpeg data URL --------------
  // Pad by 5% on each side so the VLM sees a little context around the box.
  const captureBoxFrame = useCallback(
    (box: { x1: number; y1: number; x2: number; y2: number }): string | null => {
      const v = videoRef.current;
      if (!v || !v.videoWidth) return null;
      const pw = (box.x2 - box.x1) * 0.05;
      const ph = (box.y2 - box.y1) * 0.05;
      const cropSx = Math.max(0, box.x1 - pw);
      const cropSy = Math.max(0, box.y1 - ph);
      const cropW = Math.min(v.videoWidth - cropSx, (box.x2 - box.x1) + 2 * pw);
      const cropH = Math.min(v.videoHeight - cropSy, (box.y2 - box.y1) + 2 * ph);
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

  // --- Tap handler -------------------------------------------------------
  const handleTap = useCallback(
    async (e: React.PointerEvent) => {
      const v = videoRef.current;
      if (!v) return;

      // Bump generation immediately so any in-flight VLM/TTS from an earlier
      // tap drops its result.
      const gen = ++generationRef.current;
      const rectNow = v.getBoundingClientRect();
      const tapElX = e.clientX - rectNow.left;
      const tapElY = e.clientY - rectNow.top;

      // Instant fallback frame — the real one snaps in once we know the box.
      const elMin = Math.min(rectNow.width, rectNow.height);
      const fallbackEl = elMin * TAP_FRAME_FRACTION;
      setTapFrame({
        left: tapElX - fallbackEl / 2,
        top: tapElY - fallbackEl / 2,
        width: fallbackEl,
        height: fallbackEl,
        gen,
      });

      // Unlock audio inside the pointer-down gesture, before any early returns
      // or awaits, so the very first tap always clears Safari's autoplay gate.
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

      // Resolve the tapped object: reuse recent continuous detections when
      // fresh, otherwise fire a targeted, looser-threshold inference.
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

      // If cache missed everything, do one more round with a looser threshold.
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
            if (!lockedRef.current) setDetections(fresh);
          }
        } catch {
          // fall through to rejection below
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

      // Capture the padded crop for the VLM voice-line call.
      const dataUrl = captureBoxFrame(tapped);
      if (!dataUrl) {
        showRejection("couldn't grab the frame");
        return;
      }

      stopCurrentAudio();
      setCaption(null);
      setErrorMsg(null);
      setRejection(null);
      if (rejectionTimerRef.current != null) {
        clearTimeout(rejectionTimerRef.current);
        rejectionTimerRef.current = null;
      }

      // Lock immediately on the YOLO box. Face anchor = box center (rx=0,
      // ry=0) per the current product spec — trust YOLO's tight class-aware
      // box rather than round-tripping through a VLM face-placement call.
      // The VLM is still invoked asynchronously for the voice line below.
      const lockBox: Box = makeBox(
        (tapped.x1 + tapped.x2) / 2,
        (tapped.y1 + tapped.y2) / 2,
        tapped.x2 - tapped.x1,
        tapped.y2 - tapped.y1
      );
      const anchor: Anchor = { rx: 0, ry: 0 };

      targetRef.current = {
        classId: tapped.classId,
        className: tapped.className,
        anchor,
      };
      boxEmaRef.current = newBoxEMA(BOX_EMA_ALPHA);
      seedBoxEMA(boxEmaRef.current, lockBox);
      smoothedBoxRef.current = lockBox;
      missedFramesRef.current = 0;
      faceOpacityRef.current = 0;
      lockedRef.current = true;
      setPhase("locked");

      // Clear the tap frame once we're locked — face has taken over.
      setTapFrame(null);

      // Voice: use the crop we already paid to prepare.
      speak(dataUrl, gen);
    },
    [
      captureBoxFrame,
      ensureAudioCtx,
      ensureYoloCanvas,
      errorMsg,
      phase,
      showRejection,
      speak,
      stopCurrentAudio,
    ]
  );

  // --- Derived UI --------------------------------------------------------
  const dotClass =
    phase === "error"
      ? "bg-rose-400"
      : phase === "starting"
        ? "bg-amber-300"
        : phase === "ready"
          ? "bg-fuchsia-400"
          : thinking
            ? "bg-amber-300"
            : speaking
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
          : thinking
            ? "thinking"
            : speaking
              ? "speaking"
              : `tracking · ${fps} fps`;

  // Breathing boxes are only rendered in the ready state (pre-lock). Mapped
  // to element space here, re-computed whenever `detections` changes.
  const breathingBoxes = useMemo(() => {
    const v = videoRef.current;
    if (!v || phase !== "ready") return [];
    return detections
      .map((d) => {
        if (EXCLUDED_CLASS_IDS.has(d.classId)) return null;
        const vp = sourceBoxToElement(d, v);
        if (!vp) return null;
        // Tiny boxes are usually spurious, but be lenient — the user wants
        // to SEE what YOLO sees, not a sanitized subset. Kill only truly
        // degenerate ones.
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

      {/* Face overlay. Always mounted while locked, opacity animated for
          occlusion. Rendered absolute, positioned by writeOverlay via
          a CSS transform. */}
      <div
        ref={overlayRef}
        className="pointer-events-none absolute left-0 top-0 will-change-transform"
        style={{
          transformOrigin: "0 0",
          opacity: 0,
          transition: "opacity 120ms linear",
          display: phase === "locked" ? "block" : "none",
        }}
      >
        <Face mouth={mouth} blink={blink} eyeX={eye.x} eyeY={eye.y} />
      </div>

      {/* Breathing boxes — every YOLO detection the model is willing to
          commit to, labelled with class + confidence. Visibility is
          intentional here: the user wants to see what's tappable. */}
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
          <span
            className="serif-italic absolute -top-[26px] left-1 flex items-center gap-1 rounded-full bg-[color:var(--ink)]/75 px-2.5 py-0.5 text-[11px] font-medium text-white ring-1 ring-white/20 backdrop-blur-md"
          >
            <span>{b.className}</span>
            <span className="tabular-nums text-white/60">
              {Math.round(b.score * 100)}%
            </span>
          </span>
        </div>
      ))}

      {/* Tap frame — snapped to the detected box while the VLM thinks. */}
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
        </div>
      </div>

      {/* YOLO loading overlay — visible during model download + compile. */}
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
                  ? `${Math.round(yoloStatus.bytesLoaded / 1024 / 1024 * 10) / 10}/${Math.round(yoloStatus.bytesTotal / 1024 / 1024 * 10) / 10} MB`
                  : yoloStatus.bytesLoaded > 0
                    ? `${Math.round(yoloStatus.bytesLoaded / 1024 / 1024 * 10) / 10} MB`
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

      {/* YOLO load error banner — shown if init fails. Retry button beside. */}
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

      {/* Diagnostic panel — toggled by the "i" button. Shows everything the
          user might need to send back if something misbehaves. */}
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
            <dt className="text-white/50">locked</dt>
            <dd>{targetRef.current ? targetRef.current.className : "—"}</dd>
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
              tap anything — give it a voice
            </span>
          </div>
        </div>
      )}

      {rejection && !thinking && (
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

      {thinking && (
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

      {caption && (
        <div className="pointer-events-none absolute inset-x-0 bottom-[max(env(safe-area-inset-bottom),22px)] px-5 pb-[148px] pt-10">
          <div
            className="mx-auto max-w-md rounded-[26px] bg-white/95 px-6 py-4 shadow-[0_24px_60px_-20px_rgba(0,0,0,0.55)] ring-1 ring-white/80 backdrop-blur-xl"
            style={{ animation: "bubble-in 480ms cubic-bezier(0.16,1,0.3,1) both" }}
          >
            <p className="serif-italic text-balance text-center text-[18px] leading-[1.35] text-[color:var(--ink)]">
              &ldquo;{caption}&rdquo;
            </p>
            {speaking && (
              <div className="mt-2 flex justify-center gap-1">
                <span
                  className="h-1.5 w-1.5 rounded-full bg-[color:var(--accent)]"
                  style={{ animation: "soft-pulse 1s ease-in-out infinite" }}
                />
                <span
                  className="h-1.5 w-1.5 rounded-full bg-[color:var(--accent)]"
                  style={{ animation: "soft-pulse 1s ease-in-out 0.15s infinite" }}
                />
                <span
                  className="h-1.5 w-1.5 rounded-full bg-[color:var(--accent)]"
                  style={{ animation: "soft-pulse 1s ease-in-out 0.3s infinite" }}
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Hold-to-talk button. Bottom-center, safe-area-aware. Pointer-down
          starts recording, pointer-up/leave/cancel stops. Layers, outside-in:
          voice-level ring, soft glow halo, primary pill, mic glyph, sent flash. */}
      <div
        className="absolute inset-x-0 bottom-0 flex flex-col items-center gap-2 px-5 pb-[max(env(safe-area-inset-bottom),22px)] pt-3"
        style={{
          opacity: phase === "starting" || phase === "error" ? 0.35 : 1,
          transition: "opacity 220ms ease",
          pointerEvents: phase === "starting" || phase === "error" ? "none" : "auto",
        }}
      >
        <div className="relative grid h-[96px] w-[96px] place-items-center">
          {/* Voice-level ring — scale + opacity track mic amplitude. */}
          <span
            aria-hidden
            className="pointer-events-none absolute inset-0 rounded-full"
            style={{
              transform: `scale(${isRecording ? 1 + talkLevel * 0.9 : 1})`,
              opacity: isRecording ? 0.35 + talkLevel * 0.5 : 0,
              boxShadow:
                "0 0 0 10px rgba(255,137,190,0.35), 0 0 60px rgba(255,137,190,0.45)",
              transition: isRecording
                ? "transform 90ms ease-out, opacity 120ms ease-out"
                : "opacity 220ms ease, transform 220ms ease",
            }}
          />
          {/* Ambient glow halo — when idle, gently pulses. */}
          <span
            aria-hidden
            className="pointer-events-none absolute -inset-2 rounded-full"
            style={{
              background:
                "radial-gradient(circle, rgba(255,137,190,0.28) 0%, rgba(255,137,190,0) 70%)",
              animation: isRecording ? undefined : "soft-pulse 2.4s ease-in-out infinite",
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
            className="relative grid h-[80px] w-[80px] place-items-center rounded-full shadow-[0_18px_50px_-18px_rgba(236,72,153,0.65)] ring-1 ring-white/35 backdrop-blur-xl transition-transform"
            style={{
              background: isRecording
                ? "radial-gradient(circle at 35% 30%, #ffc2dc 0%, #ff89be 60%, #ec4899 100%)"
                : "radial-gradient(circle at 35% 30%, rgba(255,255,255,0.85) 0%, rgba(255,194,220,0.65) 55%, rgba(236,72,153,0.55) 100%)",
              transform: isRecording
                ? `scale(${1.06 + talkLevel * 0.05})`
                : talkFlash
                  ? "scale(1.04)"
                  : "scale(1)",
              WebkitTouchCallout: "none",
              WebkitUserSelect: "none",
              userSelect: "none",
              touchAction: "none",
            }}
          >
            {/* Mic glyph — pure SVG, no icon lib. */}
            <svg
              width="28"
              height="28"
              viewBox="0 0 24 24"
              fill="none"
              stroke={isRecording ? "#fff" : "var(--ink)"}
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 2.5c1.7 0 3 1.35 3 3v6c0 1.65-1.3 3-3 3s-3-1.35-3-3v-6c0-1.65 1.3-3 3-3z" />
              <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0" />
              <path d="M12 17.5V21.5" />
              <path d="M8.5 21.5h7" />
            </svg>
            {/* Brief "sent" check flash on release. */}
            {talkFlash && (
              <span
                className="pointer-events-none absolute inset-0 grid place-items-center rounded-full"
                style={{ animation: "fade-in 180ms ease-out both" }}
              >
                <svg
                  width="30"
                  height="30"
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
                : "hold to speak"}
        </span>
      </div>
    </div>
  );
}

