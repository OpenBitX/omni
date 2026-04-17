"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { generateLine } from "@/app/actions";
import { Face } from "@/components/face";
import {
  applyTransform,
  detectCorners,
  estimateSimilarity,
  filterOutliers,
  invertTransform,
  toGray,
  trackLK,
  type Gray,
  type Pt,
  type Transform,
} from "@/lib/lk";

const PROC_W = 320;
const MIN_POINTS = 8;
const RESEED_THRESHOLD = 10;
const ROI_FRACTION = 0.32;
const CROP_FRACTION = 0.38;

type Phase = "starting" | "ready" | "locked" | "error";

export function Tracker() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  const [phase, setPhase] = useState<Phase>("starting");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [thinking, setThinking] = useState(false);
  const [caption, setCaption] = useState<string | null>(null);
  const [speaking, setSpeaking] = useState(false);

  const [mouth, setMouth] = useState(0);
  const [blink, setBlink] = useState(0);
  const [eye, setEye] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const prevGrayRef = useRef<Gray | null>(null);
  const pointsRef = useRef<Pt[]>([]);
  const anchorsRef = useRef<Pt[]>([]);
  const anchorCenterRef = useRef<Pt>({ x: 0, y: 0 });
  const transformRef = useRef<Transform>({ a: 1, b: 0, tx: 0, ty: 0 });
  const lockedRef = useRef(false);
  const trackRafRef = useRef<number | null>(null);
  const workCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const cropCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastFpsSampleRef = useRef(0);
  const frameCountRef = useRef(0);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const freqDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const mouthTargetRef = useRef(0);
  const mouthSmoothRef = useRef(0);
  const generationRef = useRef(0);

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
        setPhase("ready");
      } catch (e) {
        setPhase("error");
        setErrorMsg(e instanceof Error ? e.message : "camera unavailable");
      }
    })();
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
      if (trackRafRef.current) cancelAnimationFrame(trackRafRef.current);
      if (currentSourceRef.current) {
        try {
          currentSourceRef.current.stop();
        } catch {
          // already ended
        }
      }
      audioCtxRef.current?.close().catch(() => {});
    };
  }, []);

  // Animation loop: blinks, eye darts, audio-driven mouth. Runs for the
  // lifetime of the component, independent of tracking state.
  useEffect(() => {
    let raf = 0;
    let nextBlinkAt = performance.now() + 1800;
    let blinkStart = 0;
    let nextEyeMoveAt = performance.now() + 1200;
    let eyeTarget = { x: 0, y: 0 };
    let eyeCurrent = { x: 0, y: 0 };

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

      if (currentSourceRef.current && analyserRef.current && freqDataRef.current) {
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

  const captureGray = useCallback((): Gray | null => {
    const v = videoRef.current;
    if (!v || v.videoWidth === 0) return null;
    const aspect = v.videoHeight / v.videoWidth;
    const w = PROC_W;
    const h = Math.max(1, Math.round(PROC_W * aspect));
    let canvas = workCanvasRef.current;
    if (!canvas) {
      canvas = document.createElement("canvas");
      workCanvasRef.current = canvas;
    }
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(v, 0, 0, w, h);
    return toGray(ctx.getImageData(0, 0, w, h));
  }, []);

  // Tap in element space → source-video space, accounting for object-cover.
  const tapToSource = useCallback(
    (
      clientX: number,
      clientY: number
    ): { vx: number; vy: number; vw: number; vh: number } | null => {
      const v = videoRef.current;
      if (!v || !v.videoWidth) return null;
      const rect = v.getBoundingClientRect();
      const vw = v.videoWidth;
      const vh = v.videoHeight;
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
      const vx = offX + (ex / rect.width) * dispW;
      const vy = offY + (ey / rect.height) * dispH;
      return { vx, vy, vw, vh };
    },
    []
  );

  const captureCrop = useCallback(
    (clientX: number, clientY: number): string | null => {
      const v = videoRef.current;
      if (!v) return null;
      const src = tapToSource(clientX, clientY);
      if (!src) return null;
      const { vx, vy, vw, vh } = src;
      const size = Math.min(vw, vh) * CROP_FRACTION;
      const sx = Math.max(0, Math.min(vw - size, vx - size / 2));
      const sy = Math.max(0, Math.min(vh - size, vy - size / 2));
      let canvas = cropCanvasRef.current;
      if (!canvas) {
        canvas = document.createElement("canvas");
        cropCanvasRef.current = canvas;
      }
      const target = 384;
      canvas.width = target;
      canvas.height = target;
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;
      ctx.drawImage(v, sx, sy, size, size, 0, 0, target, target);
      return canvas.toDataURL("image/jpeg", 0.82);
    },
    [tapToSource]
  );

  const updateOverlay = useCallback(
    (t: Transform, procW: number, procH: number) => {
      const overlay = overlayRef.current;
      const video = videoRef.current;
      if (!overlay || !video) return;
      const rect = video.getBoundingClientRect();
      const scaleX = rect.width / procW;
      const scaleY = rect.height / procH;
      const center = applyTransform(t, anchorCenterRef.current);
      const angle = Math.atan2(t.b, t.a);
      const scale = Math.sqrt(t.a * t.a + t.b * t.b);
      overlay.style.transform = `translate(${center.x * scaleX}px, ${center.y * scaleY}px) translate(-50%, -50%) rotate(${angle}rad) scale(${scale})`;
    },
    []
  );

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

  const speak = useCallback(
    async (cropDataUrl: string) => {
      const ctx = ensureAudioCtx();
      if (!ctx || !analyserRef.current) {
        setErrorMsg("audio unavailable");
        return;
      }

      const gen = ++generationRef.current;
      stopCurrentAudio();
      setThinking(true);
      setCaption(null);
      setErrorMsg(null);

      try {
        const { line, audioDataUrl } = await generateLine(cropDataUrl);
        if (gen !== generationRef.current) return;

        const resp = await fetch(audioDataUrl);
        const buf = await resp.arrayBuffer();
        const audioBuf = await ctx.decodeAudioData(buf);
        if (gen !== generationRef.current) return;

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
        setCaption(line);
        setSpeaking(true);
        source.start();
      } catch (e) {
        if (gen !== generationRef.current) return;
        setErrorMsg(e instanceof Error ? e.message : "line failed");
      } finally {
        if (gen === generationRef.current) setThinking(false);
      }
    },
    [ensureAudioCtx, stopCurrentAudio]
  );

  const loop = useCallback(() => {
    const tick = (now: number) => {
      if (!lockedRef.current) return;
      const prev = prevGrayRef.current;
      const pts = pointsRef.current;
      const anchors = anchorsRef.current;
      if (!prev || pts.length === 0) {
        trackRafRef.current = requestAnimationFrame(tick);
        return;
      }

      const curr = captureGray();
      if (!curr) {
        trackRafRef.current = requestAnimationFrame(tick);
        return;
      }

      const tracked = trackLK(prev, curr, pts);
      const goodCurr: Pt[] = [];
      const goodAnchors: Pt[] = [];
      for (let i = 0; i < tracked.length; i++) {
        if (tracked[i].found) {
          goodCurr.push({ x: tracked[i].x, y: tracked[i].y });
          goodAnchors.push(anchors[i]);
        }
      }

      let t = estimateSimilarity(goodAnchors, goodCurr);
      if (t && goodAnchors.length >= 6) {
        const filtered = filterOutliers(goodAnchors, goodCurr, t);
        if (filtered.src.length >= Math.max(4, (goodAnchors.length * 0.5) | 0)) {
          const refined = estimateSimilarity(filtered.src, filtered.dst);
          if (refined) {
            t = refined;
            pointsRef.current = filtered.dst;
            anchorsRef.current = filtered.src;
          } else {
            pointsRef.current = goodCurr;
            anchorsRef.current = goodAnchors;
          }
        } else {
          pointsRef.current = goodCurr;
          anchorsRef.current = goodAnchors;
        }
      } else {
        pointsRef.current = goodCurr;
        anchorsRef.current = goodAnchors;
      }

      if (t) {
        transformRef.current = t;
        updateOverlay(t, curr.width, curr.height);
      }
      prevGrayRef.current = curr;

      if (pointsRef.current.length < RESEED_THRESHOLD && t) {
        const cx =
          pointsRef.current.reduce((s, p) => s + p.x, 0) /
          Math.max(1, pointsRef.current.length);
        const cy =
          pointsRef.current.reduce((s, p) => s + p.y, 0) /
          Math.max(1, pointsRef.current.length);
        const size = Math.min(curr.width, curr.height) * ROI_FRACTION;
        const newCorners = detectCorners(curr, {
          maxCorners: 30,
          minDistance: 8,
          roi: { x: cx - size / 2, y: cy - size / 2, w: size, h: size },
        });
        if (newCorners.length >= MIN_POINTS) {
          const inv = invertTransform(t);
          pointsRef.current = newCorners.map((c) => ({ x: c.x, y: c.y }));
          anchorsRef.current = newCorners.map((c) =>
            applyTransform(inv, { x: c.x, y: c.y })
          );
        }
      }

      frameCountRef.current++;
      if (now - lastFpsSampleRef.current > 500) {
        setFps(
          Math.round(
            (frameCountRef.current * 1000) / (now - lastFpsSampleRef.current)
          )
        );
        frameCountRef.current = 0;
        lastFpsSampleRef.current = now;
      }

      if (pointsRef.current.length < MIN_POINTS) {
        lockedRef.current = false;
        setPhase("ready");
        return;
      }

      trackRafRef.current = requestAnimationFrame(tick);
    };
    trackRafRef.current = requestAnimationFrame(tick);
  }, [captureGray, updateOverlay]);

  const handleTap = useCallback(
    (e: React.PointerEvent) => {
      if (phase !== "ready" && phase !== "locked") return;
      const v = videoRef.current;
      if (!v) return;
      const rect = v.getBoundingClientRect();
      const nx = (e.clientX - rect.left) / rect.width;
      const ny = (e.clientY - rect.top) / rect.height;

      // Unlock audio inside the gesture.
      ensureAudioCtx();

      const gray = captureGray();
      if (!gray) return;

      const cx = nx * gray.width;
      const cy = ny * gray.height;
      const size = Math.min(gray.width, gray.height) * ROI_FRACTION;
      const corners = detectCorners(gray, {
        maxCorners: 30,
        minDistance: 8,
        roi: { x: cx - size / 2, y: cy - size / 2, w: size, h: size },
      });

      if (corners.length < MIN_POINTS) {
        setErrorMsg(
          `need more texture here (${corners.length} corners). try a busier spot.`
        );
        setTimeout(() => setErrorMsg(null), 1800);
        return;
      }

      const pts = corners.map((c) => ({ x: c.x, y: c.y }));
      pointsRef.current = pts;
      anchorsRef.current = pts.map((p) => ({ ...p }));
      const acx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
      const acy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
      anchorCenterRef.current = { x: acx, y: acy };
      prevGrayRef.current = gray;
      transformRef.current = { a: 1, b: 0, tx: 0, ty: 0 };
      updateOverlay(transformRef.current, gray.width, gray.height);
      lockedRef.current = true;
      setPhase("locked");
      lastFpsSampleRef.current = performance.now();
      frameCountRef.current = 0;
      if (trackRafRef.current) cancelAnimationFrame(trackRafRef.current);
      loop();

      const crop = captureCrop(e.clientX, e.clientY);
      if (crop) speak(crop);
    },
    [captureCrop, captureGray, ensureAudioCtx, loop, phase, speak, updateOverlay]
  );

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
      ? "warming up"
      : phase === "error"
        ? (errorMsg ?? "camera error")
        : phase === "ready"
          ? (errorMsg ?? "tap anything")
          : thinking
            ? "thinking"
            : speaking
              ? "speaking"
              : `tracking · ${fps} fps`;

  return (
    <div className="fixed inset-0 overflow-hidden touch-none select-none bg-gradient-to-br from-[#1a0f2e] via-[#2a1540] to-[#3d1a4d]">
      <video
        ref={videoRef}
        playsInline
        muted
        className="absolute inset-0 h-full w-full object-cover"
        onPointerDown={handleTap}
      />
      {phase === "locked" && (
        <div
          ref={overlayRef}
          className="pointer-events-none absolute left-0 top-0 will-change-transform"
          style={{ transformOrigin: "0 0" }}
        >
          <Face mouth={mouth} blink={blink} eyeX={eye.x} eyeY={eye.y} />
        </div>
      )}

      {/* top: wordmark + status pill */}
      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-center justify-between px-5 pt-[max(env(safe-area-inset-top),18px)]">
        <div className="flex items-center gap-2 rounded-full bg-white/15 px-3.5 py-1.5 shadow-[0_8px_24px_-12px_rgba(0,0,0,0.6)] ring-1 ring-white/25 backdrop-blur-xl">
          <span className="h-1.5 w-1.5 rounded-full bg-[#ff89be] shadow-[0_0_0_3px_rgba(255,137,190,0.28)]" />
          <span className="serif-italic text-[17px] font-medium leading-none text-white/95">
            mirror
          </span>
        </div>
        <div
          className={
            "flex items-center gap-2 rounded-full px-3.5 py-1.5 shadow-[0_8px_24px_-12px_rgba(0,0,0,0.6)] ring-1 backdrop-blur-xl transition " +
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
      </div>

      {/* ready-state: gentle centered hint */}
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

      {/* thinking: small floating indicator */}
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

      {/* caption bubble */}
      {caption && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 px-5 pb-[max(env(safe-area-inset-bottom),28px)] pt-10">
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
    </div>
  );
}
