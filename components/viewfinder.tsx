"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { generateMeme, type MemeResult } from "@/app/actions";
import Reveal from "@/components/reveal";

type Phase = "idle" | "armed" | "working" | "done" | "error";

const HOLD_THRESHOLD_MS = 220;

export default function Viewfinder() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const holdTimerRef = useRef<number | null>(null);
  const isHoldingRef = useRef(false);
  const recordStartRef = useRef<number>(0);

  const [phase, setPhase] = useState<Phase>("idle");
  const [isHolding, setIsHolding] = useState(false);
  const [recMs, setRecMs] = useState(0);
  const [cameraReady, setCameraReady] = useState(false);
  const [facing, setFacing] = useState<"user" | "environment">("environment");
  const [photoDataUrl, setPhotoDataUrl] = useState<string | null>(null);
  const [result, setResult] = useState<MemeResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const startCamera = useCallback(async () => {
    try {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: true,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
      setCameraReady(true);
    } catch (err) {
      console.error(err);
      setErrorMsg("Enable camera and mic access to continue.");
      setPhase("error");
    }
  }, [facing]);

  useEffect(() => {
    startCamera();
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [startCamera]);

  // Recording timer
  useEffect(() => {
    if (!isHolding) return;
    recordStartRef.current = Date.now();
    setRecMs(0);
    const id = window.setInterval(() => {
      setRecMs(Date.now() - recordStartRef.current);
    }, 80);
    return () => clearInterval(id);
  }, [isHolding]);

  const capturePhoto = useCallback((): string | null => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return null;
    const canvas = document.createElement("canvas");
    const maxDim = 1280;
    const scale = Math.min(1, maxDim / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    if (facing === "user") {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.86);
  }, [facing]);

  const startAudioRecording = useCallback(() => {
    const stream = streamRef.current;
    if (!stream) return;
    const audioStream = new MediaStream(stream.getAudioTracks());
    const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/mp4")
        ? "audio/mp4"
        : "";
    const recorder = mime
      ? new MediaRecorder(audioStream, { mimeType: mime })
      : new MediaRecorder(audioStream);
    audioChunksRef.current = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunksRef.current.push(e.data);
    };
    recorder.start();
    recorderRef.current = recorder;
  }, []);

  const stopAudioRecording = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const rec = recorderRef.current;
      if (!rec || rec.state === "inactive") return resolve(null);
      rec.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: rec.mimeType });
        resolve(blob.size ? blob : null);
      };
      rec.stop();
    });
  }, []);

  const submit = useCallback(
    async (photo: string, audio: Blob | null) => {
      setPhase("working");
      try {
        const fd = new FormData();
        fd.append("photo", photo);
        if (audio) fd.append("audio", audio, "vent.webm");
        const res = await generateMeme(fd);
        if (!res.ok) throw new Error(res.error);
        setResult(res);
        setPhase("done");
      } catch (e) {
        console.error(e);
        setErrorMsg(e instanceof Error ? e.message : "Something went sideways.");
        setPhase("error");
      }
    },
    []
  );

  const onPressStart = useCallback(
    (e: React.PointerEvent) => {
      if (phase !== "idle" || !cameraReady) return;
      e.preventDefault();
      (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
      isHoldingRef.current = false;
      setIsHolding(false);
      setPhase("armed");
      holdTimerRef.current = window.setTimeout(() => {
        isHoldingRef.current = true;
        setIsHolding(true);
        startAudioRecording();
      }, HOLD_THRESHOLD_MS);
    },
    [phase, cameraReady, startAudioRecording]
  );

  const onPressEnd = useCallback(
    async (e: React.PointerEvent) => {
      if (phase !== "armed") return;
      if (holdTimerRef.current) {
        clearTimeout(holdTimerRef.current);
        holdTimerRef.current = null;
      }
      const wasHolding = isHoldingRef.current;
      setIsHolding(false);
      (e.currentTarget as HTMLElement).releasePointerCapture?.(e.pointerId);

      const photo = capturePhoto();
      if (!photo) {
        setPhase("idle");
        return;
      }
      setPhotoDataUrl(photo);

      const audio = wasHolding ? await stopAudioRecording() : null;
      await submit(photo, audio);
    },
    [phase, capturePhoto, stopAudioRecording, submit]
  );

  const reset = useCallback(() => {
    setResult(null);
    setPhotoDataUrl(null);
    setErrorMsg(null);
    setPhase("idle");
  }, []);

  const flip = useCallback(() => {
    if (phase !== "idle") return;
    setFacing((f) => (f === "user" ? "environment" : "user"));
  }, [phase]);

  if (phase === "done" && result && photoDataUrl) {
    return <Reveal original={photoDataUrl} meme={result} onReset={reset} />;
  }

  return (
    <div className="relative h-full w-full">
      <video
        ref={videoRef}
        playsInline
        muted
        className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-[600ms] ${
          cameraReady ? "opacity-100" : "opacity-0"
        } ${facing === "user" ? "-scale-x-100" : ""}`}
      />

      {!cameraReady && phase !== "error" && <CameraBooting />}

      {/* Soft vignettes */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(120%_80%_at_50%_0%,rgba(0,0,0,0.35)_0%,transparent_45%),radial-gradient(120%_80%_at_50%_100%,rgba(0,0,0,0.55)_0%,transparent_55%)]" />

      {/* Top chrome */}
      <header className="absolute inset-x-0 top-0 flex items-center justify-between px-5 pt-[calc(env(safe-area-inset-top,0px)+14px)]">
        <div className="flex items-center gap-2">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-white" />
          <span className="font-mono text-[11px] tracking-[0.18em] text-white/85">
            VENT
          </span>
        </div>
        <div className="flex items-center gap-2">
          {isHolding && <RecPill ms={recMs} />}
          <IconButton onClick={flip} disabled={phase !== "idle"} label="Flip camera">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 9h14l-3-3" />
              <path d="M21 15H7l3 3" />
            </svg>
          </IconButton>
        </div>
      </header>

      {/* Bottom controls */}
      <div className="absolute inset-x-0 bottom-0 flex flex-col items-center gap-5 pb-[calc(env(safe-area-inset-bottom,0px)+28px)]">
        <Hint phase={phase} isHolding={isHolding} />
        <ShutterButton
          onPointerDown={onPressStart}
          onPointerUp={onPressEnd}
          onPointerCancel={onPressEnd}
          phase={phase}
          isHolding={isHolding}
        />
      </div>

      {phase === "working" && <WorkingOverlay />}
      {phase === "error" && errorMsg && (
        <ErrorOverlay message={errorMsg} onRetry={reset} />
      )}
    </div>
  );
}

function CameraBooting() {
  return (
    <div className="absolute inset-0 grid place-items-center bg-black">
      <div className="flex flex-col items-center gap-3">
        <div className="h-1 w-1 rounded-full bg-white/70 breathe" />
        <div className="font-mono text-[10px] tracking-[0.22em] text-white/40">
          CONNECTING CAMERA
        </div>
      </div>
    </div>
  );
}

function IconButton({
  onClick,
  disabled,
  label,
  children,
}: {
  onClick: () => void;
  disabled?: boolean;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className="grid h-8 w-8 place-items-center rounded-full border border-white/15 bg-white/5 text-white/80 backdrop-blur-md transition hover:border-white/30 hover:bg-white/10 hover:text-white disabled:opacity-30"
    >
      {children}
    </button>
  );
}

function RecPill({ ms }: { ms: number }) {
  const s = Math.floor(ms / 1000);
  const t = `${String(Math.floor(s / 60)).padStart(1, "0")}:${String(s % 60).padStart(2, "0")}`;
  return (
    <div className="flex items-center gap-2 rounded-full border border-white/15 bg-black/40 px-2.5 py-1 backdrop-blur-md">
      <span className="inline-block h-1.5 w-1.5 rounded-full bg-red-500 pulse-dot" />
      <span className="font-mono text-[11px] tracking-[0.08em] text-white/90">{t}</span>
    </div>
  );
}

function Hint({ phase, isHolding }: { phase: Phase; isHolding: boolean }) {
  const text =
    phase === "working"
      ? "generating…"
      : isHolding
        ? "keep talking — release to finish"
        : phase === "armed"
          ? "hold to vent"
          : "tap to snap · hold to vent";
  return (
    <div className="fade-in h-5 text-[12.5px] font-light tracking-[-0.005em] text-white/70">
      {text}
    </div>
  );
}

function ShutterButton({
  onPointerDown,
  onPointerUp,
  onPointerCancel,
  phase,
  isHolding,
}: {
  onPointerDown: (e: React.PointerEvent) => void;
  onPointerUp: (e: React.PointerEvent) => void;
  onPointerCancel: (e: React.PointerEvent) => void;
  phase: Phase;
  isHolding: boolean;
}) {
  const disabled = phase === "working" || phase === "error";
  return (
    <button
      onPointerDown={onPointerDown}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerCancel}
      onContextMenu={(e) => e.preventDefault()}
      disabled={disabled}
      aria-label="Snap or hold to vent"
      className="relative grid h-[86px] w-[86px] place-items-center touch-none outline-none disabled:opacity-50"
    >
      {/* outer ring */}
      <span
        className={`absolute inset-0 rounded-full border transition-all duration-300 ease-[var(--ease-out-expo)] ${
          isHolding
            ? "scale-[1.08] border-white/85"
            : phase === "armed"
              ? "scale-[0.96] border-white/75"
              : "border-white/70"
        }`}
      />
      {/* subtle halo */}
      {isHolding && (
        <span className="pointer-events-none absolute inset-0 rounded-full shadow-[0_0_0_10px_rgba(239,68,68,0.12)]" />
      )}
      {/* core */}
      <span
        className={`relative rounded-full transition-all duration-200 ease-[var(--ease-out-expo)] ${
          isHolding
            ? "h-9 w-9 bg-red-500 breathe"
            : phase === "armed"
              ? "h-[58px] w-[58px] bg-white"
              : "h-[70px] w-[70px] bg-white"
        }`}
      />
    </button>
  );
}

function WorkingOverlay() {
  return (
    <div className="absolute inset-0 grid place-items-center bg-black/60 backdrop-blur-lg">
      <div className="flex flex-col items-center gap-6 rise">
        <div className="relative h-[2px] w-48 overflow-hidden rounded-full bg-white/10 shimmer-track">
          <div className="h-full w-full bg-white/30" />
        </div>
        <div className="flex flex-col items-center gap-1">
          <div className="text-[14px] font-light tracking-[-0.005em] text-white">
            Making it funny
          </div>
          <div className="font-mono text-[10px] tracking-[0.22em] text-white/45">
            LISTENING · LOOKING · IMAGINING
          </div>
        </div>
      </div>
    </div>
  );
}

function ErrorOverlay({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <div className="absolute inset-0 grid place-items-center bg-black/80 px-10 backdrop-blur-lg">
      <div className="flex max-w-xs flex-col items-center gap-5 text-center rise">
        <div className="h-8 w-8 rounded-full border border-white/20 grid place-items-center">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="text-white/70">
            <path d="M12 8v5" />
            <path d="M12 16.5h.01" />
          </svg>
        </div>
        <div className="text-[13.5px] font-light leading-relaxed text-white/80">
          {message}
        </div>
        <button
          onClick={onRetry}
          className="rounded-full bg-white px-5 py-2 text-[12px] font-medium tracking-[-0.005em] text-black transition hover:bg-white/90"
        >
          Try again
        </button>
      </div>
    </div>
  );
}
