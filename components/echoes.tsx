"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { interpretScene, leavePostcard, type EchoResult } from "@/app/actions";
import { VIDEOS, type SceneVideo, type Vibe } from "@/data/videos";
import type { Postcard } from "@/data/postcards";

type Phase =
  | "feed"
  | "paused"
  | "selected"
  | "recording"
  | "thinking"
  | "echo"
  | "leaving"
  | "sent";

type Rect = { x: number; y: number; w: number; h: number };

const HOLD_THRESHOLD_MS = 180;

export default function Echoes() {
  const [videoIndex, setVideoIndex] = useState(0);
  const [phase, setPhase] = useState<Phase>("feed");
  const [rect, setRect] = useState<Rect | null>(null);
  const [result, setResult] = useState<EchoResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const video = VIDEOS[videoIndex];

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const dragOrigin = useRef<{ x: number; y: number } | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const holdTimerRef = useRef<number | null>(null);
  const isHoldingRef = useRef(false);

  // Preload mic permission lazily on first hold (not on load — less creepy)
  const ensureMic = useCallback(async () => {
    if (streamRef.current) return streamRef.current;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      return stream;
    } catch {
      return null;
    }
  }, []);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      if (holdTimerRef.current) clearTimeout(holdTimerRef.current);
    };
  }, []);

  // Reset per-video
  useEffect(() => {
    setPhase("feed");
    setRect(null);
    setResult(null);
    setError(null);
    videoRef.current?.play().catch(() => {});
  }, [videoIndex]);

  const onStagePointerDown = useCallback(
    (e: React.PointerEvent) => {
      const v = videoRef.current;
      const stage = stageRef.current;
      if (!v || !stage) return;

      // FEED → PAUSED on tap
      if (phase === "feed") {
        v.pause();
        setPhase("paused");
        return;
      }

      if (phase === "paused" || phase === "selected") {
        const bb = stage.getBoundingClientRect();
        const x = ((e.clientX - bb.left) / bb.width) * 100;
        const y = ((e.clientY - bb.top) / bb.height) * 100;
        dragOrigin.current = { x, y };
        setRect({ x, y, w: 0, h: 0 });
        setPhase("selected");
        (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
      }
    },
    [phase]
  );

  const onStagePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragOrigin.current) return;
      const stage = stageRef.current;
      if (!stage) return;
      const bb = stage.getBoundingClientRect();
      const x = ((e.clientX - bb.left) / bb.width) * 100;
      const y = ((e.clientY - bb.top) / bb.height) * 100;
      const o = dragOrigin.current;
      const nx = Math.min(o.x, x);
      const ny = Math.min(o.y, y);
      const w = Math.abs(x - o.x);
      const h = Math.abs(y - o.y);
      setRect({ x: nx, y: ny, w, h });
    },
    []
  );

  const onStagePointerUp = useCallback(
    (_e: React.PointerEvent) => {
      if (!dragOrigin.current) return;
      dragOrigin.current = null;
      setRect((r) => {
        if (!r) return r;
        // Too small? Drop it.
        if (r.w < 6 || r.h < 6) {
          setPhase("paused");
          return null;
        }
        return r;
      });
    },
    []
  );

  const resumeFeed = useCallback(() => {
    setRect(null);
    setResult(null);
    setError(null);
    setPhase("feed");
    videoRef.current?.play().catch(() => {});
  }, []);

  const nextVideo = useCallback(() => {
    setVideoIndex((i) => (i + 1) % VIDEOS.length);
  }, []);

  const prevVideo = useCallback(() => {
    setVideoIndex((i) => (i - 1 + VIDEOS.length) % VIDEOS.length);
  }, []);

  const captureCrop = useCallback((): string | null => {
    const v = videoRef.current;
    if (!v || !rect) return null;
    const vw = v.videoWidth || 1080;
    const vh = v.videoHeight || 1920;
    const sx = Math.max(0, Math.round((rect.x / 100) * vw));
    const sy = Math.max(0, Math.round((rect.y / 100) * vh));
    const sw = Math.max(1, Math.round((rect.w / 100) * vw));
    const sh = Math.max(1, Math.round((rect.h / 100) * vh));
    const canvas = document.createElement("canvas");
    const maxDim = 768;
    const scale = Math.min(1, maxDim / Math.max(sw, sh));
    canvas.width = Math.max(1, Math.round(sw * scale));
    canvas.height = Math.max(1, Math.round(sh * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(v, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  }, [rect]);

  const startRecording = useCallback(async () => {
    const stream = await ensureMic();
    if (!stream) {
      setError("Enable microphone to ask aloud, or skip to continue.");
      return false;
    }
    const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/mp4")
        ? "audio/mp4"
        : "";
    const recorder = mime
      ? new MediaRecorder(stream, { mimeType: mime })
      : new MediaRecorder(stream);
    audioChunksRef.current = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunksRef.current.push(e.data);
    };
    recorder.start();
    recorderRef.current = recorder;
    return true;
  }, [ensureMic]);

  const stopRecording = useCallback(async (): Promise<Blob | null> => {
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
    async (audio: Blob | null) => {
      const frame = captureCrop();
      if (!frame) {
        setError("Couldn't capture that crop — try drawing again.");
        setPhase("paused");
        return;
      }
      setPhase("thinking");
      try {
        const fd = new FormData();
        fd.append("frame", frame);
        if (audio) fd.append("audio", audio, "voice.webm");
        fd.append("videoTitle", video.title);
        fd.append("videoLocation", video.location);
        const res = await interpretScene(fd);
        if (!res.ok) throw new Error(res.error);
        setResult(res);
        setPhase("echo");
      } catch (e) {
        setError(e instanceof Error ? e.message : "Something went quiet.");
        setPhase("selected");
      }
    },
    [captureCrop, video.title, video.location]
  );

  const onVoicePressStart = useCallback(
    async (e: React.PointerEvent) => {
      if (phase !== "selected") return;
      e.preventDefault();
      (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
      isHoldingRef.current = false;
      holdTimerRef.current = window.setTimeout(async () => {
        const started = await startRecording();
        if (!started) return;
        isHoldingRef.current = true;
        setPhase("recording");
        navigator.vibrate?.(10);
      }, HOLD_THRESHOLD_MS);
    },
    [phase, startRecording]
  );

  const onVoicePressEnd = useCallback(
    async (e: React.PointerEvent) => {
      (e.currentTarget as HTMLElement).releasePointerCapture?.(e.pointerId);
      if (holdTimerRef.current) {
        clearTimeout(holdTimerRef.current);
        holdTimerRef.current = null;
      }
      if (!isHoldingRef.current) {
        // Tap-only: submit without voice
        await submit(null);
        return;
      }
      const audio = await stopRecording();
      await submit(audio);
    },
    [stopRecording, submit]
  );

  return (
    <main
      className="relative h-[100svh] w-full overflow-hidden bg-black"
      ref={stageRef}
    >
      <Stage
        video={video}
        videoRef={videoRef}
        phase={phase}
        rect={rect}
        onPointerDown={onStagePointerDown}
        onPointerMove={onStagePointerMove}
        onPointerUp={onStagePointerUp}
      />

      <TopBar
        video={video}
        phase={phase}
        onBack={resumeFeed}
        onPrev={prevVideo}
        onNext={nextVideo}
      />

      {(phase === "feed" || phase === "paused" || phase === "selected") && (
        <BottomHint phase={phase} />
      )}

      {phase === "selected" && (
        <VoiceBar
          onPointerDown={onVoicePressStart}
          onPointerUp={onVoicePressEnd}
          onPointerCancel={onVoicePressEnd}
        />
      )}

      {phase === "recording" && <Recording />}

      {phase === "thinking" && <Thinking />}

      {phase === "echo" && result?.ok && (
        <EchoView
          result={result}
          rect={rect}
          onLeave={() => setPhase("leaving")}
          onAgain={resumeFeed}
        />
      )}

      {phase === "leaving" && result?.ok && (
        <LeavePostcard
          vibe={result.vibe}
          location={result.location}
          onCancel={() => setPhase("echo")}
          onSent={() => setPhase("sent")}
        />
      )}

      {phase === "sent" && <Sent onDone={resumeFeed} />}

      {error && (
        <div className="pointer-events-none absolute inset-x-0 bottom-28 z-40 flex justify-center px-6">
          <div className="pointer-events-auto rounded-full border border-white/15 bg-black/70 px-4 py-2 text-[12px] text-white/80 backdrop-blur-md rise">
            {error}{" "}
            <button
              onClick={() => setError(null)}
              className="ml-2 text-white/60 underline underline-offset-2"
            >
              dismiss
            </button>
          </div>
        </div>
      )}
    </main>
  );
}

/* ---------- Stage (video + selection) ---------- */

function Stage({
  video,
  videoRef,
  phase,
  rect,
  onPointerDown,
  onPointerMove,
  onPointerUp,
}: {
  video: SceneVideo;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  phase: Phase;
  rect: Rect | null;
  onPointerDown: (e: React.PointerEvent) => void;
  onPointerMove: (e: React.PointerEvent) => void;
  onPointerUp: (e: React.PointerEvent) => void;
}) {
  const [hasVideo, setHasVideo] = useState(true);

  useEffect(() => {
    setHasVideo(true);
  }, [video.src]);

  const dim = phase !== "feed";

  return (
    <div
      className="absolute inset-0 touch-none"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      {hasVideo ? (
        <video
          ref={videoRef}
          src={video.src}
          autoPlay
          loop
          muted
          playsInline
          preload="auto"
          onError={() => setHasVideo(false)}
          className="absolute inset-0 h-full w-full object-cover"
        />
      ) : (
        <div className="absolute inset-0 ambient-gradient ambient" />
      )}

      {/* Darken + vignette when interacting */}
      <div
        className={`pointer-events-none absolute inset-0 transition-opacity duration-700 ${
          dim ? "opacity-100" : "opacity-60"
        }`}
        style={{
          background:
            "radial-gradient(110% 70% at 50% 30%, rgba(0,0,0,0.2) 0%, rgba(0,0,0,0.55) 60%, rgba(0,0,0,0.85) 100%)",
        }}
      />

      {rect && (rect.w > 0 || rect.h > 0) && (
        <SelectionRect rect={rect} />
      )}
    </div>
  );
}

function SelectionRect({ rect }: { rect: Rect }) {
  return (
    <div
      className="pointer-events-none absolute"
      style={{
        left: `${rect.x}%`,
        top: `${rect.y}%`,
        width: `${rect.w}%`,
        height: `${rect.h}%`,
        boxShadow:
          "0 0 0 9999px rgba(0,0,0,0.55), inset 0 0 0 1px rgba(255,255,255,0.9)",
        borderRadius: 4,
        transition: "box-shadow 200ms var(--ease-out-expo)",
      }}
    >
      {[
        ["top-0 left-0", "-translate-x-1/2 -translate-y-1/2"],
        ["top-0 right-0", "translate-x-1/2 -translate-y-1/2"],
        ["bottom-0 left-0", "-translate-x-1/2 translate-y-1/2"],
        ["bottom-0 right-0", "translate-x-1/2 translate-y-1/2"],
      ].map(([pos, tr], i) => (
        <span
          key={i}
          className={`absolute h-2 w-2 rounded-full bg-white ${pos} ${tr}`}
        />
      ))}
    </div>
  );
}

/* ---------- Top bar ---------- */

function TopBar({
  video,
  phase,
  onBack,
  onPrev,
  onNext,
}: {
  video: SceneVideo;
  phase: Phase;
  onBack: () => void;
  onPrev: () => void;
  onNext: () => void;
}) {
  const showBack =
    phase === "paused" ||
    phase === "selected" ||
    phase === "echo" ||
    phase === "leaving";

  return (
    <header className="pointer-events-none absolute inset-x-0 top-0 z-20 flex items-center justify-between px-5 pt-[max(env(safe-area-inset-top),14px)]">
      <div className="pointer-events-auto flex items-center gap-2.5">
        {showBack ? (
          <button
            onClick={onBack}
            aria-label="Back"
            className="grid h-8 w-8 place-items-center rounded-full border border-white/15 bg-white/5 text-white/85 backdrop-blur-md transition hover:bg-white/10"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M15 18l-6-6 6-6" />
            </svg>
          </button>
        ) : (
          <div className="flex items-center gap-2">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-white" />
            <span className="font-mono text-[11px] tracking-[0.22em] text-white/85">
              ECHOES
            </span>
          </div>
        )}
      </div>

      <div className="pointer-events-auto flex items-center gap-1.5">
        {phase === "feed" && (
          <>
            <div className="font-serif text-[13px] italic text-white/80">
              {video.location}
            </div>
            <span className="mx-2 h-3 w-px bg-white/20" />
            <IconBtn onClick={onPrev} label="Previous">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M15 18l-6-6 6-6" />
              </svg>
            </IconBtn>
            <IconBtn onClick={onNext} label="Next">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 6l6 6-6 6" />
              </svg>
            </IconBtn>
          </>
        )}
      </div>
    </header>
  );
}

function IconBtn({
  onClick,
  label,
  children,
}: {
  onClick: () => void;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      className="grid h-8 w-8 place-items-center rounded-full border border-white/15 bg-white/5 text-white/80 backdrop-blur-md transition hover:bg-white/10 hover:text-white"
    >
      {children}
    </button>
  );
}

/* ---------- Bottom hint ---------- */

function BottomHint({ phase }: { phase: Phase }) {
  const hint =
    phase === "feed"
      ? "tap to pause"
      : phase === "paused"
        ? "drag over what moves you"
        : "hold to ask · or tap to listen";
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 flex flex-col items-center pb-[max(env(safe-area-inset-bottom),22px)]">
      <div
        key={phase}
        className="fade-in font-serif text-[14px] italic tracking-[-0.005em] text-white/80"
      >
        {hint}
      </div>
    </div>
  );
}

/* ---------- Voice bar ---------- */

function VoiceBar({
  onPointerDown,
  onPointerUp,
  onPointerCancel,
}: {
  onPointerDown: (e: React.PointerEvent) => void;
  onPointerUp: (e: React.PointerEvent) => void;
  onPointerCancel: (e: React.PointerEvent) => void;
}) {
  return (
    <div className="absolute inset-x-0 bottom-0 z-30 flex flex-col items-center gap-5 pb-[max(env(safe-area-inset-bottom),28px)]">
      <div className="font-serif text-[14px] italic text-white/80 rise">
        hold to ask · release when done
      </div>
      <button
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerCancel}
        onContextMenu={(e) => e.preventDefault()}
        aria-label="Hold to speak"
        className="relative grid h-[84px] w-[84px] place-items-center rounded-full border border-white/30 bg-white/5 backdrop-blur-md transition active:scale-95"
      >
        <span className="absolute inset-2 rounded-full bg-white/90 transition" />
        <svg className="relative text-black" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <rect x="9" y="3" width="6" height="12" rx="3" />
          <path d="M5 11a7 7 0 0 0 14 0" />
          <path d="M12 18v3" />
        </svg>
      </button>
    </div>
  );
}

/* ---------- Recording overlay ---------- */

function Recording() {
  const [ms, setMs] = useState(0);
  useEffect(() => {
    const start = Date.now();
    const id = window.setInterval(() => setMs(Date.now() - start), 80);
    return () => clearInterval(id);
  }, []);
  const s = Math.floor(ms / 1000);
  const t = `0:${String(s).padStart(2, "0")}`;

  return (
    <div className="absolute inset-x-0 bottom-0 z-30 flex flex-col items-center gap-5 pb-[max(env(safe-area-inset-bottom),28px)]">
      <div className="flex items-center gap-2 rounded-full border border-white/15 bg-black/50 px-3 py-1 backdrop-blur-md rise">
        <span className="relative inline-block h-2 w-2">
          <span className="absolute inset-0 rounded-full bg-red-500 breathe" />
          <span className="pulse-ring" />
        </span>
        <span className="font-mono text-[11px] tracking-[0.08em] text-white/90">
          {t}
        </span>
      </div>
      <div className="relative grid h-[84px] w-[84px] place-items-center rounded-full border border-red-400/60 bg-red-500/10">
        <span className="absolute inset-2 rounded-full bg-red-500 breathe" />
      </div>
    </div>
  );
}

/* ---------- Thinking overlay ---------- */

function Thinking() {
  const phrases = useMemo(
    () => [
      "listening to the wind…",
      "asking the place…",
      "remembering the road…",
      "reading the light…",
    ],
    []
  );
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = window.setInterval(
      () => setI((v) => (v + 1) % phrases.length),
      1400
    );
    return () => clearInterval(id);
  }, [phrases.length]);

  return (
    <div className="absolute inset-0 z-30 grid place-items-center bg-black/55 backdrop-blur-md">
      <div className="flex flex-col items-center gap-5">
        <div className="relative h-[2px] w-44 overflow-hidden rounded-full bg-white/10">
          <div className="absolute inset-0 shimmer" />
        </div>
        <div
          key={i}
          className="fade-in font-serif text-[15px] italic tracking-[-0.005em] text-white/80"
        >
          {phrases[i]}
        </div>
      </div>
    </div>
  );
}

/* ---------- Echo reveal ---------- */

function EchoView({
  result,
  rect,
  onLeave,
  onAgain,
}: {
  result: Extract<EchoResult, { ok: true }>;
  rect: Rect | null;
  onLeave: () => void;
  onAgain: () => void;
}) {
  return (
    <div className="absolute inset-0 z-40 flex flex-col bg-black/75 backdrop-blur-lg">
      {/* Retain the selection as a dim "postage stamp" at the top */}
      <div className="pointer-events-none relative flex-1">
        {rect && (
          <div
            className="absolute rounded-sm ring-1 ring-white/30"
            style={{
              left: `${rect.x}%`,
              top: `${rect.y}%`,
              width: `${rect.w}%`,
              height: `${rect.h}%`,
            }}
          />
        )}
      </div>

      <div className="relative z-10 px-6 pb-[max(env(safe-area-inset-bottom),24px)]">
        <div className="mx-auto flex max-w-md flex-col gap-6 rise-slow">
          <div className="flex flex-col items-center gap-1 text-center">
            <div className="font-mono text-[10.5px] tracking-[0.22em] text-white/45">
              FROM
            </div>
            <div className="font-serif text-[16px] italic text-white/80">
              {result.location}
            </div>
          </div>

          <p
            data-selectable
            className="ink-in text-balance text-center font-serif text-[22px] leading-[1.35] tracking-[-0.01em] text-white sm:text-[24px]"
          >
            {result.reply}
          </p>

          <div className="flex flex-col gap-2">
            <div className="text-center font-mono text-[10.5px] tracking-[0.22em] text-white/40">
              {formatCount(result.totalHere)} OTHERS FELT THE SAME
            </div>
            <PostcardStack postcards={result.postcards} />
          </div>

          <div className="flex items-center justify-center gap-2">
            <button
              onClick={onLeave}
              className="rounded-full bg-white px-5 py-2.5 text-[13px] font-medium tracking-[-0.005em] text-black transition hover:bg-white/90 active:scale-[0.98]"
            >
              Leave your own
            </button>
            <button
              onClick={onAgain}
              className="rounded-full border border-white/15 bg-white/5 px-5 py-2.5 text-[13px] font-medium tracking-[-0.005em] text-white/90 transition hover:bg-white/10"
            >
              Keep watching
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatCount(n: number): string {
  if (n < 1000) return String(n);
  if (n < 10000) return `${(n / 1000).toFixed(1)}K`;
  return `${Math.round(n / 1000)}K`;
}

/* ---------- Postcard stack ---------- */

function PostcardStack({ postcards }: { postcards: Postcard[] }) {
  const [index, setIndex] = useState(0);
  const visible = postcards.slice(index, index + 3);
  const hasMore = index < postcards.length - 1;
  const next = () => setIndex((i) => Math.min(postcards.length - 1, i + 1));

  if (!postcards.length) {
    return (
      <div className="rounded-xl border border-white/10 bg-white/[0.04] p-4 text-center font-serif text-[14px] italic text-white/55">
        You are among the first to stand here.
      </div>
    );
  }

  return (
    <div className="relative h-[184px]">
      {visible.map((p, i) => {
        const depth = i; // 0 is the top card
        const scale = 1 - depth * 0.04;
        const ty = depth * 10;
        const opacity = 1 - depth * 0.35;
        const z = 10 - depth;
        return (
          <button
            key={p.id}
            onClick={() => depth === 0 && hasMore && next()}
            style={{
              transform: `translate(-50%, ${ty}px) scale(${scale})`,
              opacity,
              zIndex: z,
            }}
            className="absolute left-1/2 top-0 w-[92%] max-w-sm rounded-[14px] border border-white/10 bg-[rgba(18,18,22,0.85)] px-4 py-4 text-left backdrop-blur-xl transition-all duration-300 ease-[var(--ease-out-expo)]"
          >
            <p
              data-selectable
              className="font-serif text-[14.5px] italic leading-[1.45] tracking-[-0.005em] text-white/90"
            >
              “{p.text}”
            </p>
            {p.signedBy && (
              <p className="mt-2 font-mono text-[10px] tracking-[0.14em] text-white/40">
                {p.signedBy.toUpperCase()}
              </p>
            )}
          </button>
        );
      })}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 text-center">
        <div className="font-mono text-[10px] tracking-[0.22em] text-white/35">
          {Math.min(index + 1, postcards.length)} / {postcards.length}
          {hasMore ? "  ·  tap to read next" : "  ·  end"}
        </div>
      </div>
    </div>
  );
}

/* ---------- Leave postcard form ---------- */

function LeavePostcard({
  vibe,
  location,
  onCancel,
  onSent,
}: {
  vibe: Vibe;
  location: string;
  onCancel: () => void;
  onSent: () => void;
}) {
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const [localErr, setLocalErr] = useState<string | null>(null);

  const send = async () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    setSending(true);
    const res = await leavePostcard({ vibe, text: trimmed });
    setSending(false);
    if (res.ok) onSent();
    else setLocalErr(res.error);
  };

  return (
    <div className="absolute inset-0 z-50 flex flex-col bg-black/85 backdrop-blur-xl">
      <div className="flex-1" />
      <div className="px-6 pb-[max(env(safe-area-inset-bottom),24px)]">
        <div className="mx-auto flex max-w-md flex-col gap-4 rise">
          <div className="text-center">
            <div className="font-mono text-[10.5px] tracking-[0.22em] text-white/40">
              LEAVE A POSTCARD AT
            </div>
            <div className="mt-1 font-serif text-[17px] italic text-white">
              {location}
            </div>
          </div>

          <div className="rounded-[16px] border border-white/10 bg-[rgba(255,255,255,0.03)] p-3">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value.slice(0, 400))}
              placeholder="Something you'd say to the next person standing here…"
              rows={5}
              autoFocus
              data-selectable
              className="w-full resize-none bg-transparent font-serif text-[15.5px] italic leading-[1.5] tracking-[-0.005em] text-white placeholder:text-white/30 focus:outline-none"
            />
            <div className="flex items-center justify-between pt-1">
              <span className="font-mono text-[10px] tracking-[0.14em] text-white/30">
                ANONYMOUS · {text.length}/400
              </span>
            </div>
          </div>

          {localErr && (
            <div className="text-center text-[12px] text-red-300/80">
              {localErr}
            </div>
          )}

          <div className="flex items-center justify-center gap-2">
            <button
              onClick={send}
              disabled={sending || !text.trim()}
              className="rounded-full bg-white px-5 py-2.5 text-[13px] font-medium tracking-[-0.005em] text-black transition hover:bg-white/90 active:scale-[0.98] disabled:opacity-50"
            >
              {sending ? "sending…" : "Leave it"}
            </button>
            <button
              onClick={onCancel}
              className="rounded-full border border-white/15 bg-white/5 px-5 py-2.5 text-[13px] font-medium tracking-[-0.005em] text-white/90 transition hover:bg-white/10"
            >
              Not now
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------- Sent ---------- */

function Sent({ onDone }: { onDone: () => void }) {
  useEffect(() => {
    const id = window.setTimeout(onDone, 1800);
    return () => clearTimeout(id);
  }, [onDone]);
  return (
    <div className="absolute inset-0 z-50 grid place-items-center bg-black/85 backdrop-blur-xl">
      <div className="flex flex-col items-center gap-3 rise">
        <div className="grid h-10 w-10 place-items-center rounded-full border border-white/20 text-white/80">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M5 12l5 5L20 7" />
          </svg>
        </div>
        <div className="font-serif text-[16px] italic text-white/85">
          left for the next person.
        </div>
      </div>
    </div>
  );
}
