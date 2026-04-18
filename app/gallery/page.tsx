"use client";

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import Link from "next/link";
import {
  cardDisplayName,
  clearSessionCards,
  removeSessionCard,
  useSessionCards,
  type SessionCard,
} from "@/lib/session-cards";
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

// Voice-in → voice-out pipeline that mirrors what the tracker does, but is
// fully independent: its own AudioContext, its own mic + MediaRecorder, its
// own MediaSource-backed <audio> element. Only one card can speak at a
// time (the audio graph is shared), so pressing a new card's mic stops the
// one currently playing. All bytes, mp3 decoding, and analyser teardown
// live inside this page — nothing is shared with the tracker module.

type CardStatus = "idle" | "recording" | "thinking" | "speaking";

export default function GalleryPage() {
  const cards = useSessionCards();

  return (
    <div className="relative min-h-dvh overflow-hidden bg-[#120822] pb-[max(env(safe-area-inset-bottom),24px)] pt-[max(env(safe-area-inset-top),18px)] text-white">
      {/* Ambient light — aurora blobs drifting behind everything. */}
      <div className="aurora-layer" aria-hidden>
        <div className="aurora-blob aurora-a" />
        <div className="aurora-blob aurora-b" />
        <div className="aurora-blob aurora-c" />
        <div className="aurora-grain" />
      </div>
      {/* Vignette so the card column reads centered without a hard frame. */}
      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 z-0"
        style={{
          background:
            "radial-gradient(ellipse at 50% 30%, transparent 0%, rgba(8,4,20,0.35) 65%, rgba(8,4,20,0.7) 100%)",
        }}
      />
      <div className="relative z-10">
        <Header hasCards={cards.length > 0} />
        <main className="mx-auto w-full max-w-[1120px] px-4 pb-12 pt-4 sm:px-6">
          {cards.length === 0 ? (
            <EmptyState />
          ) : (
            <GalleryGrid cards={cards} />
          )}
        </main>
      </div>
    </div>
  );
}

function Header({ hasCards }: { hasCards: boolean }) {
  return (
    <header className="mx-auto grid w-full max-w-[720px] grid-cols-[1fr_auto_1fr] items-center gap-3 px-4 sm:px-6">
      <div className="justify-self-start">
        <Link
          href="/"
          aria-label="back to camera"
          className="group inline-flex items-center gap-1.5 rounded-full bg-white/[0.06] px-3.5 py-2 text-[11.5px] font-medium text-white/85 ring-1 ring-white/15 backdrop-blur-xl transition hover:-translate-y-0.5 hover:bg-white/[0.12] hover:text-white"
        >
          <span aria-hidden className="transition-transform group-hover:-translate-x-0.5">←</span>
          <span>camera</span>
        </Link>
      </div>
      <div className="justify-self-center text-center">
        <h1
          className="serif-italic relative text-[30px] font-medium leading-none text-white sm:text-[38px]"
          style={{
            textShadow:
              "0 0 28px rgba(255,182,214,0.35), 0 2px 0 rgba(0,0,0,0.15)",
          }}
        >
          gallery
        </h1>
        <p className="mt-2 text-[10.5px] uppercase tracking-[0.32em] text-white/45">
          things that spoke to you
        </p>
      </div>
      <div className="justify-self-end">
        {hasCards && (
          <button
            type="button"
            onClick={() => {
              if (window.confirm("Clear every card in the gallery?")) {
                clearSessionCards();
              }
            }}
            className="inline-flex items-center gap-1.5 rounded-full bg-white/[0.06] px-3.5 py-2 text-[11px] font-medium uppercase tracking-[0.18em] text-white/75 ring-1 ring-white/15 backdrop-blur-xl transition hover:-translate-y-0.5 hover:bg-white/[0.12] hover:text-white"
          >
            clear
          </button>
        )}
      </div>
    </header>
  );
}

function EmptyState() {
  return (
    <div
      className="relative mx-auto mt-24 max-w-md overflow-hidden rounded-[32px] bg-white/[0.05] p-10 text-center ring-1 ring-white/15 backdrop-blur-2xl"
      style={{
        boxShadow:
          "0 30px 80px -30px rgba(255,137,190,0.35), inset 0 1px 0 rgba(255,255,255,0.06)",
      }}
    >
      <div
        aria-hidden
        className="pointer-events-none absolute -inset-px rounded-[32px]"
        style={{
          background:
            "radial-gradient(140% 80% at 50% 0%, rgba(255,182,214,0.18), transparent 60%)",
        }}
      />
      <div
        aria-hidden
        className="relative mx-auto mb-6 grid h-16 w-16 place-items-center rounded-[22px] bg-gradient-to-br from-pink-300/50 to-violet-400/40 text-[28px] ring-1 ring-white/20"
        style={{
          animation: "soft-pulse 3.2s ease-in-out infinite",
          boxShadow: "0 18px 42px -12px rgba(255,137,190,0.5)",
        }}
      >
        ✨
      </div>
      <h2 className="serif-italic relative text-[22px] leading-tight text-white/95">
        nothing speaking yet
      </h2>
      <p className="relative mt-3 text-[13px] leading-relaxed text-white/65">
        Point your camera at something small and tap. The first thing it says
        gets pressed like a flower — kept here, still talking, waiting for
        you to reply.
      </p>
      <Link
        href="/"
        className="relative mt-6 inline-flex items-center gap-2 rounded-full bg-gradient-to-br from-pink-300 to-pink-400 px-5 py-2.5 text-[12.5px] font-semibold text-pink-950 ring-1 ring-white/40 transition hover:-translate-y-0.5 hover:from-pink-200 hover:to-pink-300"
        style={{ boxShadow: "0 14px 34px -10px rgba(255,137,190,0.65)" }}
      >
        <span>open the camera</span>
        <span aria-hidden>→</span>
      </Link>
    </div>
  );
}

// --- Grid + shared speak pipeline -----------------------------------------

function GalleryGrid({ cards }: { cards: readonly SessionCard[] }) {
  const [activeCardId, setActiveCardId] = useState<string | null>(null);
  const [status, setStatus] = useState<CardStatus>("idle");
  const [liveReplyByCard, setLiveReplyByCard] = useState<
    Record<string, string>
  >({});
  const [heardByCard, setHeardByCard] = useState<Record<string, string>>({});
  const [shape, setShape] = useState<MouthShape>("X");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Shared audio graph. Reused across cards — only one card speaks at a
  // time, so a single analyser/gain/audio-element is sufficient.
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const audioElRef = useRef<HTMLAudioElement | null>(null);
  const audioSrcNodeRef = useRef<MediaElementAudioSourceNode | null>(null);
  const bufferSrcNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const freqDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const timeDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const lipSyncRef = useRef<LipSyncState>(createLipSyncState());
  const rafRef = useRef<number | null>(null);
  const speakGenRef = useRef(0);

  // Mic / recorder state.
  const micStreamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const turnCounterRef = useRef(0);

  // Per-card persistent conversation history kept inside the component (not
  // in the store, because the store is persistence-only — conversation
  // recency would bloat sessionStorage). Keyed by card.id.
  const historyRef = useRef<Record<string, { role: "user" | "assistant"; content: string }[]>>({});

  const ensureAudioCtx = useCallback((): AudioContext | null => {
    if (typeof window === "undefined") return null;
    let ctx = audioCtxRef.current;
    if (!ctx) {
      const Ctor =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext })
          .webkitAudioContext;
      if (!Ctor) return null;
      ctx = new Ctor();
      audioCtxRef.current = ctx;
    }
    if (ctx.state === "suspended") {
      ctx.resume().catch(() => {
        // The next tap will retry.
      });
    }
    return ctx;
  }, []);

  const ensureAnalyser = useCallback((ctx: AudioContext) => {
    if (!analyserRef.current) {
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
        new ArrayBuffer(analyser.frequencyBinCount)
      );
      timeDataRef.current = new Uint8Array(new ArrayBuffer(analyser.fftSize));
    }
    return analyserRef.current;
  }, []);

  const startLipSyncLoop = useCallback(() => {
    if (rafRef.current != null) return;
    const tick = () => {
      const analyser = analyserRef.current;
      const freq = freqDataRef.current;
      const time = timeDataRef.current;
      if (analyser && freq && time) {
        analyser.getByteFrequencyData(freq);
        analyser.getByteTimeDomainData(time);
        const next = classifyShapeSmooth(lipSyncRef.current, time, freq);
        setShape((prev) => (prev === next ? prev : next));
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  const stopLipSyncLoop = useCallback(() => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setShape("X");
  }, []);

  // Tear down any currently-playing audio, without touching the analyser
  // graph itself — the next play will reuse it.
  const stopCurrentPlayback = useCallback(() => {
    speakGenRef.current++;
    const audioEl = audioElRef.current;
    if (audioEl) {
      try {
        audioEl.pause();
      } catch {
        // already paused
      }
      try {
        audioEl.removeAttribute("src");
        audioEl.load();
      } catch {
        // ignore
      }
    }
    const src = bufferSrcNodeRef.current;
    if (src) {
      try {
        src.stop();
      } catch {
        // already stopped
      }
      try {
        src.disconnect();
      } catch {
        // ignore
      }
      bufferSrcNodeRef.current = null;
    }
    stopLipSyncLoop();
  }, [stopLipSyncLoop]);

  // Full page cleanup on unmount.
  useEffect(() => {
    const audioEl = document.createElement("audio");
    audioEl.setAttribute("playsinline", "");
    audioEl.preload = "auto";
    audioElRef.current = audioEl;
    return () => {
      stopCurrentPlayback();
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      micStreamRef.current?.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
      audioCtxRef.current?.close().catch(() => {});
      audioCtxRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Mobile Safari suspends AudioContexts when the tab goes to the
  // background. On return we proactively resume so the next card tap
  // doesn't hit a silent graph.
  useEffect(() => {
    if (typeof document === "undefined") return;
    const onVisible = () => {
      if (document.visibilityState !== "visible") return;
      const ctx = audioCtxRef.current;
      if (ctx && ctx.state === "suspended") {
        ctx.resume().catch(() => {});
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("pageshow", onVisible);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("pageshow", onVisible);
    };
  }, []);

  // --- Mic ---------------------------------------------------------------

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
      setErrorMsg(e instanceof Error ? e.message : "mic unavailable");
      return null;
    }
  }, []);

  const onMicPress = useCallback(
    async (cardId: string) => {
      setErrorMsg(null);
      // Interrupt any currently-playing card before starting a new capture.
      stopCurrentPlayback();
      const ctx = ensureAudioCtx();
      if (!ctx) {
        setErrorMsg("Web Audio unavailable in this browser");
        return;
      }
      const stream = await openMicStream();
      if (!stream) return;
      const mime =
        [
          "audio/webm;codecs=opus",
          "audio/webm",
          "audio/mp4",
          "",
        ].find((t) => !t || MediaRecorder.isTypeSupported(t)) ?? "";
      const recorder = new MediaRecorder(
        stream,
        mime ? { mimeType: mime } : undefined
      );
      recordedChunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) recordedChunksRef.current.push(e.data);
      };
      recorderRef.current = recorder;
      setActiveCardId(cardId);
      setStatus("recording");
      recorder.start();
    },
    [ensureAudioCtx, openMicStream, stopCurrentPlayback]
  );

  const onMicRelease = useCallback(
    (cardId: string) => {
      const recorder = recorderRef.current;
      if (!recorder || recorder.state !== "recording") return;
      recorder.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        recordedChunksRef.current = [];
        recorderRef.current = null;
        if (blob.size < 1024) {
          setStatus("idle");
          setActiveCardId(null);
          setErrorMsg("too short — hold the button longer");
          return;
        }
        const card = cards.find((c) => c.id === cardId);
        if (!card) {
          setStatus("idle");
          setActiveCardId(null);
          return;
        }
        setStatus("thinking");
        void runSpeakTurn(card, blob);
      };
      try {
        recorder.stop();
      } catch {
        // already stopped
      }
    },
    // runSpeakTurn is defined below via closure — intentionally not in deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [cards]
  );

  // --- Conversation turn -------------------------------------------------

  const runSpeakTurn = useCallback(
    async (card: SessionCard, blob: Blob) => {
      const ctx = ensureAudioCtx();
      if (!ctx) {
        setStatus("idle");
        setActiveCardId(null);
        setErrorMsg("Web Audio unavailable");
        return;
      }
      const callGen = ++speakGenRef.current;
      const turnId = String(++turnCounterRef.current);
      const history =
        historyRef.current[card.id] ??
        [{ role: "assistant" as const, content: card.line }];

      try {
        const form = new FormData();
        const filename = blob.type.includes("mp4")
          ? "talk.mp4"
          : blob.type.includes("ogg")
            ? "talk.ogg"
            : "talk.webm";
        form.append("audio", blob, filename);
        // Pass the VLM's specific name to the server so the persona
        // ("a chipped ceramic mug") rides through the conversation
        // instead of the coarse YOLO class ("cup").
        form.append("className", card.objectName?.trim() || card.className);
        form.append("voiceId", card.voiceId);
        form.append("description", card.description);
        form.append("history", JSON.stringify(history.slice(-32)));
        form.append("turnId", turnId);
        form.append("lang", card.learnLang ?? card.spokenLang ?? "zh");
        if (card.spokenLang) form.append("spokenLang", card.spokenLang);
        if (card.learnLang) form.append("learnLang", card.learnLang);

        const { transcript, reply, voiceId: replyVoiceId } =
          await converseWithObject(form);
        if (callGen !== speakGenRef.current) return;

        if (transcript) {
          setHeardByCard((prev) => ({ ...prev, [card.id]: transcript }));
        }
        if (!reply) {
          setStatus("idle");
          setActiveCardId(null);
          setErrorMsg("no reply");
          return;
        }
        const nextHistory = [
          ...history,
          ...(transcript
            ? [{ role: "user" as const, content: transcript }]
            : []),
          { role: "assistant" as const, content: reply },
        ].slice(-32);
        historyRef.current[card.id] = nextHistory;
        setLiveReplyByCard((prev) => ({ ...prev, [card.id]: reply }));

        setStatus("speaking");
        await streamTTS({
          ctx,
          text: reply,
          voiceId: replyVoiceId ?? card.voiceId,
          turnId,
          callGen,
        });
        if (callGen !== speakGenRef.current) return;
        setStatus("idle");
        setActiveCardId(null);
      } catch (e) {
        if (callGen !== speakGenRef.current) return;
        setErrorMsg(e instanceof Error ? e.message : "talk failed");
        setStatus("idle");
        setActiveCardId(null);
      }
    },
    // streamTTS is stable closure
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [ensureAudioCtx]
  );

  // --- TTS streaming ------------------------------------------------------

  const streamTTS = useCallback(
    async (args: {
      ctx: AudioContext;
      text: string;
      voiceId: string;
      turnId: string;
      callGen: number;
    }): Promise<void> => {
      const { ctx, text, voiceId, turnId, callGen } = args;
      const resp = await fetch("/api/tts/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          voiceId,
          turnId,
          lang: "en",
          emotion: [],
          speed: null,
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
        throw new Error(`tts stream ${resp.status}: ${err.slice(0, 120)}`);
      }

      const analyser = ensureAnalyser(ctx);
      if (ctx.state === "suspended") {
        try {
          await ctx.resume();
        } catch {
          // best-effort
        }
      }

      const canStream =
        typeof window !== "undefined" &&
        typeof window.MediaSource !== "undefined" &&
        window.MediaSource.isTypeSupported?.("audio/mpeg");

      if (canStream) {
        await playViaMediaSource({
          ctx,
          analyser,
          audioEl: audioElRef.current!,
          audioSrcNodeRef,
          respBody: resp.body,
          callGen,
          speakGenRef,
          onStart: () => {
            lipSyncRef.current = createLipSyncState();
            startLipSyncLoop();
          },
          onEnd: () => {
            stopLipSyncLoop();
          },
        });
      } else {
        const buf = await new Response(resp.body).arrayBuffer();
        if (callGen !== speakGenRef.current) return;
        let audioBuf: AudioBuffer;
        try {
          audioBuf = await ctx.decodeAudioData(buf);
        } catch (e) {
          throw new Error(
            `decodeAudioData failed: ${e instanceof Error ? e.message : String(e)}`
          );
        }
        if (callGen !== speakGenRef.current) return;
        const source = ctx.createBufferSource();
        source.buffer = audioBuf;
        source.connect(analyser);
        bufferSrcNodeRef.current = source;
        lipSyncRef.current = createLipSyncState();
        startLipSyncLoop();
        await new Promise<void>((resolve) => {
          source.onended = () => {
            stopLipSyncLoop();
            if (bufferSrcNodeRef.current === source) {
              bufferSrcNodeRef.current = null;
            }
            resolve();
          };
          try {
            source.start();
          } catch {
            resolve();
          }
        });
      }
    },
    [ensureAnalyser, startLipSyncLoop, stopLipSyncLoop]
  );

  const statusByCard = useMemo<Record<string, CardStatus>>(() => {
    if (!activeCardId) return {};
    return { [activeCardId]: status };
  }, [activeCardId, status]);

  return (
    <>
      {errorMsg && (
        <div
          className="mx-auto mb-4 w-full max-w-[820px] rounded-2xl bg-rose-500/20 px-4 py-2 text-[12.5px] font-medium text-rose-100 ring-1 ring-rose-300/30 backdrop-blur-xl"
          role="status"
        >
          {errorMsg}
        </div>
      )}
      <ul
        className="gallery-list mx-auto flex w-full max-w-[620px] flex-col gap-5"
        data-focused={activeCardId ? "true" : "false"}
        style={{ perspective: "1200px" }}
      >
        {cards.map((card, i) => {
          const cardStatus = statusByCard[card.id] ?? "idle";
          // Deterministic pseudo-random tilt per card (polaroid feel).
          const seed = hashSeed(card.id);
          const tilt = ((seed % 100) / 100 - 0.5) * 2.4; // ~±1.2deg
          const breathDelay = ((seed >> 4) % 100) / 100 * 2.2;
          return (
            <li
              key={card.id}
              data-active={cardStatus !== "idle" ? "true" : "false"}
              className="gallery-card-enter"
              style={
                {
                  ["--enter-delay"]: `${Math.min(i, 8) * 70}ms`,
                  ["--breath-delay"]: `${breathDelay}s`,
                  ["--tilt"]: `${tilt}deg`,
                } as React.CSSProperties
              }
            >
              <CardItem
                card={card}
                status={cardStatus}
                replyText={liveReplyByCard[card.id] ?? null}
                shape={cardStatus === "speaking" ? shape : "X"}
                onMicPress={onMicPress}
                onMicRelease={onMicRelease}
                onRemove={() => removeSessionCard(card.id)}
                heard={heardByCard[card.id] ?? null}
              />
            </li>
          );
        })}
      </ul>
    </>
  );
}

// --- Single card ----------------------------------------------------------

function CardItem({
  card,
  status,
  replyText,
  shape,
  onMicPress,
  onMicRelease,
  onRemove,
  heard,
}: {
  card: SessionCard;
  status: CardStatus;
  replyText: string | null;
  shape: MouthShape;
  onMicPress: (cardId: string) => void;
  onMicRelease: (cardId: string) => void;
  onRemove: () => void;
  heard: string | null;
}) {
  const isActive = status !== "idle";
  const isSpeaking = status === "speaking";
  const isRecording = status === "recording";
  const isThinking = status === "thinking";
  const [rawImgFailed, setRawImgFailed] = useState(false);
  const [genImgLoaded, setGenImgLoaded] = useState(false);
  const [genImgFailed, setGenImgFailed] = useState(false);
  const [retryDismissed, setRetryDismissed] = useState(false);

  // Pointer-driven 3D lean. Kept small (~5deg) — enough to notice, not so
  // much it feels gimmicky. Reset on leave.
  const leanRef = useRef<HTMLDivElement | null>(null);
  const onPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const el = leanRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width - 0.5;
    const py = (e.clientY - rect.top) / rect.height - 0.5;
    el.style.setProperty("--lean-x", `${(-py * 5).toFixed(2)}deg`);
    el.style.setProperty("--lean-y", `${(px * 6).toFixed(2)}deg`);
    el.style.setProperty("--glare-x", `${((px + 0.5) * 100).toFixed(1)}%`);
    el.style.setProperty("--glare-y", `${((py + 0.5) * 100).toFixed(1)}%`);
    el.style.setProperty("--lean-lift", "-3px");
  }, []);
  const onPointerLeave = useCallback(() => {
    const el = leanRef.current;
    if (!el) return;
    el.style.setProperty("--lean-x", "0deg");
    el.style.setProperty("--lean-y", "0deg");
    el.style.setProperty("--lean-lift", "0px");
  }, []);

  const genStatus = card.generatedImageStatus;
  const hasGenUrl = !!card.generatedImageUrl;
  const showGenerated =
    hasGenUrl && genStatus === "done" && !genImgFailed;
  const showShimmer = genStatus === "pending";
  const showPainting = genStatus === "pending";
  const showRetry =
    (genStatus === "failed" || genImgFailed) && !retryDismissed;

  // Reset load state when the generated URL changes so a subsequent swap
  // fades in again.
  useEffect(() => {
    setGenImgLoaded(false);
    setGenImgFailed(false);
  }, [card.generatedImageUrl]);

  const langPair =
    card.spokenLang && card.learnLang
      ? { spoken: card.spokenLang, learn: card.learnLang }
      : null;

  return (
    <article
      ref={leanRef}
      onPointerMove={onPointerMove}
      onPointerLeave={onPointerLeave}
      className={`group relative flex items-stretch gap-4 overflow-hidden rounded-[24px] bg-white/[0.06] p-3.5 ring-1 ring-white/15 backdrop-blur-2xl transition-[box-shadow,background,ring] duration-300 hover:bg-white/[0.1] hover:ring-white/25 ${
        isActive ? "active-halo" : ""
      }`}
      style={{
        transformStyle: "preserve-3d",
        transform:
          "perspective(1100px) rotate(var(--tilt, 0deg)) rotateX(var(--lean-x, 0deg)) rotateY(var(--lean-y, 0deg)) translateY(var(--lean-lift, 0px))",
        transition:
          "transform 420ms cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 300ms ease, background 300ms ease",
        boxShadow: isActive
          ? undefined
          : "0 22px 60px -28px rgba(0,0,0,0.75), inset 0 1px 0 rgba(255,255,255,0.06)",
      }}
    >
      {/* Sheen — follows the pointer. Pure CSS radial that reads as light
          catching on a glossy card. */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 rounded-[24px] opacity-0 transition-opacity duration-300 group-hover:opacity-100"
        style={{
          background:
            "radial-gradient(220px circle at var(--glare-x, 50%) var(--glare-y, 50%), rgba(255,255,255,0.12), transparent 60%)",
        }}
      />
      <button
        type="button"
        aria-label="remove from gallery"
        onClick={onRemove}
        className="absolute right-2 top-2 z-20 grid h-6 w-6 place-items-center rounded-full bg-black/40 text-[12px] leading-none text-white/85 opacity-0 ring-1 ring-white/10 backdrop-blur-md transition-opacity duration-150 hover:bg-black/65 group-hover:opacity-100 focus:opacity-100"
      >
        ×
      </button>

      <div
        className="relative aspect-square h-auto w-[108px] shrink-0 overflow-hidden rounded-[16px] bg-black/35 sm:w-[132px]"
        aria-hidden={isSpeaking ? undefined : true}
      >
        {rawImgFailed ? (
          <div className="absolute inset-0 grid place-items-center text-[10px] uppercase tracking-widest text-white/40">
            missing
          </div>
        ) : (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={card.imageDataUrl}
            alt={cardDisplayName(card)}
            className="absolute inset-0 h-full w-full object-cover"
            draggable={false}
            onError={() => setRawImgFailed(true)}
          />
        )}
        {showGenerated && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={card.generatedImageUrl}
            alt={`${cardDisplayName(card)} — illustrated`}
            data-loaded={genImgLoaded ? "true" : "false"}
            className="gen-image-fade absolute inset-0 h-full w-full object-cover"
            draggable={false}
            onLoad={() => setGenImgLoaded(true)}
            onError={() => {
              setGenImgFailed(true);
              setGenImgLoaded(false);
            }}
          />
        )}
        {showShimmer && <span aria-hidden className="shimmer-overlay" />}
        {showPainting && (
          <div
            className="bubble-btn pointer-events-none absolute right-1.5 top-1.5 z-10 inline-flex items-center gap-1 rounded-full bg-pink-400/80 px-2 py-0.5 text-[9.5px] font-semibold uppercase tracking-wider text-pink-950 ring-1 ring-white/40 backdrop-blur-md"
          >
            <span aria-hidden>✨</span>
            <span>painting…</span>
          </div>
        )}
        {showRetry && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setRetryDismissed(true);
            }}
            title="re-trigger from the camera to try again"
            className="absolute bottom-1.5 right-1.5 z-10 inline-flex items-center gap-1 rounded-full bg-black/55 px-2 py-0.5 text-[9.5px] font-medium uppercase tracking-wider text-white/85 ring-1 ring-white/15 backdrop-blur-md transition hover:bg-black/75"
          >
            retry
          </button>
        )}
        {isSpeaking && (
          <div
            className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
            style={{
              width: FACE_VOICE_WIDTH * 0.55,
              height: FACE_VOICE_HEIGHT * 0.55,
            }}
          >
            <div
              style={{
                width: FACE_VOICE_WIDTH,
                height: FACE_VOICE_HEIGHT,
                transform: "scale(0.55)",
                transformOrigin: "0 0",
              }}
            >
              <FaceVoice shape={shape} />
            </div>
          </div>
        )}
      </div>

      <div className="relative flex min-w-0 flex-1 flex-col justify-between gap-2.5 py-1">
        <div className="flex min-w-0 flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <span
              className="inline-flex items-center gap-1 rounded-full bg-white/[0.05] px-2 py-0.5 text-[9.5px] font-medium uppercase tracking-[0.2em] text-white/45 ring-1 ring-white/10"
              aria-label={`captured ${formatRelativeTime(card.createdAt)}`}
            >
              <span aria-hidden>·</span>
              <span>{formatRelativeTime(card.createdAt)}</span>
            </span>
            {langPair && (
              <span
                className={`inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[9.5px] font-semibold uppercase tracking-[0.12em] ring-1 ${
                  card.teachMode
                    ? "bg-pink-400/20 text-pink-200 ring-pink-300/30"
                    : "bg-white/[0.05] text-white/60 ring-white/10"
                }`}
                aria-label={`speaks ${langPair.spoken.toUpperCase()}, learning ${langPair.learn.toUpperCase()}`}
              >
                <span>{langPair.spoken.toUpperCase()}</span>
                <span aria-hidden className="opacity-60">→</span>
                <span>{langPair.learn.toUpperCase()}</span>
                {card.teachMode && (
                  <span
                    aria-hidden
                    className="ml-0.5 rounded-full bg-pink-300/30 px-1 text-[8.5px] tracking-wide text-pink-100"
                  >
                    teach
                  </span>
                )}
              </span>
            )}
          </div>
          <h2
            className="serif-italic truncate text-[26px] leading-[1.05] text-white/95"
            title={cardDisplayName(card)}
            style={{ textShadow: "0 0 22px rgba(255,182,214,0.18)" }}
          >
            {cardDisplayName(card)}
          </h2>
          <p
            className="line-clamp-2 text-[12.5px] italic leading-[1.4] text-white/65"
            aria-live={isActive ? "polite" : undefined}
          >
            {isRecording
              ? "listening…"
              : isThinking
                ? "thinking…"
                : replyText && isSpeaking
                  ? replyText
                  : `“${card.line}”`}
          </p>
          {heard && !isActive && (
            <p className="mt-0.5 line-clamp-1 text-[10.5px] font-medium uppercase tracking-[0.14em] text-white/35">
              you said: <span className="normal-case tracking-normal text-white/55">{heard}</span>
            </p>
          )}
        </div>

        <MicButton
          cardId={card.id}
          status={status}
          onPress={onMicPress}
          onRelease={onMicRelease}
        />
      </div>
    </article>
  );
}

function MicButton({
  cardId,
  status,
  onPress,
  onRelease,
}: {
  cardId: string;
  status: CardStatus;
  onPress: (cardId: string) => void;
  onRelease: (cardId: string) => void;
}) {
  const busy = status !== "idle";
  const label =
    status === "recording"
      ? "release to send"
      : status === "thinking"
        ? "thinking…"
        : status === "speaking"
          ? "tap to interrupt"
          : "hold to talk";

  return (
    <button
      type="button"
      aria-label={label}
      onPointerDown={(e) => {
        e.preventDefault();
        e.stopPropagation();
        try {
          (e.currentTarget as Element).setPointerCapture(e.pointerId);
        } catch {
          // ignore
        }
        onPress(cardId);
      }}
      onPointerUp={(e) => {
        e.preventDefault();
        e.stopPropagation();
        try {
          (e.currentTarget as Element).releasePointerCapture(e.pointerId);
        } catch {
          // ignore
        }
        onRelease(cardId);
      }}
      onPointerCancel={() => onRelease(cardId)}
      onPointerLeave={(e) => {
        if (e.buttons > 0) onRelease(cardId);
      }}
      className="relative inline-flex h-9 w-fit items-center gap-1.5 rounded-full pl-2.5 pr-3.5 text-[11.5px] font-semibold tracking-wide transition-[transform,box-shadow] duration-150 ease-out"
      style={{
        background:
          status === "recording"
            ? "linear-gradient(135deg, #ff6fae 0%, #ff9dbf 100%)"
            : busy
              ? "rgba(255,255,255,0.18)"
              : "linear-gradient(135deg, #ffb8d6 0%, #ffd4e3 100%)",
        color: busy && status !== "recording" ? "rgba(255,255,255,0.88)" : "#3a0a29",
        boxShadow:
          status === "recording"
            ? "0 0 0 2px rgba(255,255,255,0.55), 0 10px 22px -8px rgba(255,111,174,0.65)"
            : "0 6px 16px -8px rgba(255,111,174,0.55)",
        transform: status === "recording" ? "scale(0.97)" : undefined,
        touchAction: "none",
      }}
    >
      <span
        aria-hidden
        className="grid h-5 w-5 place-items-center rounded-full"
        style={{
          background:
            status === "recording" ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.55)",
        }}
      >
        <MicIcon />
      </span>
      <span>{label}</span>
    </button>
  );
}

function MicIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="9" y="3" width="6" height="12" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <path d="M12 18v3" />
    </svg>
  );
}

// --- Small helpers --------------------------------------------------------

function hashSeed(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function formatRelativeTime(ms: number): string {
  if (!ms || Number.isNaN(ms)) return "just now";
  const diff = Date.now() - ms;
  if (diff < 45_000) return "just now";
  if (diff < 60 * 60_000) return `${Math.round(diff / 60_000)}m ago`;
  if (diff < 24 * 60 * 60_000) return `${Math.round(diff / (60 * 60_000))}h ago`;
  const d = new Date(ms);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

// --- MediaSource streaming helper -----------------------------------------
//
// Stripped-down version of the tracker's playViaMediaSource — same shape
// but operates on refs passed in from the caller. Keeps the gallery page
// self-contained (no shared module with tracker, since the tracker's
// version is tied to its TrackRefs type).

async function playViaMediaSource(args: {
  ctx: AudioContext;
  analyser: AnalyserNode;
  audioEl: HTMLAudioElement;
  audioSrcNodeRef: React.MutableRefObject<MediaElementAudioSourceNode | null>;
  respBody: ReadableStream<Uint8Array>;
  callGen: number;
  speakGenRef: React.MutableRefObject<number>;
  onStart: () => void;
  onEnd: () => void;
}) {
  const {
    ctx,
    analyser,
    audioEl,
    audioSrcNodeRef,
    respBody,
    callGen,
    speakGenRef,
    onStart,
    onEnd,
  } = args;

  if (callGen !== speakGenRef.current) {
    try {
      await respBody.cancel();
    } catch {
      // ignore
    }
    return;
  }

  const mediaSource = new MediaSource();
  const objectUrl = URL.createObjectURL(mediaSource);

  // Connect audio element to the analyser graph (once — MediaElementSource
  // can only be created per element once per AudioContext).
  if (!audioSrcNodeRef.current) {
    try {
      audioSrcNodeRef.current = ctx.createMediaElementSource(audioEl);
      audioSrcNodeRef.current.connect(analyser);
    } catch {
      // If the element was already wired (shouldn't happen but defensive),
      // assume the connection is fine.
    }
  }

  audioEl.src = objectUrl;

  const cleanup = (sourceBuffer: SourceBuffer | null) => {
    try {
      if (sourceBuffer && !sourceBuffer.updating) {
        try {
          mediaSource.removeSourceBuffer(sourceBuffer);
        } catch {
          // ignore
        }
      }
    } catch {
      // ignore
    }
    try {
      URL.revokeObjectURL(objectUrl);
    } catch {
      // ignore
    }
  };

  let sourceBuffer: SourceBuffer | null = null;
  const sourceOpen = new Promise<void>((resolve, reject) => {
    mediaSource.addEventListener("sourceopen", () => resolve(), { once: true });
    mediaSource.addEventListener(
      "error",
      () => reject(new Error("MediaSource errored before open")),
      { once: true }
    );
    setTimeout(() => reject(new Error("MediaSource open timeout")), 4000);
  });

  try {
    await sourceOpen;
  } catch (e) {
    cleanup(null);
    throw e;
  }

  try {
    sourceBuffer = mediaSource.addSourceBuffer("audio/mpeg");
  } catch (e) {
    cleanup(null);
    throw new Error(
      `addSourceBuffer failed: ${e instanceof Error ? e.message : String(e)}`
    );
  }

  const reader = respBody.getReader();
  const queue: Uint8Array<ArrayBuffer>[] = [];
  let ended = false;
  let playing = false;

  const pump = () => {
    if (!sourceBuffer || sourceBuffer.updating) return;
    if (queue.length > 0) {
      const chunk = queue.shift()!;
      try {
        sourceBuffer.appendBuffer(chunk);
      } catch {
        // buffer full / quota — drop this chunk; later updates will retry.
      }
    } else if (ended && !sourceBuffer.updating) {
      try {
        mediaSource.endOfStream();
      } catch {
        // ignore
      }
    }
  };

  sourceBuffer.addEventListener("updateend", () => {
    if (callGen !== speakGenRef.current) return;
    if (!playing) {
      playing = true;
      onStart();
      audioEl.play().catch(() => {
        // autoplay blocked — onEnd will fire via ended/pause.
      });
    }
    pump();
  });

  const readLoop = async () => {
    while (true) {
      if (callGen !== speakGenRef.current) {
        try {
          await reader.cancel();
        } catch {
          // ignore
        }
        return;
      }
      const { done, value } = await reader.read();
      if (done) {
        ended = true;
        pump();
        return;
      }
      if (value && value.byteLength > 0) {
        const copy = new Uint8Array(new ArrayBuffer(value.byteLength));
        copy.set(value);
        queue.push(copy);
        pump();
      }
    }
  };

  const donePromise = new Promise<void>((resolve) => {
    const onEnded = () => {
      audioEl.removeEventListener("ended", onEnded);
      audioEl.removeEventListener("error", onEnded);
      cleanup(sourceBuffer);
      onEnd();
      resolve();
    };
    audioEl.addEventListener("ended", onEnded);
    audioEl.addEventListener("error", onEnded);
  });

  await readLoop();
  await donePromise;
}
