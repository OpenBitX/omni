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
    <div className="min-h-dvh bg-gradient-to-br from-[#1a0f2e] via-[#2a1540] to-[#3d1a4d] pb-[max(env(safe-area-inset-bottom),24px)] pt-[max(env(safe-area-inset-top),18px)] text-white">
      <Header hasCards={cards.length > 0} />
      <main className="mx-auto w-full max-w-[1120px] px-4 pb-12 pt-4 sm:px-6">
        {cards.length === 0 ? (
          <EmptyState />
        ) : (
          <GalleryGrid cards={cards} />
        )}
      </main>
    </div>
  );
}

function Header({ hasCards }: { hasCards: boolean }) {
  return (
    <header className="mx-auto flex w-full max-w-[1120px] items-center justify-between gap-3 px-4 sm:px-6">
      <Link
        href="/"
        className="inline-flex items-center gap-2 rounded-full bg-white/10 px-3.5 py-1.5 text-[12px] font-medium ring-1 ring-white/20 backdrop-blur-xl transition hover:bg-white/20"
      >
        <span aria-hidden>←</span>
        <span>camera</span>
      </Link>
      <h1 className="serif-italic text-[20px] font-medium leading-none text-white/95 sm:text-[24px]">
        gallery
      </h1>
      {hasCards ? (
        <button
          type="button"
          onClick={() => {
            if (window.confirm("Clear every card in the gallery?")) {
              clearSessionCards();
            }
          }}
          className="inline-flex items-center gap-1.5 rounded-full bg-white/10 px-3 py-1.5 text-[11.5px] font-semibold uppercase tracking-wider ring-1 ring-white/20 backdrop-blur-xl transition hover:bg-white/20"
        >
          clear all
        </button>
      ) : (
        <span className="inline-flex h-[30px] w-[78px]" aria-hidden />
      )}
    </header>
  );
}

function EmptyState() {
  return (
    <div className="mx-auto mt-20 max-w-md rounded-[28px] bg-white/8 p-8 text-center ring-1 ring-white/15 backdrop-blur-xl">
      <div
        aria-hidden
        className="mx-auto mb-5 grid h-14 w-14 place-items-center rounded-2xl bg-white/10 text-2xl"
      >
        ✨
      </div>
      <h2 className="text-[18px] font-semibold text-white/95">
        Nothing captured yet
      </h2>
      <p className="mt-2 text-[13.5px] leading-relaxed text-white/70">
        Point your camera at something and tap it. The first opening line
        the VLM writes gets saved here — and you can keep the conversation
        going right from this page.
      </p>
      <Link
        href="/"
        className="mt-5 inline-flex items-center gap-2 rounded-full bg-pink-400/90 px-4 py-2 text-[12.5px] font-semibold text-pink-950 shadow-lg shadow-pink-400/40 transition hover:bg-pink-300"
      >
        open camera
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
        form.append("className", card.className);
        form.append("voiceId", card.voiceId);
        form.append("description", card.description);
        form.append("history", JSON.stringify(history.slice(-32)));
        form.append("turnId", turnId);
        form.append("lang", "en");

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
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {cards.map((card) => (
          <CardItem
            key={card.id}
            card={card}
            status={statusByCard[card.id] ?? "idle"}
            heardText={heardByCard[card.id] ?? null}
            replyText={liveReplyByCard[card.id] ?? null}
            shape={
              statusByCard[card.id] === "speaking" ? shape : "X"
            }
            onMicPress={onMicPress}
            onMicRelease={onMicRelease}
            onRemove={() => removeSessionCard(card.id)}
          />
        ))}
      </div>
    </>
  );
}

// --- Single card ----------------------------------------------------------

function CardItem({
  card,
  status,
  heardText,
  replyText,
  shape,
  onMicPress,
  onMicRelease,
  onRemove,
}: {
  card: SessionCard;
  status: CardStatus;
  heardText: string | null;
  replyText: string | null;
  shape: MouthShape;
  onMicPress: (cardId: string) => void;
  onMicRelease: (cardId: string) => void;
  onRemove: () => void;
}) {
  const isActive = status !== "idle";
  const isSpeaking = status === "speaking";
  const isRecording = status === "recording";
  const isThinking = status === "thinking";

  return (
    <article
      className="group relative flex flex-col overflow-hidden rounded-[24px] bg-white/8 ring-1 ring-white/15 backdrop-blur-xl transition-[transform,box-shadow] duration-200"
      style={{
        boxShadow: isActive
          ? "0 18px 48px -18px rgba(255,137,190,0.55), 0 0 0 2px rgba(255,137,190,0.85)"
          : "0 14px 40px -22px rgba(0,0,0,0.55)",
        transform: isActive ? "translateY(-2px)" : undefined,
      }}
    >
      <button
        type="button"
        aria-label="remove from gallery"
        onClick={onRemove}
        className="absolute right-3 top-3 z-20 grid h-7 w-7 place-items-center rounded-full bg-black/45 text-[13px] leading-none text-white/85 opacity-0 ring-1 ring-white/15 backdrop-blur-md transition-opacity duration-150 hover:bg-black/70 group-hover:opacity-100"
      >
        ×
      </button>

      <div className="relative aspect-[4/3] w-full overflow-hidden bg-black/40">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={card.imageDataUrl}
          alt={card.className}
          className="absolute inset-0 h-full w-full object-cover"
          draggable={false}
        />
        {isSpeaking && (
          <div
            className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
            style={{
              width: FACE_VOICE_WIDTH,
              height: FACE_VOICE_HEIGHT,
            }}
            aria-hidden
          >
            <FaceVoice shape={shape} />
          </div>
        )}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-black/70 to-transparent" />
        <span className="absolute left-3 top-3 rounded-full bg-black/50 px-2.5 py-1 text-[10.5px] font-semibold uppercase tracking-[0.15em] text-white/90 ring-1 ring-white/15 backdrop-blur">
          {card.className}
        </span>
        {isActive && (
          <span
            className="absolute bottom-3 left-3 inline-flex items-center gap-1.5 rounded-full bg-pink-400/90 px-2.5 py-1 text-[10.5px] font-semibold uppercase tracking-[0.15em] text-pink-950 shadow-lg shadow-pink-500/30"
            aria-live="polite"
          >
            <span
              className="h-1.5 w-1.5 rounded-full bg-pink-950"
              style={{
                animation: isRecording || isThinking
                  ? "soft-pulse 1.1s ease-in-out infinite"
                  : undefined,
              }}
            />
            {isRecording ? "listening…" : isThinking ? "thinking…" : "speaking"}
          </span>
        )}
      </div>

      <div className="flex flex-col gap-3 px-4 py-4">
        <p className="text-[13.5px] leading-[1.45] text-white/85">
          {card.description}
        </p>
        <blockquote className="border-l-2 border-pink-300/60 pl-3 text-[14px] italic leading-[1.5] text-white/95">
          &ldquo;{card.line}&rdquo;
        </blockquote>

        {heardText && (
          <p className="text-[11.5px] text-white/60">
            you said: <span className="text-white/85">{heardText}</span>
          </p>
        )}
        {replyText && status !== "speaking" && (
          <p className="text-[11.5px] text-pink-200/90">
            latest: <span className="text-white/90">{replyText}</span>
          </p>
        )}

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
          ? "speaking — tap to interrupt"
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
      className="relative flex h-11 items-center justify-center gap-2 rounded-full text-[12.5px] font-semibold tracking-wide transition-[transform,box-shadow] duration-150 ease-out"
      style={{
        background:
          status === "recording"
            ? "linear-gradient(135deg, #ff6fae 0%, #ff9dbf 100%)"
            : busy
              ? "rgba(255,255,255,0.18)"
              : "linear-gradient(135deg, #ffb8d6 0%, #ffd4e3 100%)",
        color: busy && status !== "recording" ? "rgba(255,255,255,0.85)" : "#3a0a29",
        boxShadow:
          status === "recording"
            ? "0 0 0 2px rgba(255,255,255,0.55), 0 10px 22px -8px rgba(255,111,174,0.65)"
            : "0 8px 20px -10px rgba(255,111,174,0.55)",
        transform: status === "recording" ? "scale(0.97)" : undefined,
        touchAction: "none",
      }}
    >
      <MicIcon />
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
