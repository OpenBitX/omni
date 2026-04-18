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
  appendCardHistory,
  cardDisplayName,
  clearSessionCards,
  removeSessionCard,
  useSessionCards,
  type ChatTurn,
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

// === Gallery page =========================================================
//
// A wooden bookshelf sits at the top. Every card the user has captured is
// placed on the shelf as a die-cut cartoon cutout (Runware-painted PNG,
// composited onto the wood via mix-blend-mode: multiply). Tapping a cutout
// opens the conversation drawer below: conversation history, hold-to-talk
// mic, and a live speaking face. The drawer continues the persisted
// conversation — history lives in `sessionStorage` via session-cards store,
// so it survives route transitions.
//
// All Web Audio + mic + MediaSource plumbing is self-contained here (no
// imports from tracker.tsx). Only one card can speak at a time.

type CardStatus = "idle" | "recording" | "thinking" | "speaking";

// Shelf geometry — the bookshelf image has 4 shelves × 3 bays. Percentages
// are the bottom-edge of each shelf slot, measured from the TOP of the
// bookshelf image. CSS then uses `bottom: (100 - shelfTop%)%` so items sit
// on the shelf plank. Pillars run at roughly x=33% and x=67%; the bay
// centers avoid them.
const SHELF_ROWS_Y = [34, 52, 70, 89] as const; // % from top of bookshelf
const BAY_X = [17, 50, 83] as const; // % from left of bookshelf

const MAX_VISIBLE_SLOTS = SHELF_ROWS_Y.length * BAY_X.length; // 12

type LensFilter = "all" | "play" | "language" | "history";

export default function GalleryPage() {
  const cards = useSessionCards();
  const hasCards = cards.length > 0;
  const [filter, setFilter] = useState<LensFilter>("all");

  const filteredCards = useMemo(() => {
    if (filter === "all") return cards;
    return cards.filter((c) => (c.mode ?? "play") === filter);
  }, [cards, filter]);

  return (
    <div
      className="relative min-h-dvh overflow-hidden bg-[#120822] pb-[max(env(safe-area-inset-bottom),24px)] pt-[max(env(safe-area-inset-top),18px)] text-white"
    >
      <div className="aurora-layer" aria-hidden>
        <div className="aurora-blob aurora-a" />
        <div className="aurora-blob aurora-b" />
        <div className="aurora-blob aurora-c" />
        <div className="aurora-grain" />
      </div>
      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 z-0"
        style={{
          background:
            "radial-gradient(ellipse at 50% 20%, transparent 0%, rgba(8,4,20,0.35) 65%, rgba(8,4,20,0.75) 100%)",
        }}
      />
      <div className="relative z-10">
        <Header hasCards={hasCards} />
        <main className="mx-auto w-full max-w-[1280px] px-4 pb-16 pt-4 sm:px-6">
          {hasCards ? (
            <>
              <LensFilterBar
                cards={cards}
                filter={filter}
                onChange={setFilter}
              />
              {filteredCards.length === 0 ? (
                <FilterEmptyState
                  filter={filter}
                  onReset={() => setFilter("all")}
                />
              ) : (
                <BookshelfGallery cards={filteredCards} />
              )}
            </>
          ) : (
            <EmptyState />
          )}
        </main>
      </div>
    </div>
  );
}

function LensFilterBar({
  cards,
  filter,
  onChange,
}: {
  cards: readonly SessionCard[];
  filter: LensFilter;
  onChange: (next: LensFilter) => void;
}) {
  const counts = useMemo(() => {
    const c: Record<LensFilter, number> = {
      all: cards.length,
      play: 0,
      language: 0,
      history: 0,
    };
    for (const card of cards) {
      const m = (card.mode ?? "play") as LensFilter;
      if (m === "play" || m === "language" || m === "history") c[m]++;
    }
    return c;
  }, [cards]);

  const opts: { value: LensFilter; label: string; glyph: string }[] = [
    { value: "all", label: "all", glyph: "∗" },
    { value: "play", label: "play", glyph: "✿" },
    { value: "language", label: "language", glyph: "✦" },
    { value: "history", label: "history", glyph: "♡" },
  ];

  return (
    <div className="mx-auto mb-6 flex w-full max-w-[560px] flex-nowrap items-center justify-center gap-1 rounded-full bg-white/[0.06] p-1 ring-1 ring-white/15 backdrop-blur-2xl">
      {opts.map((o) => {
        const active = filter === o.value;
        const n = counts[o.value];
        return (
          <button
            key={o.value}
            type="button"
            onClick={() => onChange(o.value)}
            aria-pressed={active}
            className={
              "inline-flex min-w-0 flex-1 flex-nowrap items-center justify-center gap-1 whitespace-nowrap rounded-full px-2.5 py-1.5 text-[11.5px] font-semibold tracking-wide transition " +
              (active
                ? "bg-white/85 text-[#c23a7a] shadow-sm"
                : "text-white/75 hover:bg-white/[0.08] hover:text-white")
            }
          >
            <span aria-hidden className="shrink-0 text-[12.5px] leading-none">
              {o.glyph}
            </span>
            <span className="shrink-0 leading-none">{o.label}</span>
            <span
              className={
                "shrink-0 rounded-full px-1.5 text-[9.5px] font-semibold leading-[1.35] tabular-nums tracking-wide " +
                (active
                  ? "bg-[#c23a7a]/15 text-[#c23a7a]"
                  : "bg-white/10 text-white/55")
              }
            >
              {n}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function FilterEmptyState({
  filter,
  onReset,
}: {
  filter: LensFilter;
  onReset: () => void;
}) {
  const label = filter === "all" ? "" : filter;
  return (
    <div className="mx-auto mt-10 max-w-md rounded-[28px] bg-white/[0.05] p-8 text-center ring-1 ring-white/15 backdrop-blur-2xl">
      <h2 className="serif-italic text-[20px] text-white/95">
        nothing from the{" "}
        <span className="text-[color:var(--accent)]">{label}</span> lens yet
      </h2>
      <p className="mt-2 text-[12.5px] text-white/55">
        switch lenses on the camera and tap something — it&apos;ll land on
        this shelf.
      </p>
      <button
        type="button"
        onClick={onReset}
        className="btn-frost mt-5 px-4 py-2 text-[11.5px] font-medium"
      >
        show all
      </button>
    </div>
  );
}

// --- Header ---------------------------------------------------------------

function Header({ hasCards }: { hasCards: boolean }) {
  return (
    <header className="mx-auto grid w-full max-w-[820px] grid-cols-[1fr_auto_1fr] items-center gap-3 px-4 sm:px-6">
      <div className="justify-self-start">
        <Link
          href="/"
          aria-label="back to camera"
          className="btn-frost group px-3.5 py-2 text-[11.5px] font-medium"
        >
          <span aria-hidden className="transition-transform group-hover:-translate-x-0.5">
            ←
          </span>
          <span>camera</span>
        </Link>
      </div>
      <div className="justify-self-center text-center">
        <h1
          className="serif-italic relative text-[32px] font-medium leading-none text-white sm:text-[42px]"
          style={{
            textShadow:
              "0 0 32px rgba(255,182,214,0.38), 0 2px 0 rgba(0,0,0,0.18)",
          }}
        >
          the shelf
        </h1>
        <p className="mt-2 text-[10.5px] uppercase tracking-[0.32em] text-white/50">
          everything you spoke to
        </p>
      </div>
      <div className="justify-self-end">
        {hasCards && (
          <button
            type="button"
            onClick={() => {
              if (window.confirm("Clear every item from the shelf?")) {
                clearSessionCards();
              }
            }}
            className="btn-frost btn-frost-danger px-3.5 py-2 text-[11px] font-medium uppercase tracking-[0.18em]"
          >
            clear
          </button>
        )}
      </div>
    </header>
  );
}

// --- Empty state ----------------------------------------------------------

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
        📚
      </div>
      <h2 className="serif-italic relative text-[22px] leading-tight text-white/95">
        the shelf is empty
      </h2>
      <p className="relative mt-3 text-[13px] leading-relaxed text-white/65">
        Point your camera at something small and tap. Everything you speak
        to gets painted into a little cartoon and set down here, still
        ready to talk.
      </p>
      <Link
        href="/"
        className="btn-primary relative mt-6 px-5 py-2.5 text-[12.5px] font-semibold"
      >
        <span>open the camera</span>
        <span aria-hidden>→</span>
      </Link>
    </div>
  );
}

// --- Bookshelf + drawer ---------------------------------------------------

function BookshelfGallery({ cards }: { cards: readonly SessionCard[] }) {
  // Newest first, so fresh items land on the top-left slot — the most
  // visible spot. Slots beyond MAX_VISIBLE_SLOTS fall back to a secondary
  // row below the shelf.
  const ordered = useMemo(() => [...cards].reverse(), [cards]);
  const visible = ordered.slice(0, MAX_VISIBLE_SLOTS);
  const overflow = ordered.slice(MAX_VISIBLE_SLOTS);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [status, setStatus] = useState<CardStatus>("idle");
  const [activeCardId, setActiveCardId] = useState<string | null>(null);
  const [liveReply, setLiveReply] = useState<string | null>(null);
  const [heard, setHeard] = useState<string | null>(null);
  const [shape, setShape] = useState<MouthShape>("X");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // --- Audio graph (shared across cards) --------------------------------

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

  const micStreamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const turnCounterRef = useRef(0);

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
      ctx.resume().catch(() => {});
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

  const stopCurrentPlayback = useCallback(() => {
    speakGenRef.current++;
    const audioEl = audioElRef.current;
    if (audioEl) {
      try { audioEl.pause(); } catch {}
      try {
        audioEl.removeAttribute("src");
        audioEl.load();
      } catch {}
    }
    const src = bufferSrcNodeRef.current;
    if (src) {
      try { src.stop(); } catch {}
      try { src.disconnect(); } catch {}
      bufferSrcNodeRef.current = null;
    }
    stopLipSyncLoop();
  }, [stopLipSyncLoop]);

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

  useEffect(() => {
    if (typeof document === "undefined") return;
    const onVisible = () => {
      if (document.visibilityState !== "visible") return;
      const ctx = audioCtxRef.current;
      if (ctx && ctx.state === "suspended") ctx.resume().catch(() => {});
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
      stopCurrentPlayback();
      const ctx = ensureAudioCtx();
      if (!ctx) {
        setErrorMsg("Web Audio unavailable in this browser");
        return;
      }
      const stream = await openMicStream();
      if (!stream) return;
      const mime =
        ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", ""].find(
          (t) => !t || MediaRecorder.isTypeSupported(t)
        ) ?? "";
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
      setHeard(null);
      setLiveReply(null);
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
      try { recorder.stop(); } catch {}
    },
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
      const history: ChatTurn[] =
        card.history && card.history.length > 0
          ? card.history.slice()
          : [{ role: "assistant", content: card.line }];

      try {
        const form = new FormData();
        const filename = blob.type.includes("mp4")
          ? "talk.mp4"
          : blob.type.includes("ogg")
            ? "talk.ogg"
            : "talk.webm";
        form.append("audio", blob, filename);
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

        if (transcript) setHeard(transcript);
        if (!reply) {
          setStatus("idle");
          setActiveCardId(null);
          setErrorMsg("no reply");
          return;
        }

        // Persist the turn to the session card store so the conversation
        // survives a route transition. appendCardHistory dedupes consecutive
        // identical assistant messages so opening-line + first reply don't
        // double up.
        const turns: ChatTurn[] = [];
        if (transcript) turns.push({ role: "user", content: transcript });
        turns.push({ role: "assistant", content: reply });
        appendCardHistory({ cardId: card.id }, turns);

        setLiveReply(reply);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [ensureAudioCtx]
  );

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
        try { await resp.body?.cancel(); } catch {}
        return;
      }
      if (!resp.ok || !resp.body) {
        const err = await resp.text().catch(() => "");
        throw new Error(`tts stream ${resp.status}: ${err.slice(0, 120)}`);
      }

      const analyser = ensureAnalyser(ctx);
      if (ctx.state === "suspended") {
        try { await ctx.resume(); } catch {}
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
        const audioBuf = await ctx.decodeAudioData(buf);
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
          try { source.start(); } catch { resolve(); }
        });
      }
    },
    [ensureAnalyser, startLipSyncLoop, stopLipSyncLoop]
  );

  // --- Selection ---------------------------------------------------------

  const selectedCard = useMemo(
    () => ordered.find((c) => c.id === selectedId) ?? null,
    [ordered, selectedId]
  );

  const onSelect = useCallback(
    (cardId: string) => {
      // Clicking the currently-selected card closes the drawer.
      setSelectedId((prev) => {
        if (prev === cardId) {
          stopCurrentPlayback();
          setStatus("idle");
          setActiveCardId(null);
          return null;
        }
        // Switching cards — stop any live audio from the previous one.
        stopCurrentPlayback();
        setStatus("idle");
        setActiveCardId(null);
        setLiveReply(null);
        setHeard(null);
        return cardId;
      });
    },
    [stopCurrentPlayback]
  );

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

      <section className="mx-auto w-full max-w-[1120px]">
        <Bookshelf
          cards={visible}
          selectedId={selectedId}
          activeCardId={activeCardId}
          status={status}
          onSelect={onSelect}
        />
      </section>

      {overflow.length > 0 && (
        <section className="mx-auto mt-8 w-full max-w-[1120px]">
          <p className="mb-2 px-1 text-[10px] uppercase tracking-[0.3em] text-white/40">
            older shelf
          </p>
          <OverflowRow
            cards={overflow}
            selectedId={selectedId}
            onSelect={onSelect}
            onRemove={(id) => removeSessionCard(id)}
          />
        </section>
      )}

      {selectedCard && (
        <section
          className="mx-auto mt-8 w-full max-w-[960px] drawer-in"
          key={selectedCard.id}
        >
          <ConversationDrawer
            card={selectedCard}
            status={activeCardId === selectedCard.id ? status : "idle"}
            shape={
              activeCardId === selectedCard.id && status === "speaking"
                ? shape
                : "X"
            }
            heard={heard}
            liveReply={liveReply}
            onMicPress={onMicPress}
            onMicRelease={onMicRelease}
            onRemove={() => {
              removeSessionCard(selectedCard.id);
              setSelectedId(null);
              stopCurrentPlayback();
            }}
            onClose={() => {
              setSelectedId(null);
              stopCurrentPlayback();
              setStatus("idle");
              setActiveCardId(null);
            }}
          />
        </section>
      )}
    </>
  );
}

// --- Bookshelf ------------------------------------------------------------

function Bookshelf({
  cards,
  selectedId,
  activeCardId,
  status,
  onSelect,
}: {
  cards: readonly SessionCard[];
  selectedId: string | null;
  activeCardId: string | null;
  status: CardStatus;
  onSelect: (cardId: string) => void;
}) {
  return (
    <div className="shelf-frame">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src="/bookshelf.png"
        alt=""
        className="shelf-bg"
        draggable={false}
      />
      {cards.map((card, i) => {
        const row = Math.floor(i / BAY_X.length);
        const col = i % BAY_X.length;
        const y = SHELF_ROWS_Y[row];
        const x = BAY_X[col];
        // Tiny per-card jitter so things don't look mechanically aligned.
        const seed = hashSeed(card.id);
        const jitterX = ((seed % 100) / 100 - 0.5) * 3.2; // ±1.6%
        const isSelected = selectedId === card.id;
        const isActive = activeCardId === card.id && status !== "idle";
        return (
          <ShelfItem
            key={card.id}
            card={card}
            // Stagger the enter animation so they pop in shelf-by-shelf.
            enterDelayMs={Math.min(i, 11) * 55}
            x={x + jitterX}
            yFromTop={y}
            isSelected={isSelected}
            isActive={isActive}
            onSelect={onSelect}
          />
        );
      })}
    </div>
  );
}

function ShelfItem({
  card,
  enterDelayMs,
  x,
  yFromTop,
  isSelected,
  isActive,
  onSelect,
}: {
  card: SessionCard;
  enterDelayMs: number;
  x: number;
  yFromTop: number;
  isSelected: boolean;
  isActive: boolean;
  onSelect: (cardId: string) => void;
}) {
  const [genLoaded, setGenLoaded] = useState(false);
  const [genFailed, setGenFailed] = useState(false);
  useEffect(() => {
    setGenLoaded(false);
    setGenFailed(false);
  }, [card.generatedImageUrl]);

  const genStatus = card.generatedImageStatus;
  const hasGen = !!card.generatedImageUrl && genStatus === "done" && !genFailed;
  const isPending = genStatus === "pending";

  return (
    <button
      type="button"
      className="shelf-slot focus-visible:outline-none"
      data-selected={isSelected ? "true" : "false"}
      data-active={isActive ? "true" : "false"}
      aria-label={`${cardDisplayName(card)} — tap to talk`}
      onClick={() => onSelect(card.id)}
      style={{
        left: `${x}%`,
        bottom: `${100 - yFromTop}%`,
        animationDelay: `${enterDelayMs}ms`,
      }}
    >
      {(isSelected || isActive) && <span aria-hidden className="shelf-halo" />}
      {hasGen ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={card.generatedImageUrl}
          alt={cardDisplayName(card)}
          className="shelf-cutout"
          data-loaded={genLoaded ? "true" : "false"}
          draggable={false}
          onLoad={() => setGenLoaded(true)}
          onError={() => setGenFailed(true)}
        />
      ) : isPending ? (
        <span className="shelf-painting">
          <span aria-hidden>✨ painting…</span>
        </span>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={card.imageDataUrl}
          alt={cardDisplayName(card)}
          className="shelf-fallback"
          draggable={false}
        />
      )}
    </button>
  );
}

// --- Overflow thumb row (items beyond shelf capacity) --------------------

function OverflowRow({
  cards,
  selectedId,
  onSelect,
  onRemove,
}: {
  cards: readonly SessionCard[];
  selectedId: string | null;
  onSelect: (cardId: string) => void;
  onRemove: (cardId: string) => void;
}) {
  return (
    <ul className="flex gap-3 overflow-x-auto pb-2">
      {cards.map((card) => {
        const src =
          card.generatedImageStatus === "done" && card.generatedImageUrl
            ? card.generatedImageUrl
            : card.imageDataUrl;
        const selected = card.id === selectedId;
        return (
          <li key={card.id} className="shrink-0">
            <button
              type="button"
              className={`group relative h-16 w-16 overflow-hidden rounded-2xl ring-1 transition ${
                selected
                  ? "ring-pink-300/70"
                  : "ring-white/15 hover:ring-white/35"
              }`}
              onClick={() => onSelect(card.id)}
              aria-label={cardDisplayName(card)}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={src}
                alt={cardDisplayName(card)}
                className="absolute inset-0 h-full w-full object-cover"
                draggable={false}
              />
              <span
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove(card.id);
                }}
                className="absolute right-1 top-1 grid h-4 w-4 place-items-center rounded-full bg-black/55 text-[10px] leading-none text-white/85 opacity-0 ring-1 ring-white/10 transition group-hover:opacity-100"
                role="button"
                aria-label="remove"
              >
                ×
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

// --- Conversation drawer --------------------------------------------------

function ConversationDrawer({
  card,
  status,
  shape,
  heard,
  liveReply,
  onMicPress,
  onMicRelease,
  onRemove,
  onClose,
}: {
  card: SessionCard;
  status: CardStatus;
  shape: MouthShape;
  heard: string | null;
  liveReply: string | null;
  onMicPress: (cardId: string) => void;
  onMicRelease: (cardId: string) => void;
  onRemove: () => void;
  onClose: () => void;
}) {
  const isSpeaking = status === "speaking";

  const history = card.history ?? [];
  // Opening line always shown first even when history is empty.
  const transcript: ChatTurn[] =
    history.length > 0
      ? history
      : [{ role: "assistant", content: card.line }];

  const scrollRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [transcript.length, liveReply]);

  const displayName = cardDisplayName(card);
  const portraitSrc =
    card.generatedImageStatus === "done" && card.generatedImageUrl
      ? card.generatedImageUrl
      : card.imageDataUrl;

  return (
    <article
      className={`relative overflow-hidden rounded-[32px] bg-white/[0.05] p-5 ring-1 ring-white/15 backdrop-blur-2xl sm:p-7 ${
        status !== "idle" ? "active-halo" : ""
      }`}
      style={{
        boxShadow:
          "0 40px 120px -40px rgba(255,137,190,0.45), inset 0 1px 0 rgba(255,255,255,0.07)",
      }}
    >
      <button
        type="button"
        aria-label="close"
        onClick={onClose}
        className="absolute right-4 top-4 z-20 grid h-8 w-8 place-items-center rounded-full bg-black/40 text-[13px] text-white/85 ring-1 ring-white/10 backdrop-blur-md transition hover:bg-black/60"
      >
        ×
      </button>

      <div className="grid gap-5 sm:grid-cols-[220px_1fr] sm:gap-7">
        {/* Portrait */}
        <div className="relative mx-auto aspect-square w-[200px] overflow-hidden rounded-3xl bg-gradient-to-br from-pink-200/25 to-violet-300/20 ring-1 ring-white/15 sm:mx-0 sm:w-full">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={portraitSrc}
            alt={displayName}
            className="absolute inset-0 h-full w-full object-cover"
            draggable={false}
          />
          {isSpeaking && (
            <div
              className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
              style={{
                width: FACE_VOICE_WIDTH * 0.62,
                height: FACE_VOICE_HEIGHT * 0.62,
              }}
            >
              <div
                style={{
                  width: FACE_VOICE_WIDTH,
                  height: FACE_VOICE_HEIGHT,
                  transform: "scale(0.62)",
                  transformOrigin: "0 0",
                }}
              >
                <FaceVoice shape={shape} />
              </div>
            </div>
          )}
        </div>

        {/* Conversation panel */}
        <div className="flex min-w-0 flex-col gap-4">
          <header className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <p className="text-[10px] uppercase tracking-[0.3em] text-white/45">
                still talking · {formatRelativeTime(card.createdAt)}
              </p>
              <h2
                className="serif-italic mt-1 truncate text-[28px] leading-[1.05] text-white/95"
                title={displayName}
                style={{ textShadow: "0 0 22px rgba(255,182,214,0.22)" }}
              >
                {displayName}
              </h2>
              <p className="mt-1 line-clamp-2 text-[12px] italic text-white/55">
                {card.description}
              </p>
            </div>
            <button
              type="button"
              onClick={onRemove}
              className="shrink-0 rounded-full bg-white/[0.05] px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-white/60 ring-1 ring-white/10 transition hover:bg-rose-500/20 hover:text-rose-100 hover:ring-rose-300/30"
            >
              remove
            </button>
          </header>

          <div
            ref={scrollRef}
            className="max-h-[220px] min-h-[80px] overflow-y-auto rounded-2xl bg-black/30 p-3 ring-1 ring-white/10"
          >
            <ul className="flex flex-col gap-2">
              {transcript.map((t, i) => (
                <li
                  key={i}
                  className={
                    t.role === "assistant"
                      ? "self-start max-w-[85%]"
                      : "self-end max-w-[85%]"
                  }
                >
                  <div
                    className={
                      t.role === "assistant"
                        ? "rounded-[18px] rounded-bl-md bg-white/[0.1] px-3 py-2 text-[13px] leading-[1.4] text-white/90 ring-1 ring-white/10"
                        : "rounded-[18px] rounded-br-md bg-pink-400/30 px-3 py-2 text-[13px] leading-[1.4] text-white ring-1 ring-pink-200/30"
                    }
                  >
                    {t.content}
                  </div>
                </li>
              ))}
              {status === "thinking" && (
                <li className="self-start">
                  <div className="flex items-center gap-1 rounded-[18px] bg-white/[0.08] px-3 py-2 ring-1 ring-white/10">
                    <Dot /> <Dot delay="0.15s" /> <Dot delay="0.3s" />
                  </div>
                </li>
              )}
              {status === "speaking" && liveReply && !historyIncludes(history, liveReply) && (
                <li className="self-start max-w-[85%]">
                  <div className="rounded-[18px] rounded-bl-md bg-white/[0.1] px-3 py-2 text-[13px] italic leading-[1.4] text-white/95 ring-1 ring-white/10">
                    {liveReply}
                  </div>
                </li>
              )}
            </ul>
          </div>

          {status === "recording" && (
            <p className="text-center text-[11.5px] uppercase tracking-[0.22em] text-pink-200/90">
              listening… release to send
            </p>
          )}
          {heard && status === "idle" && (
            <p className="truncate text-center text-[10.5px] uppercase tracking-[0.18em] text-white/40">
              you said:{" "}
              <span className="normal-case tracking-normal text-white/65">
                {heard}
              </span>
            </p>
          )}

          <div className="flex items-center justify-center pt-1">
            <HoldToTalkButton
              cardId={card.id}
              status={status}
              onPress={onMicPress}
              onRelease={onMicRelease}
            />
          </div>
        </div>
      </div>
    </article>
  );
}

function historyIncludes(history: readonly ChatTurn[], text: string): boolean {
  const last = history[history.length - 1];
  return !!last && last.role === "assistant" && last.content === text;
}

function Dot({ delay }: { delay?: string }) {
  return (
    <span
      aria-hidden
      className="inline-block h-1.5 w-1.5 rounded-full bg-white/60"
      style={{
        animation: "soft-pulse 1.2s ease-in-out infinite",
        animationDelay: delay ?? "0s",
      }}
    />
  );
}

// --- Hold-to-talk -------------------------------------------------------

function HoldToTalkButton({
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
  const recording = status === "recording";
  const busy = status === "thinking" || status === "speaking";

  return (
    <button
      type="button"
      aria-label={recording ? "release to send" : "hold to talk"}
      onPointerDown={(e) => {
        e.preventDefault();
        try { (e.currentTarget as Element).setPointerCapture(e.pointerId); } catch {}
        onPress(cardId);
      }}
      onPointerUp={(e) => {
        e.preventDefault();
        try { (e.currentTarget as Element).releasePointerCapture(e.pointerId); } catch {}
        onRelease(cardId);
      }}
      onPointerCancel={() => onRelease(cardId)}
      onPointerLeave={(e) => {
        if (e.buttons > 0) onRelease(cardId);
      }}
      className={`group relative inline-flex items-center gap-2.5 rounded-full px-6 py-3 text-[13px] font-semibold tracking-wide transition-[transform,box-shadow] duration-150 ${
        recording ? "mic-recording" : ""
      }`}
      style={{
        background: recording
          ? "linear-gradient(135deg, #ff5aa0 0%, #ff86be 100%)"
          : busy
            ? "rgba(255,255,255,0.16)"
            : "linear-gradient(135deg, #ffb8d6 0%, #ffd4e3 100%)",
        color: busy && !recording ? "rgba(255,255,255,0.88)" : "#3a0a29",
        boxShadow: recording
          ? undefined
          : "0 14px 30px -12px rgba(255,111,174,0.55)",
        transform: recording ? "scale(0.98)" : undefined,
        touchAction: "none",
      }}
    >
      <span
        aria-hidden
        className="grid h-7 w-7 place-items-center rounded-full"
        style={{
          background: recording
            ? "rgba(255,255,255,0.28)"
            : "rgba(255,255,255,0.65)",
        }}
      >
        <MicIcon />
      </span>
      <span className="min-w-[90px] text-left">
        {recording
          ? "release to send"
          : status === "thinking"
            ? "thinking…"
            : status === "speaking"
              ? "tap to stop"
              : "hold to talk"}
      </span>
    </button>
  );
}

function MicIcon() {
  return (
    <svg
      width="16"
      height="16"
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

// --- Helpers --------------------------------------------------------------

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

// --- MediaSource streaming helper ----------------------------------------
//
// Same shape as the tracker's playViaMediaSource but self-contained so the
// gallery page has no dependencies on tracker internals.

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
    try { await respBody.cancel(); } catch {}
    return;
  }

  const mediaSource = new MediaSource();
  const objectUrl = URL.createObjectURL(mediaSource);

  if (!audioSrcNodeRef.current) {
    try {
      audioSrcNodeRef.current = ctx.createMediaElementSource(audioEl);
      audioSrcNodeRef.current.connect(analyser);
    } catch {
      // already wired
    }
  }

  audioEl.src = objectUrl;

  const cleanup = (sourceBuffer: SourceBuffer | null) => {
    try {
      if (sourceBuffer && !sourceBuffer.updating) {
        try { mediaSource.removeSourceBuffer(sourceBuffer); } catch {}
      }
    } catch {}
    try { URL.revokeObjectURL(objectUrl); } catch {}
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
      try { sourceBuffer.appendBuffer(chunk); } catch {}
    } else if (ended && !sourceBuffer.updating) {
      try { mediaSource.endOfStream(); } catch {}
    }
  };

  sourceBuffer.addEventListener("updateend", () => {
    if (callGen !== speakGenRef.current) return;
    if (!playing) {
      playing = true;
      onStart();
      audioEl.play().catch(() => {});
    }
    pump();
  });

  const readLoop = async () => {
    while (true) {
      if (callGen !== speakGenRef.current) {
        try { await reader.cancel(); } catch {}
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
