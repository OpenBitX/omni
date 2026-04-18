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
  setCardGeneratedImage,
  setCardImageStatus,
  updateSessionCard,
  useSessionCards,
  type AppLang,
  type ChatTurn,
  type SessionCard,
} from "@/lib/session-cards";
import { gallerizeCard, teacherSay } from "@/app/actions";
import { useOnboardingPrefs } from "@/lib/onboarding";
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
// The gallery is reframed as a language-learning surface: each card is a
// bilingual learning card, and tapping one opens a teacher conversation
// where the object plays a native speaker of the user's LEARN language
// (opposite of their onboarding `spokenLang`).
//
// All Web Audio + mic + MediaSource plumbing is self-contained here (no
// imports from tracker.tsx). Only one card can speak at a time.

type CardStatus = "idle" | "recording" | "thinking" | "speaking";

// --- Web Speech API shims -------------------------------------------------
//
// The DOM lib doesn't always expose SpeechRecognition (it's a WebKit-origin
// API still gated behind the `webkitSpeechRecognition` prefix in most TS
// targets). We define a minimal structural type so this file typechecks
// without pulling in lib.dom.iterable.d.ts additions. Mirrors the shims in
// components/tracker.tsx.
interface GallerySRResultLike {
  0: { transcript: string };
  isFinal: boolean;
}
interface GallerySRResultListLike {
  readonly length: number;
  [index: number]: GallerySRResultLike;
}
interface GallerySREventLike extends Event {
  results: GallerySRResultListLike;
}
interface GallerySRLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start: () => void;
  stop: () => void;
  abort: () => void;
  onresult: ((e: GallerySREventLike) => void) | null;
  onerror: ((e: Event) => void) | null;
  onend: (() => void) | null;
}
type GallerySRCtor = new () => GallerySRLike;

function getGallerySRCtor(): GallerySRCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: GallerySRCtor;
    webkitSpeechRecognition?: GallerySRCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

// Flip the user's native → target. Kept pure so we can call it anywhere
// without reading hooks.
function oppositeLang(lang: AppLang): AppLang {
  return lang === "zh" ? "en" : "zh";
}

// Shelf geometry — the bookshelf image has 4 shelves × 3 bays. Percentages
// are the bottom-edge of each shelf slot, measured from the TOP of the
// bookshelf image. CSS then uses `bottom: (100 - shelfTop%)%` so items sit
// on the shelf plank. Pillars run at roughly x=33% and x=67%; the bay
// centers avoid them.
const SHELF_ROWS_Y = [34, 52, 70, 89] as const; // % from top of bookshelf
const BAYS_THREE = [17, 50, 83] as const;
const BAYS_TWO = [20, 80] as const;
const BAYS_ONE = [50] as const;

const MAX_VISIBLE_SLOTS = SHELF_ROWS_Y.length * BAYS_THREE.length; // 12

// Given N cards, return N (x,y) slots distributed across as few shelves
// as possible but still spread horizontally — a single card centers on
// the top shelf instead of sliding into the top-left bay and leaving the
// rest empty. Rows fill top-down so newest (index 0) always lands where
// the eye goes first.
function assignShelfSlots(n: number): { x: number; y: number }[] {
  if (n <= 0) return [];
  const clamped = Math.min(n, MAX_VISIBLE_SLOTS);
  const rows = Math.min(SHELF_ROWS_Y.length, Math.ceil(clamped / 3));
  const perRow: number[] = [];
  let remaining = clamped;
  for (let r = 0; r < rows; r++) {
    const rowsLeft = rows - r;
    const take = Math.min(3, Math.ceil(remaining / rowsLeft));
    perRow.push(take);
    remaining -= take;
  }
  const out: { x: number; y: number }[] = [];
  for (let r = 0; r < rows; r++) {
    const count = perRow[r];
    const y = SHELF_ROWS_Y[r];
    const bays =
      count === 1 ? BAYS_ONE : count === 2 ? BAYS_TWO : BAYS_THREE;
    for (const bx of bays) out.push({ x: bx, y });
  }
  return out;
}

// Inline SVG defs — referenced by .shelf-cutout via filter: url(#whiteToAlpha).
// Maps near-white pixels to alpha=0, preserving edges on every other color.
// Alpha = -1.2·r - 1.2·g - 1.2·b + 3.6 (clamped). White → 0. Gray+ → 1.
function SvgFilters() {
  return (
    <svg
      aria-hidden
      width="0"
      height="0"
      style={{ position: "absolute", width: 0, height: 0 }}
    >
      <defs>
        <filter
          id="whiteToAlpha"
          x="-10%"
          y="-10%"
          width="120%"
          height="120%"
          colorInterpolationFilters="sRGB"
        >
          <feColorMatrix
            type="matrix"
            values="1 0 0 0 0
                    0 1 0 0 0
                    0 0 1 0 0
                    -1.2 -1.2 -1.2 0 3.6"
          />
        </filter>
      </defs>
    </svg>
  );
}

export default function GalleryPage() {
  const cards = useSessionCards();
  const prefs = useOnboardingPrefs();
  const hasCards = cards.length > 0;

  // spokenLang comes from onboarding; learnLang is always the opposite.
  // Gallery is a bilingual surface, so we derive both here and pass them
  // down to the bookshelf / drawer so every call to teacherSay + /api/tts/stream
  // reflects the user's CURRENT preference (not whatever was snapshotted
  // onto the card at capture time).
  const spokenLang: AppLang = prefs.spokenLang;
  const learnLang: AppLang = oppositeLang(spokenLang);

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
      <SvgFilters />
      <div className="relative z-10">
        <Header hasCards={hasCards} />
        <main className="mx-auto w-full max-w-[1280px] px-4 pb-16 pt-4 sm:px-6">
          {hasCards ? (
            <BookshelfGallery
              cards={cards}
              spokenLang={spokenLang}
              learnLang={learnLang}
            />
          ) : (
            <EmptyState />
          )}
        </main>
      </div>
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

function BookshelfGallery({
  cards,
  spokenLang,
  learnLang,
}: {
  cards: readonly SessionCard[];
  spokenLang: AppLang;
  learnLang: AppLang;
}) {
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

  // --- Lazy bilingual fill ---------------------------------------------
  //
  // On mount (and whenever the card list grows), fill missing
  // bilingualIntro / translatedName via the `gallerizeCard` action. Bounded
  // to 3 in flight at once so we don't blast the reply backend with ten
  // simultaneous requests on a fresh-load of a busy session. Cards that
  // already have both fields are skipped — work is persisted through
  // updateSessionCard, so we only pay it once.
  const hydratingRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    // Skip entirely when the user happens to speak both languages (never
    // happens today since onboarding only offers en + zh, but guard
    // anyway).
    if (spokenLang === learnLang) return;

    const missing = cards.filter(
      (c) =>
        !hydratingRef.current.has(c.id) &&
        (!c.bilingualIntro ||
          !c.bilingualIntro.learn ||
          !c.bilingualIntro.spoken ||
          !c.translatedName)
    );
    if (missing.length === 0) return;

    let cancelled = false;
    const MAX_CONCURRENCY = 3;
    const queue = missing.slice();

    const runOne = async (): Promise<void> => {
      while (!cancelled) {
        const card = queue.shift();
        if (!card) return;
        if (hydratingRef.current.has(card.id)) continue;
        hydratingRef.current.add(card.id);
        try {
          const result = await gallerizeCard({
            description: card.description,
            objectName: card.objectName ?? null,
            className: card.className,
            spokenLang,
            learnLang,
          });
          if (cancelled) return;
          updateSessionCard(card.id, {
            bilingualIntro: result.bilingualIntro,
            translatedName: result.translatedName,
          });
        } catch {
          // Leave the card un-hydrated; the UI shows a skeleton and we'll
          // retry next mount.
        } finally {
          hydratingRef.current.delete(card.id);
        }
      }
    };

    const workers: Promise<void>[] = [];
    for (let i = 0; i < Math.min(MAX_CONCURRENCY, queue.length); i++) {
      workers.push(runOne());
    }
    void Promise.all(workers);

    return () => {
      cancelled = true;
    };
  }, [cards, spokenLang, learnLang]);

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
  // Teach mode latches on the first user utterance that asks to learn the
  // OTHER language ("教我英文", "how do you say…"). Once latched, every
  // subsequent turn in this gallery mount stays in teach mode — primarily
  // spokenLang with one short learnLang teaching phrase embedded. Resets
  // on unmount (i.e. when the user leaves /gallery).
  const [teachModeLatched, setTeachModeLatched] = useState(false);
  // Web Speech API for live transcription — runs in parallel with
  // MediaRecorder. MediaRecorder is retained for the recording indicator
  // UX, Web Speech is what actually produces the userText we send to
  // teacherSay. Chrome + Safari back it; Firefox drops through with empty
  // string and we surface a toast.
  const recognitionRef = useRef<GallerySRLike | null>(null);
  const lastTranscriptRef = useRef<string>("");
  // Resolves on the SR's `onend`, so `stopWebSpeech` can wait for the
  // recognizer to flush its in-progress segment as a final result before
  // we read the transcript. Without this, releasing the mic right after
  // speaking returns an empty string ~half the time on Chrome.
  const speechFinishedRef = useRef<Promise<void> | null>(null);
  const resolveSpeechFinishedRef = useRef<(() => void) | null>(null);

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

  const startWebSpeech = useCallback((lang: AppLang): void => {
    const SR = getGallerySRCtor();
    if (!SR) {
      recognitionRef.current = null;
      return;
    }
    try {
      const rec = new SR();
      rec.lang = lang === "zh" ? "zh-CN" : "en-US";
      rec.interimResults = true;
      rec.continuous = true;
      rec.maxAlternatives = 1;
      lastTranscriptRef.current = "";
      speechFinishedRef.current = new Promise<void>((resolve) => {
        resolveSpeechFinishedRef.current = resolve;
      });
      rec.onresult = (e: GallerySREventLike) => {
        // Rebuild the full transcript every event (results array is
        // cumulative). Interim segments are appended to finalized ones so
        // the live caption updates word-by-word as the user speaks.
        let finalText = "";
        let interim = "";
        for (let i = 0; i < e.results.length; i++) {
          const res = e.results[i];
          const t = res[0]?.transcript ?? "";
          if (!t) continue;
          if (res.isFinal) finalText += (finalText ? " " : "") + t;
          else interim += t;
        }
        const live = (finalText + (interim ? (finalText ? " " : "") + interim : "")).trim();
        if (live) {
          lastTranscriptRef.current = live;
          setHeard(live);
        }
      };
      rec.onerror = () => {
        // Stay silent — onRelease will detect an empty transcript and
        // surface a toast.
      };
      rec.onend = () => {
        resolveSpeechFinishedRef.current?.();
        resolveSpeechFinishedRef.current = null;
      };
      rec.start();
      recognitionRef.current = rec;
    } catch {
      recognitionRef.current = null;
      speechFinishedRef.current = null;
      resolveSpeechFinishedRef.current = null;
    }
  }, []);

  const stopWebSpeech = useCallback(async (): Promise<string> => {
    const rec = recognitionRef.current;
    recognitionRef.current = null;
    if (rec) {
      try { rec.stop(); } catch {}
    }
    // Wait for the recognizer to flush its in-progress segment as a final
    // result. Capped at 600ms so a stuck recognizer can't lock the UI; the
    // last interim is still in lastTranscriptRef as a fallback.
    const finished = speechFinishedRef.current;
    if (finished) {
      await Promise.race([
        finished,
        new Promise<void>((r) => setTimeout(r, 600)),
      ]);
    }
    speechFinishedRef.current = null;
    resolveSpeechFinishedRef.current = null;
    const text = lastTranscriptRef.current.trim();
    lastTranscriptRef.current = "";
    return text;
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
      // Web Speech runs alongside MediaRecorder. We read its transcript on
      // release; the blob is discarded (teacher is text-only).
      startWebSpeech(spokenLang);
      recorder.start();
    },
    [ensureAudioCtx, openMicStream, stopCurrentPlayback, startWebSpeech, spokenLang]
  );

  // --- Conversation: TTS + teacher turn ---------------------------------

  const streamTTS = useCallback(
    async (args: {
      ctx: AudioContext;
      text: string;
      voiceId: string;
      turnId: string;
      callGen: number;
      ttsLang: AppLang;
    }): Promise<void> => {
      const { ctx, text, voiceId, turnId, callGen, ttsLang } = args;
      const resp = await fetch("/api/tts/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          voiceId,
          turnId,
          // Always the CURRENT learnLang — the card's snapshot is ignored on
          // purpose so a user who switches their learn language in
          // onboarding retroactively hears the new language.
          lang: ttsLang,
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

  // Runs a teacher turn on mic release. userText is ALWAYS non-empty here
  // (null path removed — no auto-intro). Passes the latched teach flag
  // through to the server; server re-checks detectTeachMode on the fresh
  // utterance so the latch can flip on a "teach me" ask we haven't seen
  // yet. When the server resolves teachMode=true, we latch it locally.
  const runTeacherTurn = useCallback(
    async (card: SessionCard, userText: string) => {
      const ctx = ensureAudioCtx();
      if (!ctx) {
        setStatus("idle");
        setActiveCardId(null);
        setErrorMsg("Web Audio unavailable");
        return;
      }
      const callGen = ++speakGenRef.current;
      const turnId = String(++turnCounterRef.current);
      const history: ChatTurn[] = card.history ? card.history.slice() : [];

      try {
        setStatus("thinking");
        setActiveCardId(card.id);
        const {
          line,
          voiceId: replyVoiceId,
          teachMode: resolvedTeachMode,
        } = await teacherSay({
          description: card.description,
          className: card.className,
          objectName: card.objectName ?? null,
          spokenLang,
          learnLang,
          userText,
          history,
          voiceId: card.voiceId,
          turnId,
          teachMode: teachModeLatched,
        });
        if (callGen !== speakGenRef.current) return;
        if (!line) {
          setStatus("idle");
          setActiveCardId(null);
          setErrorMsg("no reply");
          return;
        }

        // Latch teach mode on the first turn the server resolves to true.
        // Stays on for the rest of this gallery mount (natural reset on
        // navigation away).
        if (resolvedTeachMode && !teachModeLatched) {
          setTeachModeLatched(true);
        }

        // Persist both turns (user then assistant).
        appendCardHistory({ cardId: card.id }, [
          { role: "user", content: userText },
          { role: "assistant", content: line },
        ]);

        setLiveReply(line);
        setStatus("speaking");
        await streamTTS({
          ctx,
          text: line,
          voiceId: replyVoiceId,
          turnId,
          callGen,
          // TTS always renders in spokenLang — even in teach mode the
          // reply is primarily spokenLang, with at most one embedded
          // learnLang phrase that Cartesia's monolingual voice can
          // handle with mild accent.
          ttsLang: spokenLang,
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
    [ensureAudioCtx, streamTTS, spokenLang, learnLang, teachModeLatched]
  );

  const onMicRelease = useCallback(
    (cardId: string) => {
      const recorder = recorderRef.current;
      if (!recorder || recorder.state !== "recording") return;
      recorder.onstop = async () => {
        const blob = new Blob(recordedChunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        recordedChunksRef.current = [];
        recorderRef.current = null;
        // Read the live transcript collected by the Web Speech recognizer
        // while the user was holding. Awaits SR's `onend` so the
        // in-progress segment finalizes before we read it. The blob is
        // only used as a length heuristic — the teacher call is text-only.
        const transcript = await stopWebSpeech();
        if (blob.size < 1024) {
          setStatus("idle");
          setActiveCardId(null);
          setErrorMsg("too short — hold the button longer");
          return;
        }
        if (!transcript) {
          setStatus("idle");
          setActiveCardId(null);
          setErrorMsg("didn't catch that — try again");
          return;
        }
        const card = cards.find((c) => c.id === cardId);
        if (!card) {
          setStatus("idle");
          setActiveCardId(null);
          return;
        }
        setHeard(transcript);
        setStatus("thinking");
        void runTeacherTurn(card, transcript);
      };
      try { recorder.stop(); } catch {}
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [cards, stopWebSpeech]
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

  // No auto-intro: the drawer opens silently and waits for the user to
  // press hold-to-talk. The teacher only speaks in response to the user.

  // --- Paint-on-demand --------------------------------------------------
  //
  // Cards with no generated image, or whose previous generation failed,
  // get a small "paint" button (on the shelf tile and in the drawer).
  // Fires `/api/runware/generate` off the client, flipping the card's
  // status through the session-cards store so every consumer — bookshelf
  // tile, drawer portrait, tracker if it's open in another tab — updates
  // in lockstep.
  const paintingRef = useRef(new Set<string>());
  const onPaint = useCallback(async (card: SessionCard) => {
    if (paintingRef.current.has(card.id)) return;
    paintingRef.current.add(card.id);
    setCardImageStatus(card.id, "pending");
    try {
      const resp = await fetch("/api/runware/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cardId: card.id,
          className: card.objectName?.trim() || card.className,
          description: card.description,
          history: (card.history ?? []).slice(-8),
          spokenLang: card.spokenLang,
          learnLang: card.learnLang,
          imageDataUrl: card.imageDataUrl,
        }),
      });
      if (!resp.ok) {
        const err = await resp.text().catch(() => "");
        setCardImageStatus(card.id, "failed", {
          error: err.slice(0, 160) || `paint failed (${resp.status})`,
        });
        return;
      }
      const j = (await resp.json()) as {
        imageUrl?: string;
        prompt?: string;
      };
      if (!j.imageUrl) {
        setCardImageStatus(card.id, "failed", { error: "no image url" });
        return;
      }
      setCardGeneratedImage(card.id, j.imageUrl, j.prompt ?? "");
    } catch (e) {
      setCardImageStatus(card.id, "failed", {
        error: e instanceof Error ? e.message : "paint failed",
      });
    } finally {
      paintingRef.current.delete(card.id);
    }
  }, []);

  // Drop selection if the card is removed. Keeps the drawer from pointing
  // at nothing.
  useEffect(() => {
    if (!selectedId) return;
    if (!ordered.some((c) => c.id === selectedId)) {
      setSelectedId(null);
      stopCurrentPlayback();
      setStatus("idle");
      setActiveCardId(null);
    }
  }, [ordered, selectedId, stopCurrentPlayback]);

  // Esc closes the drawer or cancels an in-flight recording.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (recorderRef.current && recorderRef.current.state === "recording") {
        try { recorderRef.current.stop(); } catch {}
        recorderRef.current = null;
        recordedChunksRef.current = [];
        setStatus("idle");
        setActiveCardId(null);
        return;
      }
      if (selectedId) {
        setSelectedId(null);
        stopCurrentPlayback();
        setStatus("idle");
        setActiveCardId(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedId, stopCurrentPlayback]);

  // When a card is selected, ease the drawer into view so the mic is
  // reachable without a hunt-scroll.
  const drawerRef = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (!selectedId) return;
    const el = drawerRef.current;
    if (!el) return;
    const t = window.setTimeout(() => {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 120);
    return () => window.clearTimeout(t);
  }, [selectedId]);

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
          onPaint={onPaint}
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
          ref={drawerRef}
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
            onInterruptSpeaking={stopCurrentPlayback}
            onPaint={() => onPaint(selectedCard)}
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
  onPaint,
}: {
  cards: readonly SessionCard[];
  selectedId: string | null;
  activeCardId: string | null;
  status: CardStatus;
  onSelect: (cardId: string) => void;
  onPaint: (card: SessionCard) => void;
}) {
  const slots = useMemo(() => assignShelfSlots(cards.length), [cards.length]);
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
        const slot = slots[i];
        if (!slot) return null;
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
            x={slot.x + jitterX}
            yFromTop={slot.y}
            isSelected={isSelected}
            isActive={isActive}
            onSelect={onSelect}
            onPaint={onPaint}
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
  onPaint,
}: {
  card: SessionCard;
  enterDelayMs: number;
  x: number;
  yFromTop: number;
  isSelected: boolean;
  isActive: boolean;
  onSelect: (cardId: string) => void;
  onPaint: (card: SessionCard) => void;
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
  const canPaint = !hasGen && !isPending; // includes "idle", "failed", undefined
  const displayName = cardDisplayName(card);
  const translated = (card.translatedName ?? "").trim();
  const intro = card.bilingualIntro;

  return (
    <button
      type="button"
      className="shelf-slot focus-visible:outline-none"
      data-selected={isSelected ? "true" : "false"}
      data-active={isActive ? "true" : "false"}
      aria-label={`${displayName} — tap to talk`}
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
          alt={displayName}
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
          alt={displayName}
          className="shelf-fallback"
          draggable={false}
        />
      )}
      {/* Bilingual label + intro chip under the cutout. Pointer-events
          disabled so the whole tile still registers a tap through to the
          button. */}
      <span
        aria-hidden
        className="shelf-label"
        style={{
          position: "absolute",
          left: "50%",
          bottom: -4,
          transform: "translate(-50%, 100%)",
          width: "min(220px, 160%)",
          pointerEvents: "none",
        }}
      >
        <span
          className="block truncate text-center font-semibold"
          style={{
            fontSize: 11.5,
            lineHeight: 1.15,
            color: "rgba(255,255,255,0.95)",
            textShadow: "0 1px 2px rgba(0,0,0,0.55)",
          }}
          title={translated ? `${displayName} · ${translated}` : displayName}
        >
          {displayName}
          {translated ? (
            <>
              {" "}
              <span style={{ color: "rgba(255,200,225,0.85)" }}>· {translated}</span>
            </>
          ) : null}
        </span>
        {intro ? (
          <span
            className="mt-0.5 block text-center"
            style={{
              fontSize: 10,
              lineHeight: 1.25,
              color: "rgba(255,255,255,0.85)",
              textShadow: "0 1px 2px rgba(0,0,0,0.55)",
            }}
          >
            <span className="block truncate" title={intro.learn}>
              {intro.learn}
            </span>
            <span
              className="block truncate"
              style={{ color: "rgba(255,255,255,0.55)" }}
              title={intro.spoken}
            >
              {intro.spoken}
            </span>
          </span>
        ) : (
          <span
            className="mt-0.5 block text-center"
            style={{
              fontSize: 10,
              color: "rgba(255,255,255,0.55)",
              textShadow: "0 1px 2px rgba(0,0,0,0.55)",
            }}
          >
            <span className="inline-block h-2 w-16 animate-pulse rounded-full bg-white/20" />
          </span>
        )}
      </span>
      {canPaint && (
        <span
          role="button"
          aria-label="repaint cartoon"
          title={
            genStatus === "failed"
              ? card.generatedImageError ?? "paint failed — retry"
              : "paint a cartoon"
          }
          className="shelf-chip"
          style={{ right: 2, top: 2 }}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onPaint(card);
          }}
          onPointerDown={(e) => e.stopPropagation()}
        >
          ✦ paint
        </span>
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
        const displayName = cardDisplayName(card);
        const translated = (card.translatedName ?? "").trim();
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
              aria-label={
                translated
                  ? `${displayName} · ${translated}`
                  : displayName
              }
              title={
                translated
                  ? `${displayName} · ${translated}`
                  : displayName
              }
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={src}
                alt={displayName}
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
  onInterruptSpeaking,
  onPaint,
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
  onInterruptSpeaking: () => void;
  onPaint: () => void;
  onRemove: () => void;
  onClose: () => void;
}) {
  const isSpeaking = status === "speaking";
  const genStatus = card.generatedImageStatus;
  const genPending = genStatus === "pending";
  const genFailed = genStatus === "failed";
  const hasGen =
    !!card.generatedImageUrl && genStatus === "done";

  const history = card.history ?? [];
  // The drawer always walks the persisted history. Intro turns are pushed
  // into history by runTeacherTurn, so there's no "virtual opening line"
  // to inject — empty history means we haven't greeted yet (intro is
  // about to fire).
  const transcript: ChatTurn[] = history;

  const scrollRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [transcript.length, liveReply]);

  const displayName = cardDisplayName(card);
  const translated = (card.translatedName ?? "").trim();
  const intro = card.bilingualIntro;
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
                learning · {formatRelativeTime(card.createdAt)}
              </p>
              <h2
                className="serif-italic mt-1 truncate text-[28px] leading-[1.05] text-white/95"
                title={
                  translated
                    ? `${displayName} · ${translated}`
                    : displayName
                }
                style={{ textShadow: "0 0 22px rgba(255,182,214,0.22)" }}
              >
                {displayName}
                {translated ? (
                  <span className="ml-2 text-[18px] not-italic text-pink-200/85">
                    · {translated}
                  </span>
                ) : null}
              </h2>
              {intro ? (
                <div className="mt-1 space-y-0.5 text-[12.5px] leading-snug">
                  <p className="text-white/90">{intro.learn}</p>
                  <p className="italic text-white/55">{intro.spoken}</p>
                </div>
              ) : (
                <p className="mt-1 line-clamp-2 text-[12px] italic text-white/55">
                  {card.description}
                </p>
              )}
            </div>
            <div className="flex shrink-0 items-center gap-2">
              {(!hasGen || genFailed) && !genPending && (
                <button
                  type="button"
                  onClick={onPaint}
                  title={
                    genFailed
                      ? card.generatedImageError ?? "paint failed — retry"
                      : "paint a cartoon"
                  }
                  className="rounded-full bg-pink-400/20 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-pink-100 ring-1 ring-pink-300/30 transition hover:bg-pink-400/35"
                >
                  ✦ paint
                </button>
              )}
              {genPending && (
                <span className="rounded-full bg-white/[0.08] px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-white/70 ring-1 ring-white/10">
                  painting…
                </span>
              )}
              <button
                type="button"
                onClick={onRemove}
                className="rounded-full bg-white/[0.05] px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-white/60 ring-1 ring-white/10 transition hover:bg-rose-500/20 hover:text-rose-100 hover:ring-rose-300/30"
              >
                remove
              </button>
            </div>
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

          {/* Fixed-height status slot above the mic. The slot is always
              rendered (even when empty) so the button below it stays
              pinned to the same y-coordinate across status changes —
              otherwise "listening…" appearing would shove the mic down
              and the user's finger off it mid-press. */}
          <div className="flex min-h-[44px] flex-col items-center justify-center text-center">
            {status === "recording" ? (
              <>
                <p className="text-[11.5px] uppercase tracking-[0.22em] text-pink-200/90">
                  listening… release to send
                </p>
                {heard && (
                  <p className="mt-1 line-clamp-2 px-2 text-[12.5px] italic text-white/85">
                    {heard}
                  </p>
                )}
              </>
            ) : heard && (status === "idle" || status === "thinking") ? (
              <p className="truncate px-2 text-[10.5px] uppercase tracking-[0.18em] text-white/40">
                you said:{" "}
                <span className="normal-case tracking-normal text-white/65">
                  {heard}
                </span>
              </p>
            ) : null}
          </div>

          <div className="flex items-center justify-center pt-1">
            <HoldToTalkButton
              cardId={card.id}
              status={status}
              onPress={onMicPress}
              onRelease={onMicRelease}
              onInterruptSpeaking={onInterruptSpeaking}
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

// Max time we'll record before auto-releasing. A user holding for 30s
// has either fallen asleep on the button or hit a stuck-event bug; either
// way we should ship the audio rather than leak the mic forever.
const MAX_RECORDING_MS = 30_000;
// Sub-180ms presses on a SPEAKING object are treated as interrupt-only
// taps, not zero-length recordings. The same threshold from idle still
// commits the recording so a fast double-tap doesn't silently leak a
// mic stream.
const INTERRUPT_TAP_MS = 180;

function HoldToTalkButton({
  cardId,
  status,
  onPress,
  onRelease,
  onInterruptSpeaking,
}: {
  cardId: string;
  status: CardStatus;
  onPress: (cardId: string) => void;
  onRelease: (cardId: string) => void;
  onInterruptSpeaking: () => void;
}) {
  const recording = status === "recording";
  const speaking = status === "speaking";
  const thinking = status === "thinking";
  const busy = thinking || speaking;

  const buttonRef = useRef<HTMLButtonElement>(null);
  const pressStartRef = useRef<number>(0);
  // What the status was at the instant the user pressed. Used by the
  // release path so the interrupt-tap heuristic doesn't read a state
  // that's already changed mid-press.
  const pressStatusRef = useRef<CardStatus>("idle");
  // Single source of truth for "is a press currently held". Gates the
  // release path so onPointerUp/onPointerCancel/keyboardUp can't all
  // each fire onRelease and double-stop the recorder.
  const isHeldRef = useRef(false);
  const safetyTimerRef = useRef<number | null>(null);
  const [holdMs, setHoldMs] = useState(0);

  const beginHold = useCallback(() => {
    if (isHeldRef.current) return;
    // While a reply is in flight we treat the press as a no-op so the
    // returning audio doesn't race a fresh recording. The interrupt
    // affordance (tap during speaking) is still honored below.
    if (thinking) return;
    isHeldRef.current = true;
    pressStartRef.current = performance.now();
    pressStatusRef.current = status;
    setHoldMs(0);
    // Confirm the press registered. iOS only honors vibrate inside a
    // user-gesture handler, which this is.
    if (typeof navigator.vibrate === "function") {
      try { navigator.vibrate(8); } catch {}
    }
    if (speaking) onInterruptSpeaking();
    onPress(cardId);
    if (safetyTimerRef.current != null) {
      window.clearTimeout(safetyTimerRef.current);
    }
    safetyTimerRef.current = window.setTimeout(() => {
      // Auto-release at the cap. finishHold dedupes so a real release
      // arriving moments later is a no-op.
      finishHold();
    }, MAX_RECORDING_MS);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cardId, onPress, onInterruptSpeaking, speaking, thinking, status]);

  const finishHold = useCallback(() => {
    if (!isHeldRef.current) return;
    isHeldRef.current = false;
    if (safetyTimerRef.current != null) {
      window.clearTimeout(safetyTimerRef.current);
      safetyTimerRef.current = null;
    }
    const held = performance.now() - pressStartRef.current;
    // A short tap that began while the object was speaking is treated as
    // interrupt-only — the playback was already stopped in beginHold and
    // there's no recording to commit. Pressing from idle always commits,
    // even on a fast tap, so the parent's "too short" toast can guide the
    // user instead of silently leaking the mic stream we just opened.
    if (held < INTERRUPT_TAP_MS && pressStatusRef.current === "speaking") {
      return;
    }
    onRelease(cardId);
    setHoldMs(0);
  }, [cardId, onRelease]);

  // Live duration counter while recording. Only spins the RAF when
  // actually recording — idle button costs zero frames.
  useEffect(() => {
    if (!recording) {
      setHoldMs(0);
      return;
    }
    const start = pressStartRef.current || performance.now();
    let raf = 0;
    const tick = () => {
      setHoldMs(performance.now() - start);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [recording]);

  // Keyboard hold-to-talk on space/enter — only when the button itself is
  // focused, so it doesn't hijack typing elsewhere on the page. e.repeat
  // is filtered so OS key-repeat doesn't fire beginHold a hundred times.
  useEffect(() => {
    const target = () => document.activeElement === buttonRef.current;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.repeat) return;
      if (!target()) return;
      if (e.key !== " " && e.key !== "Enter") return;
      e.preventDefault();
      beginHold();
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (!target()) return;
      if (e.key !== " " && e.key !== "Enter") return;
      e.preventDefault();
      finishHold();
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [beginHold, finishHold]);

  // If the tab is hidden mid-record, finish so we don't leak audio while
  // the user is somewhere else. Same idea for window blur.
  useEffect(() => {
    const onHide = () => {
      if (isHeldRef.current) finishHold();
    };
    document.addEventListener("visibilitychange", onHide);
    window.addEventListener("blur", onHide);
    return () => {
      document.removeEventListener("visibilitychange", onHide);
      window.removeEventListener("blur", onHide);
    };
  }, [finishHold]);

  // Tear down the safety timer if the component unmounts mid-hold.
  useEffect(() => {
    return () => {
      if (safetyTimerRef.current != null) {
        window.clearTimeout(safetyTimerRef.current);
        safetyTimerRef.current = null;
      }
    };
  }, []);

  const seconds = Math.floor(holdMs / 1000);
  const remaining = Math.max(0, Math.ceil((MAX_RECORDING_MS - holdMs) / 1000));
  const nearMax = recording && remaining <= 5;

  return (
    <button
      ref={buttonRef}
      type="button"
      aria-pressed={recording}
      aria-label={
        recording
          ? "release to send"
          : speaking
            ? "tap to interrupt or hold to talk"
            : thinking
              ? "thinking"
              : "hold to talk"
      }
      onPointerDown={(e) => {
        // Ignore non-primary buttons (right/middle click on desktop).
        if (e.pointerType === "mouse" && e.button !== 0) return;
        e.preventDefault();
        try { (e.currentTarget as Element).setPointerCapture(e.pointerId); } catch {}
        beginHold();
      }}
      onPointerUp={(e) => {
        e.preventDefault();
        try { (e.currentTarget as Element).releasePointerCapture(e.pointerId); } catch {}
        finishHold();
      }}
      onPointerCancel={(e) => {
        try { (e.currentTarget as Element).releasePointerCapture(e.pointerId); } catch {}
        finishHold();
      }}
      // Intentionally NO onPointerLeave: with setPointerCapture above,
      // the element keeps receiving pointer events even when the finger
      // drifts off the button's box. Treating leave as a release was
      // killing recordings on tiny finger movement.
      onContextMenu={(e) => e.preventDefault()}
      // Native click handler can fire after pointerup on some Android
      // browsers; we've already handled release in pointerup, so swallow
      // it to avoid a phantom second release path.
      onClick={(e) => e.preventDefault()}
      // Prevent the browser's own keydown→click translation (which
      // would fire onClick once on space/enter), so our keyboard hold
      // handlers above are the only path.
      onKeyDown={(e) => {
        if (e.key === " " || e.key === "Enter") e.preventDefault();
      }}
      onKeyUp={(e) => {
        if (e.key === " " || e.key === "Enter") e.preventDefault();
      }}
      className={`group relative inline-flex select-none items-center gap-2.5 rounded-full px-6 py-3 text-[13px] font-semibold tracking-wide transition-[box-shadow,background-color] duration-150 ${
        recording ? "mic-recording" : ""
      } ${nearMax ? "animate-pulse" : ""}`}
      // Drag/select on the button itself is also blocked at the
      // attribute level — belt-and-braces with the CSS below — because
      // some browsers honor draggable=false even when user-select isn't
      // respected (e.g. Safari with images inside the button).
      draggable={false}
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
        // Intentionally NO transform on press: the button must read as a
        // physical object that doesn't shift under the finger. Pulse ring
        // + color swap are the press feedback.
        // touchAction:none stops the browser from interpreting the press
        // as a scroll/pan gesture and stealing the pointer away.
        touchAction: "none",
        // Bulletproof selection prevention: kills iOS's long-press
        // magnifier, the "copy/share" callout, the blue tap-highlight
        // flash, and any caret that would land on the label text.
        userSelect: "none",
        WebkitUserSelect: "none",
        MozUserSelect: "none",
        msUserSelect: "none",
        WebkitTouchCallout: "none",
        WebkitTapHighlightColor: "transparent",
        cursor: thinking ? "not-allowed" : recording ? "grabbing" : "pointer",
        opacity: thinking ? 0.85 : 1,
      }}
    >
      <span
        aria-hidden
        className="grid h-7 w-7 shrink-0 place-items-center rounded-full"
        style={{
          background: recording
            ? "rgba(255,255,255,0.28)"
            : "rgba(255,255,255,0.65)",
          // Pointer events go to the button, never the inner chip — so a
          // finger that lands on the icon edge doesn't get treated as a
          // separate target.
          pointerEvents: "none",
        }}
      >
        <MicIcon />
      </span>
      {/* Fixed-width label slot so the icon stays put as the text swaps
          between "hold to talk" / "release · Ns" / "thinking…". Centered
          so neither the longest nor shortest label nudges anything. */}
      <span
        className="block w-[120px] text-center tabular-nums"
        style={{ pointerEvents: "none" }}
      >
        {recording
          ? `release · ${seconds}s`
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
