"use client";

// Full-bleed onboarding that mounts OVER the tracker (camera feed
// already warming up underneath). Two fast picks in a "customize your
// agent" frame, then a soft fade that releases the camera view — so the
// handoff feels like the overlay was always going to dissolve back into
// the world you're pointing at.

import React, { useCallback, useEffect, useState } from "react";
import {
  DEFAULT_PREFS,
  LANGUAGES,
  readOnboardingPrefs,
  writeOnboardingPrefs,
  type LangCode,
  type Lens,
  type OnboardingPrefs,
} from "@/lib/onboarding";

type Step = 0 | 1;

type Props = {
  // Preselected lens (from URL ?lens=...), used as the Step-2 default
  // highlight. null means the user enters Step 2 with nothing pre-chosen.
  initialLens: Lens | null;
  // Called once the fade-out has fully finished. The tracker uses this
  // to unmount us AND to lift its ?onboarding=1 URL param.
  onFinished: (prefs: OnboardingPrefs) => void;
};

const LENS_COPY: Record<
  Lens,
  { kicker: string; title: string; copy: string; emoji: string; tint: string }
> = {
  play: {
    kicker: "play",
    title: "play with me",
    copy: "roast my stuff. flirt. whine. fourteen words or less.",
    emoji: "✿",
    tint: "from-[#ffe4f2] to-[#ffd9bd]",
  },
  language: {
    kicker: "language",
    title: "teach me",
    copy: "speak the other language. drill me. correct me. kindly.",
    emoji: "✦",
    tint: "from-[#cfd9ff] to-[#e4d1ff]",
  },
  history: {
    kicker: "history",
    title: "tell me your past",
    copy: "every object knows where it came from. speak it.",
    emoji: "♡",
    tint: "from-[#ffe8a8] to-[#ffc9df]",
  },
};

export function OnboardingOverlay({ initialLens, onFinished }: Props) {
  const [phase, setPhase] = useState<"entering" | "active" | "leaving">(
    "entering"
  );
  const [step, setStep] = useState<Step>(0);
  const [prefs, setPrefs] = useState<OnboardingPrefs>(() => ({
    ...DEFAULT_PREFS,
    lens: initialLens ?? DEFAULT_PREFS.lens,
  }));

  // Hydrate prior answers so returning users see their last pick highlighted.
  useEffect(() => {
    const existing = readOnboardingPrefs();
    if (existing) {
      setPrefs((p) => ({
        ...existing,
        // URL lens wins over the stored one if present — user is
        // intentionally re-onboarding into a specific lens.
        lens: initialLens ?? existing.lens,
      }));
    }
  }, [initialLens]);

  // Fade in on mount: flip entering → active on the next tick so the
  // opening transition actually plays (no animation applies if we start
  // in `active`).
  useEffect(() => {
    const id = window.requestAnimationFrame(() => setPhase("active"));
    return () => window.cancelAnimationFrame(id);
  }, []);

  const finish = useCallback(
    (lens: Lens, p: OnboardingPrefs) => {
      // The "learn" language is implicitly the OTHER one in language mode —
      // user speaks EN → learn ZH, and vice versa. In play/history mode we
      // leave learnLang null so the prompts stay in the spoken language.
      const learnLang: LangCode | null =
        lens === "language" ? (p.spokenLang === "en" ? "zh" : "en") : null;
      const completed: OnboardingPrefs = {
        ...p,
        lens,
        learnLang,
        completedAt: Date.now(),
      };
      writeOnboardingPrefs(completed);
      setPhase("leaving");
      // Wait for the fade to finish before handing control back to the
      // tracker. Keeps the camera reveal feeling intentional, not abrupt.
      window.setTimeout(() => onFinished(completed), 720);
    },
    [onFinished]
  );

  const onLangPick = useCallback((code: LangCode) => {
    setPrefs((p) => ({ ...p, spokenLang: code }));
    // Small pause so the selected-state highlight is visible before the
    // step advances. Otherwise the chip click feels swallowed.
    window.setTimeout(() => setStep(1), 140);
  }, []);

  const onLensPick = useCallback(
    (lens: Lens) => {
      finish(lens, prefs);
    },
    [finish, prefs]
  );

  const visible = phase !== "leaving";
  const entered = phase === "active";

  return (
    <div
      aria-hidden={!visible}
      className="pointer-events-auto fixed inset-0 z-[80] flex items-center justify-center"
      style={{
        opacity: entered ? 1 : 0,
        transition: "opacity 640ms cubic-bezier(0.22, 1, 0.36, 1)",
      }}
    >
      {/* Scrim — pure black fade with a gentle radial vignette, so the
          camera underneath reads as a held breath rather than a blocked
          view. Backdrop-blur slides off on leave so the world comes
          back into focus, not just into visibility. */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(120% 80% at 50% 30%, rgba(16,6,34,0.78) 0%, rgba(8,3,22,0.92) 100%)",
          backdropFilter: entered ? "blur(18px) saturate(1.05)" : "blur(0px)",
          WebkitBackdropFilter: entered
            ? "blur(18px) saturate(1.05)"
            : "blur(0px)",
          transition:
            "backdrop-filter 720ms cubic-bezier(0.22, 1, 0.36, 1), -webkit-backdrop-filter 720ms cubic-bezier(0.22, 1, 0.36, 1), opacity 640ms ease",
          opacity: entered ? 1 : 0,
        }}
      />
      {/* Ambient colour wash, same palette family as the landing blobs
          so brand carries through even on the dark scrim. */}
      <OverlayBlobs entered={entered} />

      {/* Content — fades + lifts in, fades + lifts out. */}
      <div
        className="relative z-10 flex w-full max-w-[620px] flex-col items-center px-6 pb-12 pt-10 text-center text-white sm:px-10"
        style={{
          transform: entered ? "translateY(0) scale(1)" : "translateY(14px) scale(0.98)",
          opacity: entered ? 1 : 0,
          transition:
            "transform 680ms cubic-bezier(0.22, 1, 0.36, 1), opacity 560ms cubic-bezier(0.22, 1, 0.36, 1)",
        }}
      >
        <ProgressDots current={step} total={2} />

        {step === 0 && (
          <LanguageStep
            selected={prefs.spokenLang}
            onPick={onLangPick}
          />
        )}
        {step === 1 && (
          <LensStep
            selected={prefs.lens}
            onBack={() => setStep(0)}
            onPick={onLensPick}
          />
        )}

        <p className="mt-10 text-[11.5px] uppercase tracking-[0.28em] text-white/45">
          万物皆有声 · give everything a soul
        </p>
      </div>
    </div>
  );
}

// --- Step 0 ---------------------------------------------------------------

function LanguageStep({
  selected,
  onPick,
}: {
  selected: LangCode;
  onPick: (code: LangCode) => void;
}) {
  return (
    <section className="mt-4 flex w-full flex-col items-center">
      <Kicker>your agent speaks…</Kicker>
      <h1 className="serif-italic mt-4 text-balance text-[42px] font-semibold leading-[1.02] tracking-[-0.02em] text-white sm:text-[58px]">
        pick your agent&apos;s
        <br />
        native tongue
      </h1>
      <p className="mt-4 max-w-[420px] text-[14px] leading-[1.55] text-white/65 sm:text-[15px]">
        the language every object around you will speak back to you in.
      </p>

      <div className="mt-10 grid w-full max-w-[420px] grid-cols-2 gap-3">
        {LANGUAGES.map((l) => {
          const active = selected === l.code;
          return (
            <button
              key={l.code}
              type="button"
              onClick={() => onPick(l.code)}
              className={`group relative overflow-hidden rounded-2xl px-5 py-4 text-center transition ${
                active
                  ? "bg-white/[0.14] ring-2 ring-[color:var(--accent)]/70 shadow-[0_14px_34px_-18px_rgba(255,137,190,0.55)]"
                  : "bg-white/[0.06] ring-1 ring-white/15 hover:bg-white/[0.1] hover:ring-white/30"
              }`}
            >
              <span className="serif-italic block text-[26px] font-medium leading-tight text-white">
                {l.native}
              </span>
              <span className="mt-1 block text-[10.5px] font-medium uppercase tracking-[0.2em] text-white/55">
                {l.english}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

// --- Step 1 ---------------------------------------------------------------

function LensStep({
  selected,
  onBack,
  onPick,
}: {
  selected: Lens;
  onBack: () => void;
  onPick: (lens: Lens) => void;
}) {
  const lenses: Lens[] = ["play", "language", "history"];

  return (
    <section className="mt-4 flex w-full flex-col items-center">
      <Kicker>your agent should…</Kicker>
      <h1 className="serif-italic mt-4 text-balance text-[42px] font-semibold leading-[1.02] tracking-[-0.02em] text-white sm:text-[58px]">
        what do you
        <br />
        want it to do?
      </h1>
      <p className="mt-4 max-w-[480px] text-[14px] leading-[1.55] text-white/65 sm:text-[15px]">
        same camera, different soul. switch any time from the top bar.
      </p>

      <div className="mt-9 grid w-full gap-3 sm:grid-cols-3">
        {lenses.map((lens) => {
          const c = LENS_COPY[lens];
          const active = selected === lens;
          return (
            <button
              key={lens}
              type="button"
              onClick={() => onPick(lens)}
              className={`group relative overflow-hidden rounded-[22px] p-5 text-left transition hover:-translate-y-0.5 ${
                active
                  ? "bg-white/[0.14] ring-2 ring-[color:var(--accent)]/70 shadow-[0_18px_42px_-18px_rgba(255,137,190,0.55)]"
                  : "bg-white/[0.06] ring-1 ring-white/15 hover:bg-white/[0.1] hover:ring-white/30"
              }`}
            >
              <div
                aria-hidden
                className={`absolute -right-10 -top-10 h-32 w-32 rounded-full bg-gradient-to-br ${c.tint} opacity-40 blur-2xl transition group-hover:scale-110 group-hover:opacity-60`}
              />
              <div className="relative flex items-center justify-between">
                <span className="text-[10.5px] font-medium uppercase tracking-[0.28em] text-white/55">
                  {c.kicker}
                </span>
                <span className="text-[22px] text-[color:var(--accent)] transition-transform group-hover:rotate-12 group-hover:scale-110">
                  {c.emoji}
                </span>
              </div>
              <h3 className="serif-italic relative mt-3 text-[24px] font-medium leading-tight text-white">
                {c.title}
              </h3>
              <p className="relative mt-2 text-[13px] leading-[1.5] text-white/65">
                {c.copy}
              </p>
            </button>
          );
        })}
      </div>

      <div className="mt-7 flex items-center gap-3">
        <button
          type="button"
          onClick={onBack}
          className="btn-frost px-4 py-2 text-[11.5px] font-medium"
        >
          ← back
        </button>
        <span className="text-[11px] uppercase tracking-[0.22em] text-white/35">
          pick one to begin
        </span>
      </div>
    </section>
  );
}

// --- Shared ---------------------------------------------------------------

function Kicker({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full bg-white/[0.08] px-3 py-1 text-[10.5px] font-medium uppercase tracking-[0.28em] text-white/65 ring-1 ring-white/15 backdrop-blur-md">
      <span className="inline-block h-1 w-1 rounded-full bg-[color:var(--accent)]" />
      {children}
    </span>
  );
}

function ProgressDots({ current, total }: { current: number; total: number }) {
  return (
    <div
      className="flex items-center gap-1.5"
      aria-label={`step ${current + 1} of ${total}`}
    >
      {Array.from({ length: total }, (_, i) => {
        const state =
          i === current ? "active" : i < current ? "done" : "pending";
        return (
          <span
            key={i}
            className={`h-1.5 rounded-full transition-all duration-300 ${
              state === "active"
                ? "w-7 bg-[color:var(--accent)] shadow-[0_0_0_4px_rgba(236,72,153,0.18)]"
                : state === "done"
                  ? "w-2 bg-[color:var(--accent-deep)]"
                  : "w-2 bg-white/25"
            }`}
          />
        );
      })}
    </div>
  );
}

function OverlayBlobs({ entered }: { entered: boolean }) {
  const base = entered ? 0.45 : 0;
  return (
    <>
      <div
        className="blob"
        style={{
          width: 480,
          height: 480,
          top: -180,
          left: -160,
          background: "radial-gradient(circle, #ff89be 0%, #c026d3 70%)",
          opacity: base,
          mixBlendMode: "screen",
          transition: "opacity 720ms ease",
        }}
      />
      <div
        className="blob"
        style={{
          width: 520,
          height: 520,
          bottom: -200,
          right: -160,
          background: "radial-gradient(circle, #8b5cf6 0%, #6ee7ff 70%)",
          animationDelay: "-6s",
          opacity: base * 0.85,
          mixBlendMode: "screen",
          transition: "opacity 720ms ease",
        }}
      />
    </>
  );
}
