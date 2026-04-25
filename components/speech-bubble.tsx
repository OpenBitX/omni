"use client";

import { useEffect, useRef, useState } from "react";

// Cute speech bubble that rides above a locked face. Owns its own enter/exit
// animation so the parent can just flip `caption` on and off — when caption
// goes null, we keep the last text pinned while the out-animation plays, then
// unmount. Three-dot "thinking" state renders while the voice line is being
// cooked (speaking=true, caption still null).

const EXIT_MS = 260;

// Typewriter cadence. Faster than natural typing so short lines (~6-12 words)
// finish well inside the audio playback, but slow enough that the bubble
// visibly grows char-by-char. We also cap total reveal time so paragraph-
// length replies don't lag behind the voice.
const REVEAL_MS_PER_CHAR = 24;
const MAX_REVEAL_MS = 1400;

type SpeechBubbleProps = {
  caption: string | null;
  thinking: boolean;
  speaking: boolean;
  // Max bubble width in CSS px at scale=1. Parent sizes this from the face's
  // displayed width so a mug gets a pill and a sofa gets a paragraph.
  maxWidth: number | string;
};

export function SpeechBubble({ caption, thinking, speaking, maxWidth }: SpeechBubbleProps) {
  // Persist last non-null caption through the exit animation so the text
  // doesn't swap to empty mid-fade.
  const [shownCaption, setShownCaption] = useState<string | null>(caption);
  const [phase, setPhase] = useState<"hidden" | "in" | "out">(
    caption || thinking ? "in" : "hidden"
  );
  // How many chars of shownCaption to reveal right now. Driven by a timer
  // that kicks off each time a new caption arrives — the bubble's width
  // grows naturally as React reflows with each added char.
  const [revealedChars, setRevealedChars] = useState(0);
  const exitTimerRef = useRef<number | null>(null);
  const revealTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (caption) {
      setShownCaption((prev) => (prev === caption ? prev : caption));
      setPhase("in");
      if (exitTimerRef.current != null) {
        clearTimeout(exitTimerRef.current);
        exitTimerRef.current = null;
      }
      return;
    }
    if (thinking) {
      setPhase("in");
      if (exitTimerRef.current != null) {
        clearTimeout(exitTimerRef.current);
        exitTimerRef.current = null;
      }
      return;
    }
    // Nothing to show — schedule exit. Keep shownCaption so the text stays
    // legible while fading out.
    if (phase === "in") {
      setPhase("out");
      exitTimerRef.current = window.setTimeout(() => {
        setPhase("hidden");
        setShownCaption(null);
        exitTimerRef.current = null;
      }, EXIT_MS);
    }
  }, [caption, thinking, phase]);

  // Typewriter reveal. Restart from 0 whenever the caption text changes;
  // tick up with setInterval and stop at full length. Cadence scales down
  // for long lines so we don't trail the audio.
  useEffect(() => {
    if (revealTimerRef.current != null) {
      clearInterval(revealTimerRef.current);
      revealTimerRef.current = null;
    }
    if (!shownCaption) {
      setRevealedChars(0);
      return;
    }
    setRevealedChars(0);
    const len = shownCaption.length;
    const perChar = Math.max(12, Math.min(REVEAL_MS_PER_CHAR, MAX_REVEAL_MS / Math.max(1, len)));
    revealTimerRef.current = window.setInterval(() => {
      setRevealedChars((n) => {
        const next = n + 1;
        if (next >= len && revealTimerRef.current != null) {
          clearInterval(revealTimerRef.current);
          revealTimerRef.current = null;
        }
        return next;
      });
    }, perChar);
    return () => {
      if (revealTimerRef.current != null) {
        clearInterval(revealTimerRef.current);
        revealTimerRef.current = null;
      }
    };
  }, [shownCaption]);

  useEffect(() => {
    return () => {
      if (exitTimerRef.current != null) clearTimeout(exitTimerRef.current);
      if (revealTimerRef.current != null) clearInterval(revealTimerRef.current);
    };
  }, []);

  if (phase === "hidden") return null;

  const isThinking = !shownCaption && thinking;

  return (
    <div
      className="relative"
      style={{
        maxWidth,
        transformOrigin: "50% 90%",
        // Chunky hard offset — the classic newsprint comic drop. No blur.
        filter:
          "drop-shadow(5px 5px 0 #1a1024) drop-shadow(0 8px 18px rgba(236,72,153,0.18))",
        animation:
          phase === "out"
            ? `bubble-out ${EXIT_MS}ms cubic-bezier(0.4,0,1,1) both`
            : "bubble-in 520ms cubic-bezier(0.22,1.3,0.36,1) both",
      }}
    >
      <div
        className="relative rounded-[44px] px-[34px] py-[22px]"
        style={{
          // Faint cream tint reads as old newsprint paper, not screen-white.
          background: "#fffdf2",
          // Thick ink outline — the load-bearing comic-book signal.
          border: "4px solid #1a1024",
        }}
      >
        {isThinking ? (
          <span className="flex items-center gap-[12px] py-[6px] px-[2px]">
            <span
              className="h-[13px] w-[13px] rounded-full bg-[#1a1024]"
              style={{ animation: "dot-bounce 1.1s ease-in-out infinite" }}
            />
            <span
              className="h-[13px] w-[13px] rounded-full bg-[#1a1024]"
              style={{ animation: "dot-bounce 1.1s ease-in-out 0.15s infinite" }}
            />
            <span
              className="h-[13px] w-[13px] rounded-full bg-[#1a1024]"
              style={{ animation: "dot-bounce 1.1s ease-in-out 0.3s infinite" }}
            />
          </span>
        ) : (
          <p
            className="text-center text-[30px] leading-[1.1] uppercase text-[#1a1024] whitespace-pre-wrap break-words"
            style={{
              fontFamily: "var(--font-comic), 'Bangers', 'Comic Sans MS', system-ui, sans-serif",
              letterSpacing: "0.04em",
              textShadow: "1px 1px 0 rgba(26,16,36,0.08)",
            }}
          >
            {shownCaption ? shownCaption.slice(0, revealedChars) : ""}
            {shownCaption && revealedChars < shownCaption.length && (
              <span
                aria-hidden
                className="ml-[3px] inline-block h-[0.78em] w-[4px] bg-[#1a1024] align-baseline translate-y-[2px]"
                style={{ animation: "caret-blink 0.9s steps(2) infinite" }}
              />
            )}
          </p>
        )}
      </div>
      {/* Comic tail — open V drawn as SVG so the ink outline stays continuous
          with the bubble border. A small cream-fill mask covers the bubble's
          bottom border under the tail so the V appears to open into it. */}
      <span
        aria-hidden
        className="absolute left-1/2"
        style={{
          bottom: -4,
          transform: "translateX(-50%)",
          width: 36,
          height: 8,
          background: "#fffdf2",
          animation: speaking ? "tail-bob 1.6s ease-in-out infinite" : undefined,
        }}
      />
      <svg
        aria-hidden
        width="44"
        height="26"
        viewBox="0 0 44 26"
        className="absolute left-1/2"
        style={{
          bottom: -25,
          transform: "translateX(-50%)",
          overflow: "visible",
          animation: speaking ? "tail-bob 1.6s ease-in-out infinite" : undefined,
        }}
      >
        <path
          d="M4 0 L22 24 L40 0"
          fill="#fffdf2"
          stroke="#1a1024"
          strokeWidth="4"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
}
