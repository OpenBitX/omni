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
  maxWidth: number;
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
        transformOrigin: "50% 100%",
        // One soft drop-shadow wraps the bubble + tail as a single blob. Two
        // layered shadows: a deep pink halo for the pastel language, and a
        // tighter ink shadow for grounding.
        filter:
          "drop-shadow(0 12px 22px rgba(236,72,153,0.18)) drop-shadow(0 4px 10px rgba(42,21,64,0.14))",
        animation:
          phase === "out"
            ? `bubble-out ${EXIT_MS}ms cubic-bezier(0.4,0,1,1) both`
            : "bubble-in 520ms cubic-bezier(0.22,1.3,0.36,1) both",
      }}
    >
      <div
        className="relative rounded-[26px] bg-white px-[20px] py-[13px]"
        style={{
          // Inset highlight at the top reads as a glossy ceramic rim. The
          // second inset traces a barely-there pink edge so the bubble feels
          // intentional without a hard ring.
          boxShadow:
            "inset 0 1px 0 rgba(255,255,255,0.9), inset 0 0 0 1px rgba(255,192,219,0.55)",
        }}
      >
        {isThinking ? (
          <span className="flex items-center gap-[6px] py-[3px] px-[2px]">
            <span
              className="h-[7px] w-[7px] rounded-full bg-[color:var(--accent)]/80"
              style={{ animation: "dot-bounce 1.1s ease-in-out infinite" }}
            />
            <span
              className="h-[7px] w-[7px] rounded-full bg-[color:var(--accent)]/80"
              style={{ animation: "dot-bounce 1.1s ease-in-out 0.15s infinite" }}
            />
            <span
              className="h-[7px] w-[7px] rounded-full bg-[color:var(--accent)]/80"
              style={{ animation: "dot-bounce 1.1s ease-in-out 0.3s infinite" }}
            />
          </span>
        ) : (
          <p className="serif-italic text-balance text-center text-[15px] leading-[1.38] tracking-[-0.005em] text-[color:var(--ink)] whitespace-pre-wrap break-words">
            {shownCaption ? shownCaption.slice(0, revealedChars) : ""}
            {shownCaption && revealedChars < shownCaption.length && (
              <span
                aria-hidden
                className="ml-[2px] inline-block h-[0.85em] w-[3px] rounded-full bg-[color:var(--accent)]/70 align-baseline translate-y-[2px]"
                style={{ animation: "caret-blink 0.9s steps(2) infinite" }}
              />
            )}
          </p>
        )}
      </div>
      {/* Tail — a small rounded square rotated 45°, sharing the fill and
          inset highlight so it reads as part of the bubble. The parent's
          drop-shadow wraps it continuously. */}
      <span
        aria-hidden
        className="absolute left-1/2 h-[12px] w-[12px] rounded-[3px] bg-white"
        style={{
          bottom: -5,
          transform: "translateX(-50%) rotate(45deg)",
          boxShadow: "inset 0 0 0 1px rgba(255,192,219,0.55)",
          animation: speaking ? "tail-bob 1.6s ease-in-out infinite" : undefined,
        }}
      />
    </div>
  );
}
