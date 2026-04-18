"use client";

import { useEffect, useRef, useState } from "react";

// Cute speech bubble that rides above a locked face. Owns its own enter/exit
// animation so the parent can just flip `caption` on and off — when caption
// goes null, we keep the last text pinned while the out-animation plays, then
// unmount. Three-dot "thinking" state renders while the voice line is being
// cooked (speaking=true, caption still null).

const EXIT_MS = 260;

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
  const exitTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (caption) {
      setShownCaption(caption);
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

  useEffect(() => {
    return () => {
      if (exitTimerRef.current != null) clearTimeout(exitTimerRef.current);
    };
  }, []);

  if (phase === "hidden") return null;

  const isThinking = !shownCaption && thinking;

  return (
    <div
      className="relative"
      style={{
        maxWidth,
        animation:
          phase === "out"
            ? `bubble-out ${EXIT_MS}ms cubic-bezier(0.4,0,1,1) both`
            : "bubble-in 440ms cubic-bezier(0.16,1,0.3,1) both",
      }}
    >
      <div
        className="relative rounded-[22px] bg-white/95 px-[18px] py-[12px] shadow-[0_18px_40px_-14px_rgba(236,72,153,0.35),0_6px_18px_-8px_rgba(42,21,64,0.25)] ring-1 ring-[rgba(255,192,219,0.9)] backdrop-blur-xl"
      >
        {isThinking ? (
          <span className="flex items-center gap-[5px] py-[2px]">
            <span
              className="h-[7px] w-[7px] rounded-full bg-[color:var(--accent)]"
              style={{ animation: "dot-bounce 1.1s ease-in-out infinite" }}
            />
            <span
              className="h-[7px] w-[7px] rounded-full bg-[color:var(--accent)]"
              style={{ animation: "dot-bounce 1.1s ease-in-out 0.15s infinite" }}
            />
            <span
              className="h-[7px] w-[7px] rounded-full bg-[color:var(--accent)]"
              style={{ animation: "dot-bounce 1.1s ease-in-out 0.3s infinite" }}
            />
          </span>
        ) : (
          <p className="serif-italic text-balance text-center text-[15px] leading-[1.35] text-[color:var(--ink)] whitespace-pre-wrap break-words">
            {shownCaption}
          </p>
        )}
      </div>
      {/* Tail — small rotated square tucked under the bubble, matching fill
          + ring so it reads as one blob. Wobbles gently while speaking. */}
      <span
        aria-hidden
        className="absolute left-1/2 -translate-x-1/2 h-[14px] w-[14px] rotate-45 rounded-[3px] bg-white/95 ring-1 ring-[rgba(255,192,219,0.9)]"
        style={{
          bottom: -6,
          animation: speaking ? "tail-bob 1.6s ease-in-out infinite" : undefined,
          transformOrigin: "center",
        }}
      />
    </div>
  );
}
