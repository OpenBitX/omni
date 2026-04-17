"use client";

import { useCallback, useState } from "react";
import type { MemeResult } from "@/app/actions";
import MemeCanvas from "@/components/meme-canvas";

export default function Reveal({
  original,
  meme,
  onReset,
}: {
  original: string;
  meme: MemeResult;
  onReset: () => void;
}) {
  const [showOriginal, setShowOriginal] = useState(false);
  const [composited, setComposited] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const onReady = useCallback((url: string) => setComposited(url), []);

  if (!meme.ok) return null;

  const displayTop = meme.format === "bottom-only" ? "" : meme.topText;
  const displayBottom = meme.format === "top-only" ? "" : meme.bottomText;

  const share = async () => {
    const src = composited ?? meme.imageDataUrl;
    try {
      const blob = await (await fetch(src)).blob();
      const file = new File([blob], "vent-meme.png", { type: blob.type });
      if (navigator.canShare?.({ files: [file] })) {
        await navigator.share({
          files: [file],
          text: meme.caption || displayBottom || displayTop,
        });
        return;
      }
    } catch {}
    const a = document.createElement("a");
    a.href = src;
    a.download = "vent-meme.png";
    a.click();
  };

  const copyCaption = async () => {
    const text = meme.caption || [displayTop, displayBottom].filter(Boolean).join(" / ");
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {}
  };

  return (
    <div className="relative h-full w-full bg-black">
      {/* Hidden canvas composite — emits the ready data URL into <img> */}
      <div
        className={`absolute inset-0 transition-opacity duration-500 ${
          showOriginal ? "opacity-0" : "opacity-100"
        }`}
      >
        <MemeCanvas
          imageSrc={meme.imageDataUrl}
          topText={displayTop}
          bottomText={displayBottom}
          onReady={onReady}
          className="h-full w-full object-cover"
        />
        {!composited && (
          <div className="absolute inset-0 grid place-items-center bg-black">
            <div className="h-1 w-1 rounded-full bg-white/70 breathe" />
          </div>
        )}
      </div>

      <img
        src={original}
        alt="original"
        className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-200 ${
          showOriginal ? "opacity-100" : "opacity-0"
        }`}
      />

      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(120%_80%_at_50%_0%,rgba(0,0,0,0.45)_0%,transparent_45%),radial-gradient(120%_80%_at_50%_100%,rgba(0,0,0,0.75)_0%,transparent_55%)]" />

      <header className="absolute inset-x-0 top-0 flex items-center justify-between px-5 pt-[max(env(safe-area-inset-top),14px)]">
        <button
          onClick={onReset}
          aria-label="Back"
          className="grid h-8 w-8 place-items-center rounded-full border border-white/15 bg-white/5 text-white/80 backdrop-blur-md transition hover:border-white/30 hover:bg-white/10 hover:text-white"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M19 12H5" />
            <path d="M12 19l-7-7 7-7" />
          </svg>
        </button>

        <button
          onPointerDown={() => setShowOriginal(true)}
          onPointerUp={() => setShowOriginal(false)}
          onPointerLeave={() => setShowOriginal(false)}
          onPointerCancel={() => setShowOriginal(false)}
          className="rounded-full border border-white/15 bg-white/5 px-3 py-1.5 font-mono text-[10.5px] tracking-[0.14em] text-white/75 backdrop-blur-md transition hover:border-white/30 hover:text-white"
        >
          HOLD · ORIGINAL
        </button>
      </header>

      <div className="absolute inset-x-0 bottom-0 px-3 pb-[max(env(safe-area-inset-bottom),16px)] sm:px-4">
        <div
          className="mx-auto flex max-w-md flex-col gap-3 rounded-[var(--radius-card)] border border-white/10 bg-[rgba(14,14,14,0.72)] p-4 backdrop-blur-xl rise"
          style={{ boxShadow: "0 20px 60px -20px rgba(0,0,0,0.6)" }}
        >
          {meme.transcript && (
            <div className="flex items-start gap-2.5 rounded-xl bg-white/[0.04] px-3 py-2.5">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="mt-0.5 shrink-0 text-white/50">
                <path d="M9 9a3 3 0 0 1 6 0v4a3 3 0 0 1-6 0z" />
                <path d="M5 12a7 7 0 0 0 14 0" />
                <path d="M12 19v3" />
              </svg>
              <div
                data-selectable
                className="line-clamp-2 text-[12px] font-light leading-snug italic text-white/60"
              >
                “{meme.transcript}”
              </div>
            </div>
          )}

          <div className="flex items-center gap-2">
            <button
              onClick={share}
              disabled={!composited}
              className="flex flex-1 items-center justify-center gap-2 rounded-full bg-white px-4 py-2.5 text-[13px] font-medium tracking-[-0.005em] text-black transition hover:bg-white/90 active:scale-[0.98] disabled:opacity-50"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 3v12" />
                <path d="M7 8l5-5 5 5" />
                <path d="M5 15v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" />
              </svg>
              Share meme
            </button>
            <button
              onClick={copyCaption}
              aria-label="Copy caption"
              className="grid h-10 w-10 place-items-center rounded-full border border-white/15 bg-white/5 text-white/80 transition hover:border-white/30 hover:bg-white/10 hover:text-white"
            >
              {copied ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M20 6L9 17l-5-5" />
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="9" y="9" width="11" height="11" rx="2" />
                  <path d="M5 15V5a2 2 0 0 1 2-2h10" />
                </svg>
              )}
            </button>
            <button
              onClick={onReset}
              className="rounded-full border border-white/15 bg-white/5 px-4 py-2.5 text-[13px] font-medium tracking-[-0.005em] text-white transition hover:border-white/30 hover:bg-white/10"
            >
              Again
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
