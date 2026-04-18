"use client";

import { useCallback } from "react";

export type SessionCard = {
  id: string;
  trackId: string;
  createdAt: number;
  className: string;
  description: string;
  voiceId: string;
  line: string;
  imageDataUrl: string;
};

export type CardLiveStatus = "alive" | "lost";

export type SessionGalleryProps = {
  cards: readonly SessionCard[];
  activeCardId: string | null;
  cardStatus: Readonly<Record<string, CardLiveStatus>>;
  onMicPress: (cardId: string) => void;
  onMicRelease: (cardId: string) => void;
  onDismiss?: (cardId: string) => void;
};

export function SessionGallery({
  cards,
  activeCardId,
  cardStatus,
  onMicPress,
  onMicRelease,
  onDismiss,
}: SessionGalleryProps) {
  if (cards.length === 0) return null;

  return (
    <div
      className="pointer-events-none absolute inset-x-0 bottom-[116px] z-20 flex justify-center px-3"
      aria-label="session gallery"
    >
      <div
        className="pointer-events-auto flex w-full max-w-[640px] gap-3 overflow-x-auto overflow-y-hidden py-2 scrollbar-none"
        style={{ scrollbarWidth: "none" }}
      >
        {cards.map((card) => (
          <GalleryCard
            key={card.id}
            card={card}
            isActive={activeCardId === card.id}
            status={cardStatus[card.trackId] ?? "lost"}
            onMicPress={onMicPress}
            onMicRelease={onMicRelease}
            onDismiss={onDismiss}
          />
        ))}
      </div>
    </div>
  );
}

type GalleryCardProps = {
  card: SessionCard;
  isActive: boolean;
  status: CardLiveStatus;
  onMicPress: (cardId: string) => void;
  onMicRelease: (cardId: string) => void;
  onDismiss?: (cardId: string) => void;
};

function GalleryCard({
  card,
  isActive,
  status,
  onMicPress,
  onMicRelease,
  onDismiss,
}: GalleryCardProps) {
  const alive = status === "alive";

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLButtonElement>) => {
      e.preventDefault();
      e.stopPropagation();
      try {
        (e.currentTarget as Element).setPointerCapture(e.pointerId);
      } catch {
        // ignore
      }
      onMicPress(card.id);
    },
    [card.id, onMicPress]
  );

  const handlePointerUp = useCallback(
    (e: React.PointerEvent<HTMLButtonElement>) => {
      e.preventDefault();
      e.stopPropagation();
      try {
        (e.currentTarget as Element).releasePointerCapture(e.pointerId);
      } catch {
        // ignore
      }
      onMicRelease(card.id);
    },
    [card.id, onMicRelease]
  );

  return (
    <div
      className="group relative flex w-[172px] shrink-0 flex-col gap-2 rounded-[20px] bg-white/10 p-2 ring-1 ring-white/15 backdrop-blur-xl transition-[transform,box-shadow] duration-200"
      style={{
        boxShadow: isActive
          ? "0 10px 30px -10px rgba(255,137,190,0.55), 0 0 0 2px rgba(255,137,190,0.9)"
          : "0 8px 24px -14px rgba(0,0,0,0.55)",
        transform: isActive ? "translateY(-2px)" : undefined,
      }}
    >
      {onDismiss && (
        <button
          type="button"
          aria-label="remove from gallery"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onDismiss(card.id);
          }}
          className="absolute right-1.5 top-1.5 z-10 grid h-5 w-5 place-items-center rounded-full bg-black/40 text-[11px] leading-none text-white/80 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-black/60"
        >
          ×
        </button>
      )}

      <div className="relative aspect-square w-full overflow-hidden rounded-[14px] bg-black/40">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={card.imageDataUrl}
          alt={card.className}
          className="absolute inset-0 h-full w-full object-cover"
          draggable={false}
        />
        <div
          className="pointer-events-none absolute inset-x-0 bottom-0 h-8 bg-gradient-to-t from-black/55 to-transparent"
          aria-hidden
        />
        <span
          className="absolute left-2 top-2 rounded-full bg-black/55 px-2 py-[2px] text-[10px] font-semibold uppercase tracking-wider text-white/90"
          aria-label={`class ${card.className}`}
        >
          {card.className}
        </span>
        {!alive && (
          <span
            className="absolute right-2 top-2 rounded-full bg-white/85 px-2 py-[2px] text-[9px] font-semibold uppercase tracking-wider text-pink-700"
            aria-label="out of view"
          >
            out of view
          </span>
        )}
      </div>

      <p
        className="line-clamp-2 min-h-[2.4em] text-[11.5px] italic leading-[1.15] text-white/85"
        title={card.line}
      >
        &ldquo;{card.line}&rdquo;
      </p>

      <button
        type="button"
        aria-label={
          alive
            ? isActive
              ? "recording — release to send"
              : `hold to speak to ${card.className}`
            : "out of view — tap the object again"
        }
        disabled={!alive}
        onPointerDown={alive ? handlePointerDown : undefined}
        onPointerUp={alive ? handlePointerUp : undefined}
        onPointerCancel={alive ? handlePointerUp : undefined}
        onPointerLeave={
          alive
            ? (e) => {
                if (e.buttons > 0) onMicRelease(card.id);
              }
            : undefined
        }
        className="relative flex h-9 items-center justify-center gap-1.5 rounded-full text-[11.5px] font-semibold tracking-wide transition-[transform,box-shadow,opacity] duration-150 ease-out disabled:cursor-not-allowed disabled:opacity-50"
        style={{
          background: alive
            ? isActive
              ? "linear-gradient(135deg, #ff6fae 0%, #ff9dbf 100%)"
              : "linear-gradient(135deg, #ffb8d6 0%, #ffd4e3 100%)"
            : "rgba(255,255,255,0.18)",
          color: alive ? "#3a0a29" : "rgba(255,255,255,0.75)",
          boxShadow: isActive
            ? "0 0 0 2px rgba(255,255,255,0.65), 0 8px 22px -6px rgba(255,111,174,0.7)"
            : alive
              ? "0 6px 16px -8px rgba(255,111,174,0.55)"
              : "none",
          transform: isActive ? "scale(0.97)" : undefined,
          touchAction: "none",
        }}
      >
        <MicIcon />
        <span>{isActive ? "listening…" : alive ? "hold to talk" : "out of view"}</span>
      </button>
    </div>
  );
}

function MicIcon() {
  return (
    <svg
      width="12"
      height="12"
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
