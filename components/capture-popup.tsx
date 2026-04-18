"use client";

// Collect-to-gallery popup. Spawns the moment the user taps a YOLO box
// (same moment the face appears on the object) and shows a floating card
// at the bottom of the camera view with the crop + class name + a big
// "collect to gallery" button.
//
// The item has two possible phases:
//   - "preparing": the VLM hasn't returned yet, so there's no SessionCard
//     in the store. We still show the thumbnail + class name + a muted
//     button so the user sees an instant, consistent response to their
//     tap — the collect action itself waits until the card is ready.
//   - "ready" (status idle/pending/done/failed): a full SessionCard
//     exists; collect can fire, generation status drives the button.

export type CaptureItem = {
  // Stable id used for dismissal tracking. We use the trackId so the
  // popup survives pending→ready transitions without flicker.
  trackId: string;
  className: string;
  imageDataUrl: string;
  cardId: string | null;
  // "preparing" = no card yet (VLM in flight). Otherwise follows
  // GeneratedImageStatus from the session-cards store.
  status: "preparing" | "idle" | "pending" | "done" | "failed";
};

type CapturePopupProps = {
  item: CaptureItem | null;
  onCollect: (trackId: string) => void;
  onDismiss: (trackId: string) => void;
};

export function CapturePopup({ item, onCollect, onDismiss }: CapturePopupProps) {
  if (!item) return null;

  const { status } = item;
  const primaryLabel =
    status === "preparing"
      ? "preparing…"
      : status === "done"
        ? "saved to gallery"
        : status === "pending"
          ? "generating artwork…"
          : status === "failed"
            ? "try again"
            : "collect to gallery";
  const primaryDisabled =
    status === "preparing" || status === "pending" || status === "done";

  const primarySubline =
    status === "preparing"
      ? "sizing it up"
      : status === "pending"
        ? "painting comic"
        : status === "done"
          ? "open the gallery"
          : status === "failed"
            ? "try again"
            : "just captured";

  return (
    <div
      className="pointer-events-none absolute inset-x-0 bottom-[112px] z-30 flex justify-center px-4 sm:bottom-[124px]"
      role="dialog"
      aria-label={`collect ${item.className} to gallery`}
    >
      <div
        key={item.trackId}
        className="capture-popup-enter pointer-events-auto flex w-full max-w-[420px] items-center gap-3 rounded-[22px] p-2 pr-3 shadow-[0_24px_60px_-20px_rgba(236,72,153,0.55)] ring-1 ring-white/70 backdrop-blur-2xl"
        style={{
          background:
            "linear-gradient(135deg, rgba(255,238,247,0.96) 0%, rgba(255,220,234,0.94) 100%)",
        }}
      >
        <div className="relative h-[68px] w-[68px] shrink-0 overflow-hidden rounded-[16px] bg-black/10 ring-1 ring-white/60">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={item.imageDataUrl}
            alt={item.className}
            className="absolute inset-0 h-full w-full object-cover"
            draggable={false}
          />
          {(status === "preparing" || status === "pending") && (
            <span
              aria-hidden
              className="pointer-events-none absolute inset-0 rounded-[16px]"
              style={{
                background:
                  "linear-gradient(90deg, transparent 20%, rgba(255,255,255,0.55) 50%, transparent 80%)",
                animation: "shimmer-sweep 1.6s linear infinite",
              }}
            />
          )}
        </div>

        <div className="flex min-w-0 flex-1 flex-col gap-1.5">
          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0">
              <div className="serif-italic truncate text-[16px] font-medium leading-tight text-[#3a0a29]">
                {item.className}
              </div>
              <div className="truncate text-[10.5px] font-medium uppercase tracking-[0.12em] text-[#7a1a4a]/75">
                {primarySubline}
              </div>
            </div>
            <button
              type="button"
              aria-label="dismiss"
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                onDismiss(item.trackId);
              }}
              className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-black/5 text-[13px] leading-none text-[#5a0a29]/70 transition hover:bg-black/10 hover:text-[#3a0a29]"
            >
              ×
            </button>
          </div>

          <button
            type="button"
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              if (primaryDisabled) return;
              onCollect(item.trackId);
            }}
            disabled={primaryDisabled}
            aria-label={primaryLabel}
            className="group/collect relative inline-flex h-9 items-center justify-center gap-1.5 rounded-full px-3.5 text-[12.5px] font-semibold tracking-tight transition-[transform,box-shadow,filter] duration-150 ease-out disabled:cursor-not-allowed"
            style={{
              background:
                status === "done"
                  ? "linear-gradient(135deg, #bff2d1 0%, #a8ead9 100%)"
                  : status === "pending" || status === "preparing"
                    ? "rgba(255,255,255,0.9)"
                    : status === "failed"
                      ? "linear-gradient(135deg, #ffd69a 0%, #ffc3cf 100%)"
                      : "linear-gradient(135deg, #ff6fae 0%, #ff9dbf 100%)",
              color:
                status === "done"
                  ? "#1a5245"
                  : status === "pending" || status === "preparing"
                    ? "#7a1a4a"
                    : "#3a0a29",
              boxShadow:
                status === "pending" || status === "preparing"
                  ? "none"
                  : status === "done"
                    ? "0 8px 22px -10px rgba(34,197,94,0.55)"
                    : "0 10px 26px -10px rgba(236,72,153,0.7), inset 0 1px 0 rgba(255,255,255,0.55)",
              transform: "none",
            }}
          >
            {status === "idle" && <SparkleIcon />}
            {(status === "preparing" || status === "pending") && <SpinnerIcon />}
            {status === "done" && <CheckIcon />}
            {status === "failed" && <RetryIcon />}
            <span>{primaryLabel}</span>
          </button>
        </div>
      </div>
    </div>
  );
}

function SparkleIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M12 2l2.1 6.3L20.4 10l-6.3 2.1L12 18l-2.1-5.9L3.6 10l6.3-1.7L12 2z" />
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.4"
      strokeLinecap="round"
      aria-hidden
      style={{ animation: "spin 900ms linear infinite" }}
    >
      <path d="M21 12a9 9 0 1 1-6.2-8.55" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M4 12l5 5L20 6" />
    </svg>
  );
}

function RetryIcon() {
  return (
    <svg
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M3 12a9 9 0 1 0 3-6.7" />
      <path d="M3 4v5h5" />
    </svg>
  );
}
