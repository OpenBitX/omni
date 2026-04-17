import Viewfinder from "@/components/viewfinder";

export default function Home() {
  return (
    <main className="paper-grain relative min-h-[100svh] w-full overflow-hidden bg-[color:var(--color-paper)]">
      {/* Desktop chrome: brand + tagline */}
      <div className="pointer-events-none absolute inset-x-0 top-0 z-10 hidden items-center justify-between px-10 pt-8 md:flex">
        <div className="flex items-center gap-2.5">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-[color:var(--color-ink)]" />
          <span className="text-[13px] font-medium tracking-[-0.01em] text-[color:var(--color-ink)]">
            vent
          </span>
        </div>
        <div className="text-[12px] tracking-[-0.005em] text-[color:var(--color-mute)]">
          Snap it. Say it. Meme it.
        </div>
      </div>

      {/* Mobile: full-bleed. Desktop: centered device frame. */}
      <div className="flex min-h-[100svh] w-full items-center justify-center md:px-6 md:py-20">
        <div
          className="relative h-[100svh] w-full overflow-hidden bg-black shadow-[var(--shadow-frame)]
            md:aspect-[9/16] md:h-[min(86svh,880px)] md:w-auto md:rounded-[var(--radius-frame)]"
        >
          <Viewfinder />
          {/* Hairline inner border for the frame on desktop */}
          <div className="pointer-events-none absolute inset-0 hidden md:block md:rounded-[var(--radius-frame)] md:[box-shadow:inset_0_0_0_1px_rgba(255,255,255,0.06)]" />
        </div>
      </div>

      {/* Desktop footer hint */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 hidden items-center justify-between px-10 pb-8 md:flex">
        <div className="font-mono text-[11px] tracking-[0] text-[color:var(--color-mute)]">
          tap · snap   /   hold · vent
        </div>
        <div className="text-[11px] tracking-[-0.005em] text-[color:var(--color-mute)]">
          powered by Runware
        </div>
      </div>
    </main>
  );
}
