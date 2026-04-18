import Link from "next/link";

export const metadata = {
  title: "Omni-Personal · Give everything a soul",
  description:
    "Everything has a voice — choose what it teaches you. Tap anything in frame and it talks back in Play, Language, or History mode.",
};

type Lens = {
  kicker: string;
  title: string;
  copy: string;
  emoji: string;
  tint: string;
  rotate: string;
  href: string;
};

const LENSES: Lens[] = [
  {
    kicker: "play",
    title: "the cheeky one",
    copy: "your boba, your trash can, your running shoes — all of them with a mouth and an opinion.",
    emoji: "✿",
    tint: "from-[#ffe4f2] to-[#ffd9bd]",
    rotate: "-rotate-[1.5deg]",
    href: "/?onboarding=1&lens=play",
  },
  {
    kicker: "language",
    title: "the private tutor",
    copy: "your desk plant teaches you mandarin. the mug drills your french. fluency, one tap at a time.",
    emoji: "✦",
    tint: "from-[#cfd9ff] to-[#e4d1ff]",
    rotate: "rotate-[1.5deg]",
    href: "/?onboarding=1&lens=language",
  },
  {
    kicker: "history",
    title: "the quiet historian",
    copy: "the park bench tells you who sat here a century ago. the lamp remembers the war.",
    emoji: "♡",
    tint: "from-[#ffe8a8] to-[#ffc9df]",
    rotate: "-rotate-[1deg]",
    href: "/?onboarding=1&lens=history",
  },
];

type Cast = {
  emoji: string;
  name: string;
  state: string;
  line: string;
  tint: string;
  rotate: string;
};

const CAST: Cast[] = [
  {
    emoji: "🧋",
    name: "boba cup",
    state: "play · last pearls",
    line: "keep sipping. go on. drain me dry.",
    tint: "from-[#ffe4f2] to-[#ffd9bd]",
    rotate: "-rotate-[2deg]",
  },
  {
    emoji: "🪴",
    name: "desk plant",
    state: "language · mandarin · a1",
    line: "我叫小绿。你呢？ — my name is xiao lü. yours?",
    tint: "from-[#d9f5e4] to-[#cfe4ff]",
    rotate: "rotate-[1.5deg]",
  },
  {
    emoji: "🪑",
    name: "park bench",
    state: "history · 1924",
    line: "before you, a tram clerk ate lunch here every tuesday for thirty years.",
    tint: "from-[#ffe8a8] to-[#ffc9df]",
    rotate: "-rotate-[1deg]",
  },
  {
    emoji: "👟",
    name: "running shoes",
    state: "play · dusty",
    line: "r.i.p. 0 km. cause of death: your laziness.",
    tint: "from-[#e4d1ff] to-[#ffd1e8]",
    rotate: "rotate-[2deg]",
  },
  {
    emoji: "📘",
    name: "calculus book",
    state: "language · french",
    line: "dérivée. repeat after me — déri·vée. un peu plus lent, oui?",
    tint: "from-[#cfe4ff] to-[#ffe4f2]",
    rotate: "-rotate-[1.5deg]",
  },
  {
    emoji: "🏛️",
    name: "museum column",
    state: "history · doric",
    line: "i am older than every idea in your pocket. and yet — here we both are.",
    tint: "from-[#ffd1e8] to-[#e5d4ff]",
    rotate: "rotate-[1deg]",
  },
];

const PERSONAS = [
  {
    kicker: "curious travellers",
    title: "every street, a museum",
    copy: "walk a new city. let the statues, menus, and trams tell you what they know.",
    emoji: "✿",
    tint: "from-[#ffe4f2] to-[#ffd9bd]",
  },
  {
    kicker: "language learners",
    title: "immersion on your desk",
    copy: "no app, no flashcards. your mug drills vocabulary. your plant holds conversations.",
    emoji: "✦",
    tint: "from-[#cfd9ff] to-[#e4d1ff]",
  },
  {
    kicker: "chronically online",
    title: "a camera that films back",
    copy: "ten seconds of stapler slander. instant serotonin. your group chat will not recover.",
    emoji: "♡",
    tint: "from-[#ffe8a8] to-[#ffc9df]",
  },
];

export default function LandingPage() {
  return (
    <main className="relative min-h-[100svh] overflow-hidden">
      {/* Ambient blobs */}
      <div
        className="blob"
        style={{
          width: 460,
          height: 460,
          top: -160,
          left: -160,
          background: "radial-gradient(circle, #ffc4de 0%, #ffd9bd 70%)",
        }}
      />
      <div
        className="blob"
        style={{
          width: 560,
          height: 560,
          top: "28%",
          right: -240,
          background: "radial-gradient(circle, #cfd9ff 0%, #e4d1ff 70%)",
          animationDelay: "-6s",
        }}
      />
      <div
        className="blob"
        style={{
          width: 380,
          height: 380,
          bottom: -140,
          left: "14%",
          background: "radial-gradient(circle, #ffe8a8 0%, #ffc9df 70%)",
          animationDelay: "-12s",
        }}
      />
      <div
        className="blob"
        style={{
          width: 320,
          height: 320,
          top: "6%",
          right: "10%",
          background: "radial-gradient(circle, #d9f5e4 0%, #cfe4ff 70%)",
          animationDelay: "-3s",
        }}
      />

      <div className="relative z-10 mx-auto flex w-full max-w-[1040px] flex-col">
        {/* Top rail */}
        <header className="flex items-center justify-between px-6 pt-7 sm:px-8 sm:pt-9">
          <div className="flex items-baseline gap-2">
            <span className="h-2 w-2 rounded-full bg-[color:var(--accent)] shadow-[0_0_0_4px_rgba(236,72,153,0.18)]" />
            <span className="serif-italic text-[26px] font-semibold leading-none text-[color:var(--ink)] sm:text-[30px]">
              omni-personal
            </span>
            <span className="hidden text-[12px] font-medium tracking-[0.22em] text-[color:var(--ink-muted)] sm:inline">
              · 万物皆有声
            </span>
          </div>
          <div className="flex items-center gap-2 rounded-full bg-white/70 px-3 py-1.5 shadow-[0_2px_10px_-4px_rgba(42,21,64,0.15)] ring-1 ring-white/80 backdrop-blur-md">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            <span className="text-[11px] font-medium tracking-wide text-[color:var(--ink-soft)]">
              beta ✿ three lenses
            </span>
          </div>
        </header>

        {/* Hero */}
        <section className="px-6 pb-6 pt-10 text-center sm:px-8 sm:pb-10 sm:pt-14">
          <h1 className="serif-italic mx-auto max-w-[900px] text-balance text-[52px] font-semibold leading-[0.95] tracking-[-0.02em] text-[color:var(--ink)] sm:text-[96px]">
            give everything
            <br />a{" "}
            <span className="relative inline-block">
              soul
              <span className="absolute -right-6 -top-2 inline-block animate-[wiggle_2.2s_ease-in-out_infinite] text-[28px] text-[color:var(--accent)] sm:-right-9 sm:text-[44px]">
                ✿
              </span>
            </span>
          </h1>

          <p className="mx-auto mt-6 max-w-[620px] text-balance text-[16px] leading-[1.55] text-[color:var(--ink-soft)] sm:text-[19px]">
            everything has a voice —{" "}
            <span className="serif-italic text-[color:var(--ink)]">
              choose what it teaches you.
            </span>{" "}
            point your camera at the world; decide whether it plays with you,
            tutors you, or tells you who lived here before.
          </p>

          {/* CTA */}
          <div className="relative mx-auto mt-9 inline-flex">
            <div
              aria-hidden
              className="absolute -inset-8 rounded-[48px] opacity-70 blur-2xl"
              style={{
                background:
                  "conic-gradient(from 140deg, #ffd1e8, #e5d4ff, #d6e6ff, #ffe5d0, #ffd1e8)",
              }}
            />
            <div className="relative flex flex-col items-center gap-3 sm:flex-row">
              <Link
                href="/?onboarding=1"
                className="btn-primary px-8 py-4 text-[15px] font-semibold tracking-wide"
              >
                begin →
              </Link>
              <Link
                href="/?lens=play"
                className="btn-ghost px-7 py-4 text-[14px] font-semibold tracking-wide"
              >
                skip, just play ♡
              </Link>
            </div>
          </div>

          {/* sparkles */}
          <div className="mt-8 flex items-center justify-center gap-5 text-[color:var(--ink-muted)]">
            <span className="text-[22px] animate-[blob-float_14s_ease-in-out_infinite]">
              ✿
            </span>
            <span className="text-[18px] animate-[blob-float_9s_ease-in-out_infinite] [animation-delay:-3s]">
              ✦
            </span>
            <span className="text-[22px] animate-[blob-float_11s_ease-in-out_infinite] [animation-delay:-6s]">
              ♡
            </span>
            <span className="text-[18px] animate-[blob-float_13s_ease-in-out_infinite] [animation-delay:-2s]">
              ✦
            </span>
            <span className="text-[22px] animate-[blob-float_10s_ease-in-out_infinite] [animation-delay:-5s]">
              ✿
            </span>
          </div>
        </section>

        {/* Three lenses — the core packaging */}
        <section className="px-6 pb-10 sm:px-8">
          <div className="mb-5 flex items-end justify-between">
            <div className="flex flex-col gap-0.5">
              <span className="text-[10.5px] font-medium uppercase tracking-[0.28em] text-[color:var(--ink-muted)]">
                three lenses
              </span>
              <span className="serif-italic text-[22px] font-medium leading-none text-[color:var(--ink)] sm:text-[26px]">
                choose what the world teaches you
              </span>
            </div>
            <span className="hidden text-[11px] font-medium tracking-[0.22em] text-[color:var(--ink-muted)] sm:inline">
              same camera · different soul
            </span>
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            {LENSES.map((l) => (
              <Link
                key={l.kicker}
                href={l.href}
                className={`group relative overflow-hidden rounded-[28px] bg-white/75 p-6 shadow-[0_18px_36px_-22px_rgba(42,21,64,0.28)] ring-1 ring-white/80 backdrop-blur-md transition hover:-translate-y-1 hover:bg-white ${l.rotate} hover:rotate-0`}
              >
                <div
                  aria-hidden
                  className={`absolute -right-12 -top-12 h-36 w-36 rounded-full bg-gradient-to-br ${l.tint} opacity-80 blur-2xl transition group-hover:scale-110`}
                />
                <div className="relative flex items-center justify-between">
                  <span className="text-[10.5px] font-medium uppercase tracking-[0.28em] text-[color:var(--ink-muted)]">
                    {l.kicker}
                  </span>
                  <span className="text-[22px] text-[color:var(--accent)] transition-transform group-hover:rotate-12">
                    {l.emoji}
                  </span>
                </div>
                <h3 className="serif-italic relative mt-3 text-[26px] font-medium leading-tight text-[color:var(--ink)]">
                  {l.title}
                </h3>
                <p className="relative mt-2 text-[13.5px] leading-[1.5] text-[color:var(--ink-soft)]">
                  {l.copy}
                </p>
                <div className="relative mt-5 flex items-center gap-2 text-[color:var(--accent-deep)]">
                  <span className="text-[11.5px] font-semibold uppercase tracking-[0.18em]">
                    try this lens
                  </span>
                  <span
                    aria-hidden
                    className="transition-transform group-hover:translate-x-1"
                  >
                    →
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* The cast — object gallery across all three lenses */}
        <section className="px-6 pb-10 sm:px-8">
          <div className="mb-5 flex items-end justify-between">
            <div className="flex flex-col gap-0.5">
              <span className="text-[10.5px] font-medium uppercase tracking-[0.28em] text-[color:var(--ink-muted)]">
                the cast
              </span>
              <span className="serif-italic text-[22px] font-medium leading-none text-[color:var(--ink)] sm:text-[26px]">
                a few things, each with something to say
              </span>
            </div>
            <span className="hidden text-[11px] font-medium tracking-[0.22em] text-[color:var(--ink-muted)] sm:inline">
              tap · lock · listen — &lt;500ms
            </span>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {CAST.map((c) => (
              <article
                key={c.name}
                className={`group relative overflow-hidden rounded-[28px] bg-white/75 p-5 shadow-[0_18px_36px_-22px_rgba(42,21,64,0.28)] ring-1 ring-white/80 backdrop-blur-md transition hover:-translate-y-1 hover:bg-white ${c.rotate} hover:rotate-0`}
              >
                <div
                  aria-hidden
                  className={`absolute -right-12 -top-12 h-32 w-32 rounded-full bg-gradient-to-br ${c.tint} opacity-80 blur-2xl transition group-hover:scale-110`}
                />
                <div className="relative flex items-center justify-between">
                  <span className="grid h-12 w-12 place-items-center rounded-2xl bg-white/85 text-[26px] shadow-[0_6px_16px_-10px_rgba(42,21,64,0.3)] ring-1 ring-white/90">
                    {c.emoji}
                  </span>
                  <span className="rounded-full bg-white/80 px-2.5 py-1 text-[10px] font-medium uppercase tracking-[0.22em] text-[color:var(--ink-muted)] ring-1 ring-white/80">
                    {c.name}
                  </span>
                </div>
                <p className="relative mt-4 text-[11px] font-medium uppercase tracking-[0.22em] text-[color:var(--ink-muted)]">
                  {c.state}
                </p>
                <p className="serif-italic relative mt-2 text-balance text-[19px] leading-[1.25] text-[color:var(--ink)] sm:text-[20px]">
                  &ldquo;{c.line}&rdquo;
                </p>
                <div className="relative mt-4 flex items-center gap-2">
                  <span className="inline-block h-1.5 w-1.5 rounded-full bg-[color:var(--accent)]" />
                  <span className="text-[11px] font-medium tracking-wide text-[color:var(--ink-soft)]">
                    voiced. bubble-synced. remembers you.
                  </span>
                </div>
              </article>
            ))}

            {/* "your turn" card */}
            <Link
              href="/?onboarding=1"
              className="group relative grid min-h-[220px] place-items-center overflow-hidden rounded-[28px] bg-gradient-to-br from-[#ff89be] via-[#ec4899] to-[#c026d3] p-6 text-center text-white shadow-[0_24px_46px_-20px_rgba(236,72,153,0.55)] transition hover:-translate-y-1 hover:scale-[1.01]"
            >
              <div
                aria-hidden
                className="absolute inset-0 opacity-60 blur-2xl"
                style={{
                  background:
                    "conic-gradient(from 140deg, #ffd1e8, #e5d4ff, #d6e6ff, #ffe5d0, #ffd1e8)",
                }}
              />
              <div className="relative flex flex-col items-center gap-3">
                <span className="text-[30px] animate-[wiggle_2.4s_ease-in-out_infinite]">
                  ✿
                </span>
                <span className="serif-italic text-[24px] font-semibold leading-tight">
                  what speaks
                  <br />
                  to you next?
                </span>
                <span className="text-[12px] font-medium uppercase tracking-[0.24em] opacity-90">
                  pick a lens →
                </span>
              </div>
            </Link>
          </div>
        </section>

        {/* Built for */}
        <section className="px-6 pb-10 sm:px-8">
          <div className="mb-5 flex items-end justify-between">
            <div className="flex flex-col gap-0.5">
              <span className="text-[10.5px] font-medium uppercase tracking-[0.28em] text-[color:var(--ink-muted)]">
                built for
              </span>
              <span className="serif-italic text-[22px] font-medium leading-none text-[color:var(--ink)] sm:text-[26px]">
                everyone one tap from a conversation
              </span>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            {PERSONAS.map((p) => (
              <div
                key={p.kicker}
                className="group relative overflow-hidden rounded-[26px] bg-white/75 p-5 shadow-[0_14px_30px_-18px_rgba(42,21,64,0.22)] ring-1 ring-white/80 backdrop-blur-md transition hover:bg-white"
              >
                <div
                  aria-hidden
                  className={`absolute -right-10 -top-10 h-28 w-28 rounded-full bg-gradient-to-br ${p.tint} opacity-80 blur-2xl transition group-hover:scale-110`}
                />
                <div className="relative flex items-center justify-between">
                  <span className="text-[10.5px] font-medium uppercase tracking-[0.28em] text-[color:var(--ink-muted)]">
                    {p.kicker}
                  </span>
                  <span className="text-[18px] text-[color:var(--accent)]">
                    {p.emoji}
                  </span>
                </div>
                <h3 className="serif-italic relative mt-2 text-[22px] font-medium leading-tight text-[color:var(--ink)]">
                  {p.title}
                </h3>
                <p className="relative mt-1.5 text-[13.5px] leading-[1.5] text-[color:var(--ink-soft)]">
                  {p.copy}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* How it works */}
        <section className="px-6 pb-10 sm:px-8">
          <div className="rounded-[32px] bg-white/70 p-6 shadow-[0_18px_36px_-22px_rgba(42,21,64,0.22)] ring-1 ring-white/80 backdrop-blur-md sm:p-8">
            <div className="grid gap-6 sm:grid-cols-3">
              {[
                {
                  k: "one · choose",
                  t: "pick your lens",
                  c: "play, language, or history. a 10-second onboarding tunes the world to you. switch lenses any time.",
                },
                {
                  k: "two · tap",
                  t: "one tap, one lock",
                  c: "point the camera, tap the thing. a face locks on in under half a second. it sees what you see.",
                },
                {
                  k: "three · listen",
                  t: "it speaks. you reply.",
                  c: "voice streams live, bubble text crawls in sync. hold the mic, talk back, keep the thread going.",
                },
              ].map((s) => (
                <div key={s.k} className="flex flex-col gap-2">
                  <span className="text-[10.5px] font-medium uppercase tracking-[0.28em] text-[color:var(--ink-muted)]">
                    {s.k}
                  </span>
                  <h4 className="serif-italic text-[20px] font-medium leading-tight text-[color:var(--ink)]">
                    {s.t}
                  </h4>
                  <p className="text-[13.5px] leading-[1.55] text-[color:var(--ink-soft)]">
                    {s.c}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="px-6 pb-[max(env(safe-area-inset-bottom),28px)] pt-2 sm:px-8">
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-full bg-white/70 px-5 py-2.5 shadow-[0_2px_10px_-4px_rgba(42,21,64,0.15)] ring-1 ring-white/80 backdrop-blur-md">
            <span className="serif-italic text-[13px] text-[color:var(--ink-soft)]">
              万物皆有声 · everything has a voice
            </span>
            <div className="flex items-center gap-2">
              <Link
                href="/gallery"
                className="btn-ghost px-3.5 py-1.5 text-[11.5px] font-medium"
              >
                gallery
              </Link>
              <Link
                href="/?onboarding=1"
                className="btn-primary px-4 py-1.5 text-[12px] font-semibold"
              >
                begin
              </Link>
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
}
