import { Link } from "react-router-dom";
import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import MermaidDiagram from "@/components/mermaid-diagram";
import { DIAGRAM_CN, DIAGRAM_EN } from "@/i18n/diagrams";

const GITHUB_URL = "https://github.com/OpenBitX/omni";
const CDN = "https://phonicsmaker-storage.sfo3.cdn.digitaloceanspaces.com/omni/landing";

export default function LandingContent() {
  const { t, i18n } = useTranslation();
  const isCN = i18n.language.startsWith("zh");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("omni-lang");
      if (saved === "en" || saved === "zh") i18n.changeLanguage(saved);
    } catch {}
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("omni-lang", i18n.language);
    } catch {}
    document.documentElement.lang = isCN ? "zh-CN" : "en";
  }, [i18n.language]);

  const diagram = isCN ? DIAGRAM_CN : DIAGRAM_EN;

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

      <div className="relative z-10 mx-auto flex w-full max-w-[920px] flex-col">
        {/* Top rail */}
        <header className="flex flex-wrap items-center justify-between gap-2 px-4 pt-3 sm:px-6 sm:pt-5">
          <div className="flex items-baseline gap-2">
            <span className="h-2 w-2 rounded-full bg-[color:var(--accent)] shadow-[0_0_0_4px_rgba(236,72,153,0.18)]" />
            <span className="serif-italic text-[24px] font-semibold leading-none text-[color:var(--ink)] sm:text-[30px]">
              omni
            </span>
            <span className="hidden text-[12px] font-medium tracking-[0.22em] text-[color:var(--ink-muted)] sm:inline">
              · 万物拟人局            </span>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {/* Lang toggle */}
            <div
              role="group"
              aria-label="language"
              className="flex items-center gap-0.5 rounded-full bg-white/70 p-0.5 shadow-[0_2px_10px_-4px_rgba(42,21,64,0.15)] ring-1 ring-white/80 backdrop-blur-md"
            >
              <button
                type="button"
                onClick={() => i18n.changeLanguage("zh")}
                aria-pressed={isCN}
                className={`rounded-full px-3 py-1 text-[11px] font-semibold tracking-wide transition ${
                  isCN
                    ? "bg-[color:var(--accent)] text-white shadow-[0_6px_14px_-6px_rgba(236,72,153,0.55)]"
                    : "text-[color:var(--ink-soft)] hover:text-[color:var(--ink)]"
                }`}
              >
                中文
              </button>
              <button
                type="button"
                onClick={() => i18n.changeLanguage("en")}
                aria-pressed={!isCN}
                className={`rounded-full px-3 py-1 text-[11px] font-semibold tracking-wide transition ${
                  !isCN
                    ? "bg-[color:var(--accent)] text-white shadow-[0_6px_14px_-6px_rgba(236,72,153,0.55)]"
                    : "text-[color:var(--ink-soft)] hover:text-[color:var(--ink)]"
                }`}
              >
                EN
              </button>
            </div>
            <div className="hidden items-center gap-2 rounded-full bg-white/70 px-3 py-1.5 shadow-[0_2px_10px_-4px_rgba(42,21,64,0.15)] ring-1 ring-white/80 backdrop-blur-md sm:flex">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              <span className="text-[11px] font-medium tracking-wide text-[color:var(--ink-soft)]">
                {t("betaTag")}
              </span>
            </div>
          </div>
        </header>

        {/* Hero */}
        <section className="px-4 pb-4 pt-4 sm:px-6 sm:pb-6 sm:pt-6">
          <div className="grid items-center gap-5 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] md:gap-8">
            <div className="text-center md:text-left">
              <h1 className="serif-italic text-balance text-[36px] font-semibold leading-[0.98] tracking-[-0.02em] text-[color:var(--ink)] sm:text-[54px] md:text-[64px]">
                {isCN ? (
                  <>
                    {t("heroLine1")}
                    <br />
                    <span className="relative inline-block">
                      {t("heroSoul")}
                      <span className="absolute -right-8 -top-2 inline-block animate-[wiggle_2.2s_ease-in-out_infinite] text-[28px] text-[color:var(--accent)] sm:-right-10 sm:text-[44px]">
                        ✿                      </span>
                    </span>
                  </>
                ) : (
                  <>
                    {t("heroLine1")}
                    <br />a{" "}
                    <span className="relative inline-block">
                      {t("heroSoul")}
                      <span className="absolute -right-6 -top-2 inline-block animate-[wiggle_2.2s_ease-in-out_infinite] text-[28px] text-[color:var(--accent)] sm:-right-9 sm:text-[44px]">
                        ✿                      </span>
                    </span>
                  </>
                )}
              </h1>

              <p className="mx-auto mt-3 max-w-[500px] text-balance text-[13.5px] leading-[1.5] text-[color:var(--ink-soft)] sm:text-[15px] md:mx-0">
                {t("heroSubPrefix")}
                <span className="serif-italic text-[color:var(--ink)]">
                  {t("heroSubMid")}
                </span>
                {t("heroSubSuffix")}
              </p>

              <div className="relative mt-5 inline-flex w-full sm:w-auto">
                <div
                  aria-hidden
                  className="absolute -inset-6 rounded-[48px] opacity-70 blur-2xl sm:-inset-8"
                  style={{
                    background:
                      "conic-gradient(from 140deg, #ffd1e8, #e5d4ff, #d6e6ff, #ffe5d0, #ffd1e8)",
                  }}
                />
                <div className="relative flex w-full flex-col items-stretch gap-2 sm:w-auto sm:flex-row sm:items-center">
                  <Link
                    to="/"
                    className="btn-primary px-5 py-2.5 text-center text-[13px] font-semibold tracking-wide"
                  >
                    {t("ctaBegin")}
                  </Link>
                  <Link
                    to="/?lens=play"
                    className="btn-ghost px-4 py-2.5 text-center text-[12.5px] font-semibold tracking-wide"
                  >
                    {t("ctaSkip")}
                  </Link>
                </div>
              </div>

              <div className="mt-4 flex items-center justify-center gap-3 text-[color:var(--ink-muted)] md:justify-start">
                <span className="text-[16px] animate-[blob-float_14s_ease-in-out_infinite]">
                  ✿                </span>
                <span className="text-[13px] animate-[blob-float_9s_ease-in-out_infinite] [animation-delay:-3s]">
                  ✦                </span>
                <span className="text-[16px] animate-[blob-float_11s_ease-in-out_infinite] [animation-delay:-6s]">
                  ♡                </span>
                <span className="text-[13px] animate-[blob-float_13s_ease-in-out_infinite] [animation-delay:-2s]">
                  ✦                </span>
                <span className="text-[16px] animate-[blob-float_10s_ease-in-out_infinite] [animation-delay:-5s]">
                  ✿                </span>
              </div>
            </div>

            <div className="relative mx-auto w-full max-w-[320px] md:max-w-[380px]">
              <div
                aria-hidden
                className="absolute -inset-4 rounded-[32px] opacity-70 blur-3xl"
                style={{
                  background:
                    "conic-gradient(from 140deg, #ffd1e8, #e5d4ff, #d6e6ff, #ffe5d0, #ffd1e8)",
                }}
              />
              <div className="relative overflow-hidden rounded-[22px] bg-white/80 p-1.5 shadow-[0_20px_40px_-22px_rgba(42,21,64,0.4)] ring-1 ring-white/80 backdrop-blur-md">
                <video
                  poster="/landing/chat-with-anything.jpg"
                  controls
                  playsInline
                  preload="metadata"
                  aria-label={isCN ? "看看万物开口" : "chat with anything"}
                  className="block h-auto w-full rounded-[16px] bg-black object-cover"
                  onError={(e) => {
                    const v = e.currentTarget;
                    if (!v.dataset.fellBack) {
                      v.dataset.fellBack = "1";
                      v.src = `${CDN}/tapnow.mp4`;
                      v.load();
                    }
                  }}
                >
                  <source src="/landing/tapnow.mp4" type="video/mp4" />
                  <source src={`${CDN}/tapnow.mp4`} type="video/mp4" />
                </video>
              </div>
            </div>
          </div>
        </section>

        {/* Demo video */}
        <section className="px-4 pb-5 sm:px-6">
          <div className="mb-2 flex flex-col gap-0.5 sm:mb-3 sm:flex-row sm:items-end sm:justify-between sm:gap-2">
            <div className="flex flex-col gap-0.5">
              <span className="text-[9.5px] font-medium uppercase tracking-[0.28em] text-[color:var(--ink-muted)]">
                {t("demoKicker")}
              </span>
              <span className="serif-italic text-[17px] font-medium leading-tight text-[color:var(--ink)] sm:text-[20px]">
                {t("demoTitle")}
              </span>
            </div>
            <span className="text-[10px] font-medium tracking-[0.22em] text-[color:var(--ink-muted)]">
              {t("demoHint")}
            </span>
          </div>
          <div className="overflow-hidden rounded-[16px] bg-black/5 shadow-[0_14px_28px_-20px_rgba(42,21,64,0.28)] ring-1 ring-white/80 sm:rounded-[20px]">
            <video
              controls
              playsInline
              preload="metadata"
              className="h-auto w-full"
              onError={(e) => {
                const v = e.currentTarget;
                if (!v.dataset.fellBack) {
                  v.dataset.fellBack = "1";
                  v.src = `${CDN}/demo.mp4`;
                  v.load();
                }
              }}
            >
              <source src="/landing/demo.mp4" type="video/mp4" />
              <source src={`${CDN}/demo.mp4`} type="video/mp4" />
            </video>
          </div>
        </section>

        {/* Open source */}
        <section className="px-4 pb-5 sm:px-6">
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer noopener"
            className="group relative block overflow-hidden rounded-[20px] bg-gradient-to-br from-[#24292f] via-[#1f2328] to-[#0d1117] p-4 text-center text-white shadow-[0_20px_40px_-22px_rgba(13,17,23,0.65)] ring-1 ring-white/10 transition hover:-translate-y-0.5 hover:scale-[1.003] sm:rounded-[24px] sm:p-6"
          >
            <div
              aria-hidden
              className="absolute inset-0 opacity-30 blur-3xl"
              style={{
                background:
                  "radial-gradient(circle at 20% 20%, #2f81f7 0%, transparent 55%), radial-gradient(circle at 80% 80%, #8957e5 0%, transparent 55%)",
              }}
            />
            <div className="relative flex flex-col items-center gap-2">
              <span className="inline-flex items-center gap-1.5 rounded-full bg-white/10 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.24em] text-white/90 ring-1 ring-white/15">
                <svg
                  aria-hidden
                  viewBox="0 0 24 24"
                  width="14"
                  height="14"
                  fill="currentColor"
                >
                  <path d="M12 .5C5.65.5.5 5.65.5 12.02c0 5.1 3.29 9.42 7.86 10.95.58.11.79-.25.79-.56 0-.28-.01-1.01-.02-1.98-3.2.7-3.87-1.54-3.87-1.54-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.7 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.07 0 0 .97-.31 3.18 1.18a11.06 11.06 0 0 1 5.79 0c2.21-1.49 3.18-1.18 3.18-1.18.63 1.6.23 2.78.11 3.07.74.81 1.19 1.84 1.19 3.1 0 4.43-2.69 5.4-5.26 5.69.41.35.77 1.04.77 2.1 0 1.51-.01 2.73-.01 3.1 0 .31.21.67.8.56C20.22 21.43 23.5 17.11 23.5 12 23.5 5.65 18.35.5 12 .5z" />
                </svg>
                {t("openKicker")}
              </span>
              <h3 className="serif-italic text-balance text-[22px] font-semibold leading-[1.05] text-white sm:text-[32px]">
                {t("openTitle")}
              </h3>
              <p className="max-w-[520px] text-balance text-[12.5px] leading-[1.5] text-white/80 sm:text-[14px]">
                {t("openCopy")}
              </p>
              <span className="mt-1 inline-flex items-center gap-2 rounded-full bg-[#238636] px-4 py-2 text-[12.5px] font-semibold tracking-wide text-white shadow-[0_10px_20px_-10px_rgba(35,134,54,0.7)] ring-1 ring-[#2ea043]/60 transition group-hover:gap-2.5 group-hover:bg-[#2ea043] sm:px-5 sm:py-2.5 sm:text-[13px]">
                <svg
                  aria-hidden
                  viewBox="0 0 24 24"
                  width="16"
                  height="16"
                  fill="currentColor"
                >
                  <path d="M12 .5C5.65.5.5 5.65.5 12.02c0 5.1 3.29 9.42 7.86 10.95.58.11.79-.25.79-.56 0-.28-.01-1.01-.02-1.98-3.2.7-3.87-1.54-3.87-1.54-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.7 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.07 0 0 .97-.31 3.18 1.18a11.06 11.06 0 0 1 5.79 0c2.21-1.49 3.18-1.18 3.18-1.18.63 1.6.23 2.78.11 3.07.74.81 1.19 1.84 1.19 3.1 0 4.43-2.69 5.4-5.26 5.69.41.35.77 1.04.77 2.1 0 1.51-.01 2.73-.01 3.1 0 .31.21.67.8.56C20.22 21.43 23.5 17.11 23.5 12 23.5 5.65 18.35.5 12 .5z" />
                </svg>
                {t("openBtn")}
                <span
                  aria-hidden
                  className="transition-transform group-hover:translate-x-1"
                >
                  →                </span>
              </span>
              <span className="max-w-[320px] text-balance text-[9.5px] font-medium uppercase tracking-[0.22em] text-white/70 sm:max-w-none sm:text-[10px] sm:tracking-[0.24em]">
                {t("openFoot")}
              </span>
            </div>
          </a>
        </section>

        {/* Mermaid diagram */}
        <section className="px-4 pb-5 sm:px-6">
          <div className="mb-2 flex flex-col gap-0.5 sm:mb-3 sm:flex-row sm:items-end sm:justify-between">
            <div className="flex flex-col gap-0.5">
              <span className="text-[9.5px] font-medium uppercase tracking-[0.28em] text-[color:var(--ink-muted)]">
                {t("diagramKicker")}
              </span>
              <span className="serif-italic text-[17px] font-medium leading-tight text-[color:var(--ink)] sm:text-[20px]">
                {t("diagramTitle")}
              </span>
            </div>
            <span className="text-[10px] font-medium tracking-[0.22em] text-[color:var(--ink-muted)]">
              {t("diagramHint")}
            </span>
          </div>
          <div className="-mx-4 overflow-x-auto rounded-none bg-white/80 p-3 shadow-[0_14px_28px_-22px_rgba(42,21,64,0.28)] ring-1 ring-white/80 backdrop-blur-md sm:mx-0 sm:rounded-[20px] sm:p-5">
            <MermaidDiagram
              chart={diagram}
              className="mx-auto flex min-w-[900px] justify-center [&_svg]:h-auto [&_svg]:max-w-full"
            />
          </div>
        </section>

        {/* User clips */}
        <section className="px-4 pb-5 sm:px-6">
          <div className="mb-2 flex flex-col gap-0.5 sm:mb-3 sm:flex-row sm:items-end sm:justify-between sm:gap-2">
            <div className="flex flex-col gap-0.5">
              <span className="text-[9.5px] font-medium uppercase tracking-[0.28em] text-[color:var(--ink-muted)]">
                {t("clipsKicker")}
              </span>
              <span className="serif-italic text-[17px] font-medium leading-tight text-[color:var(--ink)] sm:text-[20px]">
                {t("clipsTitle")}
              </span>
            </div>
            <span className="text-[10px] font-medium tracking-[0.22em] text-[color:var(--ink-muted)]">
              {t("clipsHint")}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-2 sm:gap-3">
            {["clip1", "clip2", "clip3"].map((name, i) => (
              <div
                key={name}
                className="group relative overflow-hidden rounded-[14px] bg-black/5 shadow-[0_12px_24px_-18px_rgba(42,21,64,0.28)] ring-1 ring-white/80 sm:rounded-[18px]"
                style={{ animationDelay: `${i * 0.15}s` }}
              >
                <video
                  autoPlay
                  loop
                  muted
                  playsInline
                  preload="metadata"
                  className="aspect-[9/16] h-auto w-full object-cover"
                  src={`${CDN}/${name}.mp4`}
                />
                <div
                  aria-hidden
                  className="pointer-events-none absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-black/30 to-transparent"
                />
              </div>
            ))}
          </div>
        </section>

        {/* Vision section */}
        <section className="px-4 pb-6 sm:px-6">
          <div
            className="relative overflow-hidden rounded-[22px] p-5 shadow-[0_18px_38px_-22px_rgba(190,24,93,0.38)] ring-1 ring-rose-200/70 backdrop-blur-md sm:rounded-[26px] sm:p-7"
            style={{
              background:
                "linear-gradient(160deg, #fff0f0 0%, #ffe2e4 45%, #fff4e8 100%)",
            }}
          >
            <div
              aria-hidden
              className="pointer-events-none absolute -top-16 -right-10 h-60 w-60 rounded-full opacity-70 blur-3xl"
              style={{ background: "radial-gradient(circle, #ffb4bf 0%, transparent 70%)" }}
            />
            <div
              aria-hidden
              className="pointer-events-none absolute -bottom-20 -left-14 h-64 w-64 rounded-full opacity-55 blur-3xl"
              style={{ background: "radial-gradient(circle, #ffc9a8 0%, transparent 70%)" }}
            />
            <div
              aria-hidden
              className="pointer-events-none absolute inset-0 rounded-[22px] ring-1 ring-rose-300/50 sm:rounded-[26px]"
              style={{
                background:
                  "repeating-linear-gradient(135deg, transparent 0 14px, rgba(244,63,94,0.035) 14px 16px)",
              }}
            />
            <span
              aria-hidden
              className="pointer-events-none absolute right-4 top-4 animate-[wiggle_2.4s_ease-in-out_infinite] text-[22px] text-rose-500 sm:right-6 sm:top-6 sm:text-[28px]"
            >
              ✿            </span>

            <div className="relative flex flex-col gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[9.5px] font-medium uppercase tracking-[0.28em] text-rose-500/80">
                  {t("visionKicker")}
                </span>
                <span className="inline-flex items-center gap-1 rounded-full bg-rose-100 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-rose-700 ring-1 ring-rose-300/80">
                  {t("visionWarn")}
                </span>
              </div>

              <h3 className="serif-italic text-balance text-[22px] font-semibold leading-[1.08] text-rose-950 sm:text-[30px]">
                {t("visionTitle")}
              </h3>

              <p className="serif-italic text-balance text-[14.5px] leading-[1.45] text-rose-900 sm:text-[17px]">
                {t("visionLede")}
              </p>

              <p className="text-balance text-[13px] leading-[1.6] text-rose-900/80 sm:text-[14.5px]">
                {t("visionBody")}
              </p>

              <p className="serif-italic text-balance text-[13.5px] leading-[1.55] text-rose-950 sm:text-[15.5px]">
                {t("visionClosing")}
              </p>

              <div className="mt-2 flex flex-col gap-2 rounded-[16px] bg-white/80 p-3 ring-1 ring-rose-200/80 sm:flex-row sm:items-start sm:gap-3 sm:p-4">
                <span className="inline-flex shrink-0 items-center gap-1.5 self-start rounded-full bg-rose-500 px-3 py-1 text-[10.5px] font-semibold uppercase tracking-[0.2em] text-white shadow-[0_6px_14px_-6px_rgba(244,63,94,0.55)]">
                  {t("visionNextBadge")} →                </span>
                <p className="text-balance text-[12.5px] leading-[1.55] text-rose-900/80 sm:text-[13.5px]">
                  {t("visionNext")}
                </p>
              </div>

              <div className="mt-1 flex items-center gap-2 text-rose-400">
                <span className="text-[14px] animate-[blob-float_12s_ease-in-out_infinite]">✿</span>
                <span className="text-[11px] animate-[blob-float_9s_ease-in-out_infinite] [animation-delay:-3s]">✦</span>
                <span className="text-[14px] animate-[blob-float_11s_ease-in-out_infinite] [animation-delay:-6s]">♡</span>
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="px-4 pb-[max(env(safe-area-inset-bottom),16px)] pt-1 sm:px-6">
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-[16px] bg-white/70 px-3 py-2 shadow-[0_2px_10px_-4px_rgba(42,21,64,0.15)] ring-1 ring-white/80 backdrop-blur-md sm:rounded-full sm:px-4">
            <span className="serif-italic text-[11.5px] text-[color:var(--ink-soft)]">
              {t("footerTagline")}
            </span>
            <div className="flex items-center gap-1.5">
              <Link
                to="/gallery"
                className="btn-ghost px-3 py-1 text-[10.5px] font-medium"
              >
                {t("footerGallery")}
              </Link>
              <Link
                to="/"
                className="btn-primary px-3.5 py-1 text-[11px] font-semibold"
              >
                {t("footerBegin")}
              </Link>
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
}
