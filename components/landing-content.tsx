"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import MermaidDiagram from "@/components/mermaid-diagram";

type Lang = "cn" | "en";

const DIAGRAM_CN = `flowchart TB
  U([你 · 用户]):::you

  subgraph DEVICE["📱 浏览器 · 端侧"]
    direction TB
    CAM[[getUserMedia<br/>后置相机]]:::box
    TAP{{手指点击 x, y}}:::event
    MIC{{按住麦克风<br/>MediaRecorder · webm/opus}}:::event

    subgraph VISION["🧠 视觉 · YOLO26n-seg · onnxruntime-web"]
      direction TB
      PREP[letterbox<br/>→ CHW float32]:::box
      ORT{{ORT 会话<br/>WebGPU → WASM SIMD}}:::model
      PROTO[(原型掩码<br/>32 × 160 × 160)]:::store
      DETS[检测框<br/>+ 32 掩码系数]:::box
      CENT[掩码中心<br/>稳定锚点原点]:::box
    end

    subgraph STT["🎙️ 端侧 STT · transformers.js · 唯一路径"]
      direction TB
      PCM[解码 + 重采样<br/>OfflineAudioContext → 16kHz mono]:::box
      WHISPER{{Whisper-tiny · Xenova<br/>encoder q8 · decoder fp32<br/>ORT · IndexedDB 缓存}}:::model
      TXT[转录文本<br/>60s 超时]:::box
    end

    subgraph TRACK["🔁 跟踪 · RAF 60fps 外推"]
      direction TB
      PICK[pickTappedDetection<br/>最小包围 → 最近中心]:::box
      MATCH[IoU ≥ 0.3 匹配<br/>3 帧后回退最近中心]:::box
      EMA[(BoxEMA<br/>位置 0.7 · 尺寸 0.25)]:::store
      VEL[速度外推<br/>EMA 0.75 · 500ms]:::box
      ANCH[(Anchor<br/>盒归一化偏移)]:::store
      LRU[LRU 淘汰<br/>MAX_FACES=3]:::box
    end

    subgraph AUDIO["🎵 Web Audio · 每轨图"]
      direction LR
      SRC[source]:::box --> ANA[analyser]:::box --> GAIN[gain ← opacity]:::box --> OUT[destination]:::box
    end

    FACE[[FaceVoice<br/>眼睛视频 + 9 嘴型 PNG]]:::box
    SHAPE{{classifyShape<br/>FFT → A..X}}:::event
  end

  subgraph NEXT["☁️ Next.js 服务器动作 · app/actions.ts"]
    direction TB
    ASSESS[[assessObject<br/>放置裁剪]]:::action
    BUNDLE[[generateLine · 首次点击<br/>描述 + 声音 + 台词 绑定]]:::action
    RETAP[[generateLine · 再点击<br/>纯文本]]:::action
    CONVO[[converseWithObject<br/>仅生成回复]]:::action
    TTSR[[/api/tts/stream<br/>直通 MediaSource]]:::action
  end

  subgraph PROV["🌐 AI 提供商（云端）"]
    direction TB
    GLM[(GLM-5v-turbo<br/>推理 VLM · ~4s)]:::prov
    GPT[(gpt-4o-mini<br/>视觉 · ~1.5s)]:::prov
    CER[(Cerebras llama3.1-8b<br/>纯文本 · ~200ms)]:::prov
    FISH[(Fish.audio s1<br/>latency=balanced)]:::prov
    OAITTS[(OpenAI tts-1 · 兜底)]:::prov
    VCAT[[VOICE_CATALOG<br/>9 种音色]]:::store
  end

  PERSONA[(🃏 人设卡<br/>voiceId + description<br/>钉在 TrackRefs)]:::pin

  CAM --> PREP --> ORT --> PROTO
  ORT --> DETS --> CENT
  U -- 点击 --> TAP --> PICK
  DETS --> PICK
  CENT --> ANCH
  PICK --> MATCH --> EMA --> VEL --> FACE
  EMA --> ANCH --> FACE
  PICK --> LRU

  PICK -. 并行 .-> ASSESS
  PICK -. 并行 .-> BUNDLE
  ASSESS --> GLM
  VCAT --> BUNDLE
  BUNDLE --> GPT
  BUNDLE --> PERSONA

  U -- 说话 --> MIC --> PCM --> WHISPER --> TXT --> CONVO
  PERSONA --> CONVO
  CONVO --> CER

  U -- 再点击 --> RETAP
  PERSONA --> RETAP
  RETAP --> CER

  BUNDLE -- line --> TTSR
  RETAP -- line --> TTSR
  CONVO -- reply --> TTSR
  TTSR --> FISH
  TTSR -- 兜底 --> OAITTS
  FISH -- audio/mpeg --> SRC
  OAITTS -- audio/mpeg --> SRC
  ANA --> SHAPE --> FACE

  classDef you fill:#fff,stroke:#ec4899,stroke-width:3px,color:#2a1540,font-weight:700;
  classDef box fill:#ffe4f2,stroke:#ec4899,color:#2a1540;
  classDef event fill:#fff4d6,stroke:#d97706,color:#2a1540;
  classDef model fill:#e4d1ff,stroke:#c026d3,color:#2a1540;
  classDef action fill:#d6efff,stroke:#2563eb,color:#2a1540;
  classDef prov fill:#d9f5e4,stroke:#059669,color:#2a1540;
  classDef store fill:#ffe8a8,stroke:#b45309,color:#2a1540;
  classDef pin fill:#ffd1e8,stroke:#c026d3,stroke-width:3px,stroke-dasharray:4 3,color:#2a1540,font-weight:700;`;

const DIAGRAM_EN = `flowchart TB
  U([you · the user]):::you

  subgraph DEVICE["📱 browser · on-device"]
    direction TB
    CAM[[getUserMedia<br/>rear camera]]:::box
    TAP{{finger tap · x, y}}:::event
    MIC{{hold mic<br/>MediaRecorder · webm/opus}}:::event

    subgraph VISION["🧠 vision · YOLO26n-seg · onnxruntime-web"]
      direction TB
      PREP[letterbox<br/>→ CHW float32]:::box
      ORT{{ORT session<br/>WebGPU → WASM SIMD}}:::model
      PROTO[(prototype masks<br/>32 × 160 × 160)]:::store
      DETS[detections<br/>+ 32 mask coefs]:::box
      CENT[mask centroid<br/>stable anchor origin]:::box
    end

    subgraph STTSUB["🎙️ on-device STT · transformers.js · sole path"]
      direction TB
      PCM[decode + resample<br/>OfflineAudioContext → 16kHz mono]:::box
      WHISPER{{Whisper-tiny · Xenova<br/>encoder q8 · decoder fp32<br/>ORT · IndexedDB cached}}:::model
      TXT[transcript text<br/>60s timeout]:::box
    end

    subgraph TRACK["🔁 tracker · RAF 60fps extrapolation"]
      direction TB
      PICK[pickTappedDetection<br/>smallest-contains → nearest]:::box
      MATCH[IoU ≥ 0.3 match<br/>widen to nearest after 3 misses]:::box
      EMA[(BoxEMA<br/>pos 0.7 · size 0.25)]:::store
      VEL[velocity extrapolation<br/>EMA 0.75 · 500ms]:::box
      ANCH[(Anchor<br/>box-normalized offsets)]:::store
      LRU[LRU evict<br/>MAX_FACES=3]:::box
    end

    subgraph AUDIO["🎵 Web Audio · per-track graph"]
      direction LR
      SRC[source]:::box --> ANA[analyser]:::box --> GAIN[gain ← opacity]:::box --> OUT[destination]:::box
    end

    FACE[[FaceVoice<br/>eyes video + 9 mouth PNGs]]:::box
    SHAPE{{classifyShape<br/>FFT → A..X}}:::event
  end

  subgraph NEXT["☁️ Next.js server actions · app/actions.ts"]
    direction TB
    ASSESS[[assessObject<br/>face placement]]:::action
    BUNDLE[[generateLine · first tap<br/>describe + voice + line bundled]]:::action
    RETAP[[generateLine · retap<br/>text-only]]:::action
    CONVO[[converseWithObject<br/>reply only · no STT]]:::action
    TTSR[[/api/tts/stream<br/>MediaSource passthrough]]:::action
  end

  subgraph PROV["🌐 AI providers (cloud)"]
    direction TB
    GLM[(GLM-5v-turbo<br/>reasoning VLM · ~4s)]:::prov
    GPT[(gpt-4o-mini<br/>vision · ~1.5s)]:::prov
    CER[(Cerebras llama3.1-8b<br/>text-only · ~200ms)]:::prov
    FISH[(Fish.audio s1<br/>latency=balanced)]:::prov
    OAITTS[(OpenAI tts-1 · fallback)]:::prov
    VCAT[[VOICE_CATALOG<br/>9 curated voices]]:::store
  end

  PERSONA[(🃏 persona card<br/>voiceId + description<br/>pinned on TrackRefs)]:::pin

  CAM --> PREP --> ORT --> PROTO
  ORT --> DETS --> CENT
  U -- tap --> TAP --> PICK
  DETS --> PICK
  CENT --> ANCH
  PICK --> MATCH --> EMA --> VEL --> FACE
  EMA --> ANCH --> FACE
  PICK --> LRU

  PICK -. parallel .-> ASSESS
  PICK -. parallel .-> BUNDLE
  ASSESS --> GLM
  VCAT --> BUNDLE
  BUNDLE --> GPT
  BUNDLE --> PERSONA

  U -- talk --> MIC --> PCM --> WHISPER --> TXT --> CONVO
  PERSONA --> CONVO
  CONVO --> CER

  U -- retap --> RETAP
  PERSONA --> RETAP
  RETAP --> CER

  BUNDLE -- line --> TTSR
  RETAP -- line --> TTSR
  CONVO -- reply --> TTSR
  TTSR --> FISH
  TTSR -- fallback --> OAITTS
  FISH -- audio/mpeg --> SRC
  OAITTS -- audio/mpeg --> SRC
  ANA --> SHAPE --> FACE

  classDef you fill:#fff,stroke:#ec4899,stroke-width:3px,color:#2a1540,font-weight:700;
  classDef box fill:#ffe4f2,stroke:#ec4899,color:#2a1540;
  classDef event fill:#fff4d6,stroke:#d97706,color:#2a1540;
  classDef model fill:#e4d1ff,stroke:#c026d3,color:#2a1540;
  classDef action fill:#d6efff,stroke:#2563eb,color:#2a1540;
  classDef prov fill:#d9f5e4,stroke:#059669,color:#2a1540;
  classDef store fill:#ffe8a8,stroke:#b45309,color:#2a1540;
  classDef pin fill:#ffd1e8,stroke:#c026d3,stroke-width:3px,stroke-dasharray:4 3,color:#2a1540,font-weight:700;`;

type Copy = {
  betaTag: string;
  heroLine1: string;
  heroSoul: string;
  heroSub: (mid: string) => { prefix: string; mid: string; suffix: string };
  heroSubMid: string;
  ctaBegin: string;
  ctaSkip: string;
  demoKicker: string;
  demoTitle: string;
  demoHint: string;
  clipsKicker: string;
  clipsTitle: string;
  clipsHint: string;
  diagramKicker: string;
  diagramTitle: string;
  diagramHint: string;
  diagram: string;
  openKicker: string;
  openTitle: string;
  openCopy: string;
  openBtn: string;
  openFoot: string;
  footerTagline: string;
  footerGallery: string;
  footerBegin: string;
};

const COPY: Record<Lang, Copy> = {
  cn: {
    betaTag: "beta ✿ 三重视角",
    heroLine1: "给万物",
    heroSoul: "一个灵魂",
    heroSub: () => ({
      prefix: "万物皆有声 — ",
      mid: "选择它教你什么。",
      suffix: "把镜头对准世界；它可以陪你玩、教你学，或告诉你这里曾发生过什么。",
    }),
    heroSubMid: "选择它教你什么。",
    ctaBegin: "开始 →",
    ctaSkip: "直接玩 ♡",
    demoKicker: "演示",
    demoTitle: "看它活起来",
    demoHint: "点击 · 锁定 · 倾听",
    clipsKicker: "现场",
    clipsTitle: "用户正在玩",
    clipsHint: "真实使用 · 未剪辑",
    diagramKicker: "幕后",
    diagramTitle: "一次点击如何化为声音",
    diagramHint: "端侧识别 · 云端人设 · 流式语音",
    diagram: DIAGRAM_CN,
    openKicker: "开源",
    openTitle: "我们是开源的 ✿",
    openCopy:
      "每个模型、每段提示词、每份完整许可证都公开。克隆它、改造它，给你自己的万物一个灵魂。",
    openBtn: "前往 github",
    openFoot: "我们提供完整许可证 · 所有代码公开 · 全归你",
    footerTagline: "万物皆有声 · everything has a voice",
    footerGallery: "画廊",
    footerBegin: "开始",
  },
  en: {
    betaTag: "beta ✿ three lenses",
    heroLine1: "give everything",
    heroSoul: "soul",
    heroSub: () => ({
      prefix: "everything has a voice — ",
      mid: "choose what it teaches you.",
      suffix:
        " point your camera at the world; decide whether it plays with you, tutors you, or tells you who lived here before.",
    }),
    heroSubMid: "choose what it teaches you.",
    ctaBegin: "begin →",
    ctaSkip: "skip, just play ♡",
    demoKicker: "the demo",
    demoTitle: "watch it come alive",
    demoHint: "tap · lock · listen",
    clipsKicker: "in the wild",
    clipsTitle: "people playing right now",
    clipsHint: "real users · unedited",
    diagramKicker: "under the hood",
    diagramTitle: "how a tap becomes a voice",
    diagramHint: "on-device detect · cloud persona · streamed voice",
    diagram: DIAGRAM_EN,
    openKicker: "open source",
    openTitle: "we\u2019re open source ✿",
    openCopy:
      "every model, prompt, and full license — out in the open. fork it, remix it, give your own things a soul.",
    openBtn: "find it on github",
    openFoot: "we give full licenses · all code public · all yours",
    footerTagline: "万物皆有声 · everything has a voice",
    footerGallery: "gallery",
    footerBegin: "begin",
  },
};

const GITHUB_URL = "https://github.com/OpenBitX/duidui";

export default function LandingContent() {
  const [lang, setLang] = useState<Lang>("cn");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("omni-lang");
      if (saved === "en" || saved === "cn") setLang(saved);
    } catch {}
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("omni-lang", lang);
    } catch {}
    document.documentElement.lang = lang === "cn" ? "zh-CN" : "en";
  }, [lang]);

  const t = COPY[lang];
  const isCN = lang === "cn";
  const sub = t.heroSub(t.heroSubMid);

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
              · 万物皆有声
            </span>
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
                onClick={() => setLang("cn")}
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
                onClick={() => setLang("en")}
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
                {t.betaTag}
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
                    {t.heroLine1}
                    <br />
                    <span className="relative inline-block">
                      {t.heroSoul}
                      <span className="absolute -right-8 -top-2 inline-block animate-[wiggle_2.2s_ease-in-out_infinite] text-[28px] text-[color:var(--accent)] sm:-right-10 sm:text-[44px]">
                        ✿
                      </span>
                    </span>
                  </>
                ) : (
                  <>
                    {t.heroLine1}
                    <br />a{" "}
                    <span className="relative inline-block">
                      {t.heroSoul}
                      <span className="absolute -right-6 -top-2 inline-block animate-[wiggle_2.2s_ease-in-out_infinite] text-[28px] text-[color:var(--accent)] sm:-right-9 sm:text-[44px]">
                        ✿
                      </span>
                    </span>
                  </>
                )}
              </h1>

              <p className="mx-auto mt-3 max-w-[500px] text-balance text-[13.5px] leading-[1.5] text-[color:var(--ink-soft)] sm:text-[15px] md:mx-0">
                {sub.prefix}
                <span className="serif-italic text-[color:var(--ink)]">
                  {sub.mid}
                </span>
                {sub.suffix}
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
                    href="/?onboarding=1"
                    className="btn-primary px-5 py-2.5 text-center text-[13px] font-semibold tracking-wide"
                  >
                    {t.ctaBegin}
                  </Link>
                  <Link
                    href="/?lens=play"
                    className="btn-ghost px-4 py-2.5 text-center text-[12.5px] font-semibold tracking-wide"
                  >
                    {t.ctaSkip}
                  </Link>
                </div>
              </div>

              <div className="mt-4 flex items-center justify-center gap-3 text-[color:var(--ink-muted)] md:justify-start">
                <span className="text-[16px] animate-[blob-float_14s_ease-in-out_infinite]">
                  ✿
                </span>
                <span className="text-[13px] animate-[blob-float_9s_ease-in-out_infinite] [animation-delay:-3s]">
                  ✦
                </span>
                <span className="text-[16px] animate-[blob-float_11s_ease-in-out_infinite] [animation-delay:-6s]">
                  ♡
                </span>
                <span className="text-[13px] animate-[blob-float_13s_ease-in-out_infinite] [animation-delay:-2s]">
                  ✦
                </span>
                <span className="text-[16px] animate-[blob-float_10s_ease-in-out_infinite] [animation-delay:-5s]">
                  ✿
                </span>
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
                  src="/landing/tapnow.mp4"
                  poster="/landing/chat-with-anything.jpg"
                  controls
                  playsInline
                  preload="none"
                  aria-label={isCN ? "和万物聊天 · 点击播放" : "chat with anything · click to play"}
                  className="block h-auto w-full rounded-[16px] bg-black object-cover"
                />
              </div>
            </div>
          </div>
        </section>

        {/* Demo video */}
        <section className="px-4 pb-5 sm:px-6">
          <div className="mb-2 flex flex-col gap-0.5 sm:mb-3 sm:flex-row sm:items-end sm:justify-between sm:gap-2">
            <div className="flex flex-col gap-0.5">
              <span className="text-[9.5px] font-medium uppercase tracking-[0.28em] text-[color:var(--ink-muted)]">
                {t.demoKicker}
              </span>
              <span className="serif-italic text-[17px] font-medium leading-tight text-[color:var(--ink)] sm:text-[20px]">
                {t.demoTitle}
              </span>
            </div>
            <span className="text-[10px] font-medium tracking-[0.22em] text-[color:var(--ink-muted)]">
              {t.demoHint}
            </span>
          </div>
          <div className="overflow-hidden rounded-[16px] bg-black/5 shadow-[0_14px_28px_-20px_rgba(42,21,64,0.28)] ring-1 ring-white/80 sm:rounded-[20px]">
            <video
              controls
              playsInline
              preload="metadata"
              className="h-auto w-full"
              src="/landing/demo.mp4"
            />
          </div>
        </section>

        {/* Open source — big CTA */}
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
                {t.openKicker}
              </span>
              <h3 className="serif-italic text-balance text-[22px] font-semibold leading-[1.05] text-white sm:text-[32px]">
                {t.openTitle}
              </h3>
              <p className="max-w-[520px] text-balance text-[12.5px] leading-[1.5] text-white/80 sm:text-[14px]">
                {t.openCopy}
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
                {t.openBtn}
                <span
                  aria-hidden
                  className="transition-transform group-hover:translate-x-1"
                >
                  →
                </span>
              </span>
              <span className="max-w-[320px] text-balance text-[9.5px] font-medium uppercase tracking-[0.22em] text-white/70 sm:max-w-none sm:text-[10px] sm:tracking-[0.24em]">
                {t.openFoot}
              </span>
            </div>
          </a>
        </section>

        {/* Mermaid diagram */}
        <section className="px-4 pb-5 sm:px-6">
          <div className="mb-2 flex flex-col gap-0.5 sm:mb-3 sm:flex-row sm:items-end sm:justify-between">
            <div className="flex flex-col gap-0.5">
              <span className="text-[9.5px] font-medium uppercase tracking-[0.28em] text-[color:var(--ink-muted)]">
                {t.diagramKicker}
              </span>
              <span className="serif-italic text-[17px] font-medium leading-tight text-[color:var(--ink)] sm:text-[20px]">
                {t.diagramTitle}
              </span>
            </div>
            <span className="text-[10px] font-medium tracking-[0.22em] text-[color:var(--ink-muted)]">
              {t.diagramHint}
            </span>
          </div>
          <div className="-mx-4 overflow-x-auto rounded-none bg-white/80 p-3 shadow-[0_14px_28px_-22px_rgba(42,21,64,0.28)] ring-1 ring-white/80 backdrop-blur-md sm:mx-0 sm:rounded-[20px] sm:p-5">
            <MermaidDiagram
              chart={t.diagram}
              className="mx-auto flex min-w-[900px] justify-center [&_svg]:h-auto [&_svg]:max-w-full"
            />
          </div>
        </section>

        {/* User clips — vertical strip */}
        <section className="px-4 pb-5 sm:px-6">
          <div className="mb-2 flex flex-col gap-0.5 sm:mb-3 sm:flex-row sm:items-end sm:justify-between sm:gap-2">
            <div className="flex flex-col gap-0.5">
              <span className="text-[9.5px] font-medium uppercase tracking-[0.28em] text-[color:var(--ink-muted)]">
                {t.clipsKicker}
              </span>
              <span className="serif-italic text-[17px] font-medium leading-tight text-[color:var(--ink)] sm:text-[20px]">
                {t.clipsTitle}
              </span>
            </div>
            <span className="text-[10px] font-medium tracking-[0.22em] text-[color:var(--ink-muted)]">
              {t.clipsHint}
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
                  src={`/landing/${name}.mov`}
                />
                <div
                  aria-hidden
                  className="pointer-events-none absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-black/30 to-transparent"
                />
              </div>
            ))}
          </div>
        </section>

        {/* Footer */}
        <footer className="px-4 pb-[max(env(safe-area-inset-bottom),16px)] pt-1 sm:px-6">
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-[16px] bg-white/70 px-3 py-2 shadow-[0_2px_10px_-4px_rgba(42,21,64,0.15)] ring-1 ring-white/80 backdrop-blur-md sm:rounded-full sm:px-4">
            <span className="serif-italic text-[11.5px] text-[color:var(--ink-soft)]">
              {t.footerTagline}
            </span>
            <div className="flex items-center gap-1.5">
              <Link
                href="/gallery"
                className="btn-ghost px-3 py-1 text-[10.5px] font-medium"
              >
                {t.footerGallery}
              </Link>
              <Link
                href="/?onboarding=1"
                className="btn-primary px-3.5 py-1 text-[11px] font-semibold"
              >
                {t.footerBegin}
              </Link>
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
}
