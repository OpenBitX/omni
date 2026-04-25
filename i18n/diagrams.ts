export const DIAGRAM_CN = `flowchart TB
  U([你 · 用户]):::you

  subgraph DEVICE["浏览器 · 设备端"]
    direction TB
    CAM[[getUserMedia<br/>后置摄像头]]:::box
    TAP{{手指点击 · x, y}}:::event
    MIC{{按住说话<br/>MediaRecorder · webm/opus}}:::event

    subgraph VISION["视觉 · YOLO26n-seg · onnxruntime-web"]
      direction TB
      PREP[letterbox<br/>转CHW float32]:::box
      ORT{{ORT 会话<br/>WebGPU 或 WASM SIMD}}:::model
      PROTO[(原型蒙版<br/>32 · 160 · 160)]:::store
      DETS[检测结果<br/>+ 32 蒙版系数]:::box
      CENT[蒙版质心<br/>稳定的锚点原点]:::box
    end

    subgraph STT["设备端 STT · transformers.js · 唯一路径"]
      direction TB
      PCM[解码 + 重采样<br/>OfflineAudioContext 转16kHz mono]:::box
      WHISPER{{Whisper-tiny · Xenova<br/>encoder q8 · decoder fp32<br/>ORT · IndexedDB 缓存}}:::model
      TXT[识别文本<br/>60s 超时]:::box
    end

    subgraph TRACK["追踪器 · RAF 60fps 外推"]
      direction TB
      PICK[pickTappedDetection<br/>最小包含或最近邻居]:::box
      MATCH[IoU · 0.3 匹配<br/>3次未命中后扩大搜索]:::box
      EMA[(BoxEMA<br/>位置 0.7 · 尺寸 0.25)]:::store
      VEL[速度外推<br/>EMA 0.75 · 500ms]:::box
      ANCH[(锚点<br/>框归一化偏移)]:::store
      LRU[LRU 淘汰<br/>MAX_FACES=3]:::box
    end

    subgraph AUDIO["Web Audio · 逐轨道图"]
      direction LR
      SRC[source]:::box --> ANA[analyser]:::box --> GAIN[gain · 随透明度]:::box --> OUT[destination]:::box
    end

    FACE[[FaceVoice<br/>眼睛视频 + 9 嘴型 PNG]]:::box
    SHAPE{{classifyShape<br/>FFT 映射 A..X}}:::event
  end

  subgraph NEXT["Next.js 服务端动作 · app/actions.ts"]
    direction TB
    ASSESS[[assessObject<br/>面部放置]]:::action
    BUNDLE[[generateLine · 首次点击<br/>描述 + 声音 + 台词捆绑]]:::action
    RETAP[[generateLine · 再次点击<br/>纯文本]]:::action
    CONVO[[converseWithObject<br/>仅回复 · 无 STT]]:::action
    TTSR[[/api/tts/stream<br/>MediaSource 透传]]:::action
  end

  subgraph PROV["AI 提供商（云端）"]
    direction TB
    GLM[(GLM-5v-turbo<br/>推理 VLM · ~4秒)]:::prov
    GPT[(gpt-4o-mini<br/>视觉 · ~1.5秒)]:::prov
    CER[(Cerebras llama3.1-8b<br/>纯文本 · ~200ms)]:::prov
    FISH[(Fish.audio s1<br/>延迟=平衡)]:::prov
    OAITTS[(OpenAI tts-1 · 备用)]:::prov
    VCAT[[VOICE_CATALOG<br/>9 种精选声音]]:::store
  end

  PERSONA[(角色卡片<br/>voiceId + 描述<br/>固定在 TrackRefs)]:::pin

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

  U -- 再次点击 --> RETAP
  PERSONA --> RETAP
  RETAP --> CER

  BUNDLE -- line --> TTSR
  RETAP -- line --> TTSR
  CONVO -- reply --> TTSR
  TTSR --> FISH
  TTSR -- 备用 --> OAITTS
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

export const DIAGRAM_EN = `flowchart TB
  U([you · the user]):::you

  subgraph DEVICE["browser · on-device"]
    direction TB
    CAM[[getUserMedia<br/>rear camera]]:::box
    TAP{{finger tap · x, y}}:::event
    MIC{{hold mic<br/>MediaRecorder · webm/opus}}:::event

    subgraph VISION["vision · YOLO26n-seg · onnxruntime-web"]
      direction TB
      PREP[letterbox<br/>·CHW float32]:::box
      ORT{{ORT session<br/>WebGPU or WASM SIMD}}:::model
      PROTO[(prototype masks<br/>32 · 160 · 160)]:::store
      DETS[detections<br/>+ 32 mask coefs]:::box
      CENT[mask centroid<br/>stable anchor origin]:::box
    end

    subgraph STTSUB["on-device STT · transformers.js · sole path"]
      direction TB
      PCM[decode + resample<br/>OfflineAudioContext · 16kHz mono]:::box
      WHISPER{{Whisper-tiny · Xenova<br/>encoder q8 · decoder fp32<br/>ORT · IndexedDB cached}}:::model
      TXT[transcript text<br/>60s timeout]:::box
    end

    subgraph TRACK["tracker · RAF 60fps extrapolation"]
      direction TB
      PICK[pickTappedDetection<br/>smallest-contains · nearest]:::box
      MATCH[IoU · 0.3 match<br/>widen after 3 misses]:::box
      EMA[(BoxEMA<br/>pos 0.7 · size 0.25)]:::store
      VEL[velocity extrapolation<br/>EMA 0.75 · 500ms]:::box
      ANCH[(Anchor<br/>box-normalized offsets)]:::store
      LRU[LRU evict<br/>MAX_FACES=3]:::box
    end

    subgraph AUDIO["Web Audio · per-track graph"]
      direction LR
      SRC[source]:::box --> ANA[analyser]:::box --> GAIN[gain · opacity]:::box --> OUT[destination]:::box
    end

    FACE[[FaceVoice<br/>eyes video + 9 mouth PNGs]]:::box
    SHAPE{{classifyShape<br/>FFT · A..X}}:::event
  end

  subgraph NEXT["Next.js server actions · app/actions.ts"]
    direction TB
    ASSESS[[assessObject<br/>face placement]]:::action
    BUNDLE[[generateLine · first tap<br/>describe + voice + line bundled]]:::action
    RETAP[[generateLine · retap<br/>text-only]]:::action
    CONVO[[converseWithObject<br/>reply only · no STT]]:::action
    TTSR[[/api/tts/stream<br/>MediaSource passthrough]]:::action
  end

  subgraph PROV["AI providers (cloud)"]
    direction TB
    GLM[(GLM-5v-turbo<br/>reasoning VLM · ~4s)]:::prov
    GPT[(gpt-4o-mini<br/>vision · ~1.5s)]:::prov
    CER[(Cerebras llama3.1-8b<br/>text-only · ~200ms)]:::prov
    FISH[(Fish.audio s1<br/>latency=balanced)]:::prov
    OAITTS[(OpenAI tts-1 · fallback)]:::prov
    VCAT[[VOICE_CATALOG<br/>9 curated voices]]:::store
  end

  PERSONA[(persona card<br/>voiceId + description<br/>pinned on TrackRefs)]:::pin

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
