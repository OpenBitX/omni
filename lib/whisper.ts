// On-device speech recognition via transformers.js + whisper-tiny
// (multilingual, ~40MB quantized). Runs in the browser — no network round
// trip per transcription. Used as the second leg of the STT race in
// `components/tracker.tsx`:
//
//   1. Web Speech API (instant, runs during recording) — preferred when it
//      returns a non-empty transcript by mic-release time.
//   2. This module — the on-device fallback. Beats the old
//      `gpt-4o-mini-transcribe` server fallback on latency (no upload, no
//      network) and on offline robustness. Multilingual covers en + zh.
//
// First call downloads weights from the HuggingFace CDN; subsequent calls
// hit the browser's IndexedDB cache. Lazy — never loaded unless the user
// presses the mic button at least once.

import {
  pipeline,
  env,
  type AutomaticSpeechRecognitionPipeline,
} from "@huggingface/transformers";

import type { Lang } from "@/app/actions";

// Bundled model weights aren't shipped in /public — pull from the HF CDN.
// Cached in IndexedDB after first download.
env.allowLocalModels = false;

const WHISPER_MODEL_ID = "Xenova/whisper-tiny";

export type WhisperStage = "idle" | "loading" | "ready" | "error";

export type WhisperStatus = {
  stage: WhisperStage;
  /** Latest progress fraction in [0,1] for the file currently downloading. */
  progress?: number;
  /** File currently downloading (e.g. "encoder_model_quantized.onnx"). */
  file?: string;
  /** Set when stage === "error". */
  error?: string;
};

let currentStatus: WhisperStatus = { stage: "idle" };
const statusListeners = new Set<(s: WhisperStatus) => void>();

function setStatus(next: WhisperStatus) {
  currentStatus = next;
  for (const cb of statusListeners) {
    try {
      cb(next);
    } catch {
      // listener errors don't break the chain
    }
  }
}

export function subscribeWhisperStatus(
  cb: (s: WhisperStatus) => void
): () => void {
  cb(currentStatus);
  statusListeners.add(cb);
  return () => {
    statusListeners.delete(cb);
  };
}

let pipelinePromise: Promise<AutomaticSpeechRecognitionPipeline> | null = null;

// Kick off (or reuse) the model load. Safe to call multiple times — only the
// first call actually starts the download.
export function initWhisper(): Promise<AutomaticSpeechRecognitionPipeline> {
  if (pipelinePromise) return pipelinePromise;
  setStatus({ stage: "loading", progress: 0 });
  // eslint-disable-next-line no-console
  console.log(`[whisper] ▶ loading ${WHISPER_MODEL_ID}`);
  const t0 = performance.now();
  pipelinePromise = (async () => {
    try {
      const transcriber = (await pipeline(
        "automatic-speech-recognition",
        WHISPER_MODEL_ID,
        {
          // Per-subcomponent dtype. The decoder_model_merged q8 export of
          // Xenova/whisper-tiny trips an onnxruntime-web bug
          // (TransposeDQWeightsForMatMulNBits: missing scale for
          // model.decoder.embed_tokens.weight_merged_0) — so keep the
          // decoder in fp32 while letting the encoder stay quantized.
          dtype: {
            encoder_model: "q8",
            decoder_model_merged: "fp32",
          },
          progress_callback: (info: unknown) => {
            const i = info as {
              status?: string;
              progress?: number;
              file?: string;
            };
            if (i?.status === "progress" && typeof i.progress === "number") {
              setStatus({
                stage: "loading",
                progress: i.progress / 100,
                file: i.file,
              });
            }
          },
        }
      )) as AutomaticSpeechRecognitionPipeline;
      setStatus({ stage: "ready" });
      // eslint-disable-next-line no-console
      console.log(
        `[whisper] ✓ ready in ${Math.round(performance.now() - t0)}ms`
      );
      return transcriber;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus({ stage: "error", error: msg });
      // eslint-disable-next-line no-console
      console.warn("[whisper] ✖ load failed:", msg);
      // Allow retry on next call by clearing the cached promise.
      pipelinePromise = null;
      throw err;
    }
  })();
  return pipelinePromise;
}

// MediaRecorder gives us webm/opus (or mp4 on Safari). Whisper wants 16kHz
// mono PCM Float32. Decode + resample via OfflineAudioContext.
async function blobToPcm16k(blob: Blob): Promise<Float32Array> {
  const arrBuf = await blob.arrayBuffer();
  // Decode at the device's native rate first — Safari refuses to decode
  // straight into a non-default sampleRate context.
  const decodeCtx = new (window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext })
      .webkitAudioContext)();
  let decoded: AudioBuffer;
  try {
    decoded = await decodeCtx.decodeAudioData(arrBuf.slice(0));
  } finally {
    void decodeCtx.close();
  }
  const targetRate = 16000;
  if (decoded.sampleRate === targetRate && decoded.numberOfChannels === 1) {
    // Already in the shape we want — copy out the channel data.
    return decoded.getChannelData(0).slice();
  }
  // Mix-down + resample by feeding the decoded buffer through an
  // OfflineAudioContext targeting 16kHz mono.
  const targetLength = Math.max(
    1,
    Math.ceil((decoded.duration || decoded.length / decoded.sampleRate) * targetRate)
  );
  const offline = new OfflineAudioContext(1, targetLength, targetRate);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start(0);
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0).slice();
}

// Run on-device transcription against the recorded blob. Returns the
// transcript string, or "" if nothing was recognized.
export async function transcribeBlob(
  blob: Blob,
  lang: Lang = "en",
  turnTag = ""
): Promise<string> {
  const transcriber = await initWhisper();
  const tag = `[whisper${turnTag}]`;
  const tDecode = performance.now();
  const pcm = await blobToPcm16k(blob);
  const decodeMs = Math.round(performance.now() - tDecode);
  // eslint-disable-next-line no-console
  console.log(
    `${tag} ▶ pcm=${pcm.length} samples (${(pcm.length / 16000).toFixed(2)}s) decode=${decodeMs}ms`
  );

  const tInfer = performance.now();
  const result = (await transcriber(pcm, {
    // Whisper is multilingual — pin the language so it doesn't autodetect
    // a wrong one on a 1-2 word utterance.
    language: lang === "zh" ? "chinese" : "english",
    task: "transcribe",
    // Short utterances only — the talk button caps recordings to under
    // ~30s in practice. One chunk keeps inference simple.
    chunk_length_s: 30,
    return_timestamps: false,
  } as Parameters<AutomaticSpeechRecognitionPipeline>[1])) as
    | { text: string }
    | { text: string }[];

  const inferMs = Math.round(performance.now() - tInfer);
  const text = (Array.isArray(result) ? result.map((r) => r.text).join(" ") : result.text) ?? "";
  const trimmed = text.replace(/\s+/g, " ").trim();
  // eslint-disable-next-line no-console
  console.log(
    `${tag} ✓ infer=${inferMs}ms "${trimmed.slice(0, 120)}${trimmed.length > 120 ? "…" : ""}"`
  );
  return trimmed;
}
