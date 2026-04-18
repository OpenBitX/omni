// Thin HTTP client for the Next.js backend that hosts server actions.
// The native app never calls server actions directly (those are same-origin
// RPC) — we expose them as POST /api/* routes in the Next app and hit them
// from here. Base URL comes from EXPO_PUBLIC_API_BASE or defaults to the
// LAN IP set at build time via app.json extra.
//
// Shape of each endpoint mirrors the server action in app/actions.ts.

import Constants from "expo-constants";

function resolveBaseUrl(): string {
  const env = process.env.EXPO_PUBLIC_API_BASE;
  if (env && env.length > 0) return env.replace(/\/+$/, "");
  const extra = Constants.expoConfig?.extra as { apiBase?: string } | undefined;
  if (extra?.apiBase) return extra.apiBase.replace(/\/+$/, "");
  // Fallback: the parent browser app's Next dev server (port 3000) IS the
  // backend. Simulator-only; set EXPO_PUBLIC_API_BASE=http://<LAN-IP>:3000
  // for a physical device.
  return "http://localhost:3000";
}

export const API_BASE = resolveBaseUrl();

export type AssessResult = {
  suitable: boolean;
  cx?: number;
  cy?: number;
  bbox?: { x1: number; y1: number; x2: number; y2: number } | null;
  reason?: string;
};

export type GenerateLineResult = {
  description: string;
  voiceId: string;
  line: string;
};

export type ConverseTurn = { role: "user" | "object"; text: string };

export type ConverseResult = {
  transcript: string;
  reply: string;
  voiceId: string;
};

type Lang = "en" | "zh";

const DEFAULT_TIMEOUT_MS = 30_000;

async function postJson<T>(
  path: string,
  body: unknown,
  timeoutMs = DEFAULT_TIMEOUT_MS
): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`${path} ${res.status}: ${text.slice(0, 200)}`);
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

export async function assessObject(
  imageDataUrl: string,
  tapX: number,
  tapY: number
): Promise<AssessResult> {
  return postJson<AssessResult>("/api/assess", { imageDataUrl, tapX, tapY });
}

export async function describeObject(
  imageDataUrl: string,
  className: string,
  lang: Lang = "en"
): Promise<{ description: string }> {
  return postJson("/api/describe", { imageDataUrl, className, lang });
}

export async function generateLine(
  imageDataUrl: string | null,
  opts: {
    voiceId?: string;
    description?: string;
    className?: string;
    history?: ConverseTurn[];
    lang?: Lang;
  } = {}
): Promise<GenerateLineResult> {
  return postJson<GenerateLineResult>("/api/generate-line", {
    imageDataUrl,
    ...opts,
    lang: opts.lang ?? "en",
  });
}

// Conversation turn. Audio is uploaded as multipart so the existing server
// action signature (FormData) is preserved via /api/converse.
export async function converseWithObject(params: {
  audioUri: string;
  audioMime: string;
  transcript: string;
  className: string;
  description: string;
  voiceId: string;
  history: ConverseTurn[];
  lang?: Lang;
}): Promise<ConverseResult> {
  const form = new FormData();
  // @ts-expect-error — RN FormData file shape
  form.append("audio", {
    uri: params.audioUri,
    name: "speech.m4a",
    type: params.audioMime,
  });
  form.append("transcript", params.transcript);
  form.append("className", params.className);
  form.append("description", params.description);
  form.append("voiceId", params.voiceId);
  form.append("history", JSON.stringify(params.history));
  form.append("lang", params.lang ?? "en");

  const res = await fetch(`${API_BASE}/api/converse`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`converse ${res.status}: ${text.slice(0, 200)}`);
  }
  return (await res.json()) as ConverseResult;
}

// Streaming TTS URL — pass to expo-av's Sound.loadAsync / createAsync.
// Server streams audio/mpeg bytes; expo-av will start playback on first bytes.
export function ttsStreamUrl(): string {
  return `${API_BASE}/api/tts/stream`;
}
