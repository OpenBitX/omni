// Client-side HTTP API wrappers — drop-in replacements for the Next.js
// Server Actions that previously lived in `app/actions.ts`.
//
// Components import from here instead of `@/app/actions`. The function
// signatures are identical; internally each function calls the matching
// `/api/*` Express endpoint.

// ── Types (kept identical to app/actions.ts so components need no changes) ──

export type Lang = "en" | "zh";
export type AppMode = "play" | "language" | "history";

export type VoiceCatalogEntry = {
  id: string;
  name: string;
  vibe: string;
  lang: Lang;
};

export type Assessment = {
  suitable: boolean;
  cx: number;
  cy: number;
  bbox: [number, number, number, number];
  reason: string;
};

export type BundledResult = {
  line: string;
  voiceId: string | null;
  description: string | null;
  name: string | null;
};

export type ChatTurn = { role: "user" | "assistant"; content: string };

export type ConverseResult = {
  transcript: string;
  reply: string;
  voiceId: string | null;
  emotion: string | null;
  speed: string | null;
  teachMode?: boolean;
};

export type GroupPeer = { name: string; description: string | null };
export type GroupTurn = {
  speaker: string;
  line: string;
  role?: "user" | "assistant";
};

export type ReplyBackend = "cerebras" | "openai";

export type TeacherSayArgs = {
  description: string;
  className: string;
  objectName?: string | null;
  spokenLang: Lang;
  learnLang: Lang;
  userText: string;
  history?: ChatTurn[];
  voiceId?: string | null;
  turnId?: string | null;
  teachMode?: boolean;
};

export type TeacherSayResult = {
  line: string;
  voiceId: string;
  turnId: string;
  teachMode: boolean;
};

export type GallerizeCardArgs = {
  description: string;
  objectName?: string | null;
  className?: string | null;
  spokenLang: Lang;
  learnLang: Lang;
};

export type GallerizeCardResult = {
  translatedName: string;
  bilingualIntro: { learn: string; spoken: string };
};

// ── HTTP fetch helpers ────────────────────────────────────────────────────────

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error((err as { error?: string }).error ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

// ── Server action wrappers ────────────────────────────────────────────────────

export async function assessObject(
  imageDataUrl: string,
  tapX: number,
  tapY: number,
  tag?: string | null,
  learnLang?: Lang
): Promise<Assessment> {
  return postJson<Assessment>("/api/assess", { imageDataUrl, tapX, tapY, tag, learnLang });
}

export async function describeObject(
  imageDataUrl: string,
  className: string,
  langInput?: Lang,
  tag?: string | null
): Promise<{ description: string }> {
  return postJson<{ description: string }>("/api/describe", {
    imageDataUrl,
    className,
    lang: langInput,
    tag,
  });
}

export async function generateLine(
  imageDataUrl: string,
  voiceId?: string | null,
  description?: string | null,
  history?: ChatTurn[],
  langInput?: Lang,
  tag?: string | null,
  spokenLangInput?: Lang,
  learnLangInput?: Lang,
  _mode?: AppMode
): Promise<{ line: string; voiceId: string | null; description: string | null; name: string | null }> {
  return postJson("/api/generate-line", {
    imageDataUrl,
    voiceId,
    description,
    history,
    lang: langInput,
    tag,
    spokenLang: spokenLangInput,
    learnLang: learnLangInput,
  });
}

export async function groupLine(args: {
  speaker: { name: string; description: string | null };
  peers: GroupPeer[];
  recentTurns: GroupTurn[];
  mode?: "chat" | "followup";
  lang?: Lang;
}): Promise<{
  line: string;
  emotion: string | null;
  speed: string | null;
  addressing: string | null;
  backend: ReplyBackend;
  ms: number;
}> {
  return postJson("/api/group-line", args);
}

export async function converseWithObject(formData: FormData): Promise<ConverseResult> {
  const res = await fetch("/api/converse", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error((err as { error?: string }).error ?? res.statusText);
  }
  return res.json() as Promise<ConverseResult>;
}

export async function teacherSay(args: TeacherSayArgs): Promise<TeacherSayResult> {
  return postJson<TeacherSayResult>("/api/teacher-say", args);
}

export async function gallerizeCard(args: GallerizeCardArgs): Promise<GallerizeCardResult> {
  return postJson<GallerizeCardResult>("/api/gallerize-card", args);
}
