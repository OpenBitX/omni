"use client";

// Persisted onboarding preferences. Written once by /onboarding, read by
// tracker + gallery to set initial mode + language pair. localStorage (not
// sessionStorage) so the choice survives across days, the way a user's
// preferred lens should.

import { useEffect, useState } from "react";

export type Lens = "play" | "language" | "history";

// Two-language surface. Matches the AppLang in session-cards + the voice
// catalog split en/zh. We may widen this later; for now the product stays
// honest and ships only the pairs we've actually tuned.
export type LangCode = "en" | "zh";

export const LANGUAGES: { code: LangCode; native: string; english: string }[] = [
  { code: "en", native: "English", english: "English" },
  { code: "zh", native: "中文", english: "Mandarin" },
];

export function langLabel(code: LangCode): string {
  return LANGUAGES.find((l) => l.code === code)?.english ?? code;
}

export function langNative(code: LangCode): string {
  return LANGUAGES.find((l) => l.code === code)?.native ?? code;
}

export type OnboardingPrefs = {
  spokenLang: LangCode;
  learnLang: LangCode | null;
  lens: Lens;
  completedAt: number;
};

const STORAGE_KEY = "omni.onboarding.v1";

export const DEFAULT_PREFS: OnboardingPrefs = {
  spokenLang: "en",
  learnLang: null,
  lens: "play",
  completedAt: 0,
};

function isLens(v: unknown): v is Lens {
  return v === "play" || v === "language" || v === "history";
}

function isLangCode(v: unknown): v is LangCode {
  return (
    typeof v === "string" &&
    LANGUAGES.some((l) => l.code === v)
  );
}

export function readOnboardingPrefs(): OnboardingPrefs | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const spokenLang = isLangCode(parsed.spokenLang) ? parsed.spokenLang : "en";
    const learnLang = isLangCode(parsed.learnLang) ? parsed.learnLang : null;
    const lens = isLens(parsed.lens) ? parsed.lens : "play";
    const completedAt =
      typeof parsed.completedAt === "number" ? parsed.completedAt : 0;
    return { spokenLang, learnLang, lens, completedAt };
  } catch {
    return null;
  }
}

export function writeOnboardingPrefs(prefs: OnboardingPrefs): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
    window.dispatchEvent(
      new CustomEvent("omni:onboarding-change", { detail: prefs })
    );
  } catch {
    // quota / private mode — tracker will fall back to defaults
  }
}

// Hook — subscribes to storage events + our custom change event so the
// tracker can react if the user edits onboarding in another tab.
export function useOnboardingPrefs(): OnboardingPrefs {
  const [prefs, setPrefs] = useState<OnboardingPrefs>(DEFAULT_PREFS);
  useEffect(() => {
    const current = readOnboardingPrefs();
    if (current) setPrefs(current);
    const onStorage = (e: StorageEvent) => {
      if (e.key && e.key !== STORAGE_KEY) return;
      const next = readOnboardingPrefs();
      if (next) setPrefs(next);
    };
    const onLocal = (e: Event) => {
      const detail = (e as CustomEvent<OnboardingPrefs>).detail;
      if (detail) setPrefs(detail);
    };
    window.addEventListener("storage", onStorage);
    window.addEventListener("omni:onboarding-change", onLocal);
    return () => {
      window.removeEventListener("storage", onStorage);
      window.removeEventListener("omni:onboarding-change", onLocal);
    };
  }, []);
  return prefs;
}
