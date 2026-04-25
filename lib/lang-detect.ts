// Lightweight language detection helpers used by the Tracker for the
// "spoken vs learn" language pair. Spoken is what the user speaks; learn
// is the target language the object introduces itself in so the user can
// practice.
//
// All detection here is heuristic — good enough to seed defaults and to
// refine on the fly from short utterances. Nothing here should block a
// render or an audio call; a null/wrong guess just means we keep the
// previous value.

export type AppLang = "en" | "zh";

// Initial guess from browser locale, falling back to "en". We treat all
// Chinese variants (mainland, Taiwan, HK, Cantonese) as "zh" for our
// purposes because the downstream prompts/voice catalog have a single
// zh bucket.
export function detectLangFromNavigator(): AppLang {
  if (typeof navigator === "undefined") return "en";
  const raw =
    (navigator.language || (navigator.languages && navigator.languages[0]) || "")
      .toString()
      .toLowerCase();
  if (!raw) return "en";
  // Match zh, zh-CN, zh-TW, zh-HK, zh-Hans, zh-Hant, yue (Cantonese).
  if (raw.startsWith("zh") || raw.startsWith("yue")) return "zh";
  return "en";
}

// Detect whether a short transcript is Chinese or English. Pure character
// counting — count CJK ideographs vs Latin word characters. Whichever side
// has clearly more wins. Returns null when the text is empty, too short,
// or roughly balanced (ambiguous — callers should keep the prior value).
//
// CJK range covers the Unified Ideographs + Extension A which covers
// everyday Mandarin usage. We intentionally don't count punctuation or
// digits — "一二三" counts, "123" doesn't.
export function detectLangFromText(text: string): AppLang | null {
  if (!text) return null;
  const trimmed = text.trim();
  if (trimmed.length < 2) return null;

  let cjk = 0;
  let latin = 0;
  for (let i = 0; i < trimmed.length; i++) {
    const code = trimmed.charCodeAt(i);
    // CJK Unified Ideographs (4E00–9FFF) + Extension A (3400–4DBF).
    if ((code >= 0x4e00 && code <= 0x9fff) || (code >= 0x3400 && code <= 0x4dbf)) {
      cjk++;
    } else if (
      (code >= 0x41 && code <= 0x5a) || // A–Z
      (code >= 0x61 && code <= 0x7a) // a–z
    ) {
      latin++;
    }
  }

  const total = cjk + latin;
  if (total < 2) return null;

  // Require a clear majority — avoids flipping on things like "hello 你好"
  // where the user is code-switching. 70% threshold is deliberately loose
  // because short transcripts can have noise.
  if (cjk > latin && cjk / total >= 0.6) return "zh";
  if (latin > cjk && latin / total >= 0.6) return "en";
  return null;
}

// Default "practice language" given what the user speaks. Simple inverse —
// if you speak English you're likely here to practice Chinese, and vice
// versa. Callers can override via the UI pill.
export function defaultLearnLang(spoken: AppLang): AppLang {
  return spoken === "en" ? "zh" : "en";
}
