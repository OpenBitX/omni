"use client";

// Session-wide card store. Cards are captured at first-tap (in the tracker)
// and read on the /gallery page, so they need to survive route navigation.
// sessionStorage persists across Next.js route transitions within a tab and
// clears on tab close, which matches the "this session" semantic.
//
// Entries hold a data-URL thumbnail (~40–150 KB each). sessionStorage caps
// at a few MB on most browsers, so the store self-trims to MAX_CARDS.

import { useEffect, useState } from "react";

export type SessionCard = {
  id: string;
  trackId: string;
  createdAt: number;
  className: string;
  description: string;
  voiceId: string;
  line: string;
  imageDataUrl: string;
};

const STORAGE_KEY = "omni.sessionCards.v1";
const MAX_CARDS = 40;

let cards: readonly SessionCard[] = [];
let hydrated = false;
const listeners = new Set<(cards: readonly SessionCard[]) => void>();

function hydrate() {
  if (hydrated || typeof window === "undefined") return;
  hydrated = true;
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      cards = parsed.filter(
        (c): c is SessionCard =>
          !!c &&
          typeof c.id === "string" &&
          typeof c.trackId === "string" &&
          typeof c.className === "string" &&
          typeof c.description === "string" &&
          typeof c.voiceId === "string" &&
          typeof c.line === "string" &&
          typeof c.imageDataUrl === "string"
      );
    }
  } catch {
    // Corrupt storage — start fresh.
  }
}

function persist() {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(cards));
  } catch {
    // Quota exceeded (crops are big). Trim aggressively and retry once.
    cards = cards.slice(-Math.floor(MAX_CARDS / 2));
    try {
      window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(cards));
    } catch {
      // Give up — in-memory state is still correct, persistence just fails.
    }
  }
}

function emit() {
  for (const cb of listeners) cb(cards);
}

export function getSessionCards(): readonly SessionCard[] {
  hydrate();
  return cards;
}

export function addSessionCard(card: SessionCard): void {
  hydrate();
  if (cards.some((c) => c.trackId === card.trackId)) return;
  cards = [...cards, card].slice(-MAX_CARDS);
  persist();
  emit();
}

export function removeSessionCard(id: string): void {
  hydrate();
  const next = cards.filter((c) => c.id !== id);
  if (next.length === cards.length) return;
  cards = next;
  persist();
  emit();
}

export function clearSessionCards(): void {
  hydrate();
  if (cards.length === 0) return;
  cards = [];
  persist();
  emit();
}

export function subscribeSessionCards(
  cb: (cards: readonly SessionCard[]) => void
): () => void {
  listeners.add(cb);
  return () => {
    listeners.delete(cb);
  };
}

// React hook — hydrates on mount, resubscribes on storage change from other
// tabs so the gallery stays in sync if the tracker is open in another tab.
export function useSessionCards(): readonly SessionCard[] {
  const [state, setState] = useState<readonly SessionCard[]>(() => cards);
  useEffect(() => {
    hydrate();
    setState(cards);
    const unsub = subscribeSessionCards(setState);
    const onStorage = (e: StorageEvent) => {
      if (e.key !== STORAGE_KEY) return;
      // Another tab wrote — rehydrate from storage.
      hydrated = false;
      cards = [];
      hydrate();
      setState(cards);
      emit();
    };
    window.addEventListener("storage", onStorage);
    return () => {
      unsub();
      window.removeEventListener("storage", onStorage);
    };
  }, []);
  return state;
}
