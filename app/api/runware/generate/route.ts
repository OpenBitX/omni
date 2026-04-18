// POST /api/runware/generate
//
// Generates the comic-book illustration for a gallery card. The client
// (gallery page) calls this with the card's className + description +
// truncated conversation history, and receives back { imageUrl, prompt }.
//
// Node runtime: we use AbortController timeouts and we're already on Node
// for the other server actions. Edge buys nothing here.

import { NextResponse } from "next/server";
import { generateComicImage } from "@/lib/runware";
import type { ChatTurn } from "@/lib/session-cards";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Body = {
  cardId?: unknown;
  className?: unknown;
  description?: unknown;
  history?: unknown;
  spokenLang?: unknown;
  learnLang?: unknown;
  // Optional data URL of the cropped object. If provided, Runware prompt
  // gets crafted by gpt-4o-mini vision against the actual image.
  imageDataUrl?: unknown;
};

const MAX_HISTORY_TURNS = 8;

function isChatTurn(v: unknown): v is ChatTurn {
  if (!v || typeof v !== "object") return false;
  const t = v as { role?: unknown; content?: unknown };
  return (
    (t.role === "user" || t.role === "assistant") &&
    typeof t.content === "string"
  );
}

function parseLang(v: unknown): "en" | "zh" | undefined {
  return v === "en" || v === "zh" ? v : undefined;
}

export async function POST(req: Request) {
  let payload: Body;
  try {
    payload = (await req.json()) as Body;
  } catch {
    return NextResponse.json({ error: "bad json" }, { status: 400 });
  }

  const cardId =
    typeof payload.cardId === "string" ? payload.cardId.trim() : "";
  const className =
    typeof payload.className === "string" ? payload.className.trim() : "";
  const description =
    typeof payload.description === "string" ? payload.description.trim() : "";

  if (!cardId) {
    return NextResponse.json({ error: "cardId required" }, { status: 400 });
  }
  if (!className) {
    return NextResponse.json({ error: "className required" }, { status: 400 });
  }
  if (!description) {
    return NextResponse.json({ error: "description required" }, { status: 400 });
  }

  const historyRaw = Array.isArray(payload.history) ? payload.history : [];
  const history: ChatTurn[] = historyRaw
    .filter(isChatTurn)
    .slice(-MAX_HISTORY_TURNS);

  const tag = `[runware #${cardId.slice(-8)}]`;
  const start = Date.now();
  console.log(
    `${tag} start className=${JSON.stringify(className)} history=${history.length}`
  );

  const imageDataUrl =
    typeof payload.imageDataUrl === "string" &&
    payload.imageDataUrl.startsWith("data:image/")
      ? payload.imageDataUrl
      : undefined;

  const result = await generateComicImage({
    className,
    description,
    history,
    spokenLang: parseLang(payload.spokenLang),
    learnLang: parseLang(payload.learnLang),
    imageDataUrl,
  });

  const ms = Date.now() - start;

  if (!result.ok) {
    const status = result.error === "RUNWARE_API_KEY missing" ? 503 : 500;
    console.log(`${tag} err status=${status} err=${result.error} ${ms}ms`);
    return NextResponse.json({ error: result.error }, { status });
  }

  console.log(`${tag} ok ${ms}ms`);
  return NextResponse.json({
    imageUrl: result.imageUrl,
    prompt: result.prompt,
  });
}
