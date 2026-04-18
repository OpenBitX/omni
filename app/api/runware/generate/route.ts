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
// Overall budget from request-in to response-out. Must comfortably cover
// gpt-4o-mini vision (~1–3s) + Runware image gen (~8–15s) + slack, but
// still surface a clean failure instead of letting the request hang
// forever if one provider stalls. Client-side timeout is 45s.
const TOTAL_BUDGET_MS = 35_000;
// Reject absurd payloads at the boundary. 4 MB data URL is already ~3 MB
// of image bytes — more than any tracker crop should ever be.
const MAX_IMAGE_DATA_URL_BYTES = 4_000_000;

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

  let imageDataUrl: string | undefined;
  if (typeof payload.imageDataUrl === "string") {
    if (!payload.imageDataUrl.startsWith("data:image/")) {
      // Silently ignore malformed values — we still fall through to the
      // heuristic prompt, same outcome as if it were omitted.
    } else if (payload.imageDataUrl.length > MAX_IMAGE_DATA_URL_BYTES) {
      console.log(
        `${tag} warn dropping oversized imageDataUrl bytes=${payload.imageDataUrl.length}`
      );
    } else {
      imageDataUrl = payload.imageDataUrl;
    }
  }

  // Merge the client's abort signal with our overall budget so a dropped
  // client or a stuck provider can't leave the handler hanging.
  const ac = new AbortController();
  const budgetTimer = setTimeout(() => ac.abort(), TOTAL_BUDGET_MS);
  const onClientAbort = () => ac.abort();
  if (req.signal.aborted) ac.abort();
  else req.signal.addEventListener("abort", onClientAbort, { once: true });

  let result;
  try {
    result = await generateComicImage(
      {
        className,
        description,
        history,
        spokenLang: parseLang(payload.spokenLang),
        learnLang: parseLang(payload.learnLang),
        imageDataUrl,
      },
      ac.signal
    );
  } finally {
    clearTimeout(budgetTimer);
    req.signal.removeEventListener("abort", onClientAbort);
  }

  const ms = Date.now() - start;

  if (!result.ok) {
    const source = result.promptSource ?? "heuristic";
    const status =
      result.error === "RUNWARE_API_KEY missing"
        ? 503
        : result.error === "cancelled"
          ? 499 // client aborted
          : result.error === "timeout"
            ? 504
            : 500;
    console.log(
      `${tag} err status=${status} err=${result.error} source=${source} ${ms}ms`
    );
    return NextResponse.json({ error: result.error }, { status });
  }

  const previewPrompt = result.prompt
    .replace(/\s+/g, " ")
    .slice(0, 140);
  console.log(
    `${tag} ok source=${result.promptSource} promptLen=${result.prompt.length} ${ms}ms prompt="${previewPrompt}${result.prompt.length > 140 ? "…" : ""}"`
  );
  return NextResponse.json({
    imageUrl: result.imageUrl,
    prompt: result.prompt,
    promptSource: result.promptSource,
  });
}
