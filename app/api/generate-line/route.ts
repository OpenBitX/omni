import { NextResponse } from "next/server";
import { generateLine } from "@/app/actions";

export const runtime = "nodejs";
export const maxDuration = 60;

// Retap path never looks at the image — it runs text-only against the
// pinned persona card — but the action still validates the data URL at
// the boundary. Fall back to a 1×1 PNG when the native client doesn't
// have a fresh frame to ship.
const BLANK_PNG =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

export async function POST(req: Request) {
  const body = await req.json().catch(() => null);
  if (!body) return NextResponse.json({ error: "bad body" }, { status: 400 });
  const imageDataUrl =
    typeof body.imageDataUrl === "string" &&
    body.imageDataUrl.startsWith("data:image/")
      ? body.imageDataUrl
      : BLANK_PNG;
  const voiceId = typeof body.voiceId === "string" ? body.voiceId : null;
  const description =
    typeof body.description === "string" ? body.description : null;
  const history = Array.isArray(body.history) ? body.history : [];
  const lang = body.lang === "zh" ? "zh" : "en";
  const tag =
    typeof body.tag === "string" && body.tag.trim()
      ? body.tag.trim().slice(0, 32)
      : null;
  const tagStr = tag ? ` ${tag}` : "";
  const t0 = Date.now();
  // eslint-disable-next-line no-console
  console.log(
    `[api/generate-line${tagStr}] ▶ start voice=${voiceId ?? "(picking)"} persona=${description ? "cached" : "(new)"}`
  );
  try {
    const res = await generateLine(
      imageDataUrl,
      voiceId,
      description,
      history,
      lang,
      tag
    );
    // eslint-disable-next-line no-console
    console.log(
      `[api/generate-line${tagStr}] ◀ done total=${Date.now() - t0}ms`
    );
    return NextResponse.json({
      line: res.line,
      voiceId: res.voiceId,
      description: res.description,
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    // eslint-disable-next-line no-console
    console.log(
      `[api/generate-line${tagStr}] ✖ ${msg.slice(0, 160)} after ${Date.now() - t0}ms`
    );
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
