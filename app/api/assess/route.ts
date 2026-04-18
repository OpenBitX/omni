import { NextResponse } from "next/server";
import { assessObject } from "@/app/actions";

export const runtime = "nodejs";
export const maxDuration = 60;

// HTTP wrapper around the `assessObject` server action, exposed for the
// native v2 client under `react-native-executorch/`. Additive — the browser
// app never calls this route, it keeps invoking the server action directly.
export async function POST(req: Request) {
  const body = await req.json().catch(() => null);
  if (!body || typeof body.imageDataUrl !== "string") {
    return NextResponse.json({ error: "imageDataUrl required" }, { status: 400 });
  }
  const tag =
    typeof body.tag === "string" && body.tag.trim()
      ? body.tag.trim().slice(0, 32)
      : null;
  const tagStr = tag ? ` ${tag}` : "";
  const t0 = Date.now();
  // eslint-disable-next-line no-console
  console.log(`[api/assess${tagStr}] ▶ start`);
  try {
    const learnLang =
      body.learnLang === "zh" || body.learnLang === "en"
        ? body.learnLang
        : undefined;
    const res = await assessObject(
      body.imageDataUrl,
      Number(body.tapX) || 0,
      Number(body.tapY) || 0,
      tag,
      learnLang
    );
    // eslint-disable-next-line no-console
    console.log(
      `[api/assess${tagStr}] ◀ done suitable=${res.suitable} total=${Date.now() - t0}ms`
    );
    return NextResponse.json(res);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    // eslint-disable-next-line no-console
    console.log(
      `[api/assess${tagStr}] ✖ ${msg.slice(0, 160)} after ${Date.now() - t0}ms`
    );
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
