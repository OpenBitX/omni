import { NextResponse } from "next/server";
import { describeObject } from "@/app/actions";

export const runtime = "nodejs";
export const maxDuration = 60;

// HTTP wrapper around `describeObject` for the native v2 client.
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
  console.log(`[api/describe${tagStr}] ▶ start`);
  try {
    const res = await describeObject(
      body.imageDataUrl,
      String(body.className ?? ""),
      body.lang === "zh" ? "zh" : "en",
      tag
    );
    // eslint-disable-next-line no-console
    console.log(
      `[api/describe${tagStr}] ◀ done total=${Date.now() - t0}ms`
    );
    return NextResponse.json(res);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    // eslint-disable-next-line no-console
    console.log(
      `[api/describe${tagStr}] ✖ ${msg.slice(0, 160)} after ${Date.now() - t0}ms`
    );
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
