import { NextResponse } from "next/server";
import { converseWithObject } from "@/app/actions";

export const runtime = "nodejs";
export const maxDuration = 60;

// Thin pass-through — the server action already consumes FormData. We
// peek at the `turnId` field purely so the API-edge log line can share
// the same correlation id as the action's internal `[converse #N]` logs.
export async function POST(req: Request) {
  const t0 = Date.now();
  try {
    const form = await req.formData();
    const turnIdRaw = form.get("turnId");
    const turnId =
      typeof turnIdRaw === "string" && turnIdRaw.trim()
        ? turnIdRaw.trim().slice(0, 16)
        : "?";
    // eslint-disable-next-line no-console
    console.log(`[api/converse #${turnId}] ▶ start`);
    const res = await converseWithObject(form);
    // eslint-disable-next-line no-console
    console.log(
      `[api/converse #${turnId}] ◀ done total=${Date.now() - t0}ms`
    );
    return NextResponse.json(res);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    // eslint-disable-next-line no-console
    console.log(
      `[api/converse] ✖ ${msg.slice(0, 160)} after ${Date.now() - t0}ms`
    );
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
