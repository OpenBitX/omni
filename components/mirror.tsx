"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { generateLine } from "@/app/actions";

type BaseItem = { name: string; label: string };
type Base = { bases: BaseItem[]; current: string; current_label?: string };
type ServerEvent =
  | { event: "face" }
  | { event: "no_face" }
  | { event: "base"; ok: boolean; base?: string; error?: string };

const WS_URL = "ws://localhost:8000/ws";
const HTTP_URL = "http://localhost:8000";
const SEND_WIDTH = 640;
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const STALL_MS = 6000;
const RECONNECT_MIN_MS = 500;
const RECONNECT_MAX_MS = 8000;

type LiveState = "idle" | "connecting" | "live" | "reconnecting" | "error";

export default function Mirror() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const outRef = useRef<HTMLImageElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inFlightRef = useRef(false);
  const lastFrameUrlRef = useRef<string | null>(null);
  const frameTimesRef = useRef<number[]>([]);
  const lastFrameAtRef = useRef<number>(0);
  const watchdogRef = useRef<number | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const wantConnectedRef = useRef(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  // Non-zero while an optimistic base-swap is in flight — drop incoming WS
  // frames (which are still composites of the previous base) so they don't
  // overwrite the static preview we just painted on the <img>.
  const baseFreezeRef = useRef(0);

  const [status, setStatus] = useState<LiveState>("idle");
  const [fps, setFps] = useState(0);
  const [bases, setBases] = useState<Base | null>(null);
  const [swapTarget, setSwapTarget] = useState<string | null>(null);
  const swapGenRef = useRef(0);
  const [uploading, setUploading] = useState(false);
  const [retryingName, setRetryingName] = useState<string | null>(null);
  const [retryElapsed, setRetryElapsed] = useState(0);
  const [retryToast, setRetryToast] = useState<string | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [faceDetected, setFaceDetected] = useState<boolean | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [line, setLine] = useState<string | null>(null);
  const lineTimerRef = useRef<number | null>(null);
  const retryTickRef = useRef<number | null>(null);
  const retryToastTimerRef = useRef<number | null>(null);

  const loadBases = useCallback(async () => {
    try {
      const r = await fetch(`${HTTP_URL}/bases`);
      if (!r.ok) return;
      setBases((await r.json()) as Base);
    } catch {
      // server not up yet
    }
  }, []);

  useEffect(() => {
    loadBases();
    const id = window.setInterval(loadBases, 5000);
    return () => clearInterval(id);
  }, [loadBases]);

  const dismissLine = useCallback(() => {
    if (lineTimerRef.current != null) {
      clearTimeout(lineTimerRef.current);
      lineTimerRef.current = null;
    }
    setLine(null);
  }, []);

  const speakNow = useCallback(async () => {
    if (speaking) return;
    const img = outRef.current;
    if (!img || !img.complete || img.naturalWidth === 0) {
      setErrorText("no frame to speak yet — give it a sec");
      return;
    }
    setSpeaking(true);
    setErrorText(null);
    try {
      const c = document.createElement("canvas");
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext("2d");
      if (!ctx) throw new Error("canvas unavailable");
      ctx.drawImage(img, 0, 0);
      const dataUrl = c.toDataURL("image/jpeg", 0.82);
      const { line: newLine } = await generateLine(dataUrl);
      dismissLine();
      setLine(newLine);
      lineTimerRef.current = window.setTimeout(() => {
        setLine(null);
        lineTimerRef.current = null;
      }, 9000);
    } catch (e) {
      setErrorText(e instanceof Error ? e.message : "the mirror went quiet");
    } finally {
      setSpeaking(false);
    }
  }, [speaking, dismissLine]);

  const setBase = useCallback(
    async (name: string) => {
      dismissLine();
      // Gen counter lets a later click supersede an in-flight one: the slow
      // swap's finally-block won't clear swapTarget if it's been superseded,
      // and its error (if any) won't flash a stale message over the new swap.
      const gen = ++swapGenRef.current;
      setSwapTarget(name);
      setErrorText(null);
      // Optimistic: flip the top image to the raw base NOW, before the server
      // has swapped. The lockstep WS still has a composite-of-the-old-base in
      // flight that would overwrite us, so we also freeze incoming frames
      // for a moment (see baseFreezeRef in the WS handler).
      baseFreezeRef.current = gen;
      const img = outRef.current;
      if (img) {
        const prev = lastFrameUrlRef.current;
        img.src = `${HTTP_URL}/base-image/${encodeURIComponent(name)}?t=${gen}`;
        lastFrameUrlRef.current = null;
        if (prev) URL.revokeObjectURL(prev);
      }
      // Optimistic: mark the chip active right away so the pink highlight
      // matches the big preview without waiting for a /bases poll.
      setBases((b) => (b ? { ...b, current: name } : b));
      try {
        const r = await fetch(`${HTTP_URL}/base/${encodeURIComponent(name)}`, {
          method: "POST",
        });
        if (gen !== swapGenRef.current) return;
        if (!r.ok) {
          const t = await r.text().catch(() => String(r.status));
          throw new Error(t || `swap failed: ${r.status}`);
        }
        const payload = (await r.json().catch(() => null)) as
          | { base?: string; label?: string }
          | null;
        setBases((b) => {
          if (!b) return b;
          return {
            ...b,
            current: payload?.base || name,
            current_label: payload?.label || b.current_label,
          };
        });
      } catch (e) {
        if (gen !== swapGenRef.current) return;
        setErrorText(e instanceof Error ? e.message : "swap failed");
      } finally {
        if (gen === swapGenRef.current) {
          setSwapTarget(null);
          // Release the freeze shortly after the server confirms so the next
          // composite (now on the new base) can paint.
          baseFreezeRef.current = 0;
        }
      }
    },
    [dismissLine]
  );

  const flashRetryToast = useCallback((msg: string) => {
    if (retryToastTimerRef.current != null) {
      clearTimeout(retryToastTimerRef.current);
    }
    setRetryToast(msg);
    retryToastTimerRef.current = window.setTimeout(() => {
      setRetryToast(null);
      retryToastTimerRef.current = null;
    }, 2600);
  }, []);

  const retryBase = useCallback(
    async (name: string, initialLabel: string) => {
      dismissLine();
      setRetryingName(name);
      setRetryElapsed(0);
      setErrorText(null);
      const startedAt = performance.now();
      if (retryTickRef.current != null) clearInterval(retryTickRef.current);
      retryTickRef.current = window.setInterval(() => {
        setRetryElapsed(Math.round((performance.now() - startedAt) / 1000));
      }, 500);
      try {
        const r = await fetch(
          `${HTTP_URL}/base/${encodeURIComponent(name)}/retry`,
          { method: "POST" }
        );
        if (!r.ok) {
          const t = await r.text().catch(() => String(r.status));
          throw new Error(t || `retry failed: ${r.status}`);
        }
        const payload = (await r.json().catch(() => null)) as
          | { label?: string; activated?: boolean }
          | null;
        const finalLabel = payload?.label || initialLabel;
        flashRetryToast(
          payload?.activated
            ? `${finalLabel} · re-detected`
            : `${finalLabel} · spot refreshed`
        );
        await loadBases();
      } catch (e) {
        setErrorText(e instanceof Error ? e.message : "retry failed");
      } finally {
        if (retryTickRef.current != null) {
          clearInterval(retryTickRef.current);
          retryTickRef.current = null;
        }
        setRetryingName(null);
        setRetryElapsed(0);
      }
    },
    [dismissLine, flashRetryToast, loadBases]
  );

  const uploadFile = useCallback(
    async (file: File) => {
      setErrorText(null);
      if (file.size > MAX_UPLOAD_BYTES) {
        setErrorText(
          `file too large — ${Math.round(file.size / 1024 / 1024)} MB (max 10 MB)`
        );
        return;
      }
      setUploading(true);
      try {
        const fd = new FormData();
        fd.append("file", file, file.name);
        const r = await fetch(`${HTTP_URL}/upload`, { method: "POST", body: fd });
        if (!r.ok) {
          const text = await r.text().catch(() => `${r.status}`);
          throw new Error(text || `upload failed: ${r.status}`);
        }
        await loadBases();
      } catch (e) {
        setErrorText(e instanceof Error ? e.message : "upload failed");
      } finally {
        setUploading(false);
      }
    },
    [loadBases]
  );

  const onFilePicked = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      e.target.value = "";
      if (f) uploadFile(f);
    },
    [uploadFile]
  );

  const sendFrame = useCallback(() => {
    const v = videoRef.current;
    const c = canvasRef.current;
    const ws = wsRef.current;
    if (!v || !c || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (inFlightRef.current) return;
    if (!v.videoWidth) {
      window.setTimeout(() => sendFrame(), 100);
      return;
    }

    const aspect = v.videoHeight / v.videoWidth;
    c.width = SEND_WIDTH;
    c.height = Math.round(SEND_WIDTH * aspect);
    const ctx = c.getContext("2d");
    if (!ctx) return;
    try {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(v, -c.width, 0, c.width, c.height);
      ctx.restore();
    } catch {
      return;
    }
    c.toBlob(
      (blob) => {
        if (!blob || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        inFlightRef.current = true;
        blob
          .arrayBuffer()
          .then((buf) => {
            try {
              wsRef.current?.send(buf);
            } catch {
              inFlightRef.current = false;
            }
          })
          .catch(() => {
            inFlightRef.current = false;
          });
      },
      "image/jpeg",
      0.72
    );
  }, []);

  const clearWatchdog = useCallback(() => {
    if (watchdogRef.current != null) {
      clearInterval(watchdogRef.current);
      watchdogRef.current = null;
    }
  }, []);

  const clearReconnect = useCallback(() => {
    if (reconnectTimerRef.current != null) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const teardownWS = useCallback(() => {
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) {
      ws.onopen = null;
      ws.onerror = null;
      ws.onclose = null;
      ws.onmessage = null;
      try {
        ws.close();
      } catch {
        // already closed
      }
    }
    inFlightRef.current = false;
  }, []);

  const ensureCamera = useCallback(async () => {
    if (streamRef.current) return streamRef.current;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: "user" },
      audio: false,
    });
    streamRef.current = stream;
    const v = videoRef.current;
    if (v) {
      v.srcObject = stream;
      try {
        await v.play();
      } catch {
        // muted autoplay — ignore
      }
    }
    return stream;
  }, []);

  const scheduleReconnect = useCallback(() => {
    clearReconnect();
    if (!wantConnectedRef.current) return;
    const attempt = ++reconnectAttemptsRef.current;
    const delay = Math.min(
      RECONNECT_MAX_MS,
      RECONNECT_MIN_MS * Math.pow(2, attempt - 1)
    );
    setStatus("reconnecting");
    setErrorText(
      `server unreachable — retrying in ${Math.round(delay / 1000)}s (#${attempt})`
    );
    reconnectTimerRef.current = window.setTimeout(() => {
      openSocket();
    }, delay);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clearReconnect]);

  const openSocket = useCallback(() => {
    teardownWS();
    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      reconnectAttemptsRef.current = 0;
      setStatus("live");
      setErrorText(null);
      lastFrameAtRef.current = performance.now();
      sendFrame();
    };

    ws.onerror = () => {
      // close follows — handle reconnect there
    };

    ws.onclose = () => {
      clearWatchdog();
      inFlightRef.current = false;
      if (!wantConnectedRef.current) {
        setStatus("idle");
        return;
      }
      scheduleReconnect();
    };

    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        let payload: ServerEvent | null = null;
        try {
          payload = JSON.parse(ev.data) as ServerEvent;
        } catch {
          return;
        }
        if (payload.event === "face") setFaceDetected(true);
        else if (payload.event === "no_face") setFaceDetected(false);
        else if (payload.event === "base") {
          if (!payload.ok && payload.error) setErrorText(payload.error);
          loadBases();
        }
        return;
      }
      lastFrameAtRef.current = performance.now();
      // While the base-swap is mid-flight, the server is still returning
      // composites of the previous base for frames that were already in
      // flight. Drop them so they don't stomp the optimistic static preview
      // on <img>.
      if (baseFreezeRef.current !== 0) {
        inFlightRef.current = false;
        requestAnimationFrame(sendFrame);
        return;
      }
      const blob = new Blob([ev.data as ArrayBuffer], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);
      const img = outRef.current;
      if (img) {
        const prev = lastFrameUrlRef.current;
        img.src = url;
        lastFrameUrlRef.current = url;
        if (prev) URL.revokeObjectURL(prev);
      }

      const now = performance.now();
      const arr = frameTimesRef.current;
      arr.push(now);
      while (arr.length && now - arr[0] > 1000) arr.shift();
      setFps(arr.length);

      inFlightRef.current = false;
      requestAnimationFrame(sendFrame);
    };

    clearWatchdog();
    watchdogRef.current = window.setInterval(() => {
      if (!wantConnectedRef.current) return;
      const ws2 = wsRef.current;
      if (!ws2 || ws2.readyState !== WebSocket.OPEN) return;
      if (performance.now() - lastFrameAtRef.current > STALL_MS) {
        try {
          ws2.close();
        } catch {
          // ignore
        }
      }
    }, 1500);
  }, [sendFrame, loadBases, clearWatchdog, teardownWS, scheduleReconnect]);

  const connect = useCallback(async () => {
    setErrorText(null);
    wantConnectedRef.current = true;
    setStatus("connecting");

    try {
      await ensureCamera();
    } catch (e) {
      wantConnectedRef.current = false;
      setStatus("error");
      setErrorText(e instanceof Error ? e.message : "camera blocked");
      return;
    }

    openSocket();
  }, [ensureCamera, openSocket]);

  const disconnect = useCallback(() => {
    wantConnectedRef.current = false;
    clearWatchdog();
    clearReconnect();
    teardownWS();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    const v = videoRef.current;
    if (v) v.srcObject = null;
    setStatus("idle");
    setFps(0);
    setFaceDetected(null);
    dismissLine();
  }, [clearReconnect, clearWatchdog, teardownWS, dismissLine]);

  useEffect(() => {
    return () => {
      wantConnectedRef.current = false;
      clearWatchdog();
      clearReconnect();
      teardownWS();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      if (lastFrameUrlRef.current) URL.revokeObjectURL(lastFrameUrlRef.current);
      if (lineTimerRef.current != null) clearTimeout(lineTimerRef.current);
      if (retryTickRef.current != null) clearInterval(retryTickRef.current);
      if (retryToastTimerRef.current != null) clearTimeout(retryToastTimerRef.current);
    };
  }, [clearReconnect, clearWatchdog, teardownWS]);

  const isConnecting = status === "connecting" || status === "reconnecting";
  const dotColor =
    status === "live"
      ? "bg-emerald-400"
      : isConnecting
        ? "bg-amber-400"
        : status === "error"
          ? "bg-rose-400"
          : "bg-fuchsia-300";

  const statusLabel =
    status === "live"
      ? `${String(fps).padStart(2, "0")} fps ✨`
      : status === "connecting"
        ? "opening camera…"
        : status === "reconnecting"
          ? `reconnecting #${reconnectAttemptsRef.current}`
          : status === "error"
            ? "oops"
            : "ready when you are";

  return (
    <main className="relative flex min-h-[100svh] flex-col overflow-hidden">
      {/* Soft ambient blobs */}
      <div
        className="blob"
        style={{
          width: 380,
          height: 380,
          top: -120,
          left: -120,
          background: "radial-gradient(circle, #ffc4de 0%, #ffd9bd 70%)",
        }}
      />
      <div
        className="blob"
        style={{
          width: 460,
          height: 460,
          top: "38%",
          right: -180,
          background: "radial-gradient(circle, #cfd9ff 0%, #e4d1ff 70%)",
          animationDelay: "-6s",
        }}
      />
      <div
        className="blob"
        style={{
          width: 320,
          height: 320,
          bottom: -100,
          left: "22%",
          background: "radial-gradient(circle, #ffe8a8 0%, #ffc9df 70%)",
          animationDelay: "-12s",
        }}
      />

      <div className="relative z-10 mx-auto flex w-full max-w-[880px] flex-1 flex-col">
        {/* Top rail */}
        <header className="flex items-center justify-between px-6 pt-7 sm:px-8 sm:pt-9">
          <div className="flex items-baseline gap-2">
            <span className="h-2 w-2 rounded-full bg-[color:var(--accent)] shadow-[0_0_0_4px_rgba(236,72,153,0.18)]" />
            <span className="serif-italic text-[28px] font-semibold leading-none text-[color:var(--ink)] sm:text-[32px]">
              mirror
            </span>
          </div>
          <div className="flex items-center gap-2 rounded-full bg-white/70 px-3 py-1.5 shadow-[0_2px_10px_-4px_rgba(42,21,64,0.15)] ring-1 ring-white/80 backdrop-blur-md">
            <span className={"h-1.5 w-1.5 rounded-full " + dotColor} />
            <span className="text-[11px] font-medium tabular-nums tracking-wide text-[color:var(--ink-soft)]">
              {statusLabel}
            </span>
          </div>
        </header>

        {/* Stage */}
        <section className="flex flex-1 items-center justify-center px-4 py-6 sm:px-8 sm:py-8">
          <div className="relative aspect-square w-full max-w-[min(80vh,660px)]">
            {/* gentle halo */}
            <div
              aria-hidden
              className="absolute -inset-6 rounded-[48px] opacity-70 blur-2xl"
              style={{
                background:
                  "conic-gradient(from 140deg, #ffd1e8, #e5d4ff, #d6e6ff, #ffe5d0, #ffd1e8)",
              }}
            />
            <div className="relative h-full w-full overflow-hidden rounded-[32px] bg-white/60 shadow-[0_40px_80px_-30px_rgba(42,21,64,0.25),0_0_0_1px_rgba(255,255,255,0.9)_inset] backdrop-blur">
              <img
                ref={outRef}
                alt=""
                className={
                  "h-full w-full object-cover transition-opacity duration-500 " +
                  (status === "live" ? "opacity-100" : "opacity-0")
                }
              />

              {status !== "live" && (
                <div className="absolute inset-0 grid place-items-center">
                  {status === "idle" || status === "error" ? (
                    <button
                      onClick={connect}
                      className="group flex flex-col items-center gap-5"
                    >
                      <span className="bubble-btn grid h-[104px] w-[104px] place-items-center rounded-full bg-gradient-to-br from-[#ff89be] via-[#ec4899] to-[#c026d3] text-white transition duration-300 group-hover:scale-[1.06] group-active:scale-95">
                        <svg
                          width="30"
                          height="30"
                          viewBox="0 0 24 24"
                          fill="currentColor"
                          className="translate-x-[2px]"
                        >
                          <polygon points="7 4 20 12 7 20 7 4" />
                        </svg>
                      </span>
                      <span className="serif-italic text-[18px] font-medium text-[color:var(--ink-soft)]">
                        {status === "error" ? "try once more" : "begin"}
                      </span>
                    </button>
                  ) : (
                    <div className="flex items-center gap-2.5 rounded-full bg-white/80 px-4 py-2 text-[13px] font-medium text-[color:var(--ink-soft)] shadow-sm backdrop-blur">
                      <span className="inline-block h-1.5 w-1.5 animate-ping rounded-full bg-[color:var(--accent)]" />
                      {status === "connecting"
                        ? "opening camera"
                        : "reconnecting"}
                    </div>
                  )}
                </div>
              )}

              {status === "live" && (
                <>
                  <span
                    className={
                      "absolute left-4 top-4 rounded-full px-3 py-1 text-[11px] font-medium tracking-wide backdrop-blur-md transition " +
                      (faceDetected === false
                        ? "bg-rose-500/85 text-white"
                        : "bg-white/75 text-[color:var(--ink-soft)]")
                    }
                  >
                    {faceDetected === false ? "no face" : "live"}
                  </span>
                  <button
                    onClick={disconnect}
                    className="absolute right-4 top-4 rounded-full bg-white/75 px-3.5 py-1 text-[11px] font-medium tracking-wide text-[color:var(--ink-soft)] backdrop-blur-md transition hover:bg-white hover:text-[color:var(--ink)] active:scale-95"
                  >
                    stop
                  </button>

                  {line && (
                    <button
                      onClick={dismissLine}
                      aria-label="Dismiss"
                      className="absolute inset-x-6 top-14 mx-auto max-w-[82%] rounded-[20px] bg-white/92 px-5 py-3.5 text-left shadow-[0_16px_40px_-18px_rgba(42,21,64,0.35)] ring-1 ring-white/80 backdrop-blur-md transition hover:bg-white"
                      style={{ animation: "bubble-in 420ms cubic-bezier(0.16,1,0.3,1) both" }}
                    >
                      <span className="serif-italic text-balance text-[16px] leading-[1.35] text-[color:var(--ink)] sm:text-[17px]">
                        &ldquo;{line}&rdquo;
                      </span>
                    </button>
                  )}

                  <button
                    onClick={speakNow}
                    disabled={speaking}
                    className="bubble-btn absolute bottom-5 left-1/2 -translate-x-1/2 rounded-full bg-gradient-to-br from-[#ff89be] via-[#ec4899] to-[#c026d3] px-6 py-2.5 text-[13px] font-semibold tracking-wide text-white transition hover:scale-[1.04] active:scale-95 disabled:opacity-60"
                  >
                    {speaking ? "listening…" : "speak"}
                  </button>
                </>
              )}
            </div>

            {errorText && status !== "live" && (
              <div
                className="absolute inset-x-4 -bottom-4 text-center"
                style={{ animation: "fade-in 240ms ease-out both" }}
              >
                <span className="inline-block rounded-full bg-white/90 px-4 py-1.5 text-[11px] font-medium text-rose-500 shadow-sm backdrop-blur">
                  {errorText}
                </span>
              </div>
            )}
          </div>
        </section>

        {/* Base picker */}
        <footer className="px-6 pb-[max(env(safe-area-inset-bottom),28px)] pt-6 sm:px-8">
          <div className="mb-4 flex items-end justify-between">
            <div className="flex flex-col gap-0.5">
              <span className="text-[10.5px] font-medium uppercase tracking-[0.28em] text-[color:var(--ink-muted)]">
                wardrobe
              </span>
              <span className="serif-italic text-[20px] font-medium leading-none text-[color:var(--ink)]">
                who today?
              </span>
            </div>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="rounded-full bg-white/80 px-4 py-2 text-[12px] font-medium text-[color:var(--ink-soft)] shadow-[0_2px_10px_-4px_rgba(42,21,64,0.15)] ring-1 ring-white/80 backdrop-blur-md transition hover:bg-white hover:text-[color:var(--ink)] active:scale-95 disabled:opacity-40"
            >
              {uploading ? "analyzing" : "+ upload"}
            </button>
          </div>
          <div className="flex flex-wrap gap-2">
            {bases?.bases.map((b) => {
              const active = bases.current === b.name;
              const retrying = retryingName === b.name;
              const anotherRetrying = retryingName !== null && !retrying;
              const swapPending = swapTarget === b.name;
              // Optimistic-active: light up the target chip the instant you
              // click, even before the server confirms the swap.
              const lookActive = active || swapPending;
              const chipText = retrying
                ? retryElapsed >= 2
                  ? `detecting… ${retryElapsed}s`
                  : "detecting…"
                : swapPending
                  ? "swapping…"
                  : b.label;
              return (
                <div
                  key={b.name}
                  className={
                    "flex items-stretch overflow-hidden rounded-full transition " +
                    (retrying || swapPending
                      ? "animate-[soft-pulse_1.4s_ease-in-out_infinite] "
                      : "") +
                    (lookActive
                      ? "bg-gradient-to-br from-[#ff89be] via-[#ec4899] to-[#c026d3] text-white shadow-[0_10px_25px_-10px_rgba(236,72,153,0.6)]"
                      : "bg-white/75 text-[color:var(--ink-soft)] ring-1 ring-white/80 backdrop-blur-md hover:bg-white hover:text-[color:var(--ink)]")
                  }
                >
                  <button
                    disabled={swapPending || uploading}
                    onClick={() => setBase(b.name)}
                    className="min-w-[4.5rem] px-4 py-2 text-[13px] font-medium tracking-[0.005em] tabular-nums transition active:scale-95 disabled:opacity-40"
                    title={b.name}
                  >
                    {chipText}
                  </button>
                  <button
                    disabled={retrying || anotherRetrying || uploading}
                    onClick={() => retryBase(b.name, b.label)}
                    className={
                      "grid place-items-center px-2.5 transition active:scale-95 disabled:opacity-40 " +
                      (active ? "border-l border-white/40" : "border-l border-black/5")
                    }
                    title={retrying ? "re-detecting…" : "re-detect face spots with AI"}
                    aria-label={`retry face detection for ${b.label}`}
                  >
                    <svg
                      width="13"
                      height="13"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className={retrying ? "animate-spin" : "transition-transform group-hover:rotate-45"}
                    >
                      <path d="M21 12a9 9 0 1 1-3.2-6.9" />
                      <path d="M21 4v5h-5" />
                    </svg>
                  </button>
                </div>
              );
            })}
            {!bases && (
              <span className="rounded-full bg-white/70 px-4 py-2 text-[13px] font-medium text-[color:var(--ink-muted)] backdrop-blur">
                waiting for server
              </span>
            )}
          </div>
          {retryToast && (
            <div
              className="pointer-events-none mt-3 flex justify-center"
              style={{ animation: "fade-in 260ms ease-out both" }}
            >
              <span className="inline-flex items-center gap-1.5 rounded-full bg-white/92 px-3.5 py-1.5 text-[11.5px] font-medium text-[color:var(--ink-soft)] shadow-[0_8px_24px_-14px_rgba(42,21,64,0.4)] ring-1 ring-white/80 backdrop-blur">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                {retryToast}
              </span>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={onFilePicked}
            className="hidden"
          />
        </footer>
      </div>

      <video ref={videoRef} playsInline muted className="hidden" />
      <canvas ref={canvasRef} className="hidden" />
    </main>
  );
}
