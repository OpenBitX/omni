/**
 * BackendLoader — shows a floating loading card while the Python backend
 * warms up (YOLO model load takes ~7 s on CPU).
 *
 * Polls GET /health every 600 ms until yolo.model_loaded === true, then
 * fades out automatically.  The rest of the UI renders immediately behind it
 * so the camera starts in parallel.
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { httpUrl } from "@/lib/backend-url";

type Stage = "connecting" | "loading" | "ready" | "gone";

type HealthData = {
  ok: boolean;
  yolo?: { model_loaded: boolean };
};

const POLL_MS = 600;
const DISMISS_DELAY_MS = 1200;
const FADE_MS = 400;

export function BackendLoader() {
  const [stage, setStage] = useState<Stage>("connecting");
  const [fading, setFading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef(Date.now());
  const cancelledRef = useRef(false);
  const readyRef = useRef(false);

  // Elapsed seconds ticker
  useEffect(() => {
    const id = setInterval(
      () => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)),
      1000,
    );
    return () => clearInterval(id);
  }, []);

  const dismiss = useCallback(() => {
    setFading(true);
    setTimeout(() => setStage("gone"), FADE_MS);
  }, []);

  // Health poll loop
  useEffect(() => {
    cancelledRef.current = false;

    async function poll() {
      if (cancelledRef.current || readyRef.current) return;
      try {
        const res = await fetch(httpUrl("/health"), {
          signal: AbortSignal.timeout(2500),
        });
        if (res.ok) {
          const data: HealthData = await res.json();
          if (data.yolo?.model_loaded) {
            readyRef.current = true;
            setStage("ready");
            setTimeout(dismiss, DISMISS_DELAY_MS);
            return;
          }
          setStage("loading");
        } else {
          setStage("connecting");
        }
      } catch {
        setStage("connecting");
      }
      if (!cancelledRef.current) {
        setTimeout(poll, POLL_MS);
      }
    }

    poll();
    return () => {
      cancelledRef.current = true;
    };
  }, [dismiss]);

  if (stage === "gone") return null;

  const opacity = fading ? 0 : 1;

  return (
    <div
      style={{
        position: "fixed",
        bottom: 24,
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 9999,
        opacity,
        transition: `opacity ${FADE_MS}ms ease`,
        pointerEvents: "none",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          background: "rgba(255,255,255,0.82)",
          backdropFilter: "blur(14px)",
          WebkitBackdropFilter: "blur(14px)",
          border: "1.5px solid rgba(255,182,222,0.55)",
          borderRadius: 999,
          padding: "8px 18px 8px 12px",
          boxShadow: "0 4px 24px rgba(255,100,180,0.18), 0 1px 4px rgba(0,0,0,0.08)",
          fontFamily: "system-ui, sans-serif",
          fontSize: 13,
          fontWeight: 500,
          color: "#7a3a5a",
          whiteSpace: "nowrap",
        }}
      >
        <StatusDot stage={stage} />
        <StatusText stage={stage} elapsed={elapsed} />
      </div>
    </div>
  );
}

function StatusDot({ stage }: { stage: Stage }) {
  const base: React.CSSProperties = {
    width: 9,
    height: 9,
    borderRadius: "50%",
    flexShrink: 0,
  };

  if (stage === "connecting") {
    return (
      <span
        style={{
          ...base,
          background: "#f0a0c0",
          animation: "pulse-dot 1.1s ease-in-out infinite",
        }}
      />
    );
  }
  if (stage === "loading") {
    return (
      <span
        style={{
          ...base,
          background: "transparent",
          border: "2px solid #e060a0",
          borderTopColor: "transparent",
          animation: "spin-dot 0.75s linear infinite",
        }}
      />
    );
  }
  // ready
  return (
    <span
      style={{
        ...base,
        background: "#60d0a0",
        fontSize: 10,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#fff",
      }}
    >
      ✓
    </span>
  );
}

function StatusText({ stage, elapsed }: { stage: Stage; elapsed: number }) {
  const sec = elapsed > 0 ? ` (${elapsed}s)` : "";
  if (stage === "connecting") return <span>正在连接后端{sec}…</span>;
  if (stage === "loading")    return <span>YOLO 模型加载中{sec}…</span>;
  return <span style={{ color: "#2a9a6a" }}>后端就绪 ✓</span>;
}

// Inject keyframes once
if (typeof document !== "undefined") {
  const id = "__backend_loader_kf__";
  if (!document.getElementById(id)) {
    const s = document.createElement("style");
    s.id = id;
    s.textContent = `
      @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.45; transform: scale(0.7); }
      }
      @keyframes spin-dot {
        to { transform: rotate(360deg); }
      }
    `;
    document.head.appendChild(s);
  }
}
