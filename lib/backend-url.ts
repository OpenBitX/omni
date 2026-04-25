/**
 * Single source of truth for all backend URLs.
 *
 * Set VITE_BACKEND_URL in your .env.local to point at a remote server.
 * The default is https://localhost:8000 (matches the local dev cert setup).
 *
 * Both httpUrl() and wsUrl() derive from the same base, so switching between
 * local dev and a deployed instance is a single env var change.
 *
 * Usage:
 *   import { httpUrl, wsUrl } from "@/lib/backend-url";
 *   const ws = new WebSocket(wsUrl("/ws/yolo"));
 *   const res = await fetch(httpUrl("/bases"));
 */

const RAW =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_BACKEND_URL) ||
  "https://localhost:8000";

/** Normalised base URL (trailing slash stripped). */
export const BACKEND_BASE = RAW.replace(/\/+$/, "");

/**
 * Build an HTTP/HTTPS URL for a backend path.
 * @example httpUrl("/bases")  →  "https://localhost:8000/bases"
 */
export function httpUrl(path: string): string {
  const normalised = path.startsWith("/") ? path : `/${path}`;
  return `${BACKEND_BASE}${normalised}`;
}

/**
 * Build a WS/WSS URL for a backend path.
 * Automatically converts https → wss and http → ws.
 * @example wsUrl("/ws/yolo")  →  "wss://localhost:8000/ws/yolo"
 */
export function wsUrl(path: string): string {
  const normalised = path.startsWith("/") ? path : `/${path}`;
  const base = BACKEND_BASE.replace(/^https:\/\//, "wss://").replace(
    /^http:\/\//,
    "ws://"
  );
  return `${base}${normalised}`;
}
