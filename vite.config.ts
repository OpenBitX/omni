import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { readFileSync, existsSync } from "fs";
import path from "path";

const certPath = path.resolve(__dirname, "certificates/cert.pem");
const keyPath = path.resolve(__dirname, "certificates/key.pem");
const hasCerts = existsSync(certPath) && existsSync(keyPath);

export default defineConfig({
  plugins: [react()],

  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },

  server: {
    host: "0.0.0.0",
    port: 3000,
    ...(hasCerts
      ? {
          https: {
            cert: readFileSync(certPath),
            key: readFileSync(keyPath),
          },
        }
      : {}),
    proxy: {
      // All /api/* requests go to the Python FastAPI backend (uvicorn :8000).
      // There is no longer a separate Node.js API server. /ws/* (YOLO WS)
      // is also served by the same uvicorn process, hence the same target.
      //
      // The backend (`backend/scripts/start.py`) auto-enables SSL when
      // `certificates/{cert,key}.pem` exist — the SAME condition we use
      // above for the Vite dev server. If we don't mirror the protocol
      // here, Node's http-agent talks plaintext to a TLS socket and every
      // request dies as `socket hang up` / `Parse Error: Expected HTTP/`.
      "/api": {
        target: hasCerts ? "https://localhost:8000" : "http://localhost:8000",
        changeOrigin: true,
        secure: false,
      },
      "/ws": {
        target: hasCerts ? "wss://localhost:8000" : "ws://localhost:8000",
        ws: true,
        changeOrigin: true,
        secure: false,
      },
    },
  },

  build: {
    outDir: "dist",
    target: "esnext",
  },

  optimizeDeps: {
    // `onnxruntime-node` is a native addon that ships a .node binary —
    // Vite has no business trying to pre-bundle it (and would fail
    // trying). All other ML dependencies have been moved to the Python
    // backend; the frontend is pure UI + lightweight WebSocket client.
    exclude: ["onnxruntime-node"],
  },
});
