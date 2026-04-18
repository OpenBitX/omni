/** @type {import('next').NextConfig} */
const nextConfig = {
  devIndicators: false,
  experimental: {
    serverActions: {
      // Camera frames are shipped to the GLM vision action as data URLs;
      // raise the limit so ~4 MB jpeg crops don't get truncated.
      bodySizeLimit: "8mb",
    },
  },
  async rewrites() {
    // Proxy the YOLO WebSocket through Next so the browser only ever talks
    // to one origin. Lets a phone hit http://<mac-ip>:3000 (or an HTTPS
    // tunnel like cloudflared) without needing a separate route to :8000.
    // Override the upstream with PYTHON_BACKEND_URL if it's running elsewhere.
    const upstream = process.env.PYTHON_BACKEND_URL || "http://127.0.0.1:8000";
    return [
      { source: "/ws/yolo", destination: `${upstream}/ws/yolo` },
    ];
  },
  async headers() {
    return [
      {
        // Cache the big ONNX model aggressively — once it's fetched it
        // doesn't change until we rev the filename.
        source: "/models/:path*",
        headers: [
          { key: "Cache-Control", value: "public, max-age=31536000, immutable" },
        ],
      },
      {
        // Same for the onnxruntime-web WASM runtime under /ort/.
        source: "/ort/:path*",
        headers: [
          { key: "Cache-Control", value: "public, max-age=31536000, immutable" },
        ],
      },
    ];
  },
};

export default nextConfig;
