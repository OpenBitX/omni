/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: {
      // Camera frames are shipped to the GLM vision action as data URLs;
      // raise the limit so ~4 MB jpeg crops don't get truncated.
      bodySizeLimit: "8mb",
    },
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
