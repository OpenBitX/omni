/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: { bodySizeLimit: "15mb" },
  },
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "**.runware.ai" },
      { protocol: "https", hostname: "im.runware.ai" },
    ],
  },
};

export default nextConfig;
