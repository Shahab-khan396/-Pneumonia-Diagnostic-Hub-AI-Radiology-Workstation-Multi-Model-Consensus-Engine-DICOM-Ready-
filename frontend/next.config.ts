import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const fastApiUrl = process.env.FASTAPI_URL || 'http://127.0.0.1:8000';
    return [
      {
        source: '/backend/:path*',
        destination: `${fastApiUrl}/:path*`,
      },
    ];
  },
  images: {
    remotePatterns: [
      { protocol: 'https', hostname: 'pneumonia-hub-api.onrender.com' },
      { protocol: 'https', hostname: 'shahab-khan396-pneumonia-hub.hf.space' },
      { protocol: 'http', hostname: '127.0.0.1' },
      { protocol: 'http', hostname: 'localhost' },
    ],
  },
};

export default nextConfig;

