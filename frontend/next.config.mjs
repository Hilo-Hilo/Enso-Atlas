/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable React strict mode for development
  reactStrictMode: true,
  
  // Configure image domains for WSI thumbnails from backend
  images: {
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost",
        port: "8000",
        pathname: "/api/**",
      },
      {
        protocol: "http",
        hostname: "127.0.0.1",
        port: "8000",
        pathname: "/api/**",
      },
    ],
  },
  
  // Proxy API requests to the backend server
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://100.111.126.23:8003'}/api/:path*`, // Proxy to DGX Spark backend via Tailscale
      },
    ]
  },
  
  // Suppress hydration warnings for OpenSeadragon
  compiler: {
    removeConsole: process.env.NODE_ENV === "production",
  },
};

export default nextConfig;
