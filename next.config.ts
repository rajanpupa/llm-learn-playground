import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export so `next build` produces a fully-static site (out/) that
  // opens without a server, while `next dev` still gives the dev experience.
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
