/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ["ai", "@ai-sdk/openai"],
  },
};

module.exports = nextConfig;
