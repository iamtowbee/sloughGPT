/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['@base-ui/react'],
  trailingSlash: true,
  output: 'standalone',
}

module.exports = nextConfig
