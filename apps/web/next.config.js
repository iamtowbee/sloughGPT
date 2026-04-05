/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  trailingSlash: true,
  output: 'standalone',
  transpilePackages: ['@sloughgpt/strui'],
}

module.exports = nextConfig
