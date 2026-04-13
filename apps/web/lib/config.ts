/** FastAPI backend base URL for the Next.js app (client bundles). */
export const PUBLIC_API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

/** API endpoint path - use direct backend URL to avoid proxy issues with streaming */
export const API_CHAT_ENDPOINT = `${PUBLIC_API_URL}/chat/stream`

/** Shown on the home overview; bump with apps/web/package.json when you cut a web release. */
export const WEB_UI_VERSION = process.env.NEXT_PUBLIC_WEB_UI_VERSION || '3.0.0'
