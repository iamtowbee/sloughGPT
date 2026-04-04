# SloughGPT Web UI

Modern TypeScript-based web interface for SloughGPT using Base UI components.

## Features

- 💬 **Chat Interface** - Real-time conversations with AI models
- 📊 **Dataset Management** - Create, view, and manage training datasets
- 🤖 **Model Management** - Browse and configure available AI models
- 📈 **Monitoring Dashboard** - Real-time system metrics and health checks
- 🔄 **Real-time Updates** - Live data with WebSocket support
- 📱 **Progressive Web App** - Installable with offline capabilities

## Tech Stack

- **Framework**: Next.js 14 (App Router), React 18, TypeScript
- **UI Library**: Base UI
- **State Management**: Zustand
- **Charts**: Recharts
- **Styling**: Tailwind CSS

## Getting Started

### Prerequisites

- **Node.js 20** (match repo root **`.nvmrc`**; CI uses the same major)
- npm or yarn

### Installation

```bash
# From repository root — Next.js app lives under apps/web
cd apps/web

# Install dependencies
npm install

# Or with yarn
yarn install
```

### Development

```bash
# Start development server
npm run dev

# The app will open at http://localhost:3000
```

Before pushing changes, run the same checks as CI: **`npm ci && npm run ci`** (from this directory — runs lint, typecheck, Vitest, and production `next build`).

**Talking to models:** set **`NEXT_PUBLIC_API_URL`** to your FastAPI base (default `http://localhost:8000`). Use **Models** to **`POST /models/load`** (`model_id` in JSON), then **Chat** calls **`/inference/generate/stream`** and **`/inference/generate`**. The client **`api.loadModel`** matches that contract (not `/models/{id}/load`).

**Cypress E2E** (mocked API, no Python process): after `npm run build`, run **`npm run e2e:ci`** (starts `next dev` on port **3010** so it does not clash with `output: 'standalone'` + `next start`). Or `npm run dev` on 3000 and **`npm run e2e`** / **`npm run e2e:open`**.

### Build for Production

```bash
npm run build
npm run start   # serves the production build (default port 3000)
```

## API Configuration

The web UI connects to the FastAPI backend. By default, it expects the API at `http://localhost:8000`.

To change the API URL, copy **`.env.example`** to **`.env.local`** (or edit **`.env`**) and set **`NEXT_PUBLIC_API_URL`** — see **`lib/config.ts`** (`http://localhost:8000` by default).

## Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start development server (Next.js) |
| `npm run build` | Production build |
| `npm run start` | Run production server after `build` |
| `npm run lint` | Run ESLint (`next lint`) |
| `npm run typecheck` | TypeScript `tsc --noEmit` |
| `npm run test` | Vitest unit tests (`lib/*.test.ts`) |
| `npm run ci` | Lint + typecheck + Vitest + production build (parity with CI **`test-web`**) |
| `npm run e2e` / `e2e:open` | Cypress against running app (default baseUrl `http://localhost:3000`) |
| `npm run e2e:ci` | `next dev -p 3010` + headless Cypress (use after `build` for warm compile) |
| `npm run ci:e2e` | `build` then `e2e:ci` (full UI smoke with mocked backend) |

## Project Structure

From the monorepo root, this app lives at **`apps/web/`**:

```
apps/web/
├── app/                 # Next.js App Router (routes, layouts, `globals.css`)
├── components/          # Shared React components
├── lib/                 # API client and helpers (e.g. `lib/api.ts`)
├── next.config.js
├── tailwind.config.js
├── tsconfig.json
└── package.json
```

## Connecting to Backend

Set **`NEXT_PUBLIC_API_URL`** (see **`.env.example`**, copied to **`.env.local`** in dev) to your API base URL, usually `http://localhost:8000`.

Start the API from the repo root:

```bash
python3 apps/api/server/main.py
# or: cd apps/api/server && python3 -m uvicorn main:app --reload --port 8000
```

### Troubleshooting (“web doesn’t work”)

1. **Node 20+** — match **`.nvmrc`**; run `node -v`.
2. **Install & build** — from `apps/web`: `npm ci` (or `npm install`), then `npm run dev` or `npm run ci` to match CI.
3. **Environment** — copy **`.env.example`** → **`.env.local`**. Set **`NEXTAUTH_SECRET`** (e.g. `openssl rand -base64 32`) so NextAuth can issue sessions. **`NEXTAUTH_URL`** should match where you open the app (e.g. `http://localhost:3000`).
4. **API must be up** — the UI calls **`NEXT_PUBLIC_API_URL`** (default `http://localhost:8000`). Login and most actions use the FastAPI backend; if the API is down, the home page shows **offline** and login will fail.
5. **GitHub OAuth** — optional. If **`GITHUB_ID`** / **`GITHUB_SECRET`** are unset, NextAuth still runs with a placeholder provider; use the **`/login`** form (FastAPI `/auth/login`) for username/password.

## Training console

The **Training** page calls `POST /training/start`. Native trainer `step_*.pt` files on the API host embed `stoi` / `itos` / `chars` for fair `cli.py eval`; formats and caveats are in [docs/policies/CONTRIBUTING.md](../../docs/policies/CONTRIBUTING.md) (*Checkpoint vocabulary*).

## License

MIT