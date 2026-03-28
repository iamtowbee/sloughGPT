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

- Node.js 18+
- npm or yarn

### Installation

```bash
# From repository root — Next.js app lives under apps/web/web
cd apps/web/web

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

Before pushing changes, run the same checks as CI: **`npm ci && npm run lint && npm run typecheck`** (from this directory).

### Build for Production

```bash
# Build the application
npm run build

# Preview the production build
npm run preview

# The built files will be in the dist/ directory
```

## API Configuration

The web UI connects to the FastAPI backend. By default, it expects the API at `http://localhost:8000`.

To change the API URL, create a `.env` file:

```bash
VITE_API_BASE_URL=http://your-api-server:8000
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |
| `npm run typecheck` | Run TypeScript type checking |

## Project Structure

From the monorepo root, this app lives at **`apps/web/web/`**:

```
apps/web/web/
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

## License

MIT