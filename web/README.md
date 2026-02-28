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

- **Framework**: React 18 with TypeScript
- **UI Library**: Base UI
- **State Management**: Zustand
- **Routing**: React Router DOM
- **Charts**: Recharts
- **Build Tool**: Vite
- **Styling**: Tailwind CSS

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# Navigate to web directory
cd web

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

```
web/
├── src/
│   ├── components/     # React components
│   │   ├── Chat.tsx    # Chat interface
│   │   ├── Datasets.tsx # Dataset management
│   │   ├── Models.tsx  # Model management
│   │   ├── Monitoring.tsx # System monitoring
│   │   └── Home.tsx   # Dashboard home
│   ├── store/          # Zustand state management
│   ├── utils/          # Utility functions and API client
│   ├── App.tsx        # Main app component
│   ├── main.tsx       # Entry point
│   └── index.css      # Global styles
├── public/            # Static assets
├── index.html        # HTML template
├── package.json      # Dependencies
├── vite.config.ts   # Vite configuration
├── tailwind.config.js # Tailwind configuration
└── tsconfig.json    # TypeScript configuration
```

## Connecting to Backend

The web UI proxies API requests through Vite's dev server. Make sure your FastAPI backend is running at `http://localhost:8000`.

To start the backend:

```bash
# From the project root
python -m uvicorn domains.ui.webui:app --reload --port 8000
```

## License

MIT