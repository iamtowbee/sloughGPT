# Agents

- **Repo map & commands:** [.agents/skills/SKILL.md](.agents/skills/SKILL.md). **API + web** in one terminal: repo root **`npm install && npm run dev:stack`**, or **`./scripts/dev-stack.sh`** / **`make dev-stack`** — [QUICKSTART.md](QUICKSTART.md). Root **`package.json`** contract: **`npm run test:repo-root`**, **`make test-repo-root`**, or **`python3 -m pytest tests/test_repo_root_package_json.py -q`**.
- **strui (shared UI / Storybook):** [packages/strui/README.md](packages/strui/README.md) — **Design principles** (a11y, viewports), **Foundations**, **Component gallery**; **`cd packages/strui && npm run storybook`** or **`npm run ci`** (CI job **`test-strui`**).
- **Colab notebook:** root [`sloughgpt_colab.ipynb`](sloughgpt_colab.ipynb); full execute [`scripts/run_colab_notebook_smoke.sh`](scripts/run_colab_notebook_smoke.sh) or **`make colab-smoke`** (`--help`, **`make help`**); regression **`make colab-test`** / [`tests/test_sloughgpt_colab_notebook.py`](tests/test_sloughgpt_colab_notebook.py) — [README.md](README.md) (*Google Colab*). **§11** cognitive (SM-2 + SCAMPER) must remain **one** code cell (single **`_asyncio_run`**). One pytest subtest needs **`bash`** on **`PATH`** (skipped otherwise; see [CONTRIBUTING.md](CONTRIBUTING.md)).
- **Contributing & CI parity:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Layout:** [docs/STRUCTURE.md](docs/STRUCTURE.md); runtime dirs (**`data/experiments`**, **`data/features`**, **`data/tuning`**, **`data/vector_store`**) — [data/README.md](data/README.md)
- **CLI → TUI:** Phase 1 shell [`apps/tui/`](apps/tui/README.md) (`python3 -m apps.tui`); roadmap [docs/plans/tui-cli-port.md](docs/plans/tui-cli-port.md)

## Chat UI (Phase 30-34)

### Components
Chat UI components are in [`apps/web/components/chat/`](apps/web/components/chat/):

| Component | Description |
|-----------|-------------|
| `ChatHeader` | Title + ModelStatusBar + Settings toggle |
| `ChatSettings` | Model/Temp/Max controls with animation |
| `ChatMessages` | Message list container |
| `MessageBubble` | Message with copy, markdown, images |
| `ChatInput` | Textarea + send + voice + image upload |
| `EmptyState` | Illustration + keyboard hints |
| `LoadingIndicator` | Animated typing dots |
| `TypingDots` | Reusable bouncing dots animation |
| `Toast` | Success/error/info notifications |
| `ErrorBanner` | Error with retry/dismiss |
| `VoiceInput` | Speech-to-text microphone |
| `ImageUpload` | File picker + preview |
| `Markdown` | Bold, italic, code, links |

### Features
- Markdown rendering (bold, italic, code, links)
- Entrance animations on messages
- Toast notifications
- Error handling with retry
- Keyboard shortcuts (Enter to send, Esc to close)
- Responsive design

### Pending (Multimodal Engine)
- Voice input (Web Speech API) - pending multimodal inference
- Image upload with preview - pending multimodal inference

### API Configuration
Frontend uses direct API URL: `http://localhost:8000/chat/stream`
Set via `NEXT_PUBLIC_API_URL` env var or defaults to `http://localhost:8000`.
