# strui (`@sloughgpt/strui`)

Standalone UI package: **SloughGPT web design** (pastel lattice tokens, `sl-*` utilities, Radix + CVA). **Primitives** match `apps/web/components/ui`. **Composed** layouts and patterns live under `src/components/composed/` (also exported as `@sloughgpt/strui/composed`). **AI** flows (`src/components/ai/`) include chat shell, model picker, RAG citations, reasoning panel, sources, attachments, and more.

Keep `src/styles/globals.css` aligned with `apps/web/app/globals.css` when the shell changes.

### Composed (apps & dashboards)

`PageHeader`, `AppShell`, `NavRail` / `NavRailLink`, `Toolbar`, `KpiGrid`, `StatCard`, `FormField`, `SettingsRow`, `SearchInput`, `InlineBanner`, `Skeleton`, `EmptyCard`, `CopyButton`, `KeyValueList`, `ProgressBar`, `StepIndicator`, `Chip`, `SectionHeader`, `ScrollPanel`, `ListRow`, `StatusDot`, `FoldSection`, `Timeline`.

### AI (agents & assistants)

`ChatLayout`, `ModelPicker`, `Citation`, `ReasoningPanel`, `SourceList`, `AttachmentChip`, `StreamingAssistantPlaceholder`, `JobStatus`, plus `ChatThread`, `PromptComposer`, `MessageBubble`, `ToolCallCard`, `TokenMeter`, etc.

Storybook: **Composed → Overview** / **Extras**, **StatusDot**, **FoldSection**, **Timeline**, **AI → Composed flows**, **JobStatus**, **StreamingAssistantPlaceholder**.

## Commands

```sh
cd packages/strui && npm ci && npm run storybook
```

```sh
npm run ci
```

## Consume

```tsx
import { Button, ChatThread, PromptComposer, cn } from '@sloughgpt/strui'
import '@sloughgpt/strui/styles/globals.css'
```

Subpath barrels: `@sloughgpt/strui/ai`, `@sloughgpt/strui/composed` (subset of the root export; root re-exports everything).

### `apps/web` (this repo)

The Next app depends on **`file:../../packages/strui`**, uses **`transpilePackages: ['@sloughgpt/strui']`**, and keeps stable imports via thin client shims under **`apps/web/components/ui/*`** that re-export from `@sloughgpt/strui`. **`lib/cn`** re-exports `cn` from the same package. **Do not** import `@sloughgpt/strui/styles/globals.css` in the web app: the shell keeps **`app/globals.css`** with its own token format (RGB triplets for `/opacity`); Tailwind **`content`** includes `../../packages/strui/src/**` so classes from the library are not purged.

## PWA and mobile web

This is **one React + Tailwind codebase** for browsers (including installed PWAs). It is not React Native; use the same components in Capacitor/Tauri/WebView shells.

- In the host app, use **`viewport-fit=cover`** (and optional `theme-color`) so `env(safe-area-inset-*)` applies.
- Use **`str-safe-top` / `str-safe-bottom` / `str-safe-x` / `str-safe-all`** on fixed headers, bottom composers, and full-bleed layouts.
- **`str-touch-target`** enforces a 44×44 minimum (send buttons, primary actions).
- **`str-min-h-screen`** uses `100dvh` for stable mobile viewport height.
- **`str-chat-scroll`** enables momentum scrolling and avoids scroll chaining on chat regions.

Storybook: **AI → ChatShell → iPhone** for viewport preview; **Composed → Overview** for app shell patterns.
