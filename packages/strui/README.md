# strui (`@sloughgpt/strui`)

Standalone UI package: **SloughGPT web design** (pastel lattice tokens, `sl-*` utilities, Radix + CVA components) copied from `apps/web`. Keep `src/styles/globals.css` aligned with `apps/web/app/globals.css` when the shell changes.

## Commands

```sh
cd packages/strui && npm ci && npm run storybook
```

```sh
npm run ci
```

## Consume

```tsx
import { Button, cn } from '@sloughgpt/strui'
import '@sloughgpt/strui/styles/globals.css'
```

Point your bundler at this package’s `src` (or add a build step later) and ensure Tailwind `content` includes `node_modules/@sloughgpt/strui` if you class-scan from here.
