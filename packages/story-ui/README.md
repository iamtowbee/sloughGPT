# story-ui

Design system for SloughGPT-style AI web UIs: **tokens + Tailwind**, **head** utilities (`cn`), and React primitives. Storybook is the catalog.

## Commands

```sh
cd packages/story-ui
npm ci
npm run storybook
```

Static build (output in `storybook-static/`, gitignored):

```sh
npm run ci
```

## Layout

- `src/head/` — `@sloughgpt/story-ui/head`: class merging and future token helpers (no heavy components).
- `src/components/` — components and `*.stories.tsx`.
- `src/styles/globals.css` — pastel lattice theme; keep aligned with `apps/web/app/globals.css` until apps consume this package directly.

## Branch

Develop on git branch **`story-ui`**. Replacing `apps/web` imports with this package is a follow-up once the catalog covers the shell.
