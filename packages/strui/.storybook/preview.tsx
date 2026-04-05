import type { Preview } from '@storybook/react'
import '../src/styles/globals.css'

const preview: Preview = {
  parameters: {
    layout: 'centered',
    backgrounds: {
      default: 'shell',
      values: [
        { name: 'shell', value: 'var(--background)' },
        { name: 'card', value: 'var(--card)' },
      ],
    },
  },
  decorators: [
    (Story) => (
      <div className="min-h-[200px] min-w-[320px] font-sans text-foreground antialiased">
        <Story />
      </div>
    ),
  ],
}

export default preview
