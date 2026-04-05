import { useState } from 'react'
import type { Meta, StoryObj } from '@storybook/react'
import { PromptComposer } from './prompt-composer'

const meta = {
  title: 'AI/PromptComposer',
  component: PromptComposer,
  tags: ['autodocs'],
  decorators: [
    (Story) => (
      <div className="sl-shell-main flex min-h-[280px] w-full max-w-lg flex-col justify-end rounded-none border border-border">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof PromptComposer>

export default meta

function Stateful() {
  const [value, setValue] = useState('')
  return (
    <PromptComposer
      value={value}
      onValueChange={setValue}
      onSubmit={() => setValue('')}
      placeholder="Ask anything…"
    />
  )
}

export const Interactive: StoryObj = {
  render: () => <Stateful />,
}
