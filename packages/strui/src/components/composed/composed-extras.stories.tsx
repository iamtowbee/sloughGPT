import type { Meta, StoryObj } from '@storybook/react'
import { Chip } from './chip'
import { KeyValueList } from './key-value-list'
import { ListRow } from './list-row'
import { ProgressBar } from './progress-bar'
import { ScrollPanel } from './scroll-panel'
import { SectionHeader } from './section-header'
import { StepIndicator } from './step-indicator'

const meta = {
  title: 'Composed/Extras',
  tags: ['autodocs'],
} satisfies Meta

export default meta

export const MetadataAndProgress: StoryObj = {
  render: () => (
    <div className="str-safe-x mx-auto max-w-2xl space-y-8 p-4">
      <SectionHeader
        title="Run configuration"
        description="Read-only snapshot for this job."
        action={<Chip variant="primary">v2</Chip>}
      />
      <KeyValueList
        items={[
          { label: 'Run id', value: 'run_01a2b3c', mono: true },
          { label: 'Dataset', value: 'sft-v1.jsonl' },
          { label: 'Learning rate', value: '2e-4' },
        ]}
      />
      <div className="space-y-2">
        <p className="text-xs font-medium uppercase text-muted-foreground">Progress</p>
        <ProgressBar value={62} />
        <ProgressBar indeterminate className="opacity-80" />
      </div>
      <StepIndicator steps={['Prepare', 'Train', 'Export'] as const} current={1} />
      <ScrollPanel className="p-3 font-mono text-xs">
        {Array.from({ length: 12 }, (_, i) => (
          <div key={i} className="py-0.5 text-muted-foreground">
            [{String(i).padStart(2, '0')}] log line…
          </div>
        ))}
      </ScrollPanel>
      <div className="divide-y divide-border rounded-none border border-border">
        <ListRow title="Experiment A" description="Last updated 2h ago" trailing="›" />
        <ListRow title="Experiment B" description="Queued" trailing="›" />
      </div>
      <div className="flex flex-wrap gap-2">
        <Chip>filter: active</Chip>
        <Chip variant="outline" onRemove={() => {}}>
          removable
        </Chip>
      </div>
    </div>
  ),
}
