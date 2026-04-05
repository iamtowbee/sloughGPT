import type { Meta, StoryObj } from "@storybook/react";
import { MessageBubble } from "./MessageBubble";

const meta = {
  title: "AI/MessageBubble",
  component: MessageBubble,
  tags: ["autodocs"],
  argTypes: {
    role: {
      control: "select",
      options: ["user", "assistant", "system"],
    },
  },
  decorators: [
    (Story) => (
      <div className="sl-shell-main flex w-full max-w-lg flex-col gap-3 p-6">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof MessageBubble>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Assistant: Story = {
  args: {
    role: "assistant",
    children: "Here is a concise summary of your training run…",
  },
};

export const User: Story = {
  args: {
    role: "user",
    children: "Tune learning rate and rerun for 500 steps.",
  },
};

export const System: Story = {
  args: {
    role: "system",
    children: "Model context refreshed — older tool results were dropped.",
  },
};

export const Thread: StoryObj<typeof MessageBubble> = {
  render: () => (
    <div className="flex w-full max-w-lg flex-col gap-3">
      <MessageBubble role="user">Export the checkpoint to W&B.</MessageBubble>
      <MessageBubble role="assistant">
        Done. Artifact URL is in the job panel under Outputs.
      </MessageBubble>
      <MessageBubble role="system">Session idle — reconnect to resume.</MessageBubble>
    </div>
  ),
};
