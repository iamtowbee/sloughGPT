import type { Meta, StoryObj } from "@storybook/react";
import { Button } from "./Button";

const meta = {
  title: "Primitives/Button",
  component: Button,
  tags: ["autodocs"],
  argTypes: {
    variant: {
      control: "select",
      options: ["primary", "secondary", "ghost", "destructive"],
    },
  },
} satisfies Meta<typeof Button>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    children: "Run inference",
    variant: "primary",
  },
};

export const Secondary: Story = {
  args: {
    children: "Dataset",
    variant: "secondary",
  },
};

export const Ghost: Story = {
  args: {
    children: "Cancel",
    variant: "ghost",
  },
};

export const Destructive: Story = {
  args: {
    children: "Stop job",
    variant: "destructive",
  },
};
