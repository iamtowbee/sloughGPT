import { type HTMLAttributes } from "react";
import { cn } from "../lib/utils";

export interface MessageBubbleProps extends HTMLAttributes<HTMLDivElement> {
  role: "user" | "assistant" | "system";
}

export function MessageBubble({
  className,
  role,
  children,
  ...props
}: MessageBubbleProps) {
  const tone =
    role === "user"
      ? "ml-auto border-primary/35 bg-primary/15 text-foreground"
      : role === "assistant"
        ? "mr-auto border-border bg-card/90 text-card-foreground"
        : "mx-auto border-dashed border-warning/50 bg-muted/40 text-muted-foreground";
  return (
    <div
      className={cn(
        "max-w-[var(--chat-thread-max)] border px-3 py-2 text-sm shadow-sm transition-colors duration-200 ease-smooth",
        tone,
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}
