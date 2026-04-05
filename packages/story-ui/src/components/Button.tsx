import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { forwardRef, type ButtonHTMLAttributes } from "react";
import { cn } from "../lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-none px-4 py-2 text-sm font-medium transition-all duration-200 ease-smooth disabled:pointer-events-none disabled:opacity-50 active:scale-[0.99]",
  {
    variants: {
      variant: {
        primary: "bg-primary text-primary-foreground shadow-sm hover:opacity-90",
        secondary:
          "border border-border bg-secondary text-secondary-foreground hover:bg-muted",
        ghost:
          "bg-transparent text-muted-foreground hover:bg-muted/55 hover:text-foreground",
        destructive:
          "bg-destructive text-destructive-foreground shadow-sm hover:opacity-90",
      },
    },
    defaultVariants: {
      variant: "primary",
    },
  },
);

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, asChild, type, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant }), className)}
        ref={ref}
        {...props}
        {...(!asChild ? { type: type ?? "button" } : {})}
      />
    );
  },
);
Button.displayName = "Button";

export { buttonVariants };
