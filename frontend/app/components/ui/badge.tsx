'use client';

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "./utils";

const badgeVariants = cva(
  "inline-flex min-w-0 max-w-full items-center justify-center rounded-full border px-3 py-1 text-center text-sm font-semibold leading-5 w-fit whitespace-normal break-words shrink gap-1.5 [&>svg]:size-3 [&>svg]:pointer-events-none [&>svg]:shrink-0 focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive transition-[color,background-color,box-shadow] overflow-visible shadow-[0_10px_22px_-20px_rgba(15,23,42,0.25)]",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-primary text-primary-foreground [a&]:hover:bg-primary/90",
        secondary:
          "border-transparent bg-secondary text-secondary-foreground [a&]:hover:bg-secondary/90",
        destructive:
          "border-transparent bg-destructive text-white [a&]:hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60",
        outline:
          "border-border/80 bg-card text-foreground [a&]:hover:bg-secondary [a&]:hover:text-primary",
        success:
          "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900/60 dark:bg-emerald-950/60 dark:text-emerald-300 [a&]:hover:bg-emerald-100",
        warning:
          "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900/60 dark:bg-amber-950/60 dark:text-amber-300 [a&]:hover:bg-amber-100",
        danger:
          "border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-900/60 dark:bg-rose-950/60 dark:text-rose-300 [a&]:hover:bg-rose-100",
        info:
          "border-sky-200 bg-sky-50 text-sky-700 dark:border-sky-900/60 dark:bg-sky-950/60 dark:text-sky-300 [a&]:hover:bg-sky-100",
        neutral:
          "border-slate-200 bg-slate-50 text-slate-700 dark:border-slate-800 dark:bg-slate-900/60 dark:text-slate-300 [a&]:hover:bg-slate-100",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

function Badge({
  className,
  variant,
  asChild = false,
  ...props
}: React.ComponentProps<"span"> &
  VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot : "span";

  return (
    <Comp
      data-slot="badge"
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  );
}

export { Badge, badgeVariants };
