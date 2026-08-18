"use client";

import Link from "next/link";
import * as React from "react";
import { LazyIcon } from "./LazyIcon";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./card";
import { cn } from "./utils";

type Tone = "blue" | "green" | "amber" | "rose" | "violet" | "slate";

const toneStyles: Record<
  Tone,
  {
    icon: string;
    accentText: string;
    ring: string;
    surface: string;
  }
> = {
  blue: {
    icon: "bg-sky-100 text-sky-700 dark:bg-sky-950/70 dark:text-sky-300",
    accentText: "text-sky-700 dark:text-sky-300",
    ring: "ring-sky-100 dark:ring-sky-900/50",
    surface: "from-sky-50/70 via-card to-card dark:from-sky-950/30 dark:via-card dark:to-card",
  },
  green: {
    icon: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/70 dark:text-emerald-300",
    accentText: "text-emerald-700 dark:text-emerald-300",
    ring: "ring-emerald-100 dark:ring-emerald-900/50",
    surface: "from-emerald-50/70 via-card to-card dark:from-emerald-950/30 dark:via-card dark:to-card",
  },
  amber: {
    icon: "bg-amber-100 text-amber-700 dark:bg-amber-950/70 dark:text-amber-300",
    accentText: "text-amber-700 dark:text-amber-300",
    ring: "ring-amber-100 dark:ring-amber-900/50",
    surface: "from-amber-50/70 via-card to-card dark:from-amber-950/30 dark:via-card dark:to-card",
  },
  rose: {
    icon: "bg-rose-100 text-rose-700 dark:bg-rose-950/70 dark:text-rose-300",
    accentText: "text-rose-700 dark:text-rose-300",
    ring: "ring-rose-100 dark:ring-rose-900/50",
    surface: "from-rose-50/70 via-card to-card dark:from-rose-950/30 dark:via-card dark:to-card",
  },
  violet: {
    icon: "bg-violet-100 text-violet-700 dark:bg-violet-950/70 dark:text-violet-300",
    accentText: "text-violet-700 dark:text-violet-300",
    ring: "ring-violet-100 dark:ring-violet-900/50",
    surface: "from-violet-50/70 via-card to-card dark:from-violet-950/30 dark:via-card dark:to-card",
  },
  slate: {
    icon: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
    accentText: "text-slate-700 dark:text-slate-300",
    ring: "ring-slate-100 dark:ring-slate-800",
    surface: "from-slate-50/70 via-card to-card dark:from-slate-900/40 dark:via-card dark:to-card",
  },
};

export function DashboardHero({
  eyebrow,
  title,
  description,
  actions,
  children,
  className,
}: {
  eyebrow?: string;
  title: string;
  description: string;
  actions?: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("dashboard-hero p-2.5 sm:p-3 lg:p-3 xl:p-4", className)}>
      <div className="relative z-10 flex flex-col gap-3 xl:flex-row xl:items-end xl:justify-between">
        <div className="space-y-1.5 reading-width">
          {eyebrow ? <span className="dashboard-kicker">{eyebrow}</span> : null}
          <div className="space-y-1">
            <h2 className="text-balance text-2xl font-bold tracking-normal text-foreground sm:text-3xl xl:text-[2.45rem]">
              {title}
            </h2>
            <p className="max-w-3xl text-sm leading-6 text-muted-foreground sm:text-[0.98rem]">
              {description}
            </p>
          </div>
        </div>
        {actions ? (
          <div className="action-row w-full xl:w-auto xl:max-w-[34rem] xl:justify-end">
            {actions}
          </div>
        ) : null}
      </div>
      {children ? <div className="relative z-10 mt-3">{children}</div> : null}
    </section>
  );
}

export function MetricCard({
  label,
  value,
  hint,
  icon,
  tone = "blue",
  className,
}: {
  label: string;
  value: number | string;
  hint?: string;
  icon?: React.ReactNode;
  tone?: Tone;
  className?: string;
}) {
  const toneStyle = toneStyles[tone];

  return (
    <Card className={cn("metric-card border-border/75 bg-gradient-to-br p-0", toneStyle.surface, className)}>
      <CardContent className="flex flex-col gap-2 p-2.5 sm:p-3 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0 space-y-1">
          <p className="text-xs font-bold uppercase tracking-normal text-muted-foreground">
            {label}
          </p>
          <p className="break-words text-[1.5rem] font-bold tracking-normal sm:text-[1.85rem]">
            {value}
          </p>
          {hint ? <p className="max-w-[30ch] text-xs leading-5 text-muted-foreground">{hint}</p> : null}
        </div>
        {icon ? (
          <div className={cn("self-start rounded-2xl p-2 shadow-sm ring-1 md:self-auto", toneStyle.icon, toneStyle.ring)}>
            {icon}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

export function SectionPanel({
  title,
  description,
  action,
  children,
  className,
  contentClassName,
}: {
  title: string;
  description?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  return (
    <Card className={cn("overflow-visible", className)}>
      <CardHeader className="border-b border-border/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.96),rgba(249,251,253,0.84))] py-3 sm:py-3">
        <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-0.5">
            <CardTitle>{title}</CardTitle>
            {description ? <CardDescription>{description}</CardDescription> : null}
          </div>
          {action ? <div className="w-full shrink-0 lg:w-auto lg:max-w-[20rem]">{action}</div> : null}
        </div>
      </CardHeader>
      <CardContent className={cn("pt-2.5 sm:pt-3", contentClassName)}>{children}</CardContent>
    </Card>
  );
}

export function ActionCard({
  href,
  title,
  description,
  icon,
  tone = "blue",
  trailing,
  className,
}: {
  href: string;
  title: string;
  description: string;
  icon?: React.ReactNode;
  tone?: Tone;
  trailing?: React.ReactNode;
  className?: string;
}) {
  const toneStyle = toneStyles[tone];

  return (
    <Link
      href={href}
      className={cn(
        "group data-card block w-full p-2.5 transition-[transform,border-color,box-shadow,background-color] duration-200 hover:-translate-y-0.5 hover:border-primary/20 hover:shadow-[0_20px_40px_-30px_rgba(15,23,42,0.28)] sm:p-3",
        className,
      )}
    >
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 items-start gap-2.5">
          {icon ? (
            <div className={cn("mt-0.5 rounded-2xl p-2 shadow-sm ring-1", toneStyle.icon, toneStyle.ring)}>
              {icon}
            </div>
          ) : null}
          <div className="min-w-0 space-y-0.5">
            <h3 className="text-base font-semibold leading-6 text-foreground text-balance">{title}</h3>
            <p className="text-sm leading-5 text-muted-foreground">{description}</p>
          </div>
        </div>
        {trailing ?? (
          <LazyIcon name="ArrowRight" className={cn("mt-0.5 size-4 shrink-0 self-end text-muted-foreground transition-transform duration-200 group-hover:translate-x-0.5 sm:self-auto", toneStyle.accentText)} />
        )}
      </div>
    </Link>
  );
}

export function EmptyStatePanel({
  title,
  description,
  className,
}: {
  title: string;
  description: string;
  className?: string;
}) {
  return (
    <div className={cn("empty-state-panel grid place-items-center px-3 py-4 text-center sm:px-4", className)}>
      <div className="space-y-1 reading-width">
        <p className="text-sm font-semibold text-foreground">{title}</p>
        <p className="mx-auto max-w-2xl text-xs leading-5 text-muted-foreground">{description}</p>
      </div>
    </div>
  );
}

export function SoftStat({
  label,
  value,
  tone = "slate",
  className,
}: {
  label: string;
  value: number | string;
  tone?: Tone;
  className?: string;
}) {
  const toneStyle = toneStyles[tone];

  return (
    <div className={cn("soft-panel px-2.5 py-2", className)}>
      <p className="text-[0.65rem] font-bold uppercase tracking-normal text-muted-foreground">{label}</p>
      <p className={cn("mt-0.5 break-words text-base font-bold tracking-normal sm:text-lg", toneStyle.accentText)}>
        {value}
      </p>
    </div>
  );
}

export function NoticeBanner({
  tone = "blue",
  children,
  className,
}: {
  tone?: Tone;
  children: React.ReactNode;
  className?: string;
}) {
  const toneClassName: Record<Tone, string> = {
    blue: "border-sky-200 bg-sky-50 text-sky-800 dark:border-sky-900/60 dark:bg-sky-950/50 dark:text-sky-200",
    green: "border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900/60 dark:bg-emerald-950/50 dark:text-emerald-200",
    amber: "border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/50 dark:text-amber-200",
    rose: "border-rose-200 bg-rose-50 text-rose-800 dark:border-rose-900/60 dark:bg-rose-950/50 dark:text-rose-200",
    violet: "border-violet-200 bg-violet-50 text-violet-800 dark:border-violet-900/60 dark:bg-violet-950/50 dark:text-violet-200",
    slate: "border-slate-200 bg-slate-50 text-slate-700 dark:border-slate-800 dark:bg-slate-900/60 dark:text-slate-200",
  };

  return (
    <div className={cn("rounded-[1.15rem] border px-3 py-2 text-xs leading-5 shadow-[0_12px_28px_-26px_rgba(15,23,42,0.2)]", toneClassName[tone], className)}>
      {children}
    </div>
  );
}
