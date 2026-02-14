import { cn } from "@/lib/utils";
import { type RiskLevel } from "@/data/mockData";
import { type ReactNode } from "react";

// --- Badge ---
const badgeVariants: Record<string, string> = {
  default: "border-transparent bg-primary text-primary-foreground",
  secondary: "border-transparent bg-secondary text-secondary-foreground",
  destructive: "border-transparent bg-destructive text-destructive-foreground",
  outline: "text-foreground border-border",
  success: "border-transparent bg-emerald-100 text-emerald-700",
  warning: "border-transparent bg-amber-100 text-amber-700",
  danger: "border-transparent bg-red-100 text-red-700",
  info: "border-transparent bg-blue-100 text-blue-700",
};

export function Badge({ variant = "default", className, children }: { variant?: string; className?: string; children: ReactNode }) {
  return (
    <span className={cn("inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors", badgeVariants[variant] || badgeVariants.default, className)}>
      {children}
    </span>
  );
}

// --- RiskBadge ---
const riskBadgeMap: Record<RiskLevel, string> = {
  critical: "danger",
  high: "warning",
  medium: "warning",
  low: "success",
};

export function RiskBadge({ level }: { level: RiskLevel }) {
  return <Badge variant={riskBadgeMap[level]}>{level.charAt(0).toUpperCase() + level.slice(1)}</Badge>;
}

// --- StatusDot ---
export function StatusDot({ connected }: { connected: boolean }) {
  return (
    <span className="relative flex items-center gap-1.5 text-xs font-medium">
      <span className={cn("relative flex h-2.5 w-2.5 rounded-full", connected ? "bg-success" : "bg-destructive")}>
        {connected && <span className="absolute inset-0 rounded-full bg-success animate-pulse-ring" />}
      </span>
      {connected ? "Live" : "Offline"}
    </span>
  );
}

// --- StatsCard ---
export function StatsCard({ title, value, subtitle, icon, trend }: { title: string; value: string | number; subtitle?: string; icon: ReactNode; trend?: { value: string; positive: boolean } }) {
  return (
    <div className="rounded-2xl border border-border bg-card shadow-sm p-5 hover:shadow-md hover:border-primary/30 transition-all duration-200">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-muted-foreground font-medium">{title}</span>
        <div className="h-10 w-10 rounded-2xl bg-secondary flex items-center justify-center text-muted-foreground">{icon}</div>
      </div>
      <p className="text-2xl font-bold tracking-tight animate-count-up">{value}</p>
      <div className="flex items-center gap-2 mt-1">
        {trend && (
          <span className={cn("text-xs font-medium", trend.positive ? "text-success" : "text-destructive")}>
            {trend.positive ? "↑" : "↓"} {trend.value}
          </span>
        )}
        {subtitle && <span className="text-xs text-muted-foreground">{subtitle}</span>}
      </div>
    </div>
  );
}

// --- Progress ---
export function ProgressBar({ value, max = 100, className, colorClass }: { value: number; max?: number; className?: string; colorClass?: string }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className={cn("h-2 w-full rounded-full bg-secondary overflow-hidden", className)}>
      <div className={cn("h-full rounded-full transition-all duration-500", colorClass || "bg-primary")} style={{ width: `${pct}%` }} />
    </div>
  );
}

// --- HeroBanner ---
export function HeroBanner({ label, title, description, gradient, children }: { label: string; title: string; description: string; gradient: string; children?: ReactNode }) {
  return (
    <div className={cn("rounded-3xl p-8 text-white relative overflow-hidden bg-gradient-to-r", gradient)}>
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMjAiIGN5PSIyMCIgcj0iMSIgZmlsbD0icmdiYSgyNTUsMjU1LDI1NSwwLjEpIi8+PC9zdmc+')] opacity-50" />
      <div className="relative z-10">
        <span className="inline-block px-3 py-1 rounded-full bg-white/20 text-sm font-medium mb-3">{label}</span>
        <h1 className="text-3xl font-bold mb-2">{title}</h1>
        <p className="text-white/80 max-w-xl">{description}</p>
        {children}
      </div>
    </div>
  );
}
