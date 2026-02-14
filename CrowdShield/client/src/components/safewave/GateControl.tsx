import { useState } from "react";
import { Badge } from "@/components/safewave/common";
import { ShieldAlert, Unlock, Loader2, Lock } from "lucide-react";

interface GateData {
    _id: string;
    gateNumber: number;
    gateName: string;
    turnstiles: boolean[];  // true = unlocked, false = locked
}

interface GateControlProps {
    eventId: string;
    gates: GateData[];
    onToggleTurnstile: (gateNumber: number, turnstileIndex: number) => Promise<void>;
    onAutoLockdown: () => Promise<void>;
    onOpenAll: () => Promise<void>;
}

// SVG Turnstile icon — rotating gate visual
function TurnstileIcon({ unlocked, size = 36 }: { unlocked: boolean; size?: number }) {
    const color = unlocked ? "#22c55e" : "#ef4444";
    const bgColor = unlocked ? "#dcfce7" : "#fee2e2";
    return (
        <svg width={size} height={size} viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
            {/* Base circle */}
            <circle cx="20" cy="20" r="18" fill={bgColor} stroke={color} strokeWidth="2" />
            {/* Center hub */}
            <circle cx="20" cy="20" r="4" fill={color} />
            {/* Rotating arms */}
            <line x1="20" y1="4" x2="20" y2="16" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
            <line x1="20" y1="24" x2="20" y2="36" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
            <line x1="4" y1="20" x2="16" y2="20" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
            <line x1="24" y1="20" x2="36" y2="20" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
            {/* Lock icon for locked state */}
            {!unlocked && (
                <>
                    <rect x="15" y="14" width="10" height="8" rx="1.5" fill="#ef4444" />
                    <path d="M18 14v-2a2 2 0 014 0v2" stroke="white" strokeWidth="1.5" fill="none" strokeLinecap="round" />
                    <circle cx="20" cy="18.5" r="1" fill="white" />
                </>
            )}
        </svg>
    );
}

export default function GateControl({ eventId, gates, onToggleTurnstile, onAutoLockdown, onOpenAll }: GateControlProps) {
    const [loadingTurnstile, setLoadingTurnstile] = useState<string | null>(null);
    const [bulkLoading, setBulkLoading] = useState<string | null>(null);

    const totalTurnstiles = gates.reduce((s, g) => s + g.turnstiles.length, 0);
    const lockedTurnstiles = gates.reduce((s, g) => s + g.turnstiles.filter(t => !t).length, 0);
    const openTurnstiles = totalTurnstiles - lockedTurnstiles;

    const handleToggle = async (gateNumber: number, idx: number) => {
        const key = `${gateNumber}-${idx}`;
        setLoadingTurnstile(key);
        try {
            await onToggleTurnstile(gateNumber, idx);
        } finally {
            setLoadingTurnstile(null);
        }
    };

    const handleAutoLockdown = async () => {
        setBulkLoading("lockdown");
        try { await onAutoLockdown(); }
        finally { setBulkLoading(null); }
    };

    const handleOpenAll = async () => {
        setBulkLoading("openAll");
        try { await onOpenAll(); }
        finally { setBulkLoading(null); }
    };

    const inflowPercent = totalTurnstiles > 0 ? Math.round((openTurnstiles / totalTurnstiles) * 100) : 100;

    return (
        <div className="rounded-2xl border border-border bg-card shadow-sm">
            {/* Header */}
            <div className="p-6">
                <div className="flex items-center justify-between flex-wrap gap-3">
                    <div>
                        <h3 className="text-lg font-semibold flex items-center gap-2">
                            <Lock className="h-5 w-5 text-muted-foreground" />
                            Rotating Gate Control
                        </h3>
                        <p className="text-sm text-muted-foreground mt-0.5">
                            Click individual turnstiles to lock/unlock — control crowd inflow per gate
                        </p>
                    </div>
                    <div className="flex items-center gap-3">
                        {/* Inflow indicator */}
                        <div className="text-center px-3">
                            <div className="text-2xl font-bold tracking-tight">
                                <span className={inflowPercent < 50 ? "text-red-600" : inflowPercent < 80 ? "text-amber-600" : "text-emerald-600"}>
                                    {inflowPercent}%
                                </span>
                            </div>
                            <span className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">Inflow</span>
                        </div>
                        <div className="flex gap-2">
                            <button
                                onClick={handleAutoLockdown}
                                disabled={bulkLoading !== null}
                                className="flex items-center gap-1.5 px-3 py-1.5 rounded-2xl bg-red-600 text-white text-sm font-semibold hover:bg-red-700 disabled:opacity-50 transition-colors"
                            >
                                {bulkLoading === "lockdown" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ShieldAlert className="h-3.5 w-3.5" />}
                                Lock 50%
                            </button>
                            <button
                                onClick={handleOpenAll}
                                disabled={bulkLoading !== null}
                                className="flex items-center gap-1.5 px-3 py-1.5 rounded-2xl bg-emerald-600 text-white text-sm font-semibold hover:bg-emerald-700 disabled:opacity-50 transition-colors"
                            >
                                {bulkLoading === "openAll" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Unlock className="h-3.5 w-3.5" />}
                                Unlock All
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Gate rows */}
            <div className="px-6 pb-6 space-y-4">
                {gates.map((gate) => {
                    const gateOpen = gate.turnstiles.filter(t => t).length;
                    const gateLocked = gate.turnstiles.length - gateOpen;
                    return (
                        <div key={gate.gateNumber} className="rounded-xl border border-border bg-secondary/30 p-4">
                            {/* Gate header */}
                            <div className="flex items-center justify-between mb-3">
                                <div className="flex items-center gap-2">
                                    <span className="font-semibold text-sm">Gate {gate.gateNumber}</span>
                                    <span className="text-xs text-muted-foreground">— {gate.gateName}</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <Badge variant={gateLocked === 0 ? "success" : gateLocked === gate.turnstiles.length ? "danger" : "warning"}>
                                        {gateOpen}/{gate.turnstiles.length} open
                                    </Badge>
                                    {gateLocked > 0 && (
                                        <span className="text-xs text-red-500 font-medium">
                                            {Math.round((gateLocked / gate.turnstiles.length) * 100)}% restricted
                                        </span>
                                    )}
                                </div>
                            </div>

                            {/* Turnstile row — horizontal line of rotation gates */}
                            <div className="flex items-center gap-1.5 overflow-x-auto pb-1">
                                {/* Left barrier wall */}
                                <div className="w-1.5 h-12 bg-slate-400 rounded-full shrink-0" />

                                {gate.turnstiles.map((unlocked, idx) => {
                                    const loadKey = `${gate.gateNumber}-${idx}`;
                                    const isLoading = loadingTurnstile === loadKey;
                                    return (
                                        <button
                                            key={idx}
                                            onClick={() => handleToggle(gate.gateNumber, idx)}
                                            disabled={isLoading}
                                            className={`shrink-0 rounded-lg p-1 transition-all duration-200 hover:scale-110 active:scale-95 ${unlocked
                                                ? "hover:bg-red-50"
                                                : "hover:bg-emerald-50"
                                                }`}
                                            title={`Turnstile ${idx + 1}: ${unlocked ? "Unlocked ✓ (click to lock)" : "LOCKED ✕ (click to unlock)"}`}
                                        >
                                            {isLoading ? (
                                                <div className="w-9 h-9 flex items-center justify-center">
                                                    <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                                                </div>
                                            ) : (
                                                <TurnstileIcon unlocked={unlocked} size={36} />
                                            )}
                                        </button>
                                    );
                                })}

                                {/* Right barrier wall */}
                                <div className="w-1.5 h-12 bg-slate-400 rounded-full shrink-0" />
                            </div>

                            {/* Flow bar */}
                            <div className="mt-2 h-1.5 w-full rounded-full bg-secondary overflow-hidden">
                                <div
                                    className={`h-full rounded-full transition-all duration-500 ${gateLocked === 0 ? "bg-emerald-500" : gateLocked === gate.turnstiles.length ? "bg-red-500" : "bg-amber-500"
                                        }`}
                                    style={{ width: `${(gateOpen / gate.turnstiles.length) * 100}%` }}
                                />
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Legend */}
            <div className="px-6 pb-4 flex items-center gap-4 text-xs text-muted-foreground">
                <span className="flex items-center gap-1.5">
                    <TurnstileIcon unlocked={true} size={18} /> Unlocked — people can pass
                </span>
                <span className="flex items-center gap-1.5">
                    <TurnstileIcon unlocked={false} size={18} /> Locked — inflow blocked
                </span>
            </div>

            {/* Warning if any locked */}
            {lockedTurnstiles > 0 && (
                <div className="mx-6 mb-4 p-3 rounded-xl bg-amber-50 border border-amber-200 text-sm text-amber-700 flex items-center gap-2">
                    <ShieldAlert className="h-4 w-4 shrink-0" />
                    <span>
                        <strong>{lockedTurnstiles} of {totalTurnstiles}</strong> turnstiles locked — crowd inflow at <strong>{inflowPercent}%</strong> capacity
                    </span>
                </div>
            )}
        </div>
    );
}
