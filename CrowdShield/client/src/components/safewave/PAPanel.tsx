import { useState, type FormEvent } from "react";
import { Badge } from "@/components/safewave/common";
import { Volume2, VolumeX, Radio, Send, Loader2, Megaphone, Square } from "lucide-react";

interface Broadcast {
    _id: string;
    type: string;
    message: string;
    zones: string[];
    priority: "normal" | "urgent" | "critical";
    triggeredBy: "auto" | "manual";
    status: "active" | "completed";
    timestamp: string;
}

interface PAPanelProps {
    eventId: string;
    zones: { id: string; name: string }[];
    activeBroadcasts: Broadcast[];
    broadcastHistory: Broadcast[];
    onBroadcast: (data: { type: string; message: string; zones: string[]; priority: string }) => Promise<void>;
    onStop: (broadcastId?: string) => Promise<void>;
}

const BROADCAST_TYPES = [
    { value: "evacuation", label: "🚨 Evacuation Order" },
    { value: "stampede_alert", label: "⚠️ Stampede Alert" },
    { value: "missing_person", label: "👤 Missing Person" },
    { value: "custom", label: "📢 Custom Message" },
];

const TEMPLATES: Record<string, string> = {
    evacuation: "ATTENTION: Immediate evacuation required. Please proceed to the nearest exit calmly. Follow the directions of security personnel.",
    stampede_alert: "ATTENTION: Crowd surge detected. All personnel maintain positions. Entry gates are being restricted. Do not push. Stay calm.",
    missing_person: "ATTENTION: A missing person alert has been issued. Please report any sightings to the nearest security checkpoint.",
    custom: "",
};

export default function PAPanel({ eventId, zones, activeBroadcasts, broadcastHistory, onBroadcast, onStop }: PAPanelProps) {
    const [type, setType] = useState("evacuation");
    const [message, setMessage] = useState(TEMPLATES.evacuation);
    const [selectedZones, setSelectedZones] = useState<string[]>(["all"]);
    const [priority, setPriority] = useState("critical");
    const [sending, setSending] = useState(false);
    const [stopping, setStopping] = useState(false);

    const handleTypeChange = (newType: string) => {
        setType(newType);
        setMessage(TEMPLATES[newType] || "");
    };

    const toggleZone = (zoneId: string) => {
        if (zoneId === "all") {
            setSelectedZones(["all"]);
        } else {
            const filtered = selectedZones.filter(z => z !== "all");
            if (filtered.includes(zoneId)) {
                setSelectedZones(filtered.filter(z => z !== zoneId));
            } else {
                setSelectedZones([...filtered, zoneId]);
            }
        }
    };

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if (!message.trim()) return;
        setSending(true);
        try {
            await onBroadcast({ type, message, zones: selectedZones.length ? selectedZones : ["all"], priority });
        } finally {
            setSending(false);
        }
    };

    const handleStop = async (broadcastId?: string) => {
        setStopping(true);
        try {
            await onStop(broadcastId);
        } finally {
            setStopping(false);
        }
    };

    return (
        <div className="rounded-2xl border border-border bg-card shadow-sm">
            {/* Header */}
            <div className="p-6 flex flex-col space-y-1.5">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                    <Megaphone className="h-5 w-5 text-muted-foreground" />
                    PA Override System
                </h3>
                <p className="text-sm text-muted-foreground">
                    Override all microphones and speakers — broadcast emergency alerts across zones
                </p>
            </div>

            {/* Active Broadcasts */}
            {activeBroadcasts.length > 0 && (
                <div className="px-6 pb-4">
                    {activeBroadcasts.map((b) => (
                        <div key={b._id} className="flex items-start gap-3 p-4 rounded-2xl bg-red-50 border-2 border-red-300 animate-pulse mb-2">
                            <div className="h-10 w-10 rounded-2xl bg-red-100 flex items-center justify-center shrink-0">
                                <Radio className="h-5 w-5 text-red-600" />
                            </div>
                            <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 flex-wrap">
                                    <span className="font-semibold text-red-800 text-sm">LIVE BROADCAST</span>
                                    <Badge variant="danger">{b.priority}</Badge>
                                    {b.triggeredBy === "auto" && <Badge variant="warning">Auto-triggered</Badge>}
                                </div>
                                <p className="text-sm text-red-700 mt-1">{b.message}</p>
                                <p className="text-xs text-red-500 mt-1">Zones: {b.zones.join(", ")} • {new Date(b.timestamp).toLocaleTimeString()}</p>
                            </div>
                            <button
                                onClick={() => handleStop(b._id)}
                                disabled={stopping}
                                className="flex items-center gap-1 px-3 py-1.5 rounded-2xl bg-red-600 text-white text-sm font-semibold hover:bg-red-700 shrink-0"
                            >
                                {stopping ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Square className="h-3.5 w-3.5" />}
                                Stop
                            </button>
                        </div>
                    ))}
                </div>
            )}

            {/* Broadcast Form */}
            <form onSubmit={handleSubmit} className="p-6 pt-0 space-y-4">
                {/* Type Selector */}
                <div>
                    <label className="text-sm font-medium text-foreground mb-1.5 block">Broadcast Type</label>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                        {BROADCAST_TYPES.map((t) => (
                            <button
                                key={t.value}
                                type="button"
                                onClick={() => handleTypeChange(t.value)}
                                className={`rounded-xl border px-3 py-2 text-sm font-medium transition-all ${type === t.value
                                    ? "border-primary bg-primary/10 text-primary"
                                    : "border-border bg-background text-muted-foreground hover:border-primary/30"
                                    }`}
                            >
                                {t.label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Message */}
                <div>
                    <label className="text-sm font-medium text-foreground mb-1.5 block">Message</label>
                    <textarea
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        rows={3}
                        className="w-full rounded-2xl border border-border bg-background px-4 py-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary"
                        placeholder="Enter your broadcast message..."
                    />
                </div>

                {/* Zone Selector */}
                <div>
                    <label className="text-sm font-medium text-foreground mb-1.5 block">Target Zones</label>
                    <div className="flex flex-wrap gap-2">
                        <button
                            type="button"
                            onClick={() => toggleZone("all")}
                            className={`rounded-full border px-3 py-1 text-xs font-medium transition-all ${selectedZones.includes("all")
                                ? "border-primary bg-primary text-primary-foreground"
                                : "border-border bg-background text-muted-foreground hover:border-primary/30"
                                }`}
                        >
                            All Zones
                        </button>
                        {zones.map((zone) => (
                            <button
                                key={zone.id}
                                type="button"
                                onClick={() => toggleZone(zone.id)}
                                className={`rounded-full border px-3 py-1 text-xs font-medium transition-all ${selectedZones.includes(zone.id)
                                    ? "border-primary bg-primary text-primary-foreground"
                                    : "border-border bg-background text-muted-foreground hover:border-primary/30"
                                    }`}
                            >
                                {zone.id} — {zone.name}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Priority */}
                <div>
                    <label className="text-sm font-medium text-foreground mb-1.5 block">Priority Level</label>
                    <div className="flex gap-2">
                        {["normal", "urgent", "critical"].map((p) => (
                            <button
                                key={p}
                                type="button"
                                onClick={() => setPriority(p)}
                                className={`rounded-full border px-4 py-1.5 text-xs font-semibold transition-all ${priority === p
                                    ? p === "critical"
                                        ? "border-red-300 bg-red-100 text-red-700"
                                        : p === "urgent"
                                            ? "border-amber-300 bg-amber-100 text-amber-700"
                                            : "border-blue-300 bg-blue-100 text-blue-700"
                                    : "border-border bg-background text-muted-foreground hover:border-primary/30"
                                    }`}
                            >
                                {p.charAt(0).toUpperCase() + p.slice(1)}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Submit */}
                <button
                    type="submit"
                    disabled={sending || !message.trim()}
                    className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-2xl bg-red-600 text-white font-bold text-sm hover:bg-red-700 disabled:opacity-50 transition-all active:scale-[0.98]"
                >
                    {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Volume2 className="h-4 w-4" />}
                    Broadcast Now — Override All Speakers
                </button>
            </form>

            {/* Recent History */}
            {broadcastHistory.length > 0 && (
                <div className="p-6 pt-0">
                    <h4 className="text-sm font-semibold text-muted-foreground mb-3">Recent Broadcasts</h4>
                    <div className="space-y-2">
                        {broadcastHistory.slice(0, 5).map((b) => (
                            <div key={b._id} className="flex items-center gap-3 p-3 rounded-xl bg-secondary/50 text-sm">
                                <VolumeX className="h-4 w-4 text-muted-foreground shrink-0" />
                                <div className="flex-1 min-w-0">
                                    <p className="font-medium truncate">{b.message}</p>
                                    <p className="text-xs text-muted-foreground">{b.type} • {new Date(b.timestamp).toLocaleTimeString()}</p>
                                </div>
                                <Badge variant={b.status === "active" ? "danger" : "secondary"}>
                                    {b.status}
                                </Badge>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
