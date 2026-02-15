import { useState, useEffect, useCallback } from "react";
import DashboardLayout from "@/components/safewave/DashboardLayout";
import { HeroBanner, Badge } from "@/components/safewave/common";
import GateControl from "@/components/safewave/GateControl";
import PAPanel from "@/components/safewave/PAPanel";
import { mockEvent, mockEmergencyLogs } from "@/data/mockData";
import { AlertTriangle, UserX, TrendingUp, Construction, LogOut } from "lucide-react";

const API_BASE = "http://localhost:5000/api";

const GATE_NAMES = ["Main Entry", "North Gate", "East Gate", "West Gate", "South Gate", "VIP Entry"];

const emergencyActions = [
  { id: "critical_risk", title: "Critical Risk Alert", desc: "Auto-locks 50% turnstiles & starts PA override", icon: AlertTriangle, color: "border-red-500/30 bg-red-950/30 hover:border-red-500/50 hover:bg-red-950/50", iconColor: "text-red-400 bg-red-950/60" },
  { id: "missing_person", title: "Missing Person", desc: "Broadcast missing person alert across all zones", icon: UserX, color: "border-amber-500/30 bg-amber-950/30 hover:border-amber-500/50 hover:bg-amber-950/50", iconColor: "text-amber-400 bg-amber-950/60" },
  { id: "emergency_escalation", title: "Escalation Protocol", desc: "Escalate to law enforcement & emergency services", icon: TrendingUp, color: "border-orange-500/30 bg-orange-950/30 hover:border-orange-500/50 hover:bg-orange-950/50", iconColor: "text-orange-400 bg-orange-950/60" },
  { id: "barricade_control", title: "Barricade Control", desc: "Deploy or retract crowd barricades remotely", icon: Construction, color: "border-cyan-500/30 bg-cyan-950/30 hover:border-cyan-500/50 hover:bg-cyan-950/50", iconColor: "text-cyan-400 bg-cyan-950/60" },
  { id: "evacuation", title: "Full Evacuation", desc: "Auto-locks 50% turnstiles & starts PA evacuation order", icon: LogOut, color: "border-purple-500/30 bg-purple-950/30 hover:border-purple-500/50 hover:bg-purple-950/50", iconColor: "text-purple-400 bg-purple-950/60" },
];

interface GateData {
  _id: string;
  gateNumber: number;
  gateName: string;
  turnstiles: boolean[];
}

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

export default function Emergency() {
  const [activeAlert, setActiveAlert] = useState<string | null>(null);
  const [triggerLoading, setTriggerLoading] = useState<string | null>(null);

  // Gate state — each gate has 8 turnstiles
  const [gates, setGates] = useState<GateData[]>(() =>
    Array.from({ length: mockEvent.gates || 6 }, (_, i) => ({
      _id: `gate-${i + 1}`,
      gateNumber: i + 1,
      gateName: GATE_NAMES[i] || `Gate ${i + 1}`,
      turnstiles: Array(8).fill(true),  // all unlocked
    }))
  );

  // PA state
  const [activeBroadcasts, setActiveBroadcasts] = useState<Broadcast[]>([]);
  const [broadcastHistory, setBroadcastHistory] = useState<Broadcast[]>([]);

  const eventId = mockEvent._id;

  // Fetch gates from backend
  const fetchGates = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/gates/${eventId}`);
      if (res.ok) {
        const data = await res.json();
        if (data.length > 0) setGates(data);
      }
    } catch { /* Use local state */ }
  }, [eventId]);

  // Fetch PA broadcasts
  const fetchBroadcasts = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/pa/${eventId}`);
      if (res.ok) {
        const data = await res.json();
        setActiveBroadcasts(data.filter((b: Broadcast) => b.status === "active"));
        setBroadcastHistory(data);
      }
    } catch { /* Use local state */ }
  }, [eventId]);

  useEffect(() => {
    fetchGates();
    fetchBroadcasts();
  }, [fetchGates, fetchBroadcasts]);

  // --- API Handlers ---

  const handleTriggerEmergency = async (actionId: string) => {
    setTriggerLoading(actionId);
    setActiveAlert(actionId);
    try {
      const res = await fetch(`${API_BASE}/emergency`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ eventId, type: actionId, zoneId: "all" }),
      });
      if (res.ok) {
        await fetchGates();
        await fetchBroadcasts();
      }
    } catch {
      // Simulate locally if backend is down
      if (actionId === "critical_risk" || actionId === "evacuation") {
        // Lock every other turnstile across all gates
        setGates(prev => prev.map(g => ({
          ...g,
          turnstiles: g.turnstiles.map((_, i) => i % 2 !== 0),  // lock 0,2,4,6
        })));
        setActiveBroadcasts([{
          _id: `sim-${Date.now()}`,
          type: actionId === "critical_risk" ? "stampede_alert" : "evacuation",
          message: actionId === "critical_risk"
            ? "EMERGENCY: Crowd surge detected. Turnstiles are being locked."
            : "EMERGENCY EVACUATION: Proceed to nearest exit.",
          zones: ["all"], priority: "critical", triggeredBy: "auto",
          status: "active", timestamp: new Date().toISOString(),
        }]);
      }
    } finally {
      setTriggerLoading(null);
    }
  };

  const handleToggleTurnstile = async (gateNumber: number, turnstileIndex: number) => {
    try {
      const res = await fetch(`${API_BASE}/gates/toggle-turnstile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ eventId, gateNumber, turnstileIndex }),
      });
      if (res.ok) { await fetchGates(); return; }
    } catch { /* fallback */ }
    // Toggle locally
    setGates(prev => prev.map(g =>
      g.gateNumber === gateNumber
        ? { ...g, turnstiles: g.turnstiles.map((t, i) => i === turnstileIndex ? !t : t) }
        : g
    ));
  };

  const handleAutoLockdown = async () => {
    try {
      const res = await fetch(`${API_BASE}/gates/auto-lockdown`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ eventId, reason: "Manual lockdown" }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.length > 0) setGates(data);
        return;
      }
    } catch { /* fallback */ }
    setGates(prev => prev.map(g => ({
      ...g,
      turnstiles: g.turnstiles.map((_, i) => i % 2 !== 0),
    })));
  };

  const handleOpenAll = async () => {
    try {
      const res = await fetch(`${API_BASE}/gates/open-all`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ eventId }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.length > 0) setGates(data);
        return;
      }
    } catch { /* fallback */ }
    setGates(prev => prev.map(g => ({
      ...g,
      turnstiles: g.turnstiles.map(() => true),
    })));
  };

  const handleBroadcast = async (data: { type: string; message: string; zones: string[]; priority: string }) => {
    try {
      const res = await fetch(`${API_BASE}/pa/broadcast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ eventId, ...data }),
      });
      if (res.ok) { await fetchBroadcasts(); return; }
    } catch { /* fallback */ }
    const sim: Broadcast = {
      _id: `sim-${Date.now()}`, ...data,
      priority: data.priority as Broadcast["priority"],
      triggeredBy: "manual", status: "active",
      timestamp: new Date().toISOString(),
    };
    setActiveBroadcasts(prev => [sim, ...prev]);
    setBroadcastHistory(prev => [sim, ...prev]);
  };

  const handleStopBroadcast = async (broadcastId?: string) => {
    try {
      const res = await fetch(`${API_BASE}/pa/stop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ eventId, broadcastId }),
      });
      if (res.ok) { await fetchBroadcasts(); return; }
    } catch { /* fallback */ }
    broadcastId
      ? setActiveBroadcasts(prev => prev.filter(b => b._id !== broadcastId))
      : setActiveBroadcasts([]);
  };

  return (
    <DashboardLayout title="Emergency Panel">
      <HeroBanner
        label="Emergency Response"
        title="Emergency Panel"
        description="Rapid response tools for crowd safety incidents. Every action is blockchain-logged."
        gradient="from-red-700 via-orange-700 to-amber-700"
      />

      {/* Active Alert */}
      {activeAlert && (
        <div className="rounded-2xl border-2 border-red-500/40 bg-red-950/40 p-5 animate-fade-in-up">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-3 w-3 rounded-full bg-red-500 animate-pulse" />
              <div>
                <p className="font-semibold text-red-300">Active: {emergencyActions.find(a => a.id === activeAlert)?.title}</p>
                <p className="text-sm text-red-600">
                  {activeAlert === "critical_risk" || activeAlert === "evacuation"
                    ? "50% turnstiles auto-locked • PA system broadcasting"
                    : "Emergency protocol active"}
                </p>
              </div>
            </div>
            <button
              onClick={() => setActiveAlert(null)}
              className="px-3 py-1.5 rounded-2xl bg-red-600 text-white text-sm font-semibold hover:bg-red-700"
            >
              Deactivate
            </button>
          </div>
        </div>
      )}

      {/* Action Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {emergencyActions.map((action) => (
          <button
            key={action.id}
            onClick={() => handleTriggerEmergency(action.id)}
            disabled={triggerLoading === action.id}
            className={`rounded-2xl border p-5 text-left transition-all duration-200 hover:shadow-lg active:scale-[0.98] ${action.color}`}
          >
            <div className={`h-12 w-12 rounded-2xl flex items-center justify-center mb-4 ${action.iconColor}`}>
              <action.icon className="h-5 w-5" />
            </div>
            <h3 className="font-semibold mb-1">{action.title}</h3>
            <p className="text-sm text-muted-foreground">{action.desc}</p>
            {(action.id === "critical_risk" || action.id === "evacuation") && (
              <div className="mt-2 flex gap-1">
                <Badge variant="danger">Auto Gates</Badge>
                <Badge variant="warning">Auto PA</Badge>
              </div>
            )}
          </button>
        ))}
      </div>

      {/* Rotating Gate Control */}
      <GateControl
        eventId={eventId}
        gates={gates}
        onToggleTurnstile={handleToggleTurnstile}
        onAutoLockdown={handleAutoLockdown}
        onOpenAll={handleOpenAll}
      />

      {/* PA Override */}
      <PAPanel
        eventId={eventId}
        zones={mockEvent.zones}
        activeBroadcasts={activeBroadcasts}
        broadcastHistory={broadcastHistory}
        onBroadcast={handleBroadcast}
        onStop={handleStopBroadcast}
      />

      {/* Emergency Logs */}
      <div className="rounded-2xl border border-border bg-card shadow-sm">
        <div className="p-6 flex flex-col space-y-1.5">
          <h3 className="text-lg font-semibold">Recent Emergency Logs</h3>
          <p className="text-sm text-muted-foreground">Blockchain-verified incident records</p>
        </div>
        <div className="p-6 pt-0">
          <div className="space-y-3">
            {mockEmergencyLogs.map((log) => (
              <div key={log._id} className="flex items-start gap-4 p-4 rounded-2xl bg-secondary/50">
                <div className="flex flex-col items-center gap-1 shrink-0">
                  <Badge variant={log.severity === "critical" ? "danger" : log.severity === "high" ? "warning" : "info"}>
                    {log.severity}
                  </Badge>
                  {log.resolved ? <Badge variant="success">Resolved</Badge> : <Badge variant="danger">Active</Badge>}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm">{log.type.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}</p>
                  <p className="text-xs text-muted-foreground mt-1">{log.actionTaken}</p>
                  <p className="text-xs text-muted-foreground mt-1 font-mono truncate">Zone: {log.zoneId} • {new Date(log.timestamp).toLocaleTimeString()}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
