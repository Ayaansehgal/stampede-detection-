import { useState } from "react";
import DashboardLayout from "@/components/safewave/DashboardLayout";
import { HeroBanner, StatsCard, ProgressBar, RiskBadge } from "@/components/safewave/common";
import { mockEvent, mockZoneReadings } from "@/data/mockData";
import { Map, Users, Gauge, ShieldCheck, Lightbulb, Play } from "lucide-react";

const totalCap = mockEvent.zones.reduce((s, z) => s + z.capacity, 0);
const totalPeople = mockZoneReadings.reduce((s, z) => s + z.personCount, 0);

export default function Planning() {
  const [simulating, setSimulating] = useState(false);
  const [simResult, setSimResult] = useState<string | null>(null);

  const runSimulation = () => {
    setSimulating(true);
    setSimResult(null);
    setTimeout(() => {
      setSimulating(false);
      setSimResult("Simulation complete: Zone A1 will exceed critical threshold in ~22 minutes at current inflow rate. Recommended action: Activate overflow routing to Gate A2 and Gate C2.");
    }, 2000);
  };

  return (
    <DashboardLayout title="Planning & Zones">
      <HeroBanner
        label="Zone Management"
        title="Planning & Zones"
        description="Configure zone capacities, run simulations, and optimize crowd flow for maximum safety."
        gradient="from-violet-600 via-purple-600 to-indigo-600"
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatsCard title="Total Zones" value={mockEvent.zones.length} icon={<Map className="h-5 w-5" />} />
        <StatsCard title="Total Capacity" value={totalCap.toLocaleString()} icon={<Users className="h-5 w-5" />} />
        <StatsCard title="Current Load" value={`${Math.round((totalPeople / totalCap) * 100)}%`} icon={<Gauge className="h-5 w-5" />} />
        <StatsCard title="Safety Score" value={`${mockEvent.safetyScore}%`} icon={<ShieldCheck className="h-5 w-5" />} />
      </div>

      {/* Zone Capacity Grid */}
      <div>
        <h2 className="text-2xl font-semibold mb-4">Zone Capacities</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {mockEvent.zones.map((zone) => {
            const reading = mockZoneReadings.find((r) => r.zoneId === zone.id);
            const pct = reading ? Math.round((reading.personCount / zone.capacity) * 100) : 0;
            const color = pct >= 80 ? "bg-risk-critical" : pct >= 60 ? "bg-risk-medium" : "bg-risk-low";
            return (
              <div key={zone.id} className="rounded-2xl border border-border bg-card shadow-sm p-5 hover:shadow-md hover:border-primary/30 transition-all duration-200">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <p className="font-semibold">{zone.name}</p>
                    <p className="text-xs text-muted-foreground">Zone {zone.id}</p>
                  </div>
                  {reading && <RiskBadge level={reading.riskLevel} />}
                </div>
                <div className="mb-2">
                  <div className="flex justify-between text-sm mb-1">
                    <span>{reading?.personCount.toLocaleString() || 0}</span>
                    <span className="text-muted-foreground">{zone.capacity.toLocaleString()}</span>
                  </div>
                  <ProgressBar value={pct} colorClass={color} />
                </div>
                <p className="text-xs text-muted-foreground">{pct}% occupied</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Safety Gauge */}
      <div className="rounded-2xl border border-border bg-card shadow-sm p-6">
        <h3 className="text-lg font-semibold mb-4">Overall Safety Score</h3>
        <div className="flex items-center justify-center">
          <svg width="180" height="180" viewBox="0 0 180 180">
            <circle cx="90" cy="90" r="75" fill="none" stroke="hsl(0 0% 89.8%)" strokeWidth="12" />
            <circle
              cx="90" cy="90" r="75" fill="none"
              stroke={mockEvent.safetyScore >= 80 ? "#22c55e" : mockEvent.safetyScore >= 50 ? "#eab308" : "#ef4444"}
              strokeWidth="12"
              strokeDasharray={`${(mockEvent.safetyScore / 100) * 471} 471`}
              strokeLinecap="round"
              transform="rotate(-90 90 90)"
              className="transition-all duration-1000"
            />
            <text x="90" y="85" textAnchor="middle" className="text-3xl font-bold fill-foreground">{mockEvent.safetyScore}%</text>
            <text x="90" y="108" textAnchor="middle" className="text-sm fill-muted-foreground">Safety</text>
          </svg>
        </div>
      </div>

      {/* Simulation */}
      <div className="rounded-2xl border border-border bg-card shadow-sm p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold">Crowd Simulation</h3>
            <p className="text-sm text-muted-foreground">Run AI-powered crowd flow simulation</p>
          </div>
          <button
            onClick={runSimulation}
            disabled={simulating}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-2xl bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            <Play className="h-4 w-4" />
            {simulating ? "Running..." : "Run Simulation"}
          </button>
        </div>
        {simResult && (
          <div className="flex items-start gap-3 p-4 rounded-2xl bg-amber-50 border border-amber-200">
            <Lightbulb className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
            <p className="text-sm text-amber-800">{simResult}</p>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
