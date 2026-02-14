import DashboardLayout from "@/components/safewave/DashboardLayout";
import { HeroBanner, StatsCard, RiskBadge, ProgressBar } from "@/components/safewave/common";
import OccupancyChart from "@/components/safewave/OccupancyChart";
import RiskTrendChart from "@/components/safewave/RiskTrendChart";
import { mockEvent, mockZoneReadings } from "@/data/mockData";
import { Users, Activity, AlertTriangle, ShieldCheck, Lightbulb } from "lucide-react";

const totalPeople = mockZoneReadings.reduce((s, z) => s + z.personCount, 0);
const avgInstability = (mockZoneReadings.reduce((s, z) => s + z.instabilityScore, 0) / mockZoneReadings.length).toFixed(2);
const criticalZones = mockZoneReadings.filter((z) => z.riskLevel === "critical" || z.riskLevel === "high").length;

export default function Dashboard() {
  return (
    <DashboardLayout title="Dashboard">
      <HeroBanner
        label="Live Monitoring"
        title="Event Dashboard"
        description={`Real-time overview of ${mockEvent.name} — ${mockEvent.zones.length} active zones, ${mockEvent.gates} gates.`}
        gradient="from-indigo-600 via-blue-600 to-cyan-600"
      />

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatsCard title="Total Crowd" value={totalPeople.toLocaleString()} subtitle={`of ${mockEvent.expectedCrowd.toLocaleString()}`} icon={<Users className="h-5 w-5" />} trend={{ value: "+12%", positive: false }} />
        <StatsCard title="Avg Risk Score" value={avgInstability} subtitle="instability index" icon={<Activity className="h-5 w-5" />} trend={{ value: "0.03", positive: false }} />
        <StatsCard title="Critical Zones" value={criticalZones} subtitle={`of ${mockZoneReadings.length} total`} icon={<AlertTriangle className="h-5 w-5" />} />
        <StatsCard title="Safety Score" value={`${mockEvent.safetyScore}%`} subtitle="compliance" icon={<ShieldCheck className="h-5 w-5" />} trend={{ value: "+2%", positive: true }} />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-2xl border border-border bg-card shadow-sm">
          <div className="p-6 flex flex-col space-y-1.5">
            <h3 className="text-lg font-semibold">Zone Occupancy</h3>
            <p className="text-sm text-muted-foreground">Current person count by zone with risk coloring</p>
          </div>
          <div className="p-6 pt-0">
            <OccupancyChart />
          </div>
        </div>
        <div className="rounded-2xl border border-border bg-card shadow-sm">
          <div className="p-6 flex flex-col space-y-1.5">
            <h3 className="text-lg font-semibold">Risk Trend</h3>
            <p className="text-sm text-muted-foreground">Instability score & crowd count over time</p>
          </div>
          <div className="p-6 pt-0">
            <RiskTrendChart />
          </div>
        </div>
      </div>

      {/* AI Recommendation */}
      <div className="rounded-2xl border border-border bg-card shadow-sm">
        <div className="p-6 flex items-start gap-4">
          <div className="h-10 w-10 rounded-2xl bg-amber-100 flex items-center justify-center shrink-0">
            <Lightbulb className="h-5 w-5 text-amber-600" />
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-1">AI Recommendation</h3>
            <p className="text-sm text-muted-foreground">
              Zone A1 (Main Entry) is approaching critical density at <strong>84% capacity</strong>. Consider diverting incoming crowds through Gate 3 (North Gate) which is at 60% capacity. Historical data suggests peak influx in the next 30 minutes.
            </p>
          </div>
        </div>
      </div>

      {/* Zone Grid */}
      <div>
        <h2 className="text-2xl font-semibold mb-4">Zone Status</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {mockZoneReadings.map((zone) => {
            const capacity = mockEvent.zones.find((z) => z.id === zone.zoneId)?.capacity || 1;
            const pct = Math.round((zone.personCount / capacity) * 100);
            const color = zone.riskLevel === "critical" ? "bg-risk-critical" : zone.riskLevel === "high" ? "bg-risk-high" : zone.riskLevel === "medium" ? "bg-risk-medium" : "bg-risk-low";
            return (
              <div key={zone.zoneId} className="rounded-2xl border border-border bg-card shadow-sm p-5 hover:shadow-md hover:border-primary/30 transition-all duration-200">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <p className="font-semibold">{mockEvent.zones.find((z) => z.id === zone.zoneId)?.name || zone.zone}</p>
                    <p className="text-xs text-muted-foreground">Zone {zone.zoneId}</p>
                  </div>
                  <RiskBadge level={zone.riskLevel} />
                </div>
                <div className="flex items-end justify-between mb-2">
                  <span className="text-2xl font-bold">{zone.personCount.toLocaleString()}</span>
                  <span className="text-xs text-muted-foreground">/ {capacity.toLocaleString()}</span>
                </div>
                <ProgressBar value={pct} colorClass={color} />
                <p className="text-xs text-muted-foreground mt-2">Instability: {(zone.instabilityScore * 100).toFixed(0)}%</p>
              </div>
            );
          })}
        </div>
      </div>
    </DashboardLayout>
  );
}
