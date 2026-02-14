import DashboardLayout from "@/components/safewave/DashboardLayout";
import { HeroBanner, StatsCard, ProgressBar, Badge } from "@/components/safewave/common";
import { mockComplianceData } from "@/data/mockData";
import { ShieldCheck, Activity, AlertTriangle, CheckCircle, XCircle, AlertCircleIcon, Hash } from "lucide-react";

const statusIcon = {
  pass: <CheckCircle className="h-4 w-4 text-emerald-600" />,
  warning: <AlertCircleIcon className="h-4 w-4 text-amber-600" />,
  fail: <XCircle className="h-4 w-4 text-red-600" />,
};
const statusBadge = { pass: "success", warning: "warning", fail: "danger" };

export default function Compliance() {
  return (
    <DashboardLayout title="Compliance">
      <HeroBanner
        label="Safety Report"
        title="Compliance & Safety"
        description="Automated safety compliance scoring against international crowd management standards."
        gradient="from-emerald-600 via-green-600 to-teal-600"
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <StatsCard title="Safety Score" value={`${mockComplianceData.safetyScore}%`} icon={<ShieldCheck className="h-5 w-5" />} />
        <StatsCard title="Total Readings" value={mockComplianceData.totalReadings.toLocaleString()} icon={<Activity className="h-5 w-5" />} />
        <StatsCard title="Emergencies" value={mockComplianceData.totalEmergencies} icon={<AlertTriangle className="h-5 w-5" />} />
      </div>

      {/* Safety Gauge */}
      <div className="rounded-2xl border border-border bg-card shadow-sm p-6">
        <h3 className="text-lg font-semibold mb-4">Overall Safety Score</h3>
        <div className="flex items-center justify-center">
          <svg width="200" height="200" viewBox="0 0 200 200">
            <circle cx="100" cy="100" r="85" fill="none" stroke="hsl(0 0% 89.8%)" strokeWidth="14" />
            <circle
              cx="100" cy="100" r="85" fill="none"
              stroke={mockComplianceData.safetyScore >= 80 ? "#22c55e" : mockComplianceData.safetyScore >= 50 ? "#eab308" : "#ef4444"}
              strokeWidth="14"
              strokeDasharray={`${(mockComplianceData.safetyScore / 100) * 534} 534`}
              strokeLinecap="round"
              transform="rotate(-90 100 100)"
              className="transition-all duration-1000"
            />
            <text x="100" y="95" textAnchor="middle" className="text-4xl font-bold" fill="currentColor">{mockComplianceData.safetyScore}%</text>
            <text x="100" y="120" textAnchor="middle" className="text-sm" fill="hsl(0 0% 45.1%)">Compliance Score</text>
          </svg>
        </div>
      </div>

      {/* Findings */}
      <div className="rounded-2xl border border-border bg-card shadow-sm">
        <div className="p-6 flex flex-col space-y-1.5">
          <h3 className="text-lg font-semibold">Safety Findings</h3>
          <p className="text-sm text-muted-foreground">Automated compliance check results</p>
        </div>
        <div className="p-6 pt-0 space-y-4">
          {mockComplianceData.findings.map((f, i) => (
            <div key={i} className="flex items-center gap-4 p-4 rounded-2xl bg-secondary/50">
              {statusIcon[f.status]}
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm">{f.category}</span>
                    <Badge variant={statusBadge[f.status]}>{f.status}</Badge>
                  </div>
                  <span className="text-sm font-bold">{f.score}%</span>
                </div>
                <p className="text-xs text-muted-foreground mb-2">{f.message}</p>
                <ProgressBar
                  value={f.score}
                  colorClass={f.score >= 80 ? "bg-emerald-500" : f.score >= 50 ? "bg-amber-500" : "bg-red-500"}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Blockchain Verification */}
      <div className="rounded-2xl border border-border bg-card shadow-sm">
        <div className="p-6 flex flex-col space-y-1.5">
          <h3 className="text-lg font-semibold">Blockchain Verification</h3>
          <p className="text-sm text-muted-foreground">Recent compliance data blocks</p>
        </div>
        <div className="p-6 pt-0 space-y-3">
          {mockComplianceData.blockchainHashes.map((b, i) => (
            <div key={i} className="flex items-center gap-3 p-3 rounded-2xl bg-secondary/50">
              <div className="h-8 w-8 rounded-2xl bg-secondary flex items-center justify-center">
                <Hash className="h-4 w-4 text-muted-foreground" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium">Block #{b.block}</p>
                <p className="text-xs font-mono text-muted-foreground truncate">{b.hash}</p>
              </div>
              <span className="text-xs text-muted-foreground whitespace-nowrap">{new Date(b.timestamp).toLocaleTimeString()}</span>
            </div>
          ))}
        </div>
      </div>
    </DashboardLayout>
  );
}
