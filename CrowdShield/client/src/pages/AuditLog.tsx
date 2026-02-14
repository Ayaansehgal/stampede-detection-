import DashboardLayout from "@/components/safewave/DashboardLayout";
import { HeroBanner, StatsCard, Badge } from "@/components/safewave/common";
import { mockEmergencyLogs, mockComplianceData } from "@/data/mockData";
import { FileText, Hash, Clock, CheckCircle } from "lucide-react";

export default function AuditLog() {
  return (
    <DashboardLayout title="Audit Logs">
      <HeroBanner
        label="Blockchain Verified"
        title="Audit Logs"
        description="Immutable, tamper-proof record of all safety events and actions, secured by blockchain."
        gradient="from-teal-600 via-cyan-600 to-blue-600"
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <StatsCard title="Total Logs" value={mockEmergencyLogs.length} icon={<FileText className="h-5 w-5" />} />
        <StatsCard title="Blockchain Blocks" value={mockComplianceData.blockchainHashes.length} icon={<Hash className="h-5 w-5" />} />
        <StatsCard title="Resolved" value={mockEmergencyLogs.filter(l => l.resolved).length} icon={<CheckCircle className="h-5 w-5" />} />
      </div>

      {/* Table */}
      <div className="rounded-2xl border border-border bg-card shadow-sm overflow-hidden">
        <div className="p-6 flex flex-col space-y-1.5">
          <h3 className="text-lg font-semibold">Event Audit Trail</h3>
          <p className="text-sm text-muted-foreground">All actions are cryptographically signed and stored on-chain</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-t border-border bg-secondary/50">
                <th className="text-left px-6 py-3 font-medium text-muted-foreground">Type</th>
                <th className="text-left px-6 py-3 font-medium text-muted-foreground">Zone</th>
                <th className="text-left px-6 py-3 font-medium text-muted-foreground">Severity</th>
                <th className="text-left px-6 py-3 font-medium text-muted-foreground">Action</th>
                <th className="text-left px-6 py-3 font-medium text-muted-foreground">SHA-256 Hash</th>
                <th className="text-left px-6 py-3 font-medium text-muted-foreground">TX Hash</th>
                <th className="text-left px-6 py-3 font-medium text-muted-foreground">Time</th>
                <th className="text-left px-6 py-3 font-medium text-muted-foreground">Status</th>
              </tr>
            </thead>
            <tbody>
              {mockEmergencyLogs.map((log) => (
                <tr key={log._id} className="border-t border-border hover:bg-secondary/30 transition-colors">
                  <td className="px-6 py-4 font-medium">{log.type.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}</td>
                  <td className="px-6 py-4">{log.zoneId}</td>
                  <td className="px-6 py-4">
                    <Badge variant={log.severity === "critical" ? "danger" : log.severity === "high" ? "warning" : "info"}>{log.severity}</Badge>
                  </td>
                  <td className="px-6 py-4 max-w-xs truncate">{log.actionTaken}</td>
                  <td className="px-6 py-4 font-mono text-xs text-muted-foreground max-w-[120px] truncate">{log.hash}</td>
                  <td className="px-6 py-4 font-mono text-xs text-muted-foreground max-w-[120px] truncate">{log.txHash}</td>
                  <td className="px-6 py-4 text-muted-foreground whitespace-nowrap">
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {new Date(log.timestamp).toLocaleString()}
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    {log.resolved ? <Badge variant="success">Resolved</Badge> : <Badge variant="danger">Active</Badge>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </DashboardLayout>
  );
}
