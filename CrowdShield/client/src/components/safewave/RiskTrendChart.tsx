import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { mockTrendData } from "@/data/mockData";

export default function RiskTrendChart() {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={mockTrendData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <defs>
          <linearGradient id="colorInstability" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="colorCrowd" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(0 0% 89.8%)" />
        <XAxis dataKey="time" tick={{ fontSize: 11 }} stroke="hsl(0 0% 45.1%)" interval={4} />
        <YAxis yAxisId="left" tick={{ fontSize: 11 }} stroke="hsl(0 0% 45.1%)" domain={[0, 1]} />
        <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} stroke="hsl(0 0% 45.1%)" />
        <Tooltip contentStyle={{ borderRadius: "1rem", border: "1px solid hsl(0 0% 89.8%)", background: "hsl(0 0% 100%)", fontSize: 13 }} />
        <Area yAxisId="left" type="monotone" dataKey="instability" stroke="#ef4444" fill="url(#colorInstability)" strokeWidth={2} name="Instability" />
        <Area yAxisId="right" type="monotone" dataKey="crowdCount" stroke="#6366f1" fill="url(#colorCrowd)" strokeWidth={2} name="Crowd Count" />
      </AreaChart>
    </ResponsiveContainer>
  );
}
