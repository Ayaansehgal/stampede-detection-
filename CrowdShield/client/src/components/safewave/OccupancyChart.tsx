import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { mockZoneReadings, riskColors, type RiskLevel } from "@/data/mockData";

export default function OccupancyChart() {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={mockZoneReadings} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(0 0% 89.8%)" />
        <XAxis dataKey="zone" tick={{ fontSize: 12 }} stroke="hsl(0 0% 45.1%)" />
        <YAxis tick={{ fontSize: 12 }} stroke="hsl(0 0% 45.1%)" />
        <Tooltip
          contentStyle={{
            borderRadius: "1rem",
            border: "1px solid hsl(0 0% 89.8%)",
            background: "hsl(0 0% 100%)",
            fontSize: 13,
          }}
        />
        <Bar dataKey="personCount" radius={[8, 8, 0, 0]} name="Person Count">
          {mockZoneReadings.map((entry, i) => (
            <Cell key={i} fill={riskColors[entry.riskLevel as RiskLevel]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
