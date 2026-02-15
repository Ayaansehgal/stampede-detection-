import { useState } from "react";
import { useNavigate } from "react-router-dom";
import DashboardLayout from "@/components/safewave/DashboardLayout";
import { HeroBanner } from "@/components/safewave/common";

const eventTypes = ["temple", "stadium", "concert", "festival", "conference", "other"];

export default function CreateEvent() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    name: "",
    type: "temple",
    expectedCrowd: "",
    venueArea: "",
    gates: "",
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // In production, call API
    navigate("/dashboard");
  };

  return (
    <DashboardLayout title="Create Event">
      <HeroBanner
        label="New Event"
        title="Create a New Event"
        description="Configure your event parameters to activate AI-powered crowd safety monitoring."
        gradient="from-emerald-600 via-teal-600 to-cyan-600"
      />

      <div className="max-w-2xl mx-auto">
        <div className="rounded-2xl border border-border bg-card shadow-sm">
          <div className="p-6 flex flex-col space-y-1.5">
            <h3 className="text-lg font-semibold">Event Details</h3>
            <p className="text-sm text-muted-foreground">Fill in the information below to set up crowd monitoring</p>
          </div>
          <form onSubmit={handleSubmit} className="p-6 pt-0 space-y-5">
            <div>
              <label className="text-sm font-medium mb-1.5 block">Event Name</label>
              <input
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                placeholder="e.g. Kumbh Mela Safety Zone"
                required
                className="w-full rounded-2xl border border-input bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-1.5 block">Event Type</label>
              <select
                value={form.type}
                onChange={(e) => setForm({ ...form, type: e.target.value })}
                className="w-full rounded-2xl border border-input bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              >
                {eventTypes.map((t) => (
                  <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>
                ))}
              </select>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label className="text-sm font-medium mb-1.5 block">Expected Crowd</label>
                <input
                  type="number"
                  value={form.expectedCrowd}
                  onChange={(e) => setForm({ ...form, expectedCrowd: e.target.value })}
                  placeholder="15000"
                  required
                  className="w-full rounded-2xl border border-input bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </div>
              <div>
                <label className="text-sm font-medium mb-1.5 block">Venue Area (m²)</label>
                <input
                  type="number"
                  value={form.venueArea}
                  onChange={(e) => setForm({ ...form, venueArea: e.target.value })}
                  placeholder="5000"
                  required
                  className="w-full rounded-2xl border border-input bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </div>
              <div>
                <label className="text-sm font-medium mb-1.5 block">Number of Gates</label>
                <input
                  type="number"
                  value={form.gates}
                  onChange={(e) => setForm({ ...form, gates: e.target.value })}
                  placeholder="6"
                  required
                  className="w-full rounded-2xl border border-input bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </div>
            </div>
            <div className="flex justify-end pt-2">
              <button
                type="submit"
                className="px-6 py-2.5 rounded-2xl bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-colors"
              >
                Create Event & Start Monitoring
              </button>
            </div>
          </form>
        </div>
      </div>
    </DashboardLayout>
  );
}
