import { Link } from "react-router-dom";
import { Shield, BarChart3, Zap, Lock, Users, ArrowRight, ChevronRight, Globe, Cpu } from "lucide-react";
import { motion } from "framer-motion";
import heroBg from "@/assets/hero-bg.jpg";

const features = [
  { icon: Users, title: "Real-Time Monitoring", desc: "AI-powered crowd density tracking across all zones with instant alerts." },
  { icon: BarChart3, title: "Predictive Analytics", desc: "Forecast crowd surges before they happen with ML-driven risk models." },
  { icon: Zap, title: "Emergency Response", desc: "One-click emergency protocols with automated PA and marshal dispatch." },
  { icon: Lock, title: "Blockchain Audit", desc: "Immutable audit trail for every safety event with SHA-256 verification." },
  { icon: Globe, title: "Multi-Zone Coverage", desc: "Manage unlimited zones with independent capacity and risk thresholds." },
  { icon: Cpu, title: "AI Compliance", desc: "Automated safety compliance scoring against international standards." },
];

const arch = [
  { label: "IoT Sensors", sub: "Edge Devices" },
  { label: "AI Engine", sub: "Risk Analysis" },
  { label: "Blockchain", sub: "Audit Trail" },
  { label: "Dashboard", sub: "Live Ops" },
];

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (i: number) => ({ opacity: 1, y: 0, transition: { delay: i * 0.1, duration: 0.5 } }),
};

export default function Landing() {
  return (
    <div className="min-h-screen bg-background">
      {/* Nav */}
      <nav className="sticky top-0 z-50 bg-card/90 backdrop-blur-md border-b border-border">
        <div className="max-w-7xl mx-auto px-4 md:px-8 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-2xl bg-primary flex items-center justify-center">
              <Shield className="h-4 w-4 text-primary-foreground" />
            </div>
            <span className="font-bold text-lg">CrowdShield</span>
          </Link>
          <div className="flex items-center gap-3">
            <Link to="/dashboard" className="text-sm text-muted-foreground hover:text-foreground transition-colors hidden sm:block">Dashboard</Link>
            <Link
              to="/create-event"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-2xl bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-colors"
            >
              Get Started <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0">
          <img src={heroBg} alt="" className="w-full h-full object-cover" />
          <div className="absolute inset-0 bg-gradient-to-b from-primary/90 via-primary/80 to-background" />
        </div>
        <div className="relative max-w-7xl mx-auto px-4 md:px-8 py-24 md:py-36">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
            <span className="inline-block px-3 py-1 rounded-full bg-white/10 text-primary-foreground text-sm font-medium mb-6 border border-white/20">
              AI-Powered Crowd Safety
            </span>
            <h1 className="text-4xl md:text-6xl font-bold text-primary-foreground max-w-3xl leading-tight mb-6">
              Protect Every Life in{" "}
              <span className="bg-gradient-to-r from-cyan-300 to-blue-300 bg-clip-text text-transparent">Every Crowd</span>
            </h1>
            <p className="text-lg text-primary-foreground/70 max-w-xl mb-8">
              Real-time crowd monitoring, AI risk prediction, and blockchain-verified compliance — all in one operating system.
            </p>
            <div className="flex flex-col sm:flex-row gap-3">
              <Link
                to="/create-event"
                className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded-2xl bg-white text-primary text-sm font-bold hover:bg-white/90 transition-colors"
              >
                Create Event <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                to="/dashboard"
                className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded-2xl border border-white/30 text-primary-foreground text-sm font-medium hover:bg-white/10 transition-colors"
              >
                View Demo Dashboard
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Architecture Strip */}
      <section className="border-b border-border bg-card">
        <div className="max-w-7xl mx-auto px-4 md:px-8 py-10">
          <p className="text-center text-xs uppercase tracking-widest text-muted-foreground mb-8">System Architecture</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {arch.map((item, i) => (
              <div key={i} className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-2xl bg-secondary flex items-center justify-center text-primary font-bold text-sm">
                  {String(i + 1).padStart(2, "0")}
                </div>
                <div>
                  <p className="text-sm font-semibold">{item.label}</p>
                  <p className="text-xs text-muted-foreground">{item.sub}</p>
                </div>
                {i < arch.length - 1 && <ChevronRight className="h-4 w-4 text-muted-foreground ml-auto hidden md:block" />}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="max-w-7xl mx-auto px-4 md:px-8 py-20">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-3">Built for Safety at Scale</h2>
          <p className="text-muted-foreground max-w-lg mx-auto">Every tool you need to monitor, predict, and respond to crowd safety events — backed by AI and blockchain.</p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              custom={i}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={fadeUp}
              className="rounded-2xl border border-border bg-card p-6 hover:shadow-md hover:border-primary/30 transition-all duration-200"
            >
              <div className="h-12 w-12 rounded-2xl bg-secondary flex items-center justify-center mb-4">
                <f.icon className="h-5 w-5 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">{f.title}</h3>
              <p className="text-sm text-muted-foreground">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-7xl mx-auto px-4 md:px-8 pb-20">
        <div className="rounded-3xl bg-gradient-to-r from-indigo-600 via-blue-600 to-cyan-600 p-10 md:p-16 text-center text-white relative overflow-hidden">
          <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMjAiIGN5PSIyMCIgcj0iMSIgZmlsbD0icmdiYSgyNTUsMjU1LDI1NSwwLjEpIi8+PC9zdmc+')] opacity-50" />
          <div className="relative z-10">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Ready to Safeguard Your Next Event?</h2>
            <p className="text-white/80 max-w-lg mx-auto mb-8">
              Deploy CrowdShield in minutes. AI-powered monitoring starts protecting lives from day one.
            </p>
            <Link
              to="/create-event"
              className="inline-flex items-center gap-2 px-8 py-3.5 rounded-2xl bg-white text-primary font-bold hover:bg-white/90 transition-colors"
            >
              Create Your First Event <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border bg-card">
        <div className="max-w-7xl mx-auto px-4 md:px-8 py-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold">CrowdShield</span>
          </div>
          <p className="text-xs text-muted-foreground">© 2026 CrowdShield. AI Crowd Safety & Compliance Operating System.</p>
        </div>
      </footer>
    </div>
  );
}
