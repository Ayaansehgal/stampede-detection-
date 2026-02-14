import { useState } from "react";
import { useLocation, Link } from "react-router-dom";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  Home,
  Map,
  AlertTriangle,
  FileText,
  ShieldCheck,
  Settings,
  Plus,
  ChevronLeft,
  X,
  Shield,
} from "lucide-react";

const navSections = [
  {
    label: "Overview",
    items: [
      { title: "Home", path: "/", icon: Home },
      { title: "Dashboard", path: "/dashboard", icon: LayoutDashboard },
    ],
  },
  {
    label: "Operations",
    items: [
      { title: "Planning & Zones", path: "/planning", icon: Map },
      { title: "Emergency Panel", path: "/emergency", icon: AlertTriangle },
    ],
  },
  {
    label: "Records",
    items: [
      { title: "Audit Logs", path: "/audit", icon: FileText },
      { title: "Compliance", path: "/compliance", icon: ShieldCheck },
    ],
  },
];

interface SidebarProps {
  open: boolean;
  onClose: () => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export default function Sidebar({ open, onClose, collapsed, onToggleCollapse }: SidebarProps) {
  const location = useLocation();

  return (
    <>
      {/* Mobile overlay */}
      {open && (
        <div className="fixed inset-0 z-40 bg-black/40 md:hidden" onClick={onClose} />
      )}

      <aside
        className={cn(
          "fixed top-0 left-0 z-50 h-full bg-card border-r border-border flex flex-col transition-all duration-300",
          "md:translate-x-0",
          open ? "translate-x-0" : "-translate-x-full md:translate-x-0",
          collapsed ? "md:w-16" : "md:w-64",
          "w-64"
        )}
      >
        {/* Logo */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-border">
          <Link to="/" className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-2xl bg-primary flex items-center justify-center">
              <Shield className="h-4 w-4 text-primary-foreground" />
            </div>
            {!collapsed && <span className="font-bold text-lg">SafeWave</span>}
          </Link>
          <button onClick={onClose} className="md:hidden text-muted-foreground hover:text-foreground">
            <X className="h-5 w-5" />
          </button>
          <button onClick={onToggleCollapse} className="hidden md:block text-muted-foreground hover:text-foreground">
            <ChevronLeft className={cn("h-4 w-4 transition-transform", collapsed && "rotate-180")} />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-6">
          {navSections.map((section) => (
            <div key={section.label}>
              {!collapsed && (
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 px-2">{section.label}</p>
              )}
              <div className="space-y-1">
                {section.items.map((item) => {
                  const isActive = location.pathname === item.path;
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      onClick={onClose}
                      className={cn(
                        "flex items-center gap-3 px-3 py-2.5 rounded-2xl text-sm font-medium transition-colors",
                        isActive
                          ? "bg-primary/10 text-primary"
                          : "text-muted-foreground hover:bg-accent hover:text-foreground"
                      )}
                    >
                      <item.icon className="h-4 w-4 shrink-0" />
                      {!collapsed && <span>{item.title}</span>}
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* Bottom */}
        <div className="p-3 border-t border-border space-y-2">
          {!collapsed && (
            <div className="px-3 py-2 rounded-2xl bg-secondary text-xs">
              <p className="text-muted-foreground">Active Event</p>
              <p className="font-semibold text-foreground truncate">Kumbh Mela Safety Zone</p>
            </div>
          )}
          <Link
            to="/settings"
            className="flex items-center gap-3 px-3 py-2.5 rounded-2xl text-sm text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
          >
            <Settings className="h-4 w-4" />
            {!collapsed && <span>Settings</span>}
          </Link>
          <Link
            to="/create-event"
            className={cn(
              "flex items-center justify-center gap-2 py-2.5 rounded-2xl text-sm font-semibold transition-colors",
              "bg-primary text-primary-foreground hover:bg-primary/90",
              collapsed ? "px-2" : "px-3"
            )}
          >
            <Plus className="h-4 w-4" />
            {!collapsed && <span>New Event</span>}
          </Link>
        </div>
      </aside>
    </>
  );
}
