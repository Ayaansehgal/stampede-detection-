import { useState } from "react";
import { cn } from "@/lib/utils";
import Sidebar from "./Sidebar";
import Topbar from "./Topbar";

interface DashboardLayoutProps {
  title: string;
  children: React.ReactNode;
}

export default function DashboardLayout({ title, children }: DashboardLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="min-h-screen bg-background">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed(!collapsed)}
      />
      <div className={cn("transition-all duration-300", collapsed ? "md:ml-16" : "md:ml-64")}>
        <Topbar title={title} onMenuToggle={() => setSidebarOpen(true)} />
        <main className="p-4 md:p-6 space-y-8">
          {children}
        </main>
      </div>
    </div>
  );
}
