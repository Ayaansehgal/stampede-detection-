import { Bell, Menu } from "lucide-react";
import { Badge, StatusDot } from "./common";

interface TopbarProps {
  title: string;
  onMenuToggle: () => void;
}

export default function Topbar({ title, onMenuToggle }: TopbarProps) {
  return (
    <header className="sticky top-0 z-30 h-16 flex items-center justify-between px-4 md:px-6 bg-card/95 backdrop-blur-md border-b border-border">
      <div className="flex items-center gap-3">
        <button onClick={onMenuToggle} className="md:hidden text-muted-foreground hover:text-foreground">
          <Menu className="h-5 w-5" />
        </button>
        <h2 className="text-lg font-semibold">{title}</h2>
        <Badge variant="info">Kumbh Mela</Badge>
      </div>
      <div className="flex items-center gap-4">
        <StatusDot connected={true} />
        <button className="relative text-muted-foreground hover:text-foreground">
          <Bell className="h-5 w-5" />
          <span className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-destructive text-destructive-foreground text-[10px] font-bold flex items-center justify-center">3</span>
        </button>
      </div>
    </header>
  );
}
