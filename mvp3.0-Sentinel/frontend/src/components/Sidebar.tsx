'use client';

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, Inbox, BarChart3, Settings, Shield } from "lucide-react";
import { usePipelineStore } from "@/store/usePipelineStore";
import { useStatus } from "@/hooks/useApi";
import { cn } from "@/lib/utils";

export function Sidebar() {
  useStatus();
  const reviewCount = usePipelineStore((state) => state.reviewCount);
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r border-slate-800 bg-slate-900/50 p-4 hidden md:flex flex-col">
      <div className="flex items-center gap-2 px-2 py-4 mb-8">
        <div className="bg-emerald-500 p-1.5 rounded-lg">
          <Shield size={24} className="text-slate-950" />
        </div>
        <h1 className="text-xl font-black tracking-tighter italic">SENTINEL</h1>
      </div>
      
      <nav className="space-y-1 flex-1">
        <NavItem 
          href="/" 
          icon={<LayoutDashboard size={20} />} 
          label="Live Monitor" 
          active={pathname === "/"}
        />
        <NavItem 
          href="/review" 
          icon={<Inbox size={20} />} 
          label="Review Queue" 
          count={reviewCount}
          active={pathname === "/review"}
        />
        <NavItem 
          href="/analytics" 
          icon={<BarChart3 size={20} />} 
          label="Analytics" 
          active={pathname === "/analytics"}
        />
      </nav>

      <div className="pt-4 border-t border-slate-800">
        <NavItem 
          href="/settings" 
          icon={<Settings size={20} />} 
          label="Audit" 
          active={pathname === "/settings"}
        />
      </div>
    </aside>
  );
}

function NavItem({ 
  href, icon, label, count, active 
}: { 
  href: string; icon: React.ReactNode; label: string; count?: number; active?: boolean 
}) {
  return (
    <Link
      href={href}
      className={cn(
        "flex items-center justify-between px-3 py-2.5 rounded-lg transition-all duration-200 group",
        active 
          ? "bg-emerald-500/10 text-emerald-400 font-bold" 
          : "text-slate-400 hover:text-slate-100 hover:bg-slate-800"
      )}
    >
      <div className="flex items-center gap-3">
        <span className={cn(
          "transition-colors",
          active ? "text-emerald-400" : "text-slate-500 group-hover:text-emerald-400"
        )}>
          {icon}
        </span>
        <span className="font-medium">{label}</span>
      </div>
      {count !== undefined && count > 0 && (
        <span className={cn(
          "text-[10px] font-bold px-2 py-0.5 rounded-full border transition-all",
          active 
            ? "bg-emerald-500 text-slate-950 border-emerald-500" 
            : "bg-emerald-500/20 text-emerald-400 border-emerald-500/30 group-hover:bg-emerald-500 group-hover:text-slate-950"
        )}>
          {count}
        </span>
      )}
    </Link>
  );
}
