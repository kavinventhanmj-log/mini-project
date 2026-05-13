'use client';

import { LiveStreamTable } from "@/components/LiveStreamTable";
import { useReport, useRunPipeline } from "@/hooks/useApi";
import { Play, Activity, Target, ShieldAlert, CheckCircle2, Loader2 } from "lucide-react";
import { formatRiskScore } from "@/lib/utils";
import { usePipelineStore } from "@/store/usePipelineStore";

export default function Dashboard() {
  const { data: report } = useReport();
  const { mutate: runPipeline, isPending } = useRunPipeline();
  const isRunning = usePipelineStore((state) => state.isRunning);

  const metrics = [
    { 
      label: "Precision", 
      value: report ? formatRiskScore(report.metrics.precision) : "0%", 
      icon: <Target className="text-emerald-400" />,
      sub: "False Positive Control"
    },
    { 
      label: "Recall", 
      value: report ? formatRiskScore(report.metrics.recall) : "0%", 
      icon: <Activity className="text-blue-400" />,
      sub: "Fraud Capture Rate"
    },
    { 
      label: "Total Flagged", 
      value: report ? (report.summary.blocks + report.summary.reviews).toLocaleString() : "0", 
      icon: <ShieldAlert className="text-amber-400" />,
      sub: "Block + Review"
    },
    { 
      label: "Legit Saved", 
      value: report ? report.fp_analysis.correctly_allowed.toLocaleString() : "0", 
      icon: <CheckCircle2 className="text-emerald-500" />,
      sub: "Auto-Allowed FPs"
    },
  ];

  return (
    <div className="p-8">
      <header className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-3xl font-black text-slate-100 tracking-tight">LIVE MONITOR</h2>
          <p className="text-slate-400">Real-time transaction analysis & model inference</p>
        </div>
        
        <button
          onClick={() => {
            console.log('Button clicked, isRunning:', isRunning);
            runPipeline(1000);
          }}
          disabled={isPending || isRunning}
          className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white px-6 py-3 rounded-xl font-bold transition-all shadow-lg shadow-emerald-900/20 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isPending ? (
            <Loader2 className="animate-spin" size={20} />
          ) : isRunning ? (
            <Activity className="animate-pulse text-white" size={20} />
          ) : (
            <Play size={20} fill="currentColor" />
          )}
          {isPending ? "Starting..." : isRunning ? "Pipeline Active" : "Start Pipeline"}
        </button>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {metrics.map((m, i) => (
          <div key={i} className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl">
            <div className="flex items-center justify-between mb-4">
              <span className="text-slate-500 text-xs font-bold uppercase tracking-wider">{m.label}</span>
              {m.icon}
            </div>
            <div className="text-3xl font-black text-slate-100 mb-1">{m.value}</div>
            <div className="text-slate-500 text-xs">{m.sub}</div>
          </div>
        ))}
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold text-slate-200">Transaction Stream</h3>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs text-slate-500 font-medium uppercase tracking-tighter">Live Connection</span>
          </div>
        </div>
        <LiveStreamTable />
      </div>
    </div>
  );
}
