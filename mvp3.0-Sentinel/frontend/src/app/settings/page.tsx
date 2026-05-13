'use client';

import { Settings as SettingsIcon, Sliders, Database, Bell, Shield, History, UserCheck, CheckCircle, XCircle, AlertCircle } from "lucide-react";
import { useReviewerDecisions } from "@/hooks/useApi";
import { formatCurrency, formatRiskScore, cn } from "@/lib/utils";

export default function SettingsPage() {
  const { data: decisionData, isLoading } = useReviewerDecisions();
  const decisions = decisionData?.decisions || [];

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8">
      <header className="mb-8">
        <h2 className="text-3xl font-black text-slate-100 tracking-tight flex items-center gap-3">
          <SettingsIcon className="text-emerald-500" size={32} />
          AUDIT
        </h2>
        <p className="text-slate-400">Audit human evaluator performance and system thresholds</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-8">
          {/* Human Evaluator History */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="p-6 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <History className="text-emerald-400" size={20} />
                <h3 className="text-xl font-bold text-slate-100">Human Evaluator Data</h3>
              </div>
              <span className="text-xs font-mono text-slate-500">{decisions.length} Decisions Logged</span>
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-slate-300">
                <thead className="bg-slate-950/50 text-[10px] font-bold uppercase tracking-widest text-slate-500 border-b border-slate-800">
                  <tr>
                    <th className="px-6 py-4">Transaction ID</th>
                    <th className="px-6 py-4 text-center">Score</th>
                    <th className="px-6 py-4 text-right">Amount</th>
                    <th className="px-6 py-4 text-center">Human Decision</th>
                    <th className="px-6 py-4 text-center">Ground Truth</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50">
                  {isLoading ? (
                    <tr>
                      <td colSpan={5} className="px-6 py-12 text-center text-slate-500">
                        <div className="flex justify-center mb-2">
                          <History className="animate-spin" size={20} />
                        </div>
                        Loading audit logs...
                      </td>
                    </tr>
                  ) : decisions.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="px-6 py-12 text-center text-slate-500 italic">
                        No human decisions recorded yet.
                      </td>
                    </tr>
                  ) : (
                    decisions.map((d: any) => (
                      <tr key={d.transaction_id} className="hover:bg-slate-800/30 transition-colors">
                        <td className="px-6 py-4 font-mono text-xs text-slate-400">{d.transaction_id}</td>
                        <td className="px-6 py-4 text-center">
                          <span className={cn(
                            "text-[10px] font-bold px-2 py-0.5 rounded",
                            d.risk_score > 0.7 ? "bg-red-500/20 text-red-400" : "bg-amber-500/20 text-amber-400"
                          )}>
                            {formatRiskScore(d.risk_score)}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-right font-bold text-slate-200">{formatCurrency(d.amount)}</td>
                        <td className="px-6 py-4">
                          <div className="flex justify-center">
                            {d.decision === 'ALLOW' ? (
                              <div className="flex items-center gap-1.5 text-emerald-400 bg-emerald-500/10 px-2 py-1 rounded-lg border border-emerald-500/20">
                                <CheckCircle size={12} />
                                <span className="text-[10px] font-bold uppercase">Allowed</span>
                              </div>
                            ) : d.decision === 'BLOCK' ? (
                              <div className="flex items-center gap-1.5 text-red-400 bg-red-500/10 px-2 py-1 rounded-lg border border-red-500/20">
                                <XCircle size={12} />
                                <span className="text-[10px] font-bold uppercase">Blocked</span>
                              </div>
                            ) : (
                              <div className="flex items-center gap-1.5 text-slate-400 bg-slate-800 px-2 py-1 rounded-lg">
                                <AlertCircle size={12} />
                                <span className="text-[10px] font-bold uppercase">Skipped</span>
                              </div>
                            )}
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex justify-center">
                            {d.is_fraud === 1 ? (
                              <span className="text-[10px] font-black text-red-500 bg-red-500/10 border border-red-500/30 px-2 py-0.5 rounded italic">FRAUD</span>
                            ) : (
                              <span className="text-[10px] font-black text-emerald-500 bg-emerald-500/10 border border-emerald-500/30 px-2 py-0.5 rounded italic">LEGIT</span>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Risk Thresholds */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
            <div className="flex items-center gap-3 mb-6">
              <Sliders className="text-blue-400" size={20} />
              <h3 className="text-xl font-bold text-slate-100">Model Thresholds</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="space-y-4">
                <div className="flex justify-between items-end">
                  <label className="text-sm font-medium text-slate-400 uppercase tracking-wider">Auto-Allow</label>
                  <span className="text-emerald-400 font-mono font-bold">0.30</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500" style={{ width: '30%' }} />
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex justify-between items-end">
                  <label className="text-sm font-medium text-slate-400 uppercase tracking-wider">Auto-Block</label>
                  <span className="text-rose-400 font-mono font-bold">0.70</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-rose-500" style={{ width: '70%', marginLeft: '70%' }} />
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-8">
          {/* Stats Summary */}
          <div className="bg-emerald-500 border border-emerald-400 rounded-2xl p-6 text-slate-950">
            <div className="flex items-center gap-3 mb-4">
              <UserCheck size={24} />
              <h3 className="text-lg font-black tracking-tight">EVALUATOR STATS</h3>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between items-center border-b border-slate-950/10 pb-2">
                <span className="text-xs font-bold uppercase opacity-70">Decisions</span>
                <span className="text-2xl font-black">{decisions.length}</span>
              </div>
              <div className="flex justify-between items-center border-b border-slate-950/10 pb-2">
                <span className="text-xs font-bold uppercase opacity-70">FP Corrections</span>
                <span className="text-2xl font-black">
                  {decisions.filter((d: any) => d.decision === 'ALLOW' && d.is_fraud === 0).length}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <Database className="text-slate-500" size={20} />
              <h4 className="font-bold text-slate-200">System Info</h4>
            </div>
            <div className="space-y-3 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-500">Pipeline Version</span>
                <span className="text-slate-300 font-mono">v3.0.1</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">ML Engine</span>
                <span className="text-slate-300 font-mono">LightGBM</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Explainability</span>
                <span className="text-slate-300 font-mono">SHAP Kernel</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
