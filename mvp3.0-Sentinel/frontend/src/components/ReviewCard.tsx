import React from 'react';
import { formatCurrency, formatRiskScore, cn } from '@/lib/utils';
import { ShapBarChart } from './charts/ShapBarChart';
import { Check, X, SkipForward, MapPin, Monitor, Clock, Landmark, User, ShieldAlert } from 'lucide-react';
import { useSubmitReview } from '@/hooks/useApi';

interface ReviewCardProps {
  transaction: any;
}

export function ReviewCard({ transaction }: ReviewCardProps) {
  const { mutate: submit, isPending } = useSubmitReview();

  const handleDecision = (decision: 'ALLOW' | 'BLOCK' | 'SKIP') => {
    submit({ txnId: transaction.transaction_id, decision });
  };

  const isHighRisk = transaction.risk_score > 0.7;

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/40 backdrop-blur-sm overflow-hidden shadow-2xl">
      {/* Header Banner */}
      <div className={cn(
        "px-8 py-6 flex items-center justify-between",
        isHighRisk ? "bg-red-500/10 border-b border-red-500/20" : "bg-amber-500/10 border-b border-amber-500/20"
      )}>
        <div className="flex items-center gap-4">
          <div className={cn(
            "p-3 rounded-xl",
            isHighRisk ? "bg-red-500/20 text-red-400" : "bg-amber-500/20 text-amber-400"
          )}>
            <ShieldAlert size={28} />
          </div>
          <div>
            <h3 className="text-2xl font-black text-slate-100 tracking-tight">{transaction.transaction_id}</h3>
            <div className="flex items-center gap-2 text-slate-400 text-xs font-medium uppercase tracking-widest mt-1">
              <User size={12} />
              <span>User: {transaction.user_id}</span>
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="text-3xl font-black text-slate-100">{formatCurrency(transaction.amount)}</div>
          <div className={cn(
            "text-sm font-bold mt-1",
            isHighRisk ? "text-red-400" : "text-amber-400"
          )}>
            Risk Score: {formatRiskScore(transaction.risk_score)}
          </div>
        </div>
      </div>

      <div className="p-8">
        {/* Context Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
          <div className="space-y-1">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-tighter">Merchant</span>
            <div className="flex items-center gap-2 text-slate-200">
              <Landmark size={14} className="text-slate-600" />
              <span className="text-sm font-medium truncate">{transaction.merchant}</span>
            </div>
          </div>
          <div className="space-y-1">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-tighter">Location</span>
            <div className="flex items-center gap-2 text-slate-200">
              <MapPin size={14} className="text-slate-600" />
              <span className="text-sm font-medium truncate">{transaction.location}</span>
            </div>
          </div>
          <div className="space-y-1">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-tighter">Device</span>
            <div className="flex items-center gap-2 text-slate-200">
              <Monitor size={14} className="text-slate-600" />
              <span className="text-sm font-medium truncate">{transaction.device}</span>
            </div>
          </div>
          <div className="space-y-1">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-tighter">Timestamp</span>
            <div className="flex items-center gap-2 text-slate-200">
              <Clock size={14} className="text-slate-600" />
              <span className="text-sm font-medium truncate">{transaction.timestamp}</span>
            </div>
          </div>
        </div>

        {/* Analytics Section */}
        <div className="space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <h4 className="text-xs font-black uppercase tracking-widest text-emerald-500">ML ANALYTICS & EXPLANATION</h4>
            <span className="text-[10px] text-slate-500 font-mono italic">SHAP Value Impact</span>
          </div>
          <div className="bg-slate-950/50 rounded-xl p-6 border border-slate-800/50">
            <ShapBarChart data={transaction.shap} />
          </div>
          <p className="text-[10px] text-slate-500 leading-relaxed italic px-2">
            Positive values indicate features that increased the fraud risk score. Negative values indicate features that decreased it.
          </p>
        </div>

        {/* Action Bar */}
        <div className="mt-10 flex gap-4">
          <button
            onClick={() => handleDecision('ALLOW')}
            disabled={isPending}
            className="flex-1 flex items-center justify-center gap-3 rounded-xl bg-emerald-600 px-6 py-4 font-black text-white transition-all hover:bg-emerald-500 hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-emerald-900/20 disabled:opacity-50 disabled:hover:scale-100"
          >
            <Check size={20} />
            ALLOW TRANSACTION
          </button>
          <button
            onClick={() => handleDecision('BLOCK')}
            disabled={isPending}
            className="flex-1 flex items-center justify-center gap-3 rounded-xl bg-red-600 px-6 py-4 font-black text-white transition-all hover:bg-red-500 hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-red-900/20 disabled:opacity-50 disabled:hover:scale-100"
          >
            <X size={20} />
            BLOCK TRANSACTION
          </button>
          <button
            onClick={() => handleDecision('SKIP')}
            disabled={isPending}
            className="flex items-center justify-center gap-3 rounded-xl bg-slate-800 px-6 py-4 font-black text-slate-300 transition-all hover:bg-slate-700 active:scale-[0.98] disabled:opacity-50"
          >
            <SkipForward size={20} />
            SKIP
          </button>
        </div>
      </div>
    </div>
  );
}
