import React from 'react';
import { useSocket, Transaction } from '@/hooks/useSocket';
import { cn, formatCurrency, formatRiskScore } from '@/lib/utils';
import { AlertCircle, CheckCircle, ShieldAlert } from 'lucide-react';

export function LiveStreamTable() {
  const { transactions } = useSocket();

  return (
    <div className="overflow-hidden rounded-lg border border-slate-800 bg-slate-950">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm text-slate-300">
          <thead className="border-b border-slate-800 bg-slate-900/50 text-xs font-medium uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-4 py-3">ID</th>
              <th className="px-4 py-3">User</th>
              <th className="px-4 py-3 text-right">Amount</th>
              <th className="px-4 py-3">Merchant</th>
              <th className="px-4 py-3 text-center">Risk Score</th>
              <th className="px-4 py-3">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {transactions.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-slate-500 italic">
                  Waiting for transaction stream...
                </td>
              </tr>
            ) : (
              transactions.map((txn) => (
                <tr
                  key={txn.transaction_id}
                  className={cn(
                    "transition-colors hover:bg-slate-900/50",
                    txn.decision === 'BLOCK' && "bg-red-500/5 animate-pulse",
                    txn.decision === 'REVIEW' && "bg-amber-500/5"
                  )}
                >
                  <td className="whitespace-nowrap px-4 py-3 font-mono text-xs">
                    {txn.transaction_id}
                  </td>
                  <td className="px-4 py-3">{txn.user_id}</td>
                  <td className="px-4 py-3 text-right font-medium">
                    {formatCurrency(txn.amount)}
                  </td>
                  <td className="px-4 py-3 truncate max-w-[120px]">
                    {txn.merchant}
                  </td>
                  <td className="px-4 py-3 text-center font-mono">
                    <span className={cn(
                      "rounded px-1.5 py-0.5",
                      txn.risk_score > 0.7 ? "text-red-400 bg-red-400/10" :
                      txn.risk_score > 0.3 ? "text-amber-400 bg-amber-400/10" :
                      "text-emerald-400 bg-emerald-400/10"
                    )}>
                      {formatRiskScore(txn.risk_score)}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge decision={txn.decision} />
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StatusBadge({ decision }: { decision: Transaction['decision'] }) {
  switch (decision) {
    case 'ALLOW':
      return (
        <span className="inline-flex items-center gap-1 text-emerald-500">
          <CheckCircle size={14} />
          <span>Safe</span>
        </span>
      );
    case 'REVIEW':
      return (
        <span className="inline-flex items-center gap-1 text-amber-500">
          <AlertCircle size={14} />
          <span>Review</span>
        </span>
      );
    case 'BLOCK':
      return (
        <span className="inline-flex items-center gap-1 text-red-500 font-bold">
          <ShieldAlert size={14} />
          <span>Blocked</span>
        </span>
      );
  }
}
