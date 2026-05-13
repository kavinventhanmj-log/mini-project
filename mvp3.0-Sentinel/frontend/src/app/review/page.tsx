'use client';

import { useState, useEffect, useMemo } from "react";
import { useReviewQueue } from "@/hooks/useApi";
import { ReviewCard } from "@/components/ReviewCard";
import { Inbox, Loader2, ChevronRight, Search, Download } from "lucide-react";
import { formatCurrency, formatRiskScore, cn } from "@/lib/utils";
import { API_BASE_URL } from "@/lib/config";

export default function ReviewPage() {
  const { data, isLoading } = useReviewQueue();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const handleDownloadFeedback = () => {
    // Construct the download URL using the centralized config
    const downloadUrl = API_BASE_URL.replace('/api', '/api/feedback/download');
    window.open(downloadUrl, '_blank');
  };

  const pendingTransactions = useMemo(() => {
    const txns = data?.transactions?.filter((t: any) => t.review_decision === 'PENDING') || [];
    if (!searchQuery) return txns;
    return txns.filter((t: any) => 
      t.transaction_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      t.user_id?.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [data, searchQuery]);
  
  const selectedTransaction = useMemo(() => 
    pendingTransactions.find(t => t.transaction_id === selectedId) || pendingTransactions[0],
    [pendingTransactions, selectedId]
  );

  // Synchronize selectedId with pendingTransactions safely in useEffect
  useEffect(() => {
    if (pendingTransactions.length > 0) {
      if (!selectedId || !pendingTransactions.some(t => t.transaction_id === selectedId)) {
        setSelectedId(pendingTransactions[0].transaction_id);
      }
    } else if (selectedId) {
      setSelectedId(null);
    }
  }, [pendingTransactions, selectedId]);

  return (
    <div className="flex h-[calc(100vh-2rem)] overflow-hidden bg-slate-950">
      {/* Sidebar List */}
      <div className="w-80 border-r border-slate-800 flex flex-col bg-slate-900/20">
        <header className="p-4 border-b border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-black text-slate-100 tracking-tight">QUEUED</h2>
            <div className="flex items-center gap-2">
              <button 
                onClick={handleDownloadFeedback}
                title="Download Feedback Logs"
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-emerald-400 transition-all border border-slate-700"
              >
                <Download size={14} />
              </button>
              <span className="bg-amber-500/20 text-amber-500 text-[10px] font-bold px-2 py-0.5 rounded-full border border-amber-500/30">
                {pendingTransactions.length} PENDING
              </span>
            </div>
          </div>
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input 
              type="text" 
              placeholder="Search ID or User..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg py-2 pl-9 pr-4 text-xs text-slate-300 focus:outline-none focus:border-emerald-500/50 transition-colors"
            />
          </div>
        </header>

        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-20 text-slate-600">
              <Loader2 className="animate-spin mb-2" size={24} />
              <p className="text-xs uppercase font-bold tracking-widest">Loading...</p>
            </div>
          ) : pendingTransactions.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 px-6 text-center text-slate-600">
              <Inbox size={40} className="mb-4 opacity-20" />
              <p className="text-sm font-medium">All caught up!</p>
              <p className="text-xs mt-1">No transactions awaiting review.</p>
            </div>
          ) : (
            pendingTransactions.map((txn: any) => (
              <button
                key={txn.transaction_id}
                onClick={() => setSelectedId(txn.transaction_id)}
                className={cn(
                  "w-full text-left p-4 border-b border-slate-800/50 transition-all relative group",
                  selectedId === txn.transaction_id 
                    ? "bg-emerald-500/10" 
                    : "hover:bg-slate-800/50"
                )}
              >
                {selectedId === txn.transaction_id && (
                  <div className="absolute left-0 top-0 bottom-0 w-1 bg-emerald-500" />
                )}
                <div className="flex justify-between items-start mb-1">
                  <span className="text-[10px] font-mono text-slate-500 group-hover:text-slate-400 transition-colors">
                    {txn.transaction_id}
                  </span>
                  <span className={cn(
                    "text-[10px] font-bold px-1.5 py-0.5 rounded",
                    txn.risk_score > 0.7 ? "bg-red-500/20 text-red-400" : "bg-amber-500/20 text-amber-400"
                  )}>
                    {formatRiskScore(txn.risk_score)}
                  </span>
                </div>
                <div className="flex justify-between items-end">
                  <span className="font-bold text-slate-200">{formatCurrency(txn.amount)}</span>
                  <ChevronRight size={14} className={cn(
                    "transition-transform",
                    selectedId === txn.transaction_id ? "text-emerald-500 translate-x-1" : "text-slate-700"
                  )} />
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Detail View */}
      <div className="flex-1 overflow-y-auto bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900/50 via-slate-950 to-slate-950 p-8">
        {selectedTransaction ? (
          <div className="max-w-3xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">
            <ReviewCard transaction={selectedTransaction} />
          </div>
        ) : !isLoading && (
          <div className="h-full flex flex-col items-center justify-center text-slate-600">
            <div className="p-6 rounded-full bg-slate-900/50 mb-6">
              <Inbox size={48} className="opacity-20" />
            </div>
            <h3 className="text-xl font-bold text-slate-400">Queue is empty</h3>
            <p>New suspicious transactions will appear here automatically.</p>
          </div>
        )}
      </div>
    </div>
  );
}
