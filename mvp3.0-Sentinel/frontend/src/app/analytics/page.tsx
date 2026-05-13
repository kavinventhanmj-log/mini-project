'use client';

import { useMemo, useState, useEffect } from "react";
import { useReport } from "@/hooks/useApi";
// ... (rest of imports)
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  Cell, PieChart, Pie, Legend, AreaChart, Area, LineChart, Line
} from 'recharts';
import { 
  BarChart3, TrendingUp, ShieldCheck, AlertTriangle, 
  Target, Activity, Zap, Loader2, PieChart as PieChartIcon
} from "lucide-react";
import { formatRiskScore, cn } from "@/lib/utils";

export default function AnalyticsPage() {
  const { data: report, isLoading, error } = useReport();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const confusionData = useMemo(() => {
    if (!report) return [];
    return [
      { name: 'True Positives', value: report.confusion_matrix?.TP || 0, color: '#10b981' },
      { name: 'False Positives', value: report.confusion_matrix?.FP || 0, color: '#f43f5e' },
      { name: 'True Negatives', value: report.confusion_matrix?.TN || 0, color: '#3b82f6' },
      { name: 'False Negatives', value: report.confusion_matrix?.FN || 0, color: '#f59e0b' },
    ];
  }, [report]);

  const bucketData = useMemo(() => {
    if (!report) return [];
    return Object.entries(report.bucket_breakdown || {}).map(([key, val]: [string, any]) => ({
      name: key,
      block: val?.block || 0,
      review: val?.review || 0,
      allow: val?.allow || 0,
    }));
  }, [report]);

  if (!mounted || (isLoading && !report)) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-slate-500">
        <Loader2 className="animate-spin mb-4" size={48} />
        <p className="text-sm font-bold uppercase tracking-widest">Generating Analytics Report...</p>
      </div>
    );
  }

  if (error || !report) {
// ...
    return (
      <div className="h-full flex flex-col items-center justify-center text-slate-500 p-8 text-center">
        <AlertTriangle className="text-amber-500 mb-4" size={48} />
        <h3 className="text-xl font-bold text-slate-200">No Data Available</h3>
        <p className="max-w-md mt-2">
          Run the pipeline from the Live Monitor to generate performance analytics.
        </p>
      </div>
    );
  }

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 animate-in fade-in duration-700">
      <header>
        <div className="flex items-center gap-3 mb-2">
          <div className="bg-blue-500/20 p-2 rounded-lg text-blue-500">
            <BarChart3 size={24} />
          </div>
          <h2 className="text-3xl font-black text-slate-100 tracking-tight uppercase">Model Intelligence</h2>
        </div>
        <p className="text-slate-400">Deep-dive into fraud detection performance and decision distributions.</p>
      </header>

      {/* Primary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard 
          label="Precision" 
          value={formatRiskScore(report.metrics?.precision || 0)} 
          sub="Confidence in Blocks" 
          icon={<Target className="text-emerald-400" />}
        />
        <MetricCard 
          label="Recall" 
          value={formatRiskScore(report.metrics?.recall || 0)} 
          sub="Fraud Capture Rate" 
          icon={<Activity className="text-blue-400" />}
        />
        <MetricCard 
          label="F1 Score" 
          value={formatRiskScore(report.metrics?.f1 || 0)} 
          sub="Overall Accuracy" 
          icon={<Zap className="text-purple-400" />}
        />
        <MetricCard 
          label="FP Reduction" 
          value={formatRiskScore(report.fp_analysis?.reduction_rate || 0)} 
          sub="Friction Saved" 
          icon={<ShieldCheck className="text-amber-400" />}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Risk Distribution */}
        <div className="lg:col-span-2 bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-8">
            <h3 className="font-bold text-slate-200 flex items-center gap-2">
              <TrendingUp size={18} className="text-blue-500" />
              Risk Score Distribution
            </h3>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={report.risk_histogram || []}>
                <defs>
                  <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis 
                  dataKey="bin" 
                  stroke="#64748b" 
                  fontSize={10} 
                  tickLine={false} 
                  axisLine={false}
                  tickFormatter={(val) => val.toFixed(1)}
                />
                <YAxis hide />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
                  itemStyle={{ color: '#3b82f6' }}
                />
                <Area 
                  type="monotone" 
                  dataKey="count" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  fillOpacity={1} 
                  fill="url(#colorCount)" 
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confusion Matrix Pie */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
          <h3 className="font-bold text-slate-200 mb-6 flex items-center gap-2">
            <PieChartIcon size={18} className="text-emerald-500" />
            Outcome Breakdown
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={confusionData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {confusionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-2 gap-2 mt-4">
            {confusionData.map((d) => (
              <div key={d.name} className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: d.color }} />
                <span className="text-[10px] font-bold text-slate-500 uppercase">{d.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Decision Breakdown by Bucket */}
        <div className="lg:col-span-3 bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
          <h3 className="font-bold text-slate-200 mb-8 flex items-center gap-2">
            <TrendingUp size={18} className="text-emerald-500" />
            Decision Distribution by True Label
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={bucketData} layout="vertical" margin={{ left: 30 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                <XAxis type="number" hide />
                <YAxis 
                  dataKey="name" 
                  type="category" 
                  stroke="#64748b" 
                  fontSize={12} 
                  fontFamily="monospace"
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip 
                  cursor={{ fill: 'transparent' }}
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
                />
                <Legend iconType="circle" wrapperStyle={{ paddingTop: '20px' }} />
                <Bar dataKey="block" name="Auto-Block" stackId="a" fill="#f43f5e" radius={[0, 0, 0, 0]} barSize={40} />
                <Bar dataKey="review" name="Review" stackId="a" fill="#f59e0b" radius={[0, 0, 0, 0]} />
                <Bar dataKey="allow" name="Auto-Allow" stackId="a" fill="#10b981" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value, sub, icon }: { label: string; value: string; sub: string; icon: React.ReactNode }) {
  return (
    <div className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl hover:border-slate-700 transition-colors">
      <div className="flex items-center justify-between mb-4">
        <span className="text-slate-500 text-[10px] font-black uppercase tracking-widest">{label}</span>
        {icon}
      </div>
      <div className="text-3xl font-black text-slate-100 mb-1">{value}</div>
      <div className="text-slate-500 text-[10px] font-medium">{sub}</div>
    </div>
  );
}
