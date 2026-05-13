import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine
} from 'recharts';

interface ShapData {
  feature: string;
  label: string;
  shap_value: number;
  direction: string;
}

export function ShapBarChart({ data }: { data: ShapData[] }) {
  // Sort by absolute value for better visualization
  const sortedData = [...data].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));

  return (
    <div className="h-[300px] w-full mt-4">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={sortedData}
          margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#1e293b" />
          <XAxis type="number" hide />
          <YAxis
            dataKey="label"
            type="category"
            tick={{ fill: '#94a3b8', fontSize: 12 }}
            width={90}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
            itemStyle={{ color: '#f1f5f9' }}
            formatter={(value: any) => [Number(value).toFixed(4), 'SHAP Impact']}
          />
          <ReferenceLine x={0} stroke="#475569" />
          <Bar dataKey="shap_value" radius={[0, 4, 4, 0]}>
            {sortedData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.shap_value > 0 ? '#ef4444' : '#10b981'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex justify-between px-2 text-[10px] text-slate-500 uppercase tracking-widest mt-2">
        <span>Decreases Risk</span>
        <span>Increases Risk</span>
      </div>
    </div>
  );
}
