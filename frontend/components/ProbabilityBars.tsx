'use client';

import React from 'react';
import { Activity } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ProbabilityBarsProps {
  probabilities: {
    NORMAL: number;
    PNEUMONIA: number;
  };
}

export function ProbabilityBars({ probabilities }: ProbabilityBarsProps) {
  const pPneumonia = probabilities.PNEUMONIA ?? 0;
  const pNormal = probabilities.NORMAL ?? 0;

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Probability Distribution
        </h4>
        <span className="text-[11px] text-slate-400">Sigmoidal Softmax Output</span>
      </div>

      <div className="space-y-3">
        {/* PNEUMONIA BAR */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs">
            <span className="font-semibold text-rose-300">PNEUMONIA</span>
            <span className="font-bold text-rose-300">{pPneumonia.toFixed(2)}%</span>
          </div>
          <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-950 p-0.5 border border-slate-800">
            <div
              className="h-full rounded-full bg-gradient-to-r from-rose-600 to-rose-400 transition-all duration-700 ease-out"
              style={{ width: `${Math.max(pPneumonia, 2)}%` }}
            />
          </div>
        </div>

        {/* NORMAL BAR */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs">
            <span className="font-semibold text-emerald-300">NORMAL</span>
            <span className="font-bold text-emerald-300">{pNormal.toFixed(2)}%</span>
          </div>
          <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-950 p-0.5 border border-slate-800">
            <div
              className="h-full rounded-full bg-gradient-to-r from-emerald-600 to-emerald-400 transition-all duration-700 ease-out"
              style={{ width: `${Math.max(pNormal, 2)}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
