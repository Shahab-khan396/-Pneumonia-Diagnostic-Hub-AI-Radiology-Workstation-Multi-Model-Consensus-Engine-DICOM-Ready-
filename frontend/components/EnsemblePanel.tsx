'use client';

import React from 'react';
import { ModelResult } from '@/lib/types';
import { Sparkles, Check, AlertCircle, Clock } from 'lucide-react';
import { cn } from '@/lib/utils';

interface EnsemblePanelProps {
  models: ModelResult[];
  consensusVerdict: 'NORMAL' | 'PNEUMONIA';
  consensusConfidence: number;
  agreementLevel: 'UNANIMOUS' | 'STRONG_MAJORITY' | 'SPLIT_DECISION';
  agreementText: string;
  totalTimeMs: number;
}

export function EnsemblePanel({
  models,
  consensusVerdict,
  consensusConfidence,
  agreementLevel,
  agreementText,
  totalTimeMs,
}: EnsemblePanelProps) {
  const isPneumonia = consensusVerdict === 'PNEUMONIA';

  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-slate-800 bg-slate-900/60 p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-indigo-400" />
          <h3 className="text-sm font-semibold text-white">
            Multi-Model Consensus Matrix
          </h3>
        </div>

        <div className="flex items-center gap-2">
          <span
            className={cn(
              'rounded-full px-2.5 py-0.5 text-[11px] font-bold border',
              agreementLevel === 'UNANIMOUS'
                ? 'border-emerald-500/40 bg-emerald-500/15 text-emerald-300'
                : agreementLevel === 'STRONG_MAJORITY'
                ? 'border-cyan-500/40 bg-cyan-500/15 text-cyan-300'
                : 'border-amber-500/40 bg-amber-500/15 text-amber-300'
            )}
          >
            {agreementLevel}
          </span>
          <span className="flex items-center gap-1 text-[11px] text-slate-400">
            <Clock className="h-3 w-3" />
            {totalTimeMs.toFixed(0)} ms total
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400">
              <th className="pb-2 font-medium">Architecture</th>
              <th className="pb-2 font-medium">Weight</th>
              <th className="pb-2 font-medium">Prediction</th>
              <th className="pb-2 font-medium">Confidence</th>
              <th className="pb-2 font-medium text-right">Latency</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 text-slate-300">
            {models.map((model) => {
              const mPneumonia = model.prediction === 'PNEUMONIA';
              return (
                <tr key={model.id} className="hover:bg-slate-900/40 transition-colors">
                  <td className="py-2.5 font-medium text-white flex items-center gap-2">
                    <span>{model.name}</span>
                    {model.parameters && (
                      <span className="text-[10px] text-slate-500 font-mono">
                        ({model.parameters})
                      </span>
                    )}
                  </td>
                  <td className="py-2.5 text-slate-400 font-mono">
                    {(model.weight * 100).toFixed(0)}%
                  </td>
                  <td className="py-2.5">
                    <span
                      className={cn(
                        'inline-flex items-center gap-1 rounded-md px-2 py-0.5 font-bold text-[11px]',
                        mPneumonia
                          ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                          : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      )}
                    >
                      {model.prediction}
                    </span>
                  </td>
                  <td className="py-2.5 font-semibold font-mono">
                    {model.confidence.toFixed(1)}%
                  </td>
                  <td className="py-2.5 text-right font-mono text-slate-400">
                    {model.inference_time_ms.toFixed(1)} ms
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="text-[11px] text-slate-400 border-t border-slate-800/80 pt-2">
        Consensus formula:{' '}
        <span className="font-mono text-slate-300">
          P_consensus = ∑ (w_i × P_i)
        </span>
        . All 4 deep learning backbones evaluated in parallel.
      </p>
    </div>
  );
}
