'use client';

import React from 'react';
import { AlertTriangle, CheckCircle2, ShieldAlert, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

interface VerdictBadgeProps {
  prediction: 'NORMAL' | 'PNEUMONIA';
  confidence: number;
  isEnsemble?: boolean;
  agreementLevel?: 'UNANIMOUS' | 'STRONG_MAJORITY' | 'SPLIT_DECISION';
  agreementText?: string;
  modelName: string;
}

export function VerdictBadge({
  prediction,
  confidence,
  isEnsemble,
  agreementLevel,
  agreementText,
  modelName,
}: VerdictBadgeProps) {
  const isPneumonia = prediction === 'PNEUMONIA';

  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-2xl border p-5 transition-all shadow-xl',
        isPneumonia
          ? 'border-rose-500/40 bg-gradient-to-br from-rose-950/40 via-slate-900/60 to-slate-900/80 shadow-rose-950/20'
          : 'border-emerald-500/40 bg-gradient-to-br from-emerald-950/40 via-slate-900/60 to-slate-900/80 shadow-emerald-950/20'
      )}
    >
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3.5">
          <div
            className={cn(
              'flex h-12 w-12 items-center justify-center rounded-2xl border shadow-inner',
              isPneumonia
                ? 'border-rose-500/50 bg-rose-500/20 text-rose-400'
                : 'border-emerald-500/50 bg-emerald-500/20 text-emerald-400'
            )}
          >
            {isPneumonia ? (
              <AlertTriangle className="h-6 w-6" />
            ) : (
              <CheckCircle2 className="h-6 w-6" />
            )}
          </div>

          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Diagnostic Finding
              </span>
              {isEnsemble && (
                <span className="flex items-center gap-1 rounded-full border border-indigo-500/40 bg-indigo-500/20 px-2 py-0.5 text-[10px] font-bold text-indigo-300">
                  <Sparkles className="h-3 w-3" />
                  Consensus
                </span>
              )}
            </div>
            <h2
              className={cn(
                'text-2xl font-black tracking-tight sm:text-3xl',
                isPneumonia ? 'text-rose-400' : 'text-emerald-400'
              )}
            >
              {prediction}
            </h2>
            {agreementText && (
              <p className="text-xs text-slate-400 mt-0.5">{agreementText}</p>
            )}
          </div>
        </div>

        <div className="flex flex-col sm:items-end">
          <span className="text-xs font-medium text-slate-400">Confidence Score</span>
          <div className="flex items-baseline gap-1">
            <span
              className={cn(
                'text-3xl font-black tracking-tight sm:text-4xl',
                isPneumonia ? 'text-rose-300' : 'text-emerald-300'
              )}
            >
              {confidence.toFixed(1)}%
            </span>
          </div>
          <span className="text-[11px] text-slate-400">
            Engine: {modelName}
          </span>
        </div>
      </div>
    </div>
  );
}
