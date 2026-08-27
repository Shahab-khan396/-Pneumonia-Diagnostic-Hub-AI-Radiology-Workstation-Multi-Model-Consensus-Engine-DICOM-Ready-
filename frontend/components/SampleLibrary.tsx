'use client';

import React from 'react';
import { SampleStudy } from '@/lib/types';
import { Sparkles, FolderOpen } from 'lucide-react';
import { cn } from '@/lib/utils';

export const STATIC_SAMPLES: SampleStudy[] = [
  {
    id: 'sample_normal',
    label: 'Normal Clear CXR',
    filename: 'sample_normal.jpg',
    description: 'Clear bilateral lung fields — no consolidation.',
    category: 'normal',
    image_url: '/backend/static/samples/sample_normal.jpg',
  },
  {
    id: 'sample_bacterial',
    label: 'Bacterial Lobar',
    filename: 'sample_bacterial.jpg',
    description: 'Right lower lobe dense consolidation opacity.',
    category: 'bacterial',
    image_url: '/backend/static/samples/sample_bacterial.jpg',
  },
  {
    id: 'sample_viral',
    label: 'Viral Interstitial',
    filename: 'sample_viral.jpg',
    description: 'Bilateral diffuse interstitial reticular opacities.',
    category: 'viral',
    image_url: '/backend/static/samples/sample_viral.jpg',
  },
];

interface SampleLibraryProps {
  onSelectSample: (sample: SampleStudy) => void;
  activeSampleId?: string | null;
}

export function SampleLibrary({ onSelectSample, activeSampleId }: SampleLibraryProps) {
  return (
    <div className="flex flex-col gap-2 rounded-2xl border border-slate-800 bg-slate-900/40 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FolderOpen className="h-4 w-4 text-cyan-400" />
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Quick-Load Sample Studies
          </h4>
        </div>
        <span className="text-[11px] text-slate-400">Click to run instant demo</span>
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
        {STATIC_SAMPLES.map((sample) => {
          const isActive = activeSampleId === sample.id;
          return (
            <button
              key={sample.id}
              type="button"
              onClick={() => onSelectSample(sample)}
              className={cn(
                'flex flex-col text-left rounded-xl border p-2.5 transition-all',
                isActive
                  ? 'border-cyan-500 bg-cyan-950/40 shadow-sm'
                  : 'border-slate-800 bg-slate-950/60 hover:border-slate-700 hover:bg-slate-900/80'
              )}
            >
              <div className="flex items-center justify-between w-full">
                <span className="text-xs font-bold text-white truncate">
                  {sample.label}
                </span>
                <span
                  className={cn(
                    'rounded px-1.5 py-0.2 text-[9px] font-bold uppercase',
                    sample.category === 'normal'
                      ? 'bg-emerald-500/20 text-emerald-300'
                      : 'bg-rose-500/20 text-rose-300'
                  )}
                >
                  {sample.category}
                </span>
              </div>
              <p className="mt-1 text-[11px] text-slate-400 line-clamp-1">
                {sample.description}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
