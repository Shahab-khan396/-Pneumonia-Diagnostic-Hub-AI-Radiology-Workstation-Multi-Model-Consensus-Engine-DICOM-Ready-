'use client';

import React, { useState } from 'react';
import { Eye, Layers, Sparkles, Download, Maximize2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface GradCamViewerProps {
  originalUrl?: string;
  overlayUrl?: string;
  heatmapUrl?: string;
  compositeUrl?: string;
  reportUrl?: string;
  modelName: string;
}

type TabType = 'overlay' | 'heatmap' | 'composite' | 'original';

export function GradCamViewer({
  originalUrl,
  overlayUrl,
  heatmapUrl,
  compositeUrl,
  reportUrl,
  modelName,
}: GradCamViewerProps) {
  const [activeTab, setActiveTab] = useState<TabType>('overlay');

  const currentImageUrl = (() => {
    switch (activeTab) {
      case 'overlay':
        return overlayUrl || originalUrl;
      case 'heatmap':
        return heatmapUrl || overlayUrl || originalUrl;
      case 'composite':
        return compositeUrl || overlayUrl || originalUrl;
      case 'original':
      default:
        return originalUrl;
    }
  })();

  const tabs: { id: TabType; label: string; available: boolean }[] = [
    { id: 'overlay', label: 'Grad-CAM Overlay', available: !!overlayUrl },
    { id: 'heatmap', label: 'Raw Heatmap', available: !!heatmapUrl },
    { id: 'composite', label: '3-Panel Composite', available: !!compositeUrl },
    { id: 'original', label: 'Original CXR', available: !!originalUrl },
  ];

  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-slate-800 bg-slate-900/60 p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-white">
            Explainable AI (Grad-CAM XAI)
          </h3>
        </div>

        {reportUrl && (
          <a
            href={
              reportUrl.startsWith('data:')
                ? reportUrl
                : reportUrl.startsWith('http')
                ? reportUrl
                : `https://shahabkhan396-pneumonia-hub.hf.space${reportUrl}`
            }
            download={`Pneumonia_Diagnostic_Report_${modelName.replace(/\s+/g, '_')}.pdf`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-3 py-1.5 text-xs font-semibold text-cyan-300 hover:bg-cyan-500/20 transition-all shadow-sm"
          >
            <Download className="h-3.5 w-3.5" />
            <span>Download PDF Report</span>
          </a>
        )}
      </div>

      {/* Tabs */}
      <div className="flex flex-wrap gap-1.5 rounded-xl bg-slate-950 p-1 border border-slate-800">
        {tabs
          .filter((t) => t.available)
          .map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                'rounded-lg px-3 py-1.5 text-xs font-medium transition-all',
                activeTab === tab.id
                  ? 'bg-cyan-500/20 text-cyan-300 shadow-sm border border-cyan-500/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/50'
              )}
            >
              {tab.label}
            </button>
          ))}
      </div>

      {/* Image Display */}
      <div className="relative flex min-h-[320px] max-h-[460px] items-center justify-center overflow-hidden rounded-xl border border-slate-800 bg-slate-950 p-2">
        {currentImageUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={currentImageUrl}
            alt="Grad-CAM visualization"
            className="max-h-[440px] w-auto max-w-full rounded-lg object-contain"
          />
        ) : (
          <div className="flex flex-col items-center gap-2 text-slate-500">
            <Eye className="h-8 w-8" />
            <p className="text-xs">No image visualization available</p>
          </div>
        )}

        <div className="absolute bottom-4 left-4 rounded-lg border border-slate-800 bg-slate-950/80 px-2.5 py-1 text-[11px] font-medium text-slate-300 backdrop-blur-md">
          {activeTab === 'overlay' && 'Alpha-blended Activation Map'}
          {activeTab === 'heatmap' && 'Jet Colormap Feature Attribution'}
          {activeTab === 'composite' && 'Side-by-Side Diagnostic Triad'}
          {activeTab === 'original' && 'Source Input Radiograph'}
        </div>
      </div>
    </div>
  );
}
