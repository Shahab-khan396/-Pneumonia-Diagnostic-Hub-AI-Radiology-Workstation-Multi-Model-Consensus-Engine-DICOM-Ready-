'use client';

import React from 'react';
import { Layers, Sparkles, Zap, Shield, Cpu } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface ModelOption {
  id: string;
  name: string;
  parameters: string;
  accuracy: string;
  badge?: string;
  description: string;
  isEnsemble?: boolean;
}

export const AVAILABLE_MODELS: ModelOption[] = [
  {
    id: 'ensemble',
    name: '4-Model Weighted Ensemble',
    parameters: '97.5M Combined',
    accuracy: '~91% Consensus',
    badge: 'Gold Standard',
    description: 'Weighted soft-voting consensus across all 4 architectures with agreement rating.',
    isEnsemble: true,
  },
  {
    id: 'mobilenet',
    name: 'MobileNetV2',
    parameters: '3.5M',
    accuracy: '87.5%',
    badge: 'Recommended Fast',
    description: 'High-efficiency inverted residual bottleneck CNN for rapid screening.',
  },
  {
    id: 'resnet50',
    name: 'ResNet50',
    parameters: '25.6M',
    accuracy: '~83%',
    description: 'Deep residual learning architecture with identity shortcut connections.',
  },
  {
    id: 'efficientnet',
    name: 'EfficientNetB0',
    parameters: '5.3M',
    accuracy: '~82%',
    description: 'Compound coefficient scaling balancing depth, width, and resolution.',
  },
  {
    id: 'VGG19',
    name: 'VGG19',
    parameters: '63.1M',
    accuracy: '~79%',
    description: '19-layer deep convolutional network with uniform 3x3 filter convolutions.',
  },
];

interface ModelSelectorProps {
  selectedModel: string;
  onSelectModel: (modelId: string) => void;
}

export function ModelSelector({ selectedModel, onSelectModel }: ModelSelectorProps) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <label className="text-sm font-semibold text-slate-200">
          2. Select AI Architecture
        </label>
        <span className="text-xs text-slate-400">
          Single model or Multi-Model Consensus
        </span>
      </div>

      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
        {AVAILABLE_MODELS.map((model) => {
          const isSelected = selectedModel === model.id;

          return (
            <div
              key={model.id}
              onClick={() => onSelectModel(model.id)}
              className={cn(
                'group relative cursor-pointer overflow-hidden rounded-xl border p-3.5 transition-all duration-200',
                model.isEnsemble && 'sm:col-span-2',
                isSelected
                  ? model.isEnsemble
                    ? 'border-indigo-500/80 bg-indigo-950/30 shadow-lg shadow-indigo-500/10'
                    : 'border-cyan-500/80 bg-cyan-950/30 shadow-lg shadow-cyan-500/10'
                  : 'border-slate-800 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-900/80'
              )}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2.5">
                  <div
                    className={cn(
                      'flex h-8 w-8 items-center justify-center rounded-lg border',
                      isSelected
                        ? model.isEnsemble
                          ? 'border-indigo-500/50 bg-indigo-500/20 text-indigo-400'
                          : 'border-cyan-500/50 bg-cyan-500/20 text-cyan-400'
                        : 'border-slate-800 bg-slate-950 text-slate-400'
                    )}
                  >
                    {model.isEnsemble ? (
                      <Sparkles className="h-4 w-4" />
                    ) : (
                      <Cpu className="h-4 w-4" />
                    )}
                  </div>

                  <div>
                    <div className="flex items-center gap-2">
                      <h4 className="text-sm font-semibold text-white">
                        {model.name}
                      </h4>
                      {model.badge && (
                        <span
                          className={cn(
                            'rounded-full px-2 py-0.5 text-[10px] font-bold border',
                            model.isEnsemble
                              ? 'border-indigo-500/40 bg-indigo-500/15 text-indigo-300'
                              : 'border-cyan-500/40 bg-cyan-500/15 text-cyan-300'
                          )}
                        >
                          {model.badge}
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-slate-400">
                      {model.parameters} params • {model.accuracy}
                    </p>
                  </div>
                </div>

                <div
                  className={cn(
                    'flex h-4 w-4 items-center justify-center rounded-full border transition-all',
                    isSelected
                      ? model.isEnsemble
                        ? 'border-indigo-400 bg-indigo-500'
                        : 'border-cyan-400 bg-cyan-500'
                      : 'border-slate-700 bg-slate-950'
                  )}
                >
                  {isSelected && <div className="h-1.5 w-1.5 rounded-full bg-white" />}
                </div>
              </div>

              <p className="mt-2 text-xs text-slate-400/90 leading-relaxed">
                {model.description}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
