'use client';

import React, { useRef, useState } from 'react';
import { UploadCloud, FileText, Image as ImageIcon, X, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface DropZoneProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
  previewUrl: string | null;
}

export function DropZone({ onFileSelect, selectedFile, onClear, previewUrl }: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const isDicom = selectedFile?.name.toLowerCase().endsWith('.dcm');

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <label className="text-sm font-semibold text-slate-200">
          1. Upload Chest Radiograph
        </label>
        <span className="text-xs text-slate-400">
          DICOM (.dcm), PNG, JPG, WEBP (Max 32MB)
        </span>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".png,.jpg,.jpeg,.webp,.dcm"
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            onFileSelect(e.target.files[0]);
          }
        }}
      />

      {!selectedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={cn(
            'group relative flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 text-center transition-all duration-200',
            isDragging
              ? 'border-cyan-400 bg-cyan-950/20 shadow-lg shadow-cyan-500/10'
              : 'border-slate-800 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-900/80'
          )}
        >
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-800/80 text-cyan-400 shadow-inner group-hover:scale-105 group-hover:bg-cyan-500/20 transition-all">
            <UploadCloud className="h-7 w-7" />
          </div>

          <div className="mt-4 space-y-1">
            <p className="text-sm font-medium text-slate-200">
              Drag and drop radiograph here, or{' '}
              <span className="text-cyan-400 underline underline-offset-2">browse</span>
            </p>
            <p className="text-xs text-slate-400">
              Standard PA/AP Chest X-Rays or clinical DICOM series
            </p>
          </div>

          <div className="mt-4 flex items-center gap-2 rounded-full border border-slate-800 bg-slate-950/60 px-3 py-1 text-[11px] text-slate-400">
            <FileText className="h-3.5 w-3.5 text-cyan-400" />
            <span>Automatic DICOM pixel decoding & VOI windowing</span>
          </div>
        </div>
      ) : (
        <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="relative flex h-16 w-16 shrink-0 items-center justify-center overflow-hidden rounded-xl bg-slate-950 border border-slate-800">
                {previewUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="h-full w-full object-cover"
                  />
                ) : isDicom ? (
                  <FileText className="h-8 w-8 text-cyan-400" />
                ) : (
                  <ImageIcon className="h-8 w-8 text-slate-500" />
                )}
              </div>

              <div className="space-y-1 truncate">
                <div className="flex items-center gap-2">
                  <p className="truncate text-sm font-semibold text-slate-200">
                    {selectedFile.name}
                  </p>
                  {isDicom && (
                    <span className="rounded bg-cyan-500/20 px-1.5 py-0.5 text-[10px] font-bold text-cyan-300 border border-cyan-500/30">
                      DICOM
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-400">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • Ready for analysis
                </p>
              </div>
            </div>

            <button
              onClick={onClear}
              className="rounded-lg border border-slate-800 bg-slate-800/60 p-2 text-slate-400 hover:bg-rose-500/20 hover:border-rose-500/30 hover:text-rose-300 transition-colors"
              title="Remove file"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
