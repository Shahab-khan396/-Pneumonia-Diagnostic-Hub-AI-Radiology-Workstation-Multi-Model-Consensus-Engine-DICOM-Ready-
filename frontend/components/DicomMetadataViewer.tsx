'use client';

import React, { useState } from 'react';
import { FileText, ChevronDown, ChevronUp, Database } from 'lucide-react';
import { DicomMetadata } from '@/lib/types';

interface DicomMetadataViewerProps {
  metadata: DicomMetadata;
}

export function DicomMetadataViewer({ metadata }: DicomMetadataViewerProps) {
  const [isOpen, setIsOpen] = useState(false);

  const tags = [
    { label: 'Patient ID', value: metadata.patient_id },
    { label: 'Patient Name', value: metadata.patient_name },
    { label: 'Age / Sex', value: `${metadata.patient_age || 'N/A'} / ${metadata.patient_sex || 'N/A'}` },
    { label: 'Study Date', value: metadata.study_date },
    { label: 'Modality', value: metadata.modality },
    { label: 'Body Part', value: metadata.body_part },
    { label: 'Manufacturer', value: metadata.manufacturer },
    { label: 'KVP', value: metadata.kvp },
    { label: 'Exposure Time', value: metadata.exposure_time },
    { label: 'Photometric Interpretation', value: metadata.photometric },
    { label: 'Matrix Dimensions', value: metadata.rows && metadata.columns ? `${metadata.rows} × ${metadata.columns} px` : undefined },
  ].filter((t) => t.value && t.value !== 'N/A');

  return (
    <div className="rounded-2xl border border-cyan-500/30 bg-cyan-950/20 p-4">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between text-left text-xs font-semibold text-cyan-300"
      >
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-cyan-400" />
          <span>DICOM Clinical Metadata ({tags.length} tags extracted)</span>
        </div>
        {isOpen ? (
          <ChevronUp className="h-4 w-4 text-cyan-400" />
        ) : (
          <ChevronDown className="h-4 w-4 text-cyan-400" />
        )}
      </button>

      {isOpen && (
        <div className="mt-3 grid grid-cols-2 gap-2 border-t border-cyan-500/20 pt-3 sm:grid-cols-3">
          {tags.map((tag) => (
            <div key={tag.label} className="rounded-lg bg-slate-950/60 p-2 border border-slate-800">
              <span className="block text-[10px] font-medium uppercase tracking-wider text-slate-400">
                {tag.label}
              </span>
              <span className="block text-xs font-semibold text-slate-200 truncate font-mono">
                {tag.value}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
