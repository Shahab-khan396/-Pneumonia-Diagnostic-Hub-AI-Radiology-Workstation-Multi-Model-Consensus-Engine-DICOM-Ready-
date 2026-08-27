'use client';

import React, { useState } from 'react';
import { User, ChevronDown, ChevronUp, Stethoscope } from 'lucide-react';
import { PatientInfo } from '@/lib/types';

interface PatientFormProps {
  patientInfo: PatientInfo;
  onChange: (info: PatientInfo) => void;
}

export function PatientForm({ patientInfo, onChange }: PatientFormProps) {
  const [isOpen, setIsOpen] = useState(false);

  const updateField = (field: keyof PatientInfo, value: string) => {
    onChange({ ...patientInfo, [field]: value });
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3.5 transition-all">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between text-left text-sm font-medium text-slate-200"
      >
        <div className="flex items-center gap-2">
          <Stethoscope className="h-4 w-4 text-cyan-400" />
          <span>Patient Demographics & Indication (Optional)</span>
          {(patientInfo.patientId || patientInfo.patientAge) && (
            <span className="rounded-full bg-cyan-500/20 px-2 py-0.5 text-[10px] font-semibold text-cyan-300">
              Configured
            </span>
          )}
        </div>
        {isOpen ? (
          <ChevronUp className="h-4 w-4 text-slate-400" />
        ) : (
          <ChevronDown className="h-4 w-4 text-slate-400" />
        )}
      </button>

      {isOpen && (
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 pt-2 border-t border-slate-800/80">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">
              Patient ID
            </label>
            <input
              type="text"
              placeholder="e.g. PT-94821"
              value={patientInfo.patientId}
              onChange={(e) => updateField('patientId', e.target.value)}
              className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-1.5 text-xs text-white placeholder-slate-600 focus:border-cyan-500 focus:outline-none"
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">
                Age
              </label>
              <input
                type="text"
                placeholder="e.g. 54"
                value={patientInfo.patientAge}
                onChange={(e) => updateField('patientAge', e.target.value)}
                className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-1.5 text-xs text-white placeholder-slate-600 focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">
                Gender
              </label>
              <select
                value={patientInfo.patientGender}
                onChange={(e) => updateField('patientGender', e.target.value)}
                className="w-full rounded-lg border border-slate-800 bg-slate-950 px-2 py-1.5 text-xs text-white focus:border-cyan-500 focus:outline-none"
              >
                <option value="">Select...</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">
              Referring Physician
            </label>
            <input
              type="text"
              placeholder="e.g. Dr. Alex Mercer"
              value={patientInfo.referringPhysician}
              onChange={(e) => updateField('referringPhysician', e.target.value)}
              className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-1.5 text-xs text-white placeholder-slate-600 focus:border-cyan-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">
              Clinical Indication / History
            </label>
            <input
              type="text"
              placeholder="e.g. Acute fever, productive cough"
              value={patientInfo.clinicalHistory}
              onChange={(e) => updateField('clinicalHistory', e.target.value)}
              className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-1.5 text-xs text-white placeholder-slate-600 focus:border-cyan-500 focus:outline-none"
            />
          </div>
        </div>
      )}
    </div>
  );
}
