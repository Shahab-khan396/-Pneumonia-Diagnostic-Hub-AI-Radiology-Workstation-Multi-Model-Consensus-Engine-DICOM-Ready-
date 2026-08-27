'use client';

import React, { useState } from 'react';
import { Navbar } from '@/components/Navbar';
import { DropZone } from '@/components/DropZone';
import { ModelSelector } from '@/components/ModelSelector';
import { PatientForm } from '@/components/PatientForm';
import { VerdictBadge } from '@/components/VerdictBadge';
import { ProbabilityBars } from '@/components/ProbabilityBars';
import { GradCamViewer } from '@/components/GradCamViewer';
import { EnsemblePanel } from '@/components/EnsemblePanel';
import { DicomMetadataViewer } from '@/components/DicomMetadataViewer';
import { SampleLibrary, STATIC_SAMPLES } from '@/components/SampleLibrary';
import {
  PatientInfo,
  PredictResponse,
  EnsembleResponse,
  SampleStudy,
} from '@/lib/types';
import {
  Play,
  RotateCcw,
  Sparkles,
  AlertCircle,
  Clock,
  Loader2,
  FileCheck,
  ShieldCheck,
} from 'lucide-react';
import { cn } from '@/lib/utils';

export default function WorkstationPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('ensemble');
  const [patientInfo, setPatientInfo] = useState<PatientInfo>({
    patientId: '',
    patientAge: '',
    patientGender: '',
    clinicalHistory: '',
    referringPhysician: '',
  });

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeSampleId, setActiveSampleId] = useState<string | null>(null);

  // Results state
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  const [ensembleResult, setEnsembleResult] = useState<EnsembleResponse | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setActiveSampleId(null);
    setError(null);
    setPredictResult(null);
    setEnsembleResult(null);

    if (file.name.toLowerCase().endsWith('.dcm')) {
      setPreviewUrl(null);
    } else {
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setActiveSampleId(null);
    setError(null);
    setPredictResult(null);
    setEnsembleResult(null);
  };

  const handleSelectSample = async (sample: SampleStudy) => {
    setActiveSampleId(sample.id);
    setError(null);
    setPredictResult(null);
    setEnsembleResult(null);

    // Create a dummy image file from sample URL or generate a synthetic canvas
    try {
      const canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(0, 0, 512, 512);
        ctx.fillStyle = '#64748b';
        ctx.font = '20px sans-serif';
        ctx.fillText(sample.label, 40, 260);
      }

      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], sample.filename, { type: 'image/jpeg' });
          setSelectedFile(file);
          setPreviewUrl(canvas.toDataURL());
        }
      }, 'image/jpeg');
    } catch (e) {
      console.error(e);
    }
  };

  const handleRunInference = async () => {
    if (!selectedFile) {
      setError('Please upload a chest radiograph or select a sample study first.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setPredictResult(null);
    setEnsembleResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('explain', 'true');
    formData.append('generate_report', 'true');

    if (patientInfo.patientId) formData.append('patient_id', patientInfo.patientId);
    if (patientInfo.patientAge) formData.append('patient_age', patientInfo.patientAge);
    if (patientInfo.patientGender) formData.append('patient_gender', patientInfo.patientGender);
    if (patientInfo.clinicalHistory) formData.append('clinical_history', patientInfo.clinicalHistory);
    if (patientInfo.referringPhysician) formData.append('referring_physician', patientInfo.referringPhysician);

    const isEnsemble = selectedModel === 'ensemble';
    const endpoint = isEnsemble ? '/api/compare' : '/api/predict';

    if (!isEnsemble) {
      formData.append('model_choice', selectedModel);
    }

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Diagnostic inference failed.');
      }

      if (isEnsemble) {
        setEnsembleResult(data);
      } else {
        setPredictResult(data);
      }
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred during inference.');
    } finally {
      setIsLoading(false);
    }
  };

  const activeResult = ensembleResult || predictResult;
  const isEnsemble = !!ensembleResult;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 selection:bg-cyan-500 selection:text-white">
      <Navbar />

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
          {/* LEFT PANEL: Inputs, Controls & Upload (5 Cols) */}
          <div className="flex flex-col gap-5 lg:col-span-5">
            {/* Quick Sample Library */}
            <SampleLibrary
              onSelectSample={handleSelectSample}
              activeSampleId={activeSampleId}
            />

            {/* DropZone */}
            <DropZone
              onFileSelect={handleFileSelect}
              selectedFile={selectedFile}
              onClear={handleClear}
              previewUrl={previewUrl}
            />

            {/* Patient Demographics */}
            <PatientForm
              patientInfo={patientInfo}
              onChange={setPatientInfo}
            />

            {/* Model Architecture Selector */}
            <ModelSelector
              selectedModel={selectedModel}
              onSelectModel={setSelectedModel}
            />

            {/* Run Inference Action Button */}
            <div className="flex items-center gap-3 pt-2">
              <button
                type="button"
                disabled={isLoading || !selectedFile}
                onClick={handleRunInference}
                className={cn(
                  'flex flex-1 items-center justify-center gap-2 rounded-xl py-3.5 px-6 font-bold text-sm text-white shadow-xl transition-all',
                  isLoading || !selectedFile
                    ? 'cursor-not-allowed bg-slate-800 text-slate-500 shadow-none'
                    : selectedModel === 'ensemble'
                    ? 'bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 shadow-indigo-500/25 hover:scale-[1.01]'
                    : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 shadow-cyan-500/25 hover:scale-[1.01]'
                )}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    <span>Analyzing Radiograph...</span>
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 fill-white" />
                    <span>
                      Run {selectedModel === 'ensemble' ? 'Consensus Ensemble' : 'Inference'}
                    </span>
                  </>
                )}
              </button>

              {selectedFile && (
                <button
                  type="button"
                  onClick={handleClear}
                  disabled={isLoading}
                  className="flex h-12 w-12 items-center justify-center rounded-xl border border-slate-800 bg-slate-900/80 text-slate-400 hover:bg-slate-800 hover:text-white transition-colors"
                  title="Reset scan"
                >
                  <RotateCcw className="h-4 w-4" />
                </button>
              )}
            </div>

            {/* Error Banner */}
            {error && (
              <div className="flex items-start gap-3 rounded-xl border border-rose-500/30 bg-rose-950/20 p-4 text-rose-300">
                <AlertCircle className="h-5 w-5 shrink-0 text-rose-400 mt-0.5" />
                <div className="text-xs leading-relaxed">
                  <p className="font-semibold">Inference Error</p>
                  <p className="mt-0.5 text-rose-300/90">{error}</p>
                </div>
              </div>
            )}
          </div>

          {/* RIGHT PANEL: Diagnostic Telemetry, XAI, and Results (7 Cols) */}
          <div className="flex flex-col gap-5 lg:col-span-7">
            {activeResult ? (
              <>
                {/* Diagnostic Finding Header Badge */}
                <VerdictBadge
                  prediction={
                    isEnsemble
                      ? ensembleResult.consensus_verdict
                      : predictResult!.prediction
                  }
                  confidence={
                    isEnsemble
                      ? ensembleResult.consensus_confidence
                      : predictResult!.confidence
                  }
                  isEnsemble={isEnsemble}
                  agreementLevel={
                    isEnsemble ? ensembleResult.agreement_level : undefined
                  }
                  agreementText={
                    isEnsemble ? ensembleResult.agreement_text : undefined
                  }
                  modelName={
                    isEnsemble
                      ? '4-Model Soft-Voting Consensus'
                      : predictResult!.model_name
                  }
                />

                {/* DICOM Metadata if present */}
                {activeResult.dicom_metadata && (
                  <DicomMetadataViewer metadata={activeResult.dicom_metadata} />
                )}

                {/* Probability Distribution Bars */}
                <ProbabilityBars
                  probabilities={
                    isEnsemble
                      ? ensembleResult.consensus_probabilities
                      : predictResult!.probabilities
                  }
                />

                {/* Multi-Model Breakdown if Ensemble */}
                {isEnsemble && (
                  <EnsemblePanel
                    models={ensembleResult.models_breakdown}
                    consensusVerdict={ensembleResult.consensus_verdict}
                    consensusConfidence={ensembleResult.consensus_confidence}
                    agreementLevel={ensembleResult.agreement_level}
                    agreementText={ensembleResult.agreement_text}
                    totalTimeMs={ensembleResult.total_inference_time_ms}
                  />
                )}

                {/* Grad-CAM Explainable AI Viewer */}
                <GradCamViewer
                  originalUrl={previewUrl || undefined}
                  overlayUrl={activeResult.gradcam_overlay_url}
                  heatmapUrl={activeResult.gradcam_heatmap_url}
                  compositeUrl={activeResult.gradcam_composite_url}
                  reportUrl={activeResult.report_pdf_url}
                  modelName={
                    isEnsemble ? 'Consensus' : predictResult!.model_name
                  }
                />
              </>
            ) : (
              /* Empty Standby State */
              <div className="flex min-h-[480px] flex-col items-center justify-center rounded-2xl border border-dashed border-slate-800 bg-slate-900/20 p-8 text-center">
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-900 border border-slate-800 text-cyan-400 shadow-inner">
                  <Sparkles className="h-8 w-8" />
                </div>

                <h3 className="mt-4 text-base font-bold text-slate-200">
                  Radiology Workstation Ready
                </h3>
                <p className="mt-1 max-w-sm text-xs text-slate-400 leading-relaxed">
                  Upload a patient chest X-Ray, select a model architecture or the 4-model consensus engine, and initiate deep learning screening.
                </p>

                <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
                  <div className="flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900/60 px-3 py-1 text-[11px] text-slate-400">
                    <FileCheck className="h-3 w-3 text-cyan-400" />
                    <span>ReportLab PDF Generator</span>
                  </div>
                  <div className="flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900/60 px-3 py-1 text-[11px] text-slate-400">
                    <ShieldCheck className="h-3 w-3 text-emerald-400" />
                    <span>Grad-CAM Attention Maps</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
