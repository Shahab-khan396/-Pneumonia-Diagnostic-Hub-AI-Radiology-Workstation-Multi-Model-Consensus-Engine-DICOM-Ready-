export interface DicomMetadata {
  is_dicom: boolean;
  patient_id?: string;
  patient_name?: string;
  patient_age?: string;
  patient_sex?: string;
  study_date?: string;
  modality?: string;
  body_part?: string;
  manufacturer?: string;
  kvp?: string;
  exposure_time?: string;
  photometric?: string;
  rows?: number;
  columns?: number;
}

export interface ModelResult {
  id: string;
  name: string;
  parameters?: string;
  weight: number;
  prediction: 'NORMAL' | 'PNEUMONIA';
  confidence: number;
  inference_time_ms: number;
  has_gradcam?: boolean;
  gradcam_overlay_url?: string;
}

export interface PredictResponse {
  success: boolean;
  scan_id: string;
  prediction: 'NORMAL' | 'PNEUMONIA';
  confidence: number;
  probabilities: {
    NORMAL: number;
    PNEUMONIA: number;
  };
  raw_probabilities?: {
    NORMAL: number;
    PNEUMONIA: number;
  };
  model_id: string;
  model_name: string;
  model_parameters?: string;
  model_badge?: string;
  target_conv_layer?: string;
  inference_time_ms: number;
  has_gradcam: boolean;
  gradcam_overlay_url?: string;
  gradcam_heatmap_url?: string;
  gradcam_composite_url?: string;
  image_url?: string;
  report_pdf_url?: string;
  filename?: string;
  dicom_metadata?: DicomMetadata | null;
  error?: string;
}

export interface EnsembleResponse extends PredictResponse {
  is_ensemble: boolean;
  consensus_verdict: 'NORMAL' | 'PNEUMONIA';
  consensus_confidence: number;
  consensus_probabilities: {
    NORMAL: number;
    PNEUMONIA: number;
  };
  agreement_level: 'UNANIMOUS' | 'STRONG_MAJORITY' | 'SPLIT_DECISION';
  agreement_text: string;
  models_breakdown: ModelResult[];
  total_inference_time_ms: number;
}

export interface SampleStudy {
  id: string;
  label: string;
  filename: string;
  description: string;
  category: 'normal' | 'bacterial' | 'viral';
  image_url: string;
}

export interface PatientInfo {
  patientId: string;
  patientAge: string;
  patientGender: string;
  clinicalHistory: string;
  referringPhysician: string;
}
