"""
Pydantic v2 response schemas — single source of truth for all API response shapes.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Shared ───────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    success: bool = False
    error: str


class DicomMetadata(BaseModel):
    is_dicom: bool = True
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    patient_sex: Optional[str] = None
    study_date: Optional[str] = None
    modality: Optional[str] = None
    body_part: Optional[str] = None
    manufacturer: Optional[str] = None
    kvp: Optional[str] = None
    exposure_time: Optional[str] = None
    photometric: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None


# ─── Single-model inference ───────────────────────────────────────────────────

class PredictResponse(BaseModel):
    success: bool = True
    scan_id: str
    prediction: str                             # "NORMAL" | "PNEUMONIA"
    confidence: float
    probabilities: Dict[str, float]             # {"NORMAL": x, "PNEUMONIA": y}
    raw_probabilities: Optional[Dict[str, float]] = None
    model_id: str
    model_name: str
    model_parameters: Optional[str] = None
    model_badge: Optional[str] = None
    target_conv_layer: Optional[str] = None
    inference_time_ms: float
    has_gradcam: bool = False
    gradcam_overlay_url: Optional[str] = None
    gradcam_heatmap_url: Optional[str] = None
    gradcam_composite_url: Optional[str] = None
    image_url: Optional[str] = None
    report_pdf_url: Optional[str] = None
    filename: Optional[str] = None
    dicom_metadata: Optional[DicomMetadata] = None


# ─── Ensemble inference ───────────────────────────────────────────────────────

class ModelResult(BaseModel):
    id: str
    name: str
    parameters: Optional[str] = None
    weight: float
    prediction: str
    confidence: float
    inference_time_ms: float
    has_gradcam: bool = False
    gradcam_overlay_url: Optional[str] = None


class EnsembleResponse(BaseModel):
    success: bool = True
    scan_id: str
    is_ensemble: bool = True

    # Consensus output
    consensus_verdict: str
    consensus_confidence: float
    consensus_probabilities: Dict[str, float]
    agreement_level: str                        # UNANIMOUS | STRONG_MAJORITY | SPLIT_DECISION
    agreement_text: str

    # Per-model breakdown
    models_breakdown: List[ModelResult]
    total_inference_time_ms: float

    # Shared fields
    has_gradcam: bool = False
    gradcam_overlay_url: Optional[str] = None
    image_url: Optional[str] = None
    report_pdf_url: Optional[str] = None
    filename: Optional[str] = None
    dicom_metadata: Optional[DicomMetadata] = None


# ─── Health check ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    features: List[str]
    hf_space_url: str
    backend: str = "FastAPI + Uvicorn"


# ─── Samples catalog ──────────────────────────────────────────────────────────

class SampleItem(BaseModel):
    id: str
    label: str
    filename: str
    description: str
    category: str
    image_url: str


class SamplesResponse(BaseModel):
    success: bool = True
    samples: List[SampleItem]
