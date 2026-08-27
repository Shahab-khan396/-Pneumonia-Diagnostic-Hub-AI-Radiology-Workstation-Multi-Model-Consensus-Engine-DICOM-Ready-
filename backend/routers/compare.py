"""
POST /api/v1/compare — 4-model ensemble inference.
Same flow as predict.py but calls the HF Space /compare endpoint.
"""
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import get_settings
from core.dicom_parser import is_dicom_file, parse_dicom_bytes
from core.hf_client import call_hf_compare
from core.report_generator import generate_clinical_pdf_report
from core.validator import validate_upload_bytes
from schemas import EnsembleResponse, ErrorResponse, ModelResult

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(tags=["Inference"])


@router.post(
    "/compare",
    response_model=EnsembleResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
@limiter.limit("5/minute")
async def compare(
    request: Request,
    file: UploadFile = File(..., description="CXR image (PNG/JPG/JPEG/WEBP) or DICOM (.dcm)"),
    explain: bool = Form(True, description="Generate Grad-CAM for the consensus model"),
    generate_report: bool = Form(True, description="Generate downloadable PDF report"),
    patient_id: Optional[str] = Form(None),
    patient_age: Optional[str] = Form(None),
    patient_gender: Optional[str] = Form(None),
    clinical_history: Optional[str] = Form(None),
    referring_physician: Optional[str] = Form(None),
):
    settings = get_settings()
    scan_id  = uuid.uuid4().hex[:8].upper()

    # ── 1. Read & validate ────────────────────────────────────────────────────
    raw_bytes = await file.read()
    ok, err   = validate_upload_bytes(raw_bytes, file.filename or "upload.bin")
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err)

    # ── 2. DICOM ──────────────────────────────────────────────────────────────
    dicom_meta = None
    image_bytes = raw_bytes
    upload_filename = file.filename or "scan.jpg"

    if is_dicom_file(raw_bytes):
        try:
            result       = parse_dicom_bytes(raw_bytes)
            image_bytes  = result["jpeg_bytes"]
            dicom_meta   = result["metadata"]
            upload_filename = Path(upload_filename).stem + "_converted.jpg"
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"DICOM parsing failed: {exc}",
            )

    # ── 3. HF Space ensemble call ─────────────────────────────────────────────
    try:
        hf_result = await call_hf_compare(
            image_bytes=image_bytes,
            filename=upload_filename,
            explain=explain,
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Inference engine returned {exc.response.status_code}: {exc.response.text[:200]}",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Ensemble inference timed out. HF Space may be cold-starting; retry in 30 seconds.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Inference engine unreachable: {exc}",
        )

    if not hf_result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=hf_result.get("error", "Unknown ensemble error from HF Space."),
        )

    # ── 4. PDF ────────────────────────────────────────────────────────────────
    report_url: Optional[str] = None
    if generate_report:
        patient_meta = {
            "patient_id": patient_id,
            "patient_age": patient_age,
            "patient_gender": patient_gender,
            "clinical_history": clinical_history,
            "referring_physician": referring_physician,
            **(dicom_meta or {}),
        }
        try:
            pdf_path = generate_clinical_pdf_report(
                scan_id=scan_id,
                prediction_data={**hf_result, "is_ensemble": True},
                original_image_bytes=image_bytes,
                gradcam_b64=hf_result.get("gradcam_overlay_b64"),
                patient_metadata=patient_meta,
                output_dir=Path(settings.upload_dir),
            )
            report_url = f"/api/v1/report/{pdf_path.name}"
        except Exception:
            pass

    # ── 5. Build response ─────────────────────────────────────────────────────
    models_breakdown = [
        ModelResult(**m) for m in hf_result.get("models_breakdown", [])
    ]

    return EnsembleResponse(
        scan_id=scan_id,
        consensus_verdict=hf_result.get("consensus_verdict", "UNKNOWN"),
        consensus_confidence=hf_result.get("consensus_confidence", 0.0),
        consensus_probabilities=hf_result.get("consensus_probabilities", {}),
        agreement_level=hf_result.get("agreement_level", "UNKNOWN"),
        agreement_text=hf_result.get("agreement_text", ""),
        models_breakdown=models_breakdown,
        total_inference_time_ms=hf_result.get("total_inference_time_ms", 0.0),
        has_gradcam=hf_result.get("has_gradcam", False),
        gradcam_overlay_url=hf_result.get("gradcam_overlay_url"),
        filename=upload_filename,
        report_pdf_url=report_url,
        dicom_metadata=dicom_meta,
    )
