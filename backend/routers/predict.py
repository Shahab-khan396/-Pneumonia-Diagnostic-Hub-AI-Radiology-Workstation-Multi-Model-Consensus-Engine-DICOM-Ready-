"""
POST /api/v1/predict — Single-model inference.
Forwards the uploaded file to the HF Space, post-processes the result,
and optionally generates a clinical PDF report.
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
from core.hf_client import call_hf_predict
from core.report_generator import generate_clinical_pdf_report
from core.validator import validate_upload_bytes
from schemas import ErrorResponse, PredictResponse

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(tags=["Inference"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
@limiter.limit("10/minute")
async def predict(
    request: Request,                                   # required by SlowAPI
    file: UploadFile = File(..., description="CXR image (PNG/JPG/JPEG/WEBP) or DICOM (.dcm)"),
    model_choice: str = Form("mobilenet", description="mobilenet | resnet50 | efficientnet | VGG19"),
    explain: bool = Form(True,  description="Generate Grad-CAM heatmap"),
    generate_report: bool = Form(True, description="Generate downloadable PDF report"),
    patient_id: Optional[str] = Form(None),
    patient_age: Optional[str] = Form(None),
    patient_gender: Optional[str] = Form(None),
    clinical_history: Optional[str] = Form(None),
    referring_physician: Optional[str] = Form(None),
):
    settings = get_settings()
    scan_id  = uuid.uuid4().hex[:8].upper()

    # ── 1. Read & validate upload ──────────────────────────────────────────────
    raw_bytes = await file.read()
    ok, err   = validate_upload_bytes(raw_bytes, file.filename or "upload.bin")
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err)

    # ── 2. DICOM detection & conversion ───────────────────────────────────────
    dicom_meta = None
    image_bytes = raw_bytes
    upload_filename = file.filename or "scan.jpg"

    if is_dicom_file(raw_bytes):
        try:
            result = parse_dicom_bytes(raw_bytes)
            image_bytes  = result["jpeg_bytes"]
            dicom_meta   = result["metadata"]
            upload_filename = Path(upload_filename).stem + "_converted.jpg"
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"DICOM parsing failed: {exc}",
            )

    # ── 3. Forward to HF Space ────────────────────────────────────────────────
    try:
        hf_result = await call_hf_predict(
            image_bytes=image_bytes,
            filename=upload_filename,
            model_choice=model_choice,
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
            detail="Inference engine timed out. The HF Space may be cold-starting; please retry in 30 seconds.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Inference engine unreachable: {exc}",
        )

    if not hf_result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=hf_result.get("error", "Unknown inference error from HF Space."),
        )

    # ── 4. PDF report generation ───────────────────────────────────────────────
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
                prediction_data=hf_result,
                original_image_bytes=image_bytes,
                gradcam_b64=hf_result.get("gradcam_overlay_b64"),
                patient_metadata=patient_meta,
                output_dir=Path(settings.upload_dir),
            )
            report_url = f"/api/v1/report/{pdf_path.name}"
        except Exception:
            pass  # PDF failure must not kill the inference response

    # ── 5. Build response ─────────────────────────────────────────────────────
    return PredictResponse(
        scan_id=scan_id,
        prediction=hf_result.get("prediction", "UNKNOWN"),
        confidence=hf_result.get("confidence", 0.0),
        probabilities=hf_result.get("probabilities", {}),
        raw_probabilities=hf_result.get("raw_probabilities"),
        model_id=hf_result.get("model_id", model_choice),
        model_name=hf_result.get("model_name", model_choice),
        model_parameters=hf_result.get("model_parameters"),
        model_badge=hf_result.get("model_badge"),
        target_conv_layer=hf_result.get("target_conv_layer"),
        inference_time_ms=hf_result.get("inference_time_ms", 0.0),
        has_gradcam=hf_result.get("has_gradcam", False),
        gradcam_overlay_url=hf_result.get("gradcam_overlay_url"),
        gradcam_heatmap_url=hf_result.get("gradcam_heatmap_url"),
        gradcam_composite_url=hf_result.get("gradcam_composite_url"),
        filename=upload_filename,
        report_pdf_url=report_url,
        dicom_metadata=dicom_meta,
    )
