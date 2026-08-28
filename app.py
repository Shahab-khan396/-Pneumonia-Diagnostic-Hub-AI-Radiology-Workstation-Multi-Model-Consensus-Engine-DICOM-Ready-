"""
Pneumonia Diagnostic Hub • AI Radiology Workstation API Engine
High-performance FastAPI microservice powering multi-model pneumonia screening,
explainable AI (Grad-CAM), DICOM ingestion, and clinical consensus reporting.
"""

import os
import sys
import uuid
import base64
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

# Suppress TensorFlow verbose logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ZeroGPU compatibility hook for Hugging Face Spaces
try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except Exception:
    def GPU_DECORATOR(func):
        return func

from config import (
    BASE_DIR,
    STATIC_DIR,
    UPLOAD_FOLDER,
    SAMPLES_DIR,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    SAMPLES_CATALOG,
    ALLOWED_EXTENSIONS
)
from core.preprocessor import preprocess_image
from core.model_manager import get_model_manager
from core.ensemble import run_multi_model_comparison
from core.dicom_parser import is_dicom_file, parse_dicom_file
from core.report_generator import generate_clinical_pdf_report
from core.sample_manager import ensure_samples_generated, list_sample_catalog, get_sample_info

# Initialize sample catalog files on startup
ensure_samples_generated()

# ─── Initialize FastAPI App ───────────────────────────────────────────────────
app = FastAPI(
    title="Pneumonia Diagnostic Hub • AI Engine",
    description="Multi-Model Consensus & Explainable AI Radiology Workstation API",
    version="2.4.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ─── CORS Middleware ─────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files (for uploads, Grad-CAM overlays, sample studies, and PDFs)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── API Router ───────────────────────────────────────────────────────────────
api_router = APIRouter()


@api_router.get("/")
def root_info():
    return {
        "service": "Pneumonia Diagnostic Hub • AI Radiology Engine",
        "version": "2.4.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/hub_api/health",
            "models": "/hub_api/models",
            "samples": "/hub_api/samples",
            "predict": "/hub_api/predict",
            "compare": "/hub_api/compare"
        }
    }


@api_router.get("/health")
@api_router.get("/hub_api/health")
@api_router.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Pneumonia-Diagnostic-Hub-Engine",
        "models_available": list(AVAILABLE_MODELS.keys()),
        "default_model": DEFAULT_MODEL
    }


@api_router.get("/models")
@api_router.get("/hub_api/models")
@api_router.get("/api/v1/models")
def get_models_catalog():
    return {
        "success": True,
        "models": list(AVAILABLE_MODELS.values()),
        "default_model": DEFAULT_MODEL
    }


@api_router.get("/samples")
@api_router.get("/hub_api/samples")
@api_router.get("/api/v1/samples")
def get_samples_catalog():
    samples = list_sample_catalog()
    return {
        "success": True,
        "samples": samples
    }


@api_router.post("/predict")
@api_router.post("/hub_api/predict")
@api_router.post("/api/v1/predict")
@GPU_DECORATOR
def predict_endpoint(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    model_choice: str = Form("mobilenet"),
    explain: str = Form("true"),
    generate_report: str = Form("true"),
    patient_id: Optional[str] = Form(None),
    patient_age: Optional[str] = Form(None),
    patient_gender: Optional[str] = Form(None),
    clinical_history: Optional[str] = Form(None),
    referring_physician: Optional[str] = Form(None),
):
    scan_id = uuid.uuid4().hex[:8].upper()
    temp_dir = Path(tempfile.gettempdir()) / "pneumonia_hub"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file: Optional[Path] = None

    try:
        # 1. Resolve Input File
        if sample_id and sample_id in SAMPLES_CATALOG:
            sample_meta = get_sample_info(sample_id)
            if not sample_meta:
                raise HTTPException(status_code=404, detail="Sample study not found.")
            input_path = sample_meta["path"]
            filename = sample_meta["filename"]
        elif file is not None:
            raw_bytes = file.file.read()
            if not raw_bytes:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            temp_file = temp_dir / f"{scan_id}_{file.filename or 'scan.jpg'}"
            temp_file.write_bytes(raw_bytes)
            input_path = temp_file
            filename = file.filename or "radiograph.jpg"
        else:
            raise HTTPException(status_code=400, detail="Either 'file' or 'sample_id' must be provided.")

        # 2. Check for DICOM format
        dicom_meta = None
        if is_dicom_file(input_path):
            _, dicom_meta, converted_jpg = parse_dicom_file(input_path)
            input_path = converted_jpg

        # 3. Preprocess Image
        img_tensor = preprocess_image(input_path)
        manager = get_model_manager()

        # 4. Run Model Prediction + Grad-CAM
        should_explain = (explain.lower() in ["true", "1", "yes"])
        res = manager.predict(
            model_id=model_choice,
            image_tensor=img_tensor,
            generate_cam=should_explain,
            original_image_path=input_path,
            base_filename=f"{scan_id}_{model_choice}.jpg"
        )

        # 5. Encode Grad-CAM overlay to base64
        cam_b64 = None
        if res.get("has_gradcam") and res.get("gradcam_overlay_url"):
            cam_path = STATIC_DIR / res["gradcam_overlay_url"].replace("/static/", "").lstrip("/")
            if cam_path.exists():
                cam_b64 = base64.b64encode(cam_path.read_bytes()).decode("utf-8")

        # 6. Generate Clinical PDF Report if requested
        report_url = None
        if generate_report.lower() in ["true", "1", "yes"]:
            patient_meta = {
                "patient_id": patient_id or (dicom_meta.get("patient_id") if dicom_meta else f"PT-{scan_id}"),
                "patient_age": patient_age or (dicom_meta.get("patient_age") if dicom_meta else "N/A"),
                "patient_gender": patient_gender or (dicom_meta.get("patient_gender") if dicom_meta else "N/A"),
                "clinical_history": clinical_history or "Chest screening for respiratory infection / pneumonia.",
                "referring_physician": referring_physician or "Staff Radiologist",
            }
            gradcam_p = (STATIC_DIR / res["gradcam_overlay_url"].replace("/static/", "").lstrip("/")) if res.get("gradcam_overlay_url") else None
            report_pdf_path = generate_clinical_pdf_report(
                scan_id=scan_id,
                prediction_data=res,
                original_image_path=input_path,
                gradcam_overlay_path=gradcam_p,
                patient_metadata=patient_meta,
                output_dir=UPLOAD_FOLDER
            )
            report_url = f"/static/uploads/{report_pdf_path.name}"

        res["success"] = True
        res["scan_id"] = scan_id
        res["gradcam_overlay_b64"] = cam_b64
        res["report_pdf_url"] = report_url
        res["dicom_metadata"] = dicom_meta
        return res

    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


@api_router.post("/compare")
@api_router.post("/hub_api/compare")
@api_router.post("/api/v1/compare")
@GPU_DECORATOR
def compare_endpoint(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    explain: str = Form("true"),
    generate_report: str = Form("true"),
    patient_id: Optional[str] = Form(None),
    patient_age: Optional[str] = Form(None),
    patient_gender: Optional[str] = Form(None),
    clinical_history: Optional[str] = Form(None),
    referring_physician: Optional[str] = Form(None),
):
    scan_id = uuid.uuid4().hex[:8].upper()
    temp_dir = Path(tempfile.gettempdir()) / "pneumonia_hub"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file: Optional[Path] = None

    try:
        # 1. Resolve Input File
        if sample_id and sample_id in SAMPLES_CATALOG:
            sample_meta = get_sample_info(sample_id)
            if not sample_meta:
                raise HTTPException(status_code=404, detail="Sample study not found.")
            input_path = sample_meta["path"]
            filename = sample_meta["filename"]
        elif file is not None:
            raw_bytes = file.file.read()
            if not raw_bytes:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            temp_file = temp_dir / f"{scan_id}_{file.filename or 'scan.jpg'}"
            temp_file.write_bytes(raw_bytes)
            input_path = temp_file
            filename = file.filename or "radiograph.jpg"
        else:
            raise HTTPException(status_code=400, detail="Either 'file' or 'sample_id' must be provided.")

        # 2. Check for DICOM format
        dicom_meta = None
        if is_dicom_file(input_path):
            _, dicom_meta, converted_jpg = parse_dicom_file(input_path)
            input_path = converted_jpg

        # 3. Preprocess Image
        img_tensor = preprocess_image(input_path)

        # 4. Run 4-Model Weighted Ensemble Comparison
        should_explain = (explain.lower() in ["true", "1", "yes"])
        res = run_multi_model_comparison(
            image_tensor=img_tensor,
            original_image_path=input_path,
            base_filename=f"{scan_id}.jpg",
            generate_cams=should_explain
        )

        # 5. Encode Primary Grad-CAM overlay to base64
        cam_b64 = None
        if res.get("primary_gradcam_overlay_url"):
            cam_path = STATIC_DIR / res["primary_gradcam_overlay_url"].replace("/static/", "").lstrip("/")
            if cam_path.exists():
                cam_b64 = base64.b64encode(cam_path.read_bytes()).decode("utf-8")

        # 6. Generate Clinical PDF Report if requested
        report_url = None
        if generate_report.lower() in ["true", "1", "yes"]:
            patient_meta = {
                "patient_id": patient_id or (dicom_meta.get("patient_id") if dicom_meta else f"PT-{scan_id}"),
                "patient_age": patient_age or (dicom_meta.get("patient_age") if dicom_meta else "N/A"),
                "patient_gender": patient_gender or (dicom_meta.get("patient_gender") if dicom_meta else "N/A"),
                "clinical_history": clinical_history or "Chest screening for respiratory infection / pneumonia.",
                "referring_physician": referring_physician or "Staff Radiologist",
            }
            gradcam_p = (STATIC_DIR / res["primary_gradcam_overlay_url"].replace("/static/", "").lstrip("/")) if res.get("primary_gradcam_overlay_url") else None
            report_pdf_path = generate_clinical_pdf_report(
                scan_id=scan_id,
                prediction_data=res,
                original_image_path=input_path,
                gradcam_overlay_path=gradcam_p,
                patient_metadata=patient_meta,
                output_dir=UPLOAD_FOLDER
            )
            report_url = f"/static/uploads/{report_pdf_path.name}"

        res["success"] = True
        res["scan_id"] = scan_id
        res["gradcam_overlay_b64"] = cam_b64
        res["report_pdf_url"] = report_url
        res["dicom_metadata"] = dicom_meta
        return res

    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


@api_router.get("/reports/{filename}")
@api_router.get("/hub_api/reports/{filename}")
@api_router.get("/api/v1/report/{filename}")
def download_report(filename: str):
    file_path = UPLOAD_FOLDER / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found.")
    return FileResponse(path=str(file_path), media_type="application/pdf", filename=filename)


# Include router in FastAPI application
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)

