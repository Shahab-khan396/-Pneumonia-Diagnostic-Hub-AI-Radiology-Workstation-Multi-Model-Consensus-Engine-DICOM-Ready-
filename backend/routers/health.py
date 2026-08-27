"""Health check router."""
from fastapi import APIRouter

from config import get_settings
from core.hf_client import ping_hf_space
from schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    settings = get_settings()
    hf_status = await ping_hf_space()
    detail = hf_status.get("detail")
    hf_space_url = detail.get("hf_space_url") if isinstance(detail, dict) else settings.hf_space_url
    return HealthResponse(
        status="healthy",
        service="Pneumonia-Diagnostic-Hub-API",
        version="2.4.0",
        features=[
            "multi_model_inference",
            "gradcam_xai",
            "ensemble_consensus",
            "dicom_parser",
            "clinical_pdf_reporting",
            "sample_radiograph_library",
        ],
        hf_space_url=hf_space_url or settings.hf_space_url,
    )


