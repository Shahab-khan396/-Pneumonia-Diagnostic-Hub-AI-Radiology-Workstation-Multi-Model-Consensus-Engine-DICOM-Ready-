"""GET /api/v1/samples — Sample radiograph catalog."""
from fastapi import APIRouter

from core.sample_manager import list_sample_catalog
from schemas import SampleItem, SamplesResponse

router = APIRouter(tags=["Samples"])


@router.get("/samples", response_model=SamplesResponse)
async def get_samples():
    """Return the catalog of pre-generated synthetic CXR sample images."""
    catalog = list_sample_catalog()
    return SamplesResponse(samples=[SampleItem(**item) for item in catalog])
