"""GET /api/v1/report/{filename} — Serve generated PDF reports."""
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from config import get_settings

router = APIRouter(tags=["Reports"])


@router.get("/report/{filename}")
async def download_report(filename: str):
    """
    Stream a previously generated PDF report to the client.
    Filename must be a bare filename (no path separators) for security.
    """
    # Reject any path traversal attempts
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename.")

    pdf_path = Path(get_settings().upload_dir) / filename

    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{filename}' not found or has expired.",
        )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
