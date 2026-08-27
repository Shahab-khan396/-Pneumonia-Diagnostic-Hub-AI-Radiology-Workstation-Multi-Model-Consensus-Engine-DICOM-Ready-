"""
File upload validation — re-implemented for FastAPI without werkzeug dependency.
"""
import uuid
from pathlib import Path
from typing import Tuple

from config import get_settings


def is_allowed_extension(filename: str) -> bool:
    """Return True if the filename has an allowed extension."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in get_settings().allowed_extensions


def safe_filename(filename: str) -> str:
    """
    Sanitize a filename: strip path separators, replace spaces, lowercase extension.
    Does NOT use werkzeug — safe for FastAPI on all platforms.
    """
    # Strip any directory components
    name = Path(filename).name
    # Replace problematic characters with underscores
    for ch in (' ', '\\', '/', ':', '*', '?', '"', '<', '>', '|'):
        name = name.replace(ch, '_')
    return name or "upload.jpg"


def get_safe_filepath(filename: str, target_dir: Path) -> Tuple[str, Path]:
    """
    Generate a collision-free, sanitized filename and its absolute path.

    Returns:
        (safe_name, absolute_path)
    """
    clean = safe_filename(filename)
    unique = f"{uuid.uuid4().hex[:10]}_{clean}"
    target_dir.mkdir(parents=True, exist_ok=True)
    return unique, target_dir / unique


def validate_upload_bytes(data: bytes, filename: str) -> Tuple[bool, str]:
    """
    Validate raw uploaded file bytes.

    Returns:
        (is_valid, error_message)  — error_message is "" when valid.
    """
    settings = get_settings()

    if not data:
        return False, "No file data received."

    if len(data) > settings.max_upload_bytes:
        mb = settings.max_upload_bytes / 1_048_576
        return False, f"File size exceeds maximum allowed limit of {mb:.0f} MB."

    if not is_allowed_extension(filename):
        allowed = ", ".join(sorted(settings.allowed_extensions)).upper()
        return False, f"Invalid file format. Supported formats: {allowed}."

    return True, ""
