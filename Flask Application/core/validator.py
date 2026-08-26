import os
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename
from config import ALLOWED_EXTENSIONS, UPLOAD_FOLDER


def is_allowed_file(filename: str) -> bool:
    """Check if the filename has an allowed extension."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def get_safe_filepath(filename: str) -> tuple[str, Path]:
    """
    Generate a safe, collision-free filename and destination path.
    Returns (safe_filename, absolute_path).
    """
    clean_name = secure_filename(filename)
    if not clean_name:
        clean_name = "upload.jpg"
    
    unique_prefix = uuid.uuid4().hex[:10]
    safe_name = f"{unique_prefix}_{clean_name}"
    
    upload_dir = Path(UPLOAD_FOLDER)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    dest_path = upload_dir / safe_name
    return safe_name, dest_path


def validate_image_file(file_storage) -> tuple[bool, str]:
    """
    Validate the uploaded Flask FileStorage object.
    Returns (is_valid, error_message).
    """
    if file_storage is None or file_storage.filename == "":
        return False, "No file uploaded. Please select a Chest X-Ray image."
    
    if not is_allowed_file(file_storage.filename):
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return False, f"Invalid file format. Supported formats: {allowed_list}"
    
    return True, ""
