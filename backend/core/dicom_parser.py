"""
DICOM parser — ported from Flask Application/core/dicom_parser.py.
Adapted for FastAPI: path-free byte-stream API, async-safe (no blocking I/O in async routes).
"""
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


# ─── Detection ────────────────────────────────────────────────────────────────

def is_dicom_file(file_source: Union[str, Path, bytes]) -> bool:
    """
    Detect whether the source is a valid DICOM object.
    Accepts a filesystem path (str/Path) or raw bytes.
    Checks for .dcm extension OR the standard 'DICM' magic header at offset 128.
    """
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if path.suffix.lower() == ".dcm":
            return True
        if path.exists() and path.stat().st_size > 132:
            try:
                with open(path, "rb") as fh:
                    fh.seek(128)
                    return fh.read(4) == b"DICM"
            except Exception:
                return False
    elif isinstance(file_source, bytes):
        if len(file_source) > 132:
            return file_source[128:132] == b"DICM"
    return False


# ─── Core parsing ─────────────────────────────────────────────────────────────

def _extract_metadata(ds: pydicom.Dataset) -> Dict[str, Any]:
    return {
        "is_dicom": True,
        "patient_id": str(getattr(ds, "PatientID", "PT-DICOM-ANON")),
        "patient_name": str(getattr(ds, "PatientName", "Anonymous")),
        "patient_age": str(getattr(ds, "PatientAge", "N/A")),
        "patient_sex": str(getattr(ds, "PatientSex", "N/A")),
        "study_date": str(getattr(ds, "StudyDate", "N/A")),
        "modality": str(getattr(ds, "Modality", "CR")),
        "body_part": str(getattr(ds, "BodyPartExamined", "CHEST")),
        "manufacturer": str(getattr(ds, "Manufacturer", "Digital Radiography")),
        "kvp": str(getattr(ds, "KVP", "N/A")),
        "exposure_time": str(getattr(ds, "ExposureTime", "N/A")),
        "photometric": str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
    }


def _pixel_array_to_bgr(ds: pydicom.Dataset, metadata: Dict[str, Any]) -> np.ndarray:
    """Convert raw DICOM pixel array → 8-bit 3-channel BGR image."""
    try:
        pixel_array = ds.pixel_array.astype(np.float32)
    except Exception as exc:
        raise ValueError(f"Could not decode DICOM pixel data: {exc}") from exc

    # Rescale slope / intercept
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        pixel_array = pixel_array * slope + intercept

    # MONOCHROME1 → invert to MONOCHROME2 (0 = black)
    if metadata["photometric"] == "MONOCHROME1":
        pixel_array = np.max(pixel_array) - pixel_array

    # VOI LUT windowing
    try:
        windowed = apply_voi_lut(pixel_array, ds)
        norm = (windowed - np.min(windowed)) / (np.max(windowed) - np.min(windowed) + 1e-10)
    except Exception:
        p_min, p_max = np.min(pixel_array), np.max(pixel_array)
        norm = (pixel_array - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(pixel_array)

    img_8bit = np.uint8(255 * np.clip(norm, 0.0, 1.0))

    # Ensure 3-channel BGR
    if img_8bit.ndim == 2:
        return cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    if img_8bit.ndim == 3 and img_8bit.shape[2] == 1:
        return cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    return img_8bit


# ─── Public API ───────────────────────────────────────────────────────────────

def parse_dicom_bytes(data: bytes) -> Dict[str, Any]:
    """
    Parse a DICOM file from raw bytes (in-memory, no persistent temp files).

    Returns:
        {
            "metadata": dict,       # Clinical DICOM metadata
            "jpeg_bytes": bytes,    # Converted 8-bit JPEG image bytes
        }
    """
    # Write to a short-lived temp file (pydicom requires file-like access for pixel decoding)
    tmp_path = Path(tempfile.gettempdir()) / f"pdh_{uuid.uuid4().hex[:10]}.dcm"
    tmp_path.write_bytes(data)
    try:
        ds = pydicom.dcmread(str(tmp_path), force=True)
        metadata = _extract_metadata(ds)
        img_bgr = _pixel_array_to_bgr(ds, metadata)
        _, jpeg_buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return {
            "metadata": metadata,
            "jpeg_bytes": jpeg_buf.tobytes(),
        }
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def parse_dicom_path(
    dicom_path: Union[str, Path],
    output_jpg_path: Optional[Path] = None,
) -> Tuple[np.ndarray, Dict[str, Any], Path]:
    """
    Parse a DICOM file from a filesystem path and optionally save the JPEG.

    Returns:
        (bgr_image, metadata_dict, saved_jpg_path)
    """
    dicom_path = Path(dicom_path)
    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

    ds = pydicom.dcmread(str(dicom_path), force=True)
    metadata = _extract_metadata(ds)
    img_bgr = _pixel_array_to_bgr(ds, metadata)

    out_path = output_jpg_path or dicom_path.with_suffix(".jpg")
    cv2.imwrite(str(out_path), img_bgr)
    return img_bgr, metadata, out_path
