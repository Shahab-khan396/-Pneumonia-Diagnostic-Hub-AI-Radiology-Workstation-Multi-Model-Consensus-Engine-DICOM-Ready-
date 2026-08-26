import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


def is_dicom_file(file_source: Union[str, Path, bytes]) -> bool:
    """
    Check if a file or byte stream is a valid DICOM medical object.
    Checks for .dcm extension or standard 'DICM' header signature at offset 128.
    """
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if path.suffix.lower() == ".dcm":
            return True
        if path.exists() and path.stat().st_size > 132:
            try:
                with open(path, "rb") as f:
                    f.seek(128)
                    return f.read(4) == b"DICM"
            except Exception:
                return False
    elif isinstance(file_source, bytes):
        if len(file_source) > 132:
            return file_source[128:132] == b"DICM"
    return False


def parse_dicom_file(
    dicom_path: Union[str, Path],
    output_jpg_path: Optional[Path] = None
) -> Tuple[np.ndarray, Dict[str, Any], Path]:
    """
    Parse a DICOM radiograph (.dcm), extract clinical acquisition metadata,
    and convert the raw high-bitdepth pixel array into an optimized 8-bit radiograph image.
    
    Args:
        dicom_path: Path to the .dcm DICOM file on disk.
        output_jpg_path: Optional output path to save the converted JPEG image.
        
    Returns:
        Tuple of (converted_bgr_image, metadata_dict, converted_image_path)
    """
    dicom_path = Path(dicom_path)
    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM file not found at: {dicom_path}")

    # Read DICOM dataset with force=True for non-standard clinical headers
    ds = pydicom.dcmread(str(dicom_path), force=True)

    # Extract metadata tags
    metadata: Dict[str, Any] = {
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

    # Extract and process raw pixel data
    try:
        pixel_array = ds.pixel_array.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Could not decode pixel data from DICOM object: {e}")

    # Handle Rescale Slope & Intercept
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        pixel_array = pixel_array * slope + intercept

    # Handle Photometric Interpretation: MONOCHROME1 (0 is white, max is black) -> invert to standard MONOCHROME2
    if metadata["photometric"] == "MONOCHROME1":
        pixel_array = np.max(pixel_array) - pixel_array

    # VOI LUT Windowing for Pulmonary Anatomy
    try:
        # Attempt standard VOI LUT windowing
        windowed = apply_voi_lut(pixel_array, ds)
        norm_pixels = (windowed - np.min(windowed)) / (np.max(windowed) - np.min(windowed) + 1e-10)
    except Exception:
        # Fallback to standard min-max pulmonary normalization
        p_min, p_max = np.min(pixel_array), np.max(pixel_array)
        if p_max > p_min:
            norm_pixels = (pixel_array - p_min) / (p_max - p_min)
        else:
            norm_pixels = np.zeros_like(pixel_array)

    # Convert to 8-bit unsigned integer [0, 255]
    img_8bit = np.uint8(255 * np.clip(norm_pixels, 0.0, 1.0))

    # Convert to 3-channel BGR
    if img_8bit.ndim == 2:
        img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    elif img_8bit.ndim == 3 and img_8bit.shape[2] == 1:
        img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img_8bit

    # Determine output path
    if output_jpg_path is None:
        output_jpg_path = dicom_path.with_suffix(".jpg")
    
    cv2.imwrite(str(output_jpg_path), img_bgr)

    return img_bgr, metadata, output_jpg_path
