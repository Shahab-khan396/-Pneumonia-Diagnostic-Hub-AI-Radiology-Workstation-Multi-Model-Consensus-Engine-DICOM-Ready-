"""Core package containing ML engine, preprocessor, validator, Grad-CAM XAI, ensemble, sample manager, DICOM parser, and PDF reporting."""
from .validator import validate_image_file, get_safe_filepath, is_allowed_file
from .preprocessor import preprocess_image
from .model_manager import ModelManager, get_model_manager
from .gradcam import (
    compute_gradcam_heatmap,
    create_gradcam_overlay,
    save_gradcam_visualizations,
)
from .sample_manager import (
    list_sample_catalog,
    get_sample_info,
    ensure_samples_generated,
)
from .ensemble import run_multi_model_comparison
from .report_generator import generate_clinical_pdf_report
from .dicom_parser import parse_dicom_file, is_dicom_file

__all__ = [
    "validate_image_file",
    "get_safe_filepath",
    "is_allowed_file",
    "preprocess_image",
    "ModelManager",
    "get_model_manager",
    "compute_gradcam_heatmap",
    "create_gradcam_overlay",
    "save_gradcam_visualizations",
    "list_sample_catalog",
    "get_sample_info",
    "ensure_samples_generated",
    "run_multi_model_comparison",
    "generate_clinical_pdf_report",
    "parse_dicom_file",
    "is_dicom_file",
]
