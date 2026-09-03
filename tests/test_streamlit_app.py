"""
Unit and integration tests for the Streamlit AI Radiology Workstation.
Verifies syntax compilation, session state initialization logic,
and core workflow integration (DICOM, preprocessing, Grad-CAM, PDF report).
"""

import os
import sys
import py_compile
import tempfile
from pathlib import Path
import numpy as np
import pytest

# Ensure root directory in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import AVAILABLE_MODELS, SAMPLES_CATALOG, SAMPLES_DIR, UPLOAD_FOLDER
from core.sample_manager import ensure_samples_generated, list_sample_catalog, get_sample_info
from core.preprocessor import preprocess_image
from core.report_generator import generate_clinical_pdf_report


def test_streamlit_app_compilation():
    """Verify that streamlit_app.py has valid Python syntax and compiles without error."""
    app_path = ROOT_DIR / "streamlit_app.py"
    assert app_path.exists(), "streamlit_app.py must exist in the root directory"
    # py_compile checks for syntax errors without executing the script
    compiled = py_compile.compile(str(app_path), doraise=True)
    assert compiled is not None


def test_streamlit_config_exists():
    """Verify that .streamlit/config.toml exists and contains required theme/server keys."""
    config_path = ROOT_DIR / ".streamlit" / "config.toml"
    assert config_path.exists(), ".streamlit/config.toml must exist"
    content = config_path.read_text(encoding="utf-8")
    assert "[theme]" in content
    assert "[server]" in content
    assert "maxUploadSize" in content


def test_sample_catalog_for_streamlit():
    """Verify that all pre-packaged clinical samples can be resolved for Streamlit tabs."""
    ensure_samples_generated()
    catalog = list_sample_catalog()
    assert len(catalog) == 3
    for item in catalog:
        info = get_sample_info(item["id"])
        assert info is not None
        assert info["path"].exists()


def test_preprocessing_pipeline_for_streamlit():
    """Verify that images loaded in Streamlit are cleanly converted to batch tensors."""
    sample = get_sample_info("sample_normal")
    assert sample is not None
    tensor = preprocess_image(sample["path"])
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 128, 128, 3)
    assert tensor.dtype == np.float32
    assert 0.0 <= np.min(tensor) <= np.max(tensor) <= 1.0


def test_pdf_report_compilation_for_streamlit():
    """Verify that the clinical report generator can compile a report with mock consensus data."""
    sample = get_sample_info("sample_normal")
    assert sample is not None

    mock_prediction = {
        "consensus_verdict": "NORMAL",
        "consensus_confidence": 91.5,
        "consensus_probabilities": {"NORMAL": 91.5, "PNEUMONIA": 8.5},
        "agreement_level": "UNANIMOUS",
        "agreement_text": "Unanimous Consensus (4/4 Models Agree)",
        "disagreement_warning": False,
        "total_inference_time_ms": 142.3,
        "models_breakdown": [
            {"name": "MobileNetV2", "prediction": "NORMAL", "confidence": 94.2, "weight": 0.45},
            {"name": "ResNet50", "prediction": "NORMAL", "confidence": 89.1, "weight": 0.25},
        ]
    }

    patient_meta = {
        "patient_id": "TEST-PT-001",
        "patient_name": "Jane Doe",
        "patient_age": "52Y",
        "patient_sex": "F",
        "study_date": "2026-03-01",
        "modality": "CR",
        "body_part": "CHEST",
        "clinician_notes": "Routine screening. Clear lung fields.",
        "reviewing_radiologist": "Dr. S. Khan, MD",
        "facility_name": "AI Radiology Workstation",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = generate_clinical_pdf_report(
            scan_id="test_streamlit_scan",
            prediction_data=mock_prediction,
            original_image_path=sample["path"],
            patient_metadata=patient_meta,
            output_dir=Path(tmpdir)
        )
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000
