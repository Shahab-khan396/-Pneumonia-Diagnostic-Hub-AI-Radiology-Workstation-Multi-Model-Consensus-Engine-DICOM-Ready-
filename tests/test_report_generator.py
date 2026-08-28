from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.report_generator import generate_clinical_pdf_report
from app import app


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def dummy_images(tmp_path):
    """Create dummy original and overlay images."""
    orig = np.full((128, 128, 3), 100, dtype=np.uint8)
    overlay = np.full((128, 128, 3), 150, dtype=np.uint8)
    
    orig_path = tmp_path / "orig.jpg"
    overlay_path = tmp_path / "overlay.jpg"
    
    cv2.imwrite(str(orig_path), orig)
    cv2.imwrite(str(overlay_path), overlay)
    
    return orig_path, overlay_path


def test_generate_single_model_pdf_report(dummy_images, tmp_path):
    """Verify PDF report generation for single model inference."""
    orig_path, overlay_path = dummy_images
    
    mock_prediction = {
        "is_ensemble": False,
        "prediction": "PNEUMONIA",
        "confidence": 94.5,
        "probabilities": {"NORMAL": 5.5, "PNEUMONIA": 94.5},
        "model_name": "MobileNetV2",
        "model_parameters": "3.5M",
    }
    
    pdf_path = generate_clinical_pdf_report(
        scan_id="TEST-001",
        prediction_data=mock_prediction,
        original_image_path=orig_path,
        gradcam_overlay_path=overlay_path,
        output_dir=tmp_path
    )
    
    assert pdf_path.exists()
    assert pdf_path.suffix == ".pdf"
    assert pdf_path.stat().st_size > 5000  # Generated valid PDF content


def test_generate_ensemble_pdf_report(dummy_images, tmp_path):
    """Verify PDF report generation for multi-model ensemble consensus."""
    orig_path, overlay_path = dummy_images
    
    mock_ensemble = {
        "is_ensemble": True,
        "consensus_verdict": "PNEUMONIA",
        "consensus_confidence": 91.2,
        "consensus_probabilities": {"NORMAL": 8.8, "PNEUMONIA": 91.2},
        "agreement_text": "Unanimous Consensus (4/4 Models in Full Agreement)",
        "models_breakdown": [
            {"name": "MobileNetV2", "parameters": "3.5M", "weight": 0.45, "prediction": "PNEUMONIA", "confidence": 95.0, "inference_time_ms": 32.1},
            {"name": "ResNet50", "parameters": "25.6M", "weight": 0.25, "prediction": "PNEUMONIA", "confidence": 88.0, "inference_time_ms": 45.3},
            {"name": "EfficientNetB0", "parameters": "5.3M", "weight": 0.20, "prediction": "PNEUMONIA", "confidence": 89.5, "inference_time_ms": 40.2},
            {"name": "VGG19", "parameters": "63.1M", "weight": 0.10, "prediction": "PNEUMONIA", "confidence": 92.0, "inference_time_ms": 65.4},
        ]
    }
    
    pdf_path = generate_clinical_pdf_report(
        scan_id="TEST-ENS-002",
        prediction_data=mock_ensemble,
        original_image_path=orig_path,
        gradcam_overlay_path=overlay_path,
        output_dir=tmp_path
    )
    
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 6000


def test_download_report_endpoint(client, dummy_images, tmp_path):
    """Verify report downloading via GET /reports/<filename>."""
    orig_path, overlay_path = dummy_images
    mock_pred = {"is_ensemble": False, "prediction": "NORMAL", "confidence": 98.0, "probabilities": {"NORMAL": 98.0, "PNEUMONIA": 2.0}}
    
    pdf_path = generate_clinical_pdf_report(
        scan_id="DOWNLOAD-TEST",
        prediction_data=mock_pred,
        original_image_path=orig_path,
        gradcam_overlay_path=overlay_path
    )
    
    response = client.get(f"/hub_api/reports/{pdf_path.name}")
    assert response.status_code == 200
    assert "application/pdf" in response.headers["Content-Type"]
