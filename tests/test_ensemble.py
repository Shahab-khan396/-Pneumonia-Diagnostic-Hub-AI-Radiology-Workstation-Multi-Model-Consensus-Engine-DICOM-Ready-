import io
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

# Ensure Flask Application is in sys.path
flask_app_dir = Path(__file__).resolve().parent.parent / "Flask Application"
if str(flask_app_dir) not in sys.path:
    sys.path.insert(0, str(flask_app_dir))

from core.ensemble import run_multi_model_comparison
from config import IMG_SIZE
from app import create_app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a dummy CXR image file."""
    img = np.full((128, 128, 3), 100, dtype=np.uint8)
    cv2.circle(img, (64, 64), 30, (220, 220, 220), -1)
    file_path = tmp_path / "test_ensemble_cxr.jpg"
    cv2.imwrite(str(file_path), img)
    return file_path


def test_ensemble_multi_model_comparison(sample_image_path):
    """Verify simultaneous multi-model execution and soft-voting consensus."""
    dummy_input = np.ones((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32) * 0.5
    
    result = run_multi_model_comparison(
        image_tensor=dummy_input,
        original_image_path=sample_image_path,
        base_filename="ensemble_test.jpg",
        generate_cams=True
    )
    
    assert result["success"] is True
    assert result["is_ensemble"] is True
    assert result["consensus_verdict"] in ["NORMAL", "PNEUMONIA"]
    assert 0.0 <= result["consensus_confidence"] <= 100.0
    assert result["agreement_level"] in ["UNANIMOUS", "STRONG_MAJORITY", "SPLIT_DECISION"]
    assert len(result["models_breakdown"]) == 4
    assert result["total_inference_time_ms"] > 0
    assert "primary_gradcam_overlay_url" in result


def test_api_compare_endpoint(client, sample_image_path):
    """Verify POST /api/v1/compare endpoint returns full consensus and PDF report URL."""
    with open(sample_image_path, "rb") as f:
        img_bytes = f.read()
        
    data = {
        "file": (io.BytesIO(img_bytes), "test_compare.jpg"),
        "explain": "true"
    }
    response = client.post("/api/v1/compare", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res = response.get_json()
    assert res["success"] is True
    assert res["is_ensemble"] is True
    assert "consensus_verdict" in res
    assert "models_breakdown" in res
    assert "report_pdf_url" in res
