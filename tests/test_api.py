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

from app import create_app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def make_dummy_image_bytes():
    """Generate encoded JPEG bytes for testing."""
    dummy_img = np.full((128, 128), 128, dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", dummy_img)
    return io.BytesIO(encoded.tobytes())


def test_health_endpoint(client):
    """Test GET /api/v1/health."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["models_count"] >= 4


def test_models_catalog_endpoint(client):
    """Test GET /api/v1/models."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert len(data["models"]) >= 4


def test_web_index(client):
    """Test GET / renders HTML successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Pneumonia Diagnostic Hub" in response.data


def test_predict_api_success(client):
    """Test POST /api/v1/predict with a valid radiograph image."""
    img_bytes = make_dummy_image_bytes()
    data = {
        "file": (img_bytes, "test_xray.jpg"),
        "model_choice": "mobilenet"
    }
    response = client.post("/api/v1/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res = response.get_json()
    assert res["success"] is True
    assert res["prediction"] in ["NORMAL", "PNEUMONIA"]
    assert "probabilities" in res
    assert "inference_time_ms" in res
    assert res["inference_time_ms"] > 0


def test_predict_api_missing_file(client):
    """Test POST /api/v1/predict with missing file returns 400."""
    response = client.post("/api/v1/predict", data={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "error" in data


def test_predict_api_invalid_extension(client):
    """Test POST /api/v1/predict with non-image file returns 400."""
    text_data = io.BytesIO(b"this is not an image")
    data = {
        "file": (text_data, "bad_file.txt")
    }
    response = client.post("/api/v1/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Invalid file format" in data["error"]
