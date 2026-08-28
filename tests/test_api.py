import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


def make_dummy_image_bytes():
    """Generate encoded JPEG bytes for testing."""
    dummy_img = np.full((128, 128), 128, dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", dummy_img)
    return io.BytesIO(encoded.tobytes())


def test_health_endpoint(client):
    """Test GET /hub_api/health and GET /health."""
    response = client.get("/hub_api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert len(data["models_available"]) >= 4


def test_models_catalog_endpoint(client):
    """Test GET /hub_api/models and GET /models."""
    response = client.get("/hub_api/models")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["models"]) >= 4


def test_root_index(client):
    """Test GET / returns operational status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"


def test_predict_api_success(client):
    """Test POST /hub_api/predict with a valid radiograph image."""
    img_bytes = make_dummy_image_bytes()
    files = {"file": ("test_xray.jpg", img_bytes, "image/jpeg")}
    data = {"model_choice": "mobilenet", "explain": "true", "generate_report": "false"}
    
    response = client.post("/hub_api/predict", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["prediction"] in ["NORMAL", "PNEUMONIA"]
    assert "probabilities" in res
    assert "inference_time_ms" in res
    assert res["inference_time_ms"] > 0


def test_predict_api_missing_file(client):
    """Test POST /hub_api/predict with missing file returns 400."""
    response = client.post("/hub_api/predict", data={})
    assert response.status_code == 400


def test_predict_api_invalid_extension(client):
    """Test POST /hub_api/predict with non-image file returns 400."""
    text_data = io.BytesIO(b"this is not an image")
    files = {"file": ("bad_file.txt", text_data, "text/plain")}
    response = client.post("/hub_api/predict", files=files, data={"model_choice": "mobilenet"})
    assert response.status_code in [400, 500]
