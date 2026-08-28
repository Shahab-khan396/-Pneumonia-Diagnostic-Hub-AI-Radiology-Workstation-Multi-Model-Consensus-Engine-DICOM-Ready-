import io
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.gradcam import compute_gradcam_heatmap, create_gradcam_overlay, save_gradcam_visualizations
from core.model_manager import get_model_manager
from config import AVAILABLE_MODELS, IMG_SIZE
from app import app


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a dummy CXR image file on disk."""
    img = np.full((128, 128, 3), 100, dtype=np.uint8)
    cv2.circle(img, (64, 64), 30, (220, 220, 220), -1)
    file_path = tmp_path / "test_sample_cxr.jpg"
    cv2.imwrite(str(file_path), img)
    return file_path


def test_gradcam_heatmap_mobilenet():
    """Verify Grad-CAM heatmap calculation on MobileNetV2."""
    manager = get_model_manager()
    model = manager.get_model("mobilenet")
    target_layer = AVAILABLE_MODELS["mobilenet"]["target_conv_layer"]
    
    dummy_input = np.ones((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32) * 0.5
    heatmap = compute_gradcam_heatmap(model, target_layer, dummy_input)
    
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_gradcam_heatmap_all_models():
    """Verify Grad-CAM calculation across all 4 registered architectures."""
    manager = get_model_manager()
    dummy_input = np.ones((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32) * 0.5
    
    for model_id, meta in AVAILABLE_MODELS.items():
        model = manager.get_model(model_id)
        target_layer = meta["target_conv_layer"]
        heatmap = compute_gradcam_heatmap(model, target_layer, dummy_input)
        assert heatmap.ndim == 2
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0


def test_create_gradcam_overlay(sample_image_path):
    """Verify overlay blending and 3-panel composite generation."""
    heatmap = np.random.uniform(0.0, 1.0, (16, 16)).astype(np.float32)
    overlay, heat_color, composite = create_gradcam_overlay(sample_image_path, heatmap)
    
    assert overlay.shape == (128, 128, 3)
    assert heat_color.shape == (128, 128, 3)
    # Composite should have 3 horizontal panels: 128 x (128 * 3) = 128 x 384
    assert composite.shape == (128, 384, 3)


def test_model_manager_predict_with_gradcam(sample_image_path, tmp_path):
    """Verify ModelManager.predict() produces Grad-CAM artifact URLs."""
    manager = get_model_manager()
    dummy_input = np.ones((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32) * 0.5
    
    result = manager.predict(
        model_id="mobilenet",
        image_tensor=dummy_input,
        generate_cam=True,
        original_image_path=sample_image_path,
        base_filename="test_file.jpg"
    )
    
    assert result["success"] is True
    assert result["has_gradcam"] is True
    assert "gradcam_overlay_url" in result
    assert "gradcam_heatmap_url" in result
    assert "gradcam_composite_url" in result


def test_api_predict_with_explain(client, sample_image_path):
    """Verify POST /hub_api/predict returns Grad-CAM URLs in JSON."""
    with open(sample_image_path, "rb") as f:
        img_bytes = f.read()
        
    files = {"file": ("test_api_xray.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    data = {
        "model_choice": "mobilenet",
        "explain": "true",
        "generate_report": "false"
    }
    response = client.post("/hub_api/predict", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["has_gradcam"] is True
    assert "gradcam_overlay_url" in res
    assert "gradcam_composite_url" in res
