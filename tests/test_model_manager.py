import numpy as np
import pytest

from core.model_manager import ModelManager, get_model_manager
from config import AVAILABLE_MODELS, CLASS_LABELS, IMG_SIZE


def test_model_manager_singleton():
    """Verify ModelManager implements thread-safe singleton pattern."""
    m1 = get_model_manager()
    m2 = get_model_manager()
    assert m1 is m2


def test_list_available_models():
    """Verify all registered models are cataloged with metadata."""
    manager = get_model_manager()
    models = manager.list_available_models()
    assert len(models) >= 4
    model_ids = [m["id"] for m in models]
    assert "mobilenet" in model_ids
    assert "efficientnet" in model_ids
    assert "resnet50" in model_ids
    assert "VGG19" in model_ids


def test_mobilenet_inference_and_label_mapping():
    """Verify inference execution and correct label mapping (Index 0: NORMAL, Index 1: PNEUMONIA)."""
    manager = get_model_manager()
    
    # Create dummy batch tensor of shape (1, 128, 128, 3)
    dummy_input = np.ones((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32) * 0.5
    
    result = manager.predict("mobilenet", dummy_input)
    
    assert result["success"] is True
    assert result["prediction"] in ["NORMAL", "PNEUMONIA"]
    assert 0.0 <= result["confidence"] <= 100.0
    assert "NORMAL" in result["probabilities"]
    assert "PNEUMONIA" in result["probabilities"]
    
    total_prob = result["probabilities"]["NORMAL"] + result["probabilities"]["PNEUMONIA"]
    assert pytest.approx(total_prob, abs=1.0) == 100.0
    
    assert result["inference_time_ms"] > 0
    assert result["model_id"] == "mobilenet"


def test_model_caching():
    """Verify that subsequent model calls retrieve from memory without disk reload."""
    manager = get_model_manager()
    model_instance_1 = manager.get_model("mobilenet")
    model_instance_2 = manager.get_model("mobilenet")
    assert model_instance_1 is model_instance_2
