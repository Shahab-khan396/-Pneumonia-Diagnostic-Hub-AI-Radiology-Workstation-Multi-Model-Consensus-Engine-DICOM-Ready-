import os
import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

# Ensure Flask Application is in sys.path
flask_app_dir = Path(__file__).resolve().parent.parent / "Flask Application"
if str(flask_app_dir) not in sys.path:
    sys.path.insert(0, str(flask_app_dir))

from core.preprocessor import preprocess_image
from config import IMG_SIZE


def test_preprocess_numpy_grayscale():
    """Test preprocessing from a 2D grayscale numpy array."""
    dummy_img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    tensor = preprocess_image(dummy_img, img_size=IMG_SIZE)
    
    assert tensor.shape == (1, IMG_SIZE, IMG_SIZE, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_preprocess_numpy_rgb():
    """Test preprocessing from a 3D BGR numpy array."""
    dummy_img = np.random.randint(0, 256, (250, 300, 3), dtype=np.uint8)
    tensor = preprocess_image(dummy_img, img_size=IMG_SIZE)
    
    assert tensor.shape == (1, IMG_SIZE, IMG_SIZE, 3)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_preprocess_bytes():
    """Test preprocessing from encoded JPEG bytes."""
    dummy_img = np.zeros((100, 100), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", dummy_img)
    img_bytes = encoded.tobytes()
    
    tensor = preprocess_image(img_bytes, img_size=IMG_SIZE)
    assert tensor.shape == (1, IMG_SIZE, IMG_SIZE, 3)
    assert tensor.dtype == np.float32


def test_preprocess_invalid_input():
    """Test that invalid inputs raise ValueError."""
    with pytest.raises(ValueError):
        preprocess_image(b"not_an_image_corrupted_data")

    with pytest.raises(ValueError):
        preprocess_image(12345)  # Invalid type
