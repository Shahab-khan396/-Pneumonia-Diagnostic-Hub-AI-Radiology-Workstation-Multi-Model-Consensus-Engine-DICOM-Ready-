import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Suppress noisy TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from tensorflow.keras.models import load_model
from config import BASE_DIR, AVAILABLE_MODELS, CLASS_LABELS, DEFAULT_MODEL, UPLOAD_FOLDER
from core.gradcam import compute_gradcam_heatmap, save_gradcam_visualizations

# ZeroGPU decorator hook
try:
    import spaces
    GPU_CALL = spaces.GPU
except Exception:
    def GPU_CALL(func):
        return func


class ModelManager:
    """
    Thread-safe Singleton Model Manager that handles lazy loading,
    in-memory model caching, metadata cataloging, high-performance inference,
    and Grad-CAM visual explainability.
    """
    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._models_cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        self._base_dir = Path(BASE_DIR)
        self._upload_dir = Path(UPLOAD_FOLDER)
        self._initialized = True

    def _resolve_model_path(self, model_id: str) -> Path:
        """Resolve the absolute path to the requested model file."""
        if model_id not in AVAILABLE_MODELS:
            raise KeyError(f"Unknown model identifier: '{model_id}'. Available: {list(AVAILABLE_MODELS.keys())}")
        
        filename = AVAILABLE_MODELS[model_id]["filename"]
        model_path = self._base_dir / filename
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file '{filename}' was not found at expected location: {model_path}"
            )
        return model_path

    def get_model(self, model_id: str = DEFAULT_MODEL):
        """
        Retrieve a model from cache, or load it from disk if not yet cached.
        Thread-safe to prevent race conditions during concurrent requests.
        """
        if model_id not in AVAILABLE_MODELS:
            model_id = DEFAULT_MODEL

        with self._cache_lock:
            if model_id not in self._models_cache:
                model_path = self._resolve_model_path(model_id)
                # Load with compile=False for faster loading during pure inference
                loaded = load_model(str(model_path), compile=False)
                self._models_cache[model_id] = loaded
            return self._models_cache[model_id]

    def predict(
        self,
        model_id: str,
        image_tensor: np.ndarray,
        generate_cam: bool = False,
        original_image_path: Optional[Union[str, Path]] = None,
        base_filename: Optional[str] = None
    ) -> Dict[str, Any]:

        """
        Run inference using the specified model and optionally generate Grad-CAM heatmaps.
        
        Correct Label Mapping:
          Index 0 -> 'NORMAL'
          Index 1 -> 'PNEUMONIA'
        """
        model_meta = AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS[DEFAULT_MODEL])
        effective_model_id = model_meta["id"]
        
        model = self.get_model(effective_model_id)
        
        # High-precision inference timing
        start_time = time.perf_counter()
        raw_predictions = model.predict(image_tensor, verbose=0)
        end_time = time.perf_counter()
        
        inference_time_ms = round((end_time - start_time) * 1000.0, 2)
        
        # Extract probabilities
        pred_probs = raw_predictions[0]
        prob_normal = float(pred_probs[0])
        prob_pneumonia = float(pred_probs[1])
        
        # Argmax classification
        pred_index = int(np.argmax(pred_probs))
        predicted_label = CLASS_LABELS.get(pred_index, "UNKNOWN")
        confidence_pct = round(float(np.max(pred_probs)) * 100.0, 2)
        
        response: Dict[str, Any] = {
            "success": True,
            "prediction": predicted_label,
            "confidence": confidence_pct,
            "probabilities": {
                "NORMAL": round(prob_normal * 100.0, 2),
                "PNEUMONIA": round(prob_pneumonia * 100.0, 2),
            },
            "raw_probabilities": {
                "NORMAL": prob_normal,
                "PNEUMONIA": prob_pneumonia,
            },
            "model_id": effective_model_id,
            "model_name": model_meta["name"],
            "model_parameters": model_meta["parameters"],
            "model_badge": model_meta["badge"],
            "target_conv_layer": model_meta.get("target_conv_layer", "Unknown"),
            "inference_time_ms": inference_time_ms,
        }

        # Generate Explainable AI (Grad-CAM) visualizations if requested
        if generate_cam and original_image_path and base_filename:
            try:
                target_layer = model_meta.get("target_conv_layer", "out_relu")
                heatmap = compute_gradcam_heatmap(
                    model=model,
                    target_layer_name=target_layer,
                    img_tensor=image_tensor,
                    pred_index=pred_index
                )
                cam_urls = save_gradcam_visualizations(
                    upload_dir=self._upload_dir,
                    base_filename=base_filename,
                    original_path=Path(original_image_path),
                    heatmap=heatmap
                )
                response.update(cam_urls)
                response["has_gradcam"] = True
            except Exception as cam_err:
                response["has_gradcam"] = False
                response["gradcam_error"] = str(cam_err)

        return response

    def preload(self, model_id: str) -> bool:
        """Explicitly warm up a model in memory."""
        try:
            self.get_model(model_id)
            return True
        except Exception:
            return False

    def list_available_models(self) -> list[dict]:
        """Return list of all registered models with their availability status."""
        result = []
        for mid, meta in AVAILABLE_MODELS.items():
            model_file = self._base_dir / meta["filename"]
            info = dict(meta)
            info["file_exists"] = model_file.exists()
            info["is_loaded"] = mid in self._models_cache
            result.append(info)
        return result


def get_model_manager() -> ModelManager:
    """Convenience accessor for the ModelManager singleton."""
    return ModelManager()
