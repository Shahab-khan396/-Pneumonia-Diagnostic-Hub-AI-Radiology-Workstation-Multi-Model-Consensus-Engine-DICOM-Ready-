import os
from pathlib import Path
from typing import Union, Tuple, Dict, Optional

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    target_layer_name: str,
    img_tensor: np.ndarray,
    pred_index: Optional[int] = None
) -> np.ndarray:
    """
    Calculate the Grad-CAM activation heatmap for a given input tensor and target class.
    
    Args:
        model: Pre-trained Keras CNN model.
        target_layer_name: Name of the last convolutional/activation layer.
        img_tensor: Preprocessed batch tensor of shape (1, H, W, 3).
        pred_index: Target class index (0: NORMAL, 1: PNEUMONIA). Defaults to top prediction.
        
    Returns:
        np.ndarray: 2D normalized float32 heatmap array with values in [0.0, 1.0].
    """
    # Check if target layer is in the top-level model or inside a nested backbone
    target_layer = None
    backbone = None
    pre_layers = []
    head_layers = []
    found_backbone = False

    try:
        target_layer = model.get_layer(target_layer_name)
    except ValueError:
        pass

    if target_layer is not None:
        # Standard flat model path
        grad_model = Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)
    else:
        # Transfer-learning model with nested backbone sub-model
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) or (hasattr(layer, "layers") and len(layer.layers) > 5):
                backbone = layer
                found_backbone = True
            elif not found_backbone:
                if not isinstance(layer, tf.keras.layers.InputLayer):
                    pre_layers.append(layer)
            else:
                head_layers.append(layer)

        if backbone is not None:
            try:
                target_layer = backbone.get_layer(target_layer_name)
            except ValueError:
                for layer in reversed(backbone.layers):
                    if any(k in layer.name.lower() for k in ("conv", "relu", "activation", "sepconv")):
                        if hasattr(layer.output, "shape") and len(layer.output.shape) == 4:
                            target_layer = layer
                            break

        if target_layer is None:
            raise ValueError(f"Could not locate target layer '{target_layer_name}' in model or backbone.")

        feature_extractor = Model(inputs=backbone.inputs, outputs=[target_layer.output, backbone.output])

        with tf.GradientTape() as tape:
            x = img_tensor
            for layer in pre_layers:
                x = layer(x)
            conv_outputs, backbone_out = feature_extractor(x)
            tape.watch(conv_outputs)

            head_x = backbone_out
            for layer in head_layers:
                head_x = layer(head_x)

            predictions = head_x
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature map activations by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU: keep only features that positively correlate with the target class
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)

    return np.float32(heatmap.numpy())


def create_gradcam_overlay(
    original_image_path: Union[str, Path],
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Superimpose the 2D Grad-CAM heatmap onto the original Chest X-Ray.
    
    Args:
        original_image_path: Path to the original radiograph image file.
        heatmap: 2D float32 array in [0.0, 1.0].
        alpha: Overlay blend weight (0.0 to 1.0).
        colormap: OpenCV colormap (default cv2.COLORMAP_JET).
        
    Returns:
        Tuple of (overlay_bgr, heatmap_color_bgr, composite_bgr)
    """
    orig_img = cv2.imread(str(original_image_path))
    if orig_img is None:
        try:
            from PIL import Image
            pil_img = Image.open(original_image_path).convert("RGB")
            orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Could not read original image at {original_image_path}: {e}")

    h, w, _ = orig_img.shape

    # Ensure float32 to prevent OpenCV assertion failure when mixed_float16 is used
    heatmap = np.float32(heatmap)

    # Resize heatmap to original image dimensions with bicubic interpolation
    resized_heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Clip and convert to 8-bit unsigned int
    heatmap_uint8 = np.uint8(255 * np.clip(resized_heatmap, 0.0, 1.0))
    
    # Colorize heatmap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # Alpha blend overlay
    overlay = cv2.addWeighted(orig_img, 1.0 - alpha, heatmap_color, alpha, 0)

    # Build side-by-side composite: [ Original | Heatmap | Overlay ]
    # Add title headers to each panel
    banner_height = 40
    panel_w = w
    
    def make_panel(img: np.ndarray, title: str) -> np.ndarray:
        panel = img.copy()
        # Draw translucent top banner
        overlay_banner = panel.copy()
        cv2.rectangle(overlay_banner, (0, 0), (panel_w, banner_height), (15, 23, 42), -1)
        cv2.addWeighted(overlay_banner, 0.8, panel, 0.2, 0, panel)
        # Put title text
        cv2.putText(
            panel, title, (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        return panel

    p1 = make_panel(orig_img, "Original Radiograph")
    p2 = make_panel(heatmap_color, "Attention Heatmap")
    p3 = make_panel(overlay, "Diagnostic Overlay")

    composite = np.hstack([p1, p2, p3])

    return overlay, heatmap_color, composite


def save_gradcam_visualizations(
    upload_dir: Path,
    base_filename: str,
    original_path: Path,
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.45
) -> Dict[str, str]:
    """
    Generate and save overlay, heatmap, and composite images to disk.
    
    Returns:
        Dict with public URL paths for 'overlay_url', 'heatmap_url', 'composite_url'.
    """
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    overlay, heatmap_color, composite = create_gradcam_overlay(
        original_image_path=original_path,
        heatmap=heatmap,
        colormap=colormap,
        alpha=alpha
    )

    # Ensure valid image extension
    p = Path(base_filename)
    stem = p.stem if p.suffix else base_filename
    ext = p.suffix.lower() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"] else ".jpg"

    overlay_filename = f"gradcam_overlay_{stem}{ext}"
    heatmap_filename = f"gradcam_heat_{stem}{ext}"
    composite_filename = f"gradcam_comp_{stem}{ext}"

    cv2.imwrite(str(upload_dir / overlay_filename), overlay)
    cv2.imwrite(str(upload_dir / heatmap_filename), heatmap_color)
    cv2.imwrite(str(upload_dir / composite_filename), composite)

    return {
        "gradcam_overlay_url": f"/static/uploads/{overlay_filename}",
        "gradcam_heatmap_url": f"/static/uploads/{heatmap_filename}",
        "gradcam_composite_url": f"/static/uploads/{composite_filename}",
    }
