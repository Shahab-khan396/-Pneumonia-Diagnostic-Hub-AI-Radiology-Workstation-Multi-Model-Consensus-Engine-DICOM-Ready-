"""
Pneumonia Diagnostic Hub • YOLOv8 Anatomical Localization Module
Detects pulmonary infiltrates and draws clinical bounding boxes with confidence scores.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from config import YOLO_MODEL_PATH, UPLOAD_FOLDER

_yolo_instance = None


def get_yolo_model():
    """Lazy-load the YOLOv8 model instance."""
    global _yolo_instance
    if _yolo_instance is None and YOLO_MODEL_PATH.exists():
        try:
            from ultralytics import YOLO
            _yolo_instance = YOLO(str(YOLO_MODEL_PATH))
        except Exception as e:
            _yolo_instance = None
    return _yolo_instance


def is_yolo_available() -> bool:
    """Check if YOLO model weights and ultralytics runtime are present."""
    if not YOLO_MODEL_PATH.exists():
        return False
    try:
        import ultralytics
        return True
    except ImportError:
        return False


def run_yolo_detection(
    image_path: Path,
    conf_threshold: float = 0.20,
    iou_threshold: float = 0.45,
    img_size: int = 640
) -> Dict[str, Any]:
    """
    Run YOLOv8 inference to detect pneumonia opacities and return
    an annotated image with bounding boxes along with box coordinates and confidence.
    """
    model = get_yolo_model()
    if model is None:
        return {
            "success": False,
            "error": "YOLO model not loaded or weights missing",
            "boxes": [],
            "count": 0,
            "annotated_rgb": None,
            "annotated_image_path": None,
            "annotated_image_url": None,
        }

    orig_bgr = cv2.imread(str(image_path))
    if orig_bgr is None:
        return {
            "success": False,
            "error": "Failed to read input image for YOLO inference",
            "boxes": [],
            "count": 0,
            "annotated_rgb": None,
            "annotated_image_path": None,
            "annotated_image_url": None,
        }

    results = model.predict(
        str(image_path),
        imgsz=img_size,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    res = results[0]
    boxes_data: List[Dict[str, Any]] = []
    annotated = orig_bgr.copy()

    if res.boxes is not None and len(res.boxes) > 0:
        for idx, box in enumerate(res.boxes):
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, "pneumonia_opacity")

            boxes_data.append({
                "id": idx + 1,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf * 100.0, 1),
                "class_name": cls_name,
                "coordinates_str": f"[{x1}, {y1}] to [{x2}, {y2}]",
                "width": x2 - x1,
                "height": y2 - y1
            })

            # Draw clinical bounding box (Coral Red: (50, 50, 235))
            color = (50, 50, 235)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Label banner
            label_text = f"Pneumonia Opacity ({conf * 100:.1f}%)"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, max(th + 10, y1)), color, -1)
            cv2.putText(
                annotated,
                label_text,
                (x1 + 5, max(th + 5, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    # Save annotated image
    out_filename = f"yolo_det_{image_path.stem}.jpg"
    out_path = UPLOAD_FOLDER / out_filename
    cv2.imwrite(str(out_path), annotated)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return {
        "success": True,
        "boxes": boxes_data,
        "count": len(boxes_data),
        "annotated_bgr": annotated,
        "annotated_rgb": annotated_rgb,
        "annotated_image_path": out_path,
        "annotated_image_url": f"/static/uploads/{out_filename}",
    }
