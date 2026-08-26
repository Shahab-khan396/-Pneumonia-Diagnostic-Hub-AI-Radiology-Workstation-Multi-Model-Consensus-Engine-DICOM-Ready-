import os
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from config import SAMPLES_DIR, SAMPLES_CATALOG


def _draw_synthetic_radiograph(category: str) -> np.ndarray:
    """
    Generate a synthetic chest radiograph with realistic anatomical landmarks:
    - Thoracic rib cage & clavicle outlines
    - Mediastinum & Cardiac silhouette
    - Diaphragmatic domes & Costophrenic sulci
    - Clear lung parenchyma (Normal) vs. Lobar opacity (Bacterial) vs. Diffuse infiltrates (Viral)
    """
    canvas_size = 512
    img = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    
    # 1. Background soft gradient (chest soft tissue)
    y, x = np.ogrid[:canvas_size, :canvas_size]
    center_y, center_x = canvas_size // 2, canvas_size // 2
    body_mask = ((x - center_x) ** 2 / (210 ** 2)) + ((y - center_y) ** 2 / (240 ** 2)) <= 1.0
    img[body_mask] = 45
    
    # 2. Bilateral Lung Fields (radiolucent = darker)
    left_lung_mask = ((x - 170) ** 2 / (75 ** 2)) + ((y - 230) ** 2 / (150 ** 2)) <= 1.0
    right_lung_mask = ((x - 340) ** 2 / (80 ** 2)) + ((y - 230) ** 2 / (150 ** 2)) <= 1.0
    
    img[left_lung_mask] = 20
    img[right_lung_mask] = 20
    
    # 3. Mediastinum & Spine (radiopaque central column)
    cv2.rectangle(img, (240, 60), (272, 450), 120, -1)
    
    # 4. Cardiac silhouette (heart shadow in left hemithorax)
    cv2.ellipse(img, (225, 290), (70, 55), 30, 0, 360, 140, -1)
    
    # 5. Diaphragmatic domes
    cv2.ellipse(img, (170, 370), (85, 35), 0, 0, 180, 130, -1)
    cv2.ellipse(img, (340, 375), (90, 35), 0, 0, 180, 130, -1)
    
    # 6. Rib cage outlines
    for r_y in range(120, 360, 35):
        cv2.ellipse(img, (center_x, r_y), (190, 40), 0, 20, 160, 75, 4)
    
    # 7. Clavicles
    cv2.line(img, (110, 110), (240, 125), 110, 6)
    cv2.line(img, (400, 110), (270, 125), 110, 6)

    # 8. Pathological Manifestations
    if category == "bacterial":
        # Dense Right Middle/Lower Lobe consolidation (dense white opacity)
        cv2.circle(img, (340, 270), 45, 175, -1)
        cv2.circle(img, (365, 250), 30, 160, -1)
        cv2.circle(img, (320, 290), 35, 150, -1)
    elif category == "viral":
        # Diffuse bilateral interstitial infiltrates (patchy reticular opacities)
        for _ in range(35):
            rx = int(np.random.randint(120, 210))
            ry = int(np.random.randint(150, 320))
            rad = int(np.random.randint(8, 22))
            cv2.circle(img, (rx, ry), rad, int(np.random.randint(60, 110)), -1)
        for _ in range(35):
            rx = int(np.random.randint(300, 390))
            ry = int(np.random.randint(150, 320))
            rad = int(np.random.randint(8, 22))
            cv2.circle(img, (rx, ry), rad, int(np.random.randint(60, 110)), -1)
            
    # Apply Gaussian smoothing to blend anatomical contours naturally
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    
    # Add subtle radiological sensor noise
    noise = np.random.normal(0, 3.5, (canvas_size, canvas_size)).astype(np.float32)
    final_img = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Convert to 3-channel
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)


def ensure_samples_generated() -> Path:
    """Ensure that static/samples directory exists and sample CXR files are generated."""
    samples_dir = Path(SAMPLES_DIR)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    mapping = {
        "sample_normal": "normal",
        "sample_bacterial": "bacterial",
        "sample_viral": "viral"
    }
    
    for sid, cat in mapping.items():
        meta = SAMPLES_CATALOG[sid]
        sample_path = samples_dir / meta["filename"]
        if not sample_path.exists():
            img_bgr = _draw_synthetic_radiograph(cat)
            cv2.imwrite(str(sample_path), img_bgr)
            
    return samples_dir


def list_sample_catalog() -> List[Dict]:
    """Retrieve catalog of all available sample radiographs with public URLs."""
    ensure_samples_generated()
    catalog = []
    for sid, meta in SAMPLES_CATALOG.items():
        item = dict(meta)
        item["image_url"] = f"/static/samples/{meta['filename']}"
        catalog.append(item)
    return catalog


def get_sample_info(sample_id: str) -> Optional[Dict]:
    """Retrieve detailed sample info and filesystem path for a specific sample."""
    ensure_samples_generated()
    if sample_id not in SAMPLES_CATALOG:
        return None
    meta = dict(SAMPLES_CATALOG[sample_id])
    meta["file_path"] = Path(SAMPLES_DIR) / meta["filename"]
    meta["image_url"] = f"/static/samples/{meta['filename']}"
    return meta
