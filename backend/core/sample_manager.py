"""
Sample radiograph manager — ported from Flask Application/core/sample_manager.py.
Generates synthetic CXR samples on startup and serves the catalog.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config import get_settings


# ─── Synthetic radiograph generation ─────────────────────────────────────────

def _draw_synthetic_radiograph(category: str) -> np.ndarray:
    """
    Generate a 512×512 synthetic chest radiograph with realistic anatomical landmarks:
    - Thoracic rib cage & clavicles
    - Mediastinum & cardiac silhouette
    - Diaphragmatic domes & costophrenic sulci
    - Category-specific pathological patterns
    """
    size = 512
    img = np.zeros((size, size), dtype=np.uint8)

    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]

    # Chest soft-tissue background
    body_mask = ((x - cx) ** 2 / 210 ** 2) + ((y - cy) ** 2 / 240 ** 2) <= 1.0
    img[body_mask] = 45

    # Bilateral lung fields (radiolucent = darker)
    left_lung  = ((x - 170) ** 2 / 75 ** 2)  + ((y - 230) ** 2 / 150 ** 2) <= 1.0
    right_lung = ((x - 340) ** 2 / 80 ** 2)  + ((y - 230) ** 2 / 150 ** 2) <= 1.0
    img[left_lung]  = 20
    img[right_lung] = 20

    # Mediastinum & spine
    cv2.rectangle(img, (240, 60), (272, 450), 120, -1)

    # Cardiac silhouette
    cv2.ellipse(img, (225, 290), (70, 55), 30, 0, 360, 140, -1)

    # Diaphragmatic domes
    cv2.ellipse(img, (170, 370), (85, 35), 0, 0, 180, 130, -1)
    cv2.ellipse(img, (340, 375), (90, 35), 0, 0, 180, 130, -1)

    # Rib outlines
    for r_y in range(120, 360, 35):
        cv2.ellipse(img, (cx, r_y), (190, 40), 0, 20, 160, 75, 4)

    # Clavicles
    cv2.line(img, (110, 110), (240, 125), 110, 6)
    cv2.line(img, (400, 110), (270, 125), 110, 6)

    # Pathological overlays
    if category == "bacterial":
        # Right lower-lobe dense consolidation
        cv2.circle(img, (340, 270), 45, 175, -1)
        cv2.circle(img, (365, 250), 30, 160, -1)
        cv2.circle(img, (320, 290), 35, 150, -1)
    elif category == "viral":
        # Bilateral diffuse interstitial infiltrates
        rng = np.random.default_rng(seed=42)
        for lx_range, rx_range in [((120, 210), (300, 390))]:
            for _ in range(35):
                rx = int(rng.integers(*lx_range))
                ry = int(rng.integers(150, 320))
                rad = int(rng.integers(8, 22))
                cv2.circle(img, (rx, ry), rad, int(rng.integers(60, 110)), -1)
                rx2 = int(rng.integers(*rx_range))
                ry2 = int(rng.integers(150, 320))
                rad2 = int(rng.integers(8, 22))
                cv2.circle(img, (rx2, ry2), rad2, int(rng.integers(60, 110)), -1)

    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    noise   = np.random.default_rng(seed=7).normal(0, 3.5, (size, size)).astype(np.float32)
    final   = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)


# ─── Public API ───────────────────────────────────────────────────────────────

def ensure_samples_generated() -> Path:
    """Create static/samples/ and generate synthetic CXR images if not present."""
    settings    = get_settings()
    samples_dir = Path(settings.samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    category_map = {
        "sample_normal":   "normal",
        "sample_bacterial":"bacterial",
        "sample_viral":    "viral",
    }

    for sid, cat in category_map.items():
        meta = settings.samples_catalog.get(sid, {})
        if not meta:
            continue
        sample_path = samples_dir / meta["filename"]
        if not sample_path.exists():
            img = _draw_synthetic_radiograph(cat)
            cv2.imwrite(str(sample_path), img)

    return samples_dir


def list_sample_catalog() -> List[Dict[str, Any]]:
    """Return all sample entries with public /static/samples/ URLs."""
    ensure_samples_generated()
    catalog = []
    for sid, meta in get_settings().samples_catalog.items():
        item = dict(meta)
        item["id"]        = sid
        item["image_url"] = f"/static/samples/{meta['filename']}"
        catalog.append(item)
    return catalog


def get_sample_info(sample_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single sample's metadata and absolute filesystem path."""
    ensure_samples_generated()
    catalog = get_settings().samples_catalog
    if sample_id not in catalog:
        return None
    meta             = dict(catalog[sample_id])
    meta["id"]       = sample_id
    meta["file_path"] = Path(get_settings().samples_dir) / meta["filename"]
    meta["image_url"] = f"/static/samples/{meta['filename']}"
    return meta
