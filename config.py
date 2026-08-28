import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_FOLDER = STATIC_DIR / "uploads"
SAMPLES_DIR = STATIC_DIR / "samples"

# Ensure directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Upload and Security Settings (Includes DICOM .dcm format)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "dcm"}
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB max upload size

# Model & ML Parameters
IMG_SIZE = 128
CLASS_LABELS = {0: "NORMAL", 1: "PNEUMONIA"}
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
DEFAULT_MODEL = "mobilenet"

# Ensemble Soft-Voting Weights based on validation performance
ENSEMBLE_WEIGHTS = {
    "mobilenet": 0.45,   # Highest validation accuracy (87.5%)
    "resnet50": 0.25,    # Deep residual skip connections
    "efficientnet": 0.20,# Compound scaling feature maps
    "VGG19": 0.10,       # Baseline model
}

# Available Models Registry with Grad-CAM target convolutional layers
AVAILABLE_MODELS = {
    "mobilenet": {
        "id": "mobilenet",
        "filename": "mobilenet_model.h5",
        "name": "MobileNetV2",
        "parameters": "3.5M",
        "description": "Lightweight and highly efficient CNN. Achieved highest validation accuracy (87.5%).",
        "badge": "⭐ Recommended",
        "recommended": True,
        "target_conv_layer": "out_relu",
        "weight": 0.45,
    },
    "efficientnet": {
        "id": "efficientnet",
        "filename": "efficientnet_model.h5",
        "name": "EfficientNetB0",
        "parameters": "5.3M",
        "description": "Compound scaled architecture balancing depth, width, and resolution.",
        "badge": "High Efficiency",
        "recommended": False,
        "target_conv_layer": "top_activation",
        "weight": 0.20,
    },
    "resnet50": {
        "id": "resnet50",
        "filename": "resnet50_model.h5",
        "name": "ResNet50",
        "parameters": "25.6M",
        "description": "Deep 50-layer network utilizing identity shortcut connections.",
        "badge": "Deep Residual",
        "recommended": False,
        "target_conv_layer": "conv5_block3_out",
        "weight": 0.25,
    },
    "VGG19": {
        "id": "VGG19",
        "filename": "VGG19_model.h5",
        "name": "VGG19",
        "parameters": "63.1M",
        "description": "Standard 19-layer sequential convolutional network baseline.",
        "badge": "Heavy Baseline",
        "recommended": False,
        "target_conv_layer": "block5_conv4",
        "weight": 0.10,
    },
}

# Pre-packaged Sample Radiographs Catalog
SAMPLES_CATALOG = {
    "sample_normal": {
        "id": "sample_normal",
        "filename": "normal_clear_lungs.jpg",
        "title": "Normal Radiograph",
        "category": "NORMAL",
        "subtitle": "Clear bilateral lung fields, sharp costophrenic angles",
        "description": "Healthy adult radiograph displaying normal bronchovascular arborization without focal consolidation.",
        "badge": "Normal CXR",
        "badge_class": "badge-normal",
    },
    "sample_bacterial": {
        "id": "sample_bacterial",
        "filename": "bacterial_lobar_pneumonia.jpg",
        "title": "Bacterial Pneumonia",
        "category": "PNEUMONIA",
        "subtitle": "Dense right middle lobe alveolar consolidation",
        "description": "Demonstrates classical lobar consolidation with air bronchograms typical of bacterial Streptococcus infection.",
        "badge": "Bacterial Lobar",
        "badge_class": "badge-pneumonia",
    },
    "sample_viral": {
        "id": "sample_viral",
        "filename": "viral_interstitial_pneumonia.jpg",
        "title": "Viral Pneumonia",
        "category": "PNEUMONIA",
        "subtitle": "Bilateral diffuse interstitial & reticular opacities",
        "description": "Shows diffuse peribronchial thickening and ground-glass haziness typical of viral etiology.",
        "badge": "Viral Interstitial",
        "badge_class": "badge-pneumonia",
    },
}
