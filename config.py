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
DEFAULT_MODEL = "densenet121"

# Ensemble Soft-Voting Weights based on Kaggle training evaluation
# (From model_comparison.txt: DenseNet121 35%, MobileNetV2 25%, Xception 20%, EfficientNetB0 20%)
ENSEMBLE_WEIGHTS = {
    "densenet121": 0.35,   # Top F1-Score (91.73%), 89.42% Acc, 0.9540 AUC
    "mobilenet": 0.25,     # 89.42% Acc, Highest AUC (0.9573), 91.58% F1
    "xception": 0.20,      # 88.14% Acc, Highest Precision (94.38%), 90.08% F1
    "efficientnet": 0.20,  # Highest Clinical Recall/Sensitivity (95.38%), 0.8399 AUC
}

# YOLOv8 Localization Model Path
YOLO_MODEL_PATH = MODELS_DIR / "yolov8_pneumonia.pt"

# Available Models Registry with Grad-CAM target convolutional layers
AVAILABLE_MODELS = {
    "densenet121": {
        "id": "densenet121",
        "filename": "densenet121_model.h5",
        "name": "DenseNet121",
        "parameters": "8.1M",
        "description": "Densely Connected CNN maximizing feature reuse. Top F1-Score (91.73%) & 89.42% Accuracy.",
        "badge": "⭐ Top Performer",
        "recommended": True,
        "target_conv_layer": "relu",
        "weight": 0.35,
    },
    "mobilenet": {
        "id": "mobilenet",
        "filename": "mobilenet_model.h5",
        "name": "MobileNetV2",
        "parameters": "3.5M",
        "description": "Inverted residual architecture with linear bottlenecks. 89.42% Accuracy and highest AUC (0.9573).",
        "badge": "Highest AUC",
        "recommended": False,
        "target_conv_layer": "out_relu",
        "weight": 0.25,
    },
    "xception": {
        "id": "xception",
        "filename": "xception_model.h5",
        "name": "Xception",
        "parameters": "22.9M",
        "description": "Extreme Inception network utilizing depthwise separable convolutions. Highest precision (94.38%).",
        "badge": "Highest Precision",
        "recommended": False,
        "target_conv_layer": "block14_sepconv2_act",
        "weight": 0.20,
    },
    "efficientnet": {
        "id": "efficientnet",
        "filename": "efficientnet_model.h5",
        "name": "EfficientNetB0",
        "parameters": "5.3M",
        "description": "Compound scaled architecture balancing resolution and depth. Highest clinical sensitivity / recall (95.38%).",
        "badge": "Highest Recall",
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
        "badge": "Residual Baseline",
        "recommended": False,
        "target_conv_layer": "conv5_block3_out",
        "weight": 0.0,
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
        "weight": 0.0,
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
