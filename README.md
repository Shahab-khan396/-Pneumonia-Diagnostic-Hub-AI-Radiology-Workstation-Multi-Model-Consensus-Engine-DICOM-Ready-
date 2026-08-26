---
title: Pneumonia Diagnostic Hub AI Workstation
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Pneumonia Diagnostic Hub | AI Radiology Workstation & Decision Support System

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-3.x-red)
![Flask](https://img.shields.io/badge/Flask-3.x-black)
![XAI](https://img.shields.io/badge/Explainable%20AI-Grad--CAM-green)
![DICOM](https://img.shields.io/badge/Medical-DICOM%20Ready-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

A hospital-grade deep learning radiology platform for automated pneumonia screening, multi-model weighted consensus evaluation, Grad-CAM visual explainability, native DICOM (`.dcm`) parsing, and automated clinical PDF diagnostic report generation.

---

## 🌟 Key Features

- **⚔️ Multi-Model Consensus Battleground**: Simultaneous evaluation across 4 CNN backbones (*MobileNetV2*, *ResNet50*, *EfficientNetB0*, *VGG19*) with weighted soft-voting consensus and inter-model agreement analytics.
- **🔍 Explainable AI (Grad-CAM XAI)**: Spatial gradient activation maps backpropagated from target convolutional layers with interactive split-screen swipe sliders.
- **🏥 Native DICOM Support**: Ingests raw `.dcm` medical imaging files with automatic VOI LUT lung windowing ($W: 1500, L: -600$) and DICOM header extraction.
- **📄 Publication-Quality PDF Reports**: 1-click clinical diagnostic reports with patient demographics, side-by-side CXR/Grad-CAM figures, model telemetry tables, and clinical findings.
- **🌓 High-Contrast Dark Mode**: Radiologist-optimized slate workstation theme designed for low-light clinical reading rooms.
- **📖 Interactive Swagger REST API (`/docs`)**: OpenAPI 3.0-compliant endpoints for enterprise EHR/PACS integration.

---

## 🏗️ Model Architecture & Benchmark Matrix

| Model Architecture | Parameters | Memory Size | Target Conv Layer | Ensemble Weight | Validation Acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** ⭐ *(Recommended)* | 3.5M | 11.5 MB | `out_relu` | **45%** | **87.5%** |
| **ResNet50** | 25.6M | 101.2 MB | `conv5_block3_out` | **25%** | 50.0% (1-epoch) |
| **EfficientNetB0** | 5.3M | 20.8 MB | `top_activation` | **20%** | 50.0% |
| **VGG19** | 63.1M | 443.6 MB | `block5_conv4` | **10%** | 62.5% |

---

## 🚀 Live Deployment on Hugging Face Spaces

This repository is pre-configured for instant deployment on **Hugging Face Spaces** using Docker:

```bash
# 1. Add your Hugging Face Space as a remote
git remote add space https://huggingface.co/spaces/<your-username>/pneumonia-diagnostic-hub

# 2. Track model weights with Git LFS
git lfs track "*.h5"
git add .gitattributes

# 3. Push to Hugging Face
git add .
git commit -m "Deploy Pneumonia Diagnostic Hub"
git push space main
```

---

## 💻 Local Quickstart

### Prerequisites
```bash
Python 3.10+ / 3.11+ / 3.12+ / 3.13+
```

### Installation
```bash
# Clone the repository
git clone https://github.com/Shahab-khan396/Pneumonia-Diagnostic-Hub-Multi-Model-AI-Detector.git
cd Pneumonia-Diagnostic-Hub-Multi-Model-AI-Detector

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
python "Flask Application/app.py"
```
- Open **`http://127.0.0.1:5000`** in your browser.
- Open **`http://127.0.0.1:5000/docs`** for interactive Swagger API documentation.

### Running with Docker
```bash
docker compose up --build
```

---

## 🧪 Automated Test Suite (100% Pass Rate)

Run the full pytest suite:
```bash
pytest tests/ -v
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Disclaimer:** Developed for educational and research decision-support purposes only. All clinical outcomes must be verified by a licensed medical practitioner.