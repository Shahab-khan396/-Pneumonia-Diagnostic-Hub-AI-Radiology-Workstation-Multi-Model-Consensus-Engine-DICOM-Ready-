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

# 🫁 Pneumonia Diagnostic Hub • AI Radiology Workstation

[![Production Live Demo](https://img.shields.io/badge/Live%20Frontend-Vercel-black?style=for-the-badge&logo=vercel)](https://pneumonia-dignosis-hub.vercel.app/)
[![AI Engine](https://img.shields.io/badge/AI%20Engine-Hugging%20Face%20Spaces-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/Shahabkhan396/pneumonia-hub)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![Next.js](https://img.shields.io/badge/Next.js-15%20App%20Router-black?style=for-the-badge&logo=next.js)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An enterprise-grade, clinical-assisted decision support system for chest radiograph (CXR) pneumonia screening. Powered by a **4-Model Weighted Soft-Voting Consensus Engine**, **Explainable AI (Grad-CAM heatmaps)**, **DICOM (.dcm) metadata parsing**, and automated **Clinical PDF Report Generation**.

> **Medical Disclaimer:** This system is intended for research, demonstration, and clinical decision support purposes. It is not an FDA-cleared diagnostic device and should not replace professional radiological evaluation.

---

## 🏛️ System Architecture

```mermaid
graph TD
    User["👨‍⚕️ Radiologist / Clinician"] -->|Uploads CXR / DICOM| Vercel["🌐 Next.js 15 Radiology Workstation<br/>(Vercel Edge CDN)"]
    Vercel -->|REST API Request /hub_api| HF["⚡ High-Performance AI Microservice<br/>(Hugging Face Spaces • FastAPI • 16GB RAM)"]
    
    subgraph "Core AI Consensus Engine (core/)"
        HF --> Prep["1. Preprocessor & DICOM Parser<br/>(128x128 CLAHE Normalization)"]
        Prep --> MM["2. Thread-Safe Model Manager"]
        
        MM --> M1["MobileNetV2 (w=0.45, 87.5% Acc)"]
        MM --> M2["ResNet50 (w=0.25, Skip Residuals)"]
        MM --> M3["EfficientNetB0 (w=0.20, Compound Scale)"]
        MM --> M4["VGG19 (w=0.10, Baseline)"]
        
        M1 & M2 & M3 & M4 --> Ens["3. Weighted Soft-Voting Consensus<br/>(Agreement Scoring & Breakdown)"]
        Ens --> XAI["4. Grad-CAM Explainability<br/>(Class Activation Heatmaps & Overlays)"]
        XAI --> PDF["5. Clinical Report Generator<br/>(ReportLab Vector PDF Document)"]
    end
    
    PDF -->|JSON + Base64 Overlays + PDF Link| Vercel
    Vercel -->|Interactive Workstation View| User
```

---

## 📁 Clean Repository Structure

```text
├── app.py                     # Hugging Face Space & FastAPI ASGI Server Entrypoint
├── config.py                  # Unified Root Configuration (Paths, Model Weights, Specs)
├── core/                      # Core Machine Learning & Medical Imaging Engine
│   ├── __init__.py            # Package exports
│   ├── dicom_parser.py        # DICOM (.dcm) pixel decompression & metadata tags
│   ├── ensemble.py            # 4-Model weighted soft-voting consensus logic
│   ├── gradcam.py             # Gradient-weighted Class Activation Mapping (Grad-CAM)
│   ├── model_manager.py       # Thread-safe lazy model caching & inference manager
│   ├── preprocessor.py        # Radiograph CLAHE normalization & resizing (128x128)
│   ├── report_generator.py    # Publication-grade Clinical PDF Report generation
│   ├── sample_manager.py      # Synthetic radiograph study generator & catalog
│   └── validator.py           # Secure file validation & collision-free path generator
├── models/                    # Deep learning model weights (Tracked via Git LFS)
│   ├── mobilenet_model.h5     # MobileNetV2 (~11.5 MB)
│   ├── resnet50_model.h5      # ResNet50 (~101 MB)
│   ├── efficientnet_model.h5  # EfficientNetB0 (~20.8 MB)
│   └── VGG19_model.h5         # VGG19 (~443 MB)
├── frontend/                  # Modern Next.js 15 Medical Dashboard (Vercel-ready)
│   ├── app/                   # App Router UI & Serverless API Proxies
│   ├── components/            # Workstation components (DICOM Viewer, Grad-CAM slider)
│   └── package.json
├── static/                    # Generated output assets & sample studies
│   ├── samples/               # Pre-synthesized normal, bacterial & viral samples
│   └── uploads/               # Processed scans, heatmaps & clinical reports
├── tests/                     # 31 Comprehensive Pytest Unit & Integration Tests
│   ├── test_api.py            # API endpoint health, predict & comparison tests
│   ├── test_dicom.py          # DICOM header parsing & conversion tests
│   ├── test_docs.py           # OpenAPI / Swagger UI schema validation tests
│   ├── test_ensemble.py       # Consensus voting & weighting algorithm tests
│   ├── test_gradcam.py        # Grad-CAM heatmap & overlay blending tests
│   ├── test_model_manager.py  # ModelManager singleton & inference tests
│   ├── test_preprocessor.py   # Tensor scaling & image normalization tests
│   ├── test_report_generator.py# ReportLab PDF compilation tests
│   └── test_samples.py        # Synthetic sample radiograph generation tests
├── Dockerfile                 # Production container specification
├── docker-compose.yml         # Multi-container orchestration
└── requirements.txt           # Production Python dependencies
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Environment Setup
```powershell
git clone https://github.com/Shahab-khan396/-Pneumonia-Diagnostic-Hub-AI-Radiology-Workstation-Multi-Model-Consensus-Engine-DICOM-Ready-.git
cd -Pneumonia-Diagnostic-Hub-AI-Radiology-Workstation-Multi-Model-Consensus-Engine-DICOM-Ready-

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the AI API Server
```powershell
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```
Interactive Swagger API documentation will be available at `http://localhost:7860/docs`.

### 3. Run the Next.js Frontend
```powershell
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` to access the modern radiology workstation.

### 4. Run the Full Test Suite
```powershell
pytest tests/ -v
```
All **31 automated tests** execute in under 20 seconds.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/hub_api/health` | Service health status & loaded model catalog |
| `GET` | `/hub_api/models` | Available model metadata, parameter counts & weights |
| `GET` | `/hub_api/samples`| Pre-packaged sample studies (Normal, Bacterial, Viral) |
| `POST` | `/hub_api/predict`| Single-model inference with Grad-CAM & PDF report |
| `POST` | `/hub_api/compare`| 4-Model consensus evaluation with agreement rating |
| `GET` | `/hub_api/reports/{filename}` | Download generated Clinical PDF Report |
| `GET` | `/docs` | Interactive Swagger UI API documentation |

---

## 📜 License
Released under the [MIT License](LICENSE).