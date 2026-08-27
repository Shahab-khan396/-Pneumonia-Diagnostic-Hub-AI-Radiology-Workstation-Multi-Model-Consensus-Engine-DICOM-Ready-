# Software Requirements Specification (SRS)
## Pneumonia Diagnostic Hub — AI Radiology Workstation & Multi-Model Consensus Engine

---

| Field | Detail |
| :--- | :--- |
| **Document ID** | SRS-PDH-v2.4.0 |
| **Version** | 2.4.0 |
| **Status** | Final |
| **Author** | Shahab Khan |
| **Organization** | Independent AI Research & Development |
| **Date** | August 27, 2026 |
| **Classification** | Educational / Research Use Only |
| **Change from v2.3** | Architecture updated to 3-tier decoupled deployment: Next.js 14 (Frontend), FastAPI (Backend), Hugging Face Spaces (AI Inference) |

---

> [!CAUTION]
> **Medical Disclaimer.** This system is an experimental educational and clinical research decision-support tool. It is **not** a certified medical device under FDA 21 CFR Part 820, CE Mark, or any regulatory authority. It must **never** be used as the sole basis for clinical diagnosis or patient treatment. A licensed and qualified radiologist or physician must review all outputs and original source imaging data before any clinical decision is made.

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [System Architecture](#3-system-architecture)
4. [Functional Requirements](#4-functional-requirements)
5. [Non-Functional Requirements](#5-non-functional-requirements)
6. [External Interface Requirements](#6-external-interface-requirements)
7. [Data Requirements](#7-data-requirements)
8. [Security Requirements](#8-security-requirements)
9. [AI Model Specifications](#9-ai-model-specifications)
10. [Constraints & Assumptions](#10-constraints--assumptions)
11. [Appendix A — REST API Contract](#appendix-a--rest-api-contract)
12. [Appendix B — Glossary](#appendix-b--glossary)

---

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification defines the complete functional, non-functional, interface, and AI model requirements for the **Pneumonia Diagnostic Hub (PDH)** — an AI-powered radiology workstation designed to assist clinicians and researchers in automated chest radiograph (CXR) pneumonia screening.

### 1.2 Project Scope
The Pneumonia Diagnostic Hub is a web-based multi-model inference system that:
- Accepts chest radiograph images (PNG, JPG, JPEG, WEBP) and DICOM (.dcm) files.
- Applies Convolutional Neural Network (CNN) inference pipelines across four deep learning architectures.
- Computes a weighted multi-model soft-voting consensus verdict.
- Generates Grad-CAM (Gradient-weighted Class Activation Mapping) visual explainability overlays.
- Produces downloadable, publication-quality PDF Clinical Diagnostic Reports.
- Exposes both a Next.js Browser Workstation UI and a programmatic FastAPI REST API.
- Deploys across a 3-tier architecture: Vercel (Next.js frontend), Render (FastAPI backend), Hugging Face Spaces (TensorFlow AI inference engine).

### 1.3 Intended Audience

| Audience | Purpose |
| :--- | :--- |
| **Clinical Researchers** | Evaluate AI screening accuracy on retrospective CXR datasets |
| **Medical AI Developers** | Extend or integrate the REST API into external diagnostic pipelines |
| **Radiologists (Research Context)** | Supplement manual interpretation with AI-derived probability scores and XAI attention maps |
| **Computer Science Students** | Study applied deep learning inference systems and XAI techniques |
| **Hospital IT Administrators** | Evaluate the system for potential clinical research integration |

### 1.4 Document Conventions
- **SHALL** — Mandatory requirement.
- **SHOULD** — Recommended requirement.
- **MAY** — Optional, desirable feature.
- **FR-XXX** — Functional Requirement identifier.
- **NFR-XXX** — Non-Functional Requirement identifier.
- **SEC-XXX** — Security Requirement identifier.

### 1.5 References
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV 2017.
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
- Tan, M., Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.
- Simonyan, K., Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* ICLR 2015.
- NEMA PS 3 / ISO 12052 — DICOM Standard for Medical Imaging Communication.

---

## 2. Overall Description

### 2.1 Product Perspective

```
[Tier 1 — Vercel]          [Tier 2 — Render]             [Tier 3 — Hugging Face]

Clinician --> Next.js UI ──> FastAPI Backend ──────────> HF Spaces (TensorFlow)
                │                   │            HTTPS          │
                │             DICOM Parser                Grad-CAM Engine
                │             PDF Generator               Ensemble Engine
                │             ReportLab                   ModelManager
                │                   │                          │
                └─── Results + PDF URL ◄── JSON Response ◄─────┘
```

### 2.2 Product Functions Summary

| Function | Description |
| :--- | :--- |
| **CXR Upload & Validation** | Accept, validate, and sanitize uploaded radiograph files |
| **DICOM Parsing** | Decode DICOM (.dcm) pixel data, apply VOI LUT windowing, extract metadata |
| **CNN Inference** | Run single-model or all-four-model weighted ensemble inference |
| **Grad-CAM XAI** | Compute and visualize attention heatmaps from target convolutional layers |
| **Consensus Engine** | Aggregate weighted soft-voting probability scores from all 4 backbones |
| **PDF Report Generation** | Generate structured clinical diagnostic PDF reports with demographics |
| **Sample Radiograph Library** | Provide pre-generated synthetic CXR samples for workflow testing |
| **REST API** | Expose all diagnostic capabilities via versioned JSON API endpoints |
| **Interactive UI** | Browser-based workstation with real-time inference and visualization |

### 2.3 User Classes

#### 2.3.1 Primary Research Clinician
- Interacts via the browser workstation at `/`.
- Uploads patient CXR files or DICOM series.
- Selects model architecture, reviews probability distribution, Grad-CAM overlays, and downloads PDF reports.

#### 2.3.2 API Developer / Integration Engineer
- Interacts via the REST API at `/api/v1/`.
- Submits CXR files programmatically using `multipart/form-data` POST requests.
- Processes JSON responses for downstream integration into HIS or clinical trial databases.

#### 2.3.3 System Administrator
- Manages application configuration via `config.py`.
- Monitors application logs and container health via `/api/v1/health`.
- Adjusts model ensemble weights, upload limits, and directory paths.

### 2.4 Operating Environment

**Tier 1 — Frontend (Vercel)**

| Component | Specification |
| :--- | :--- |
| **Framework** | Next.js 14 (App Router) |
| **Language** | TypeScript |
| **Styling** | Tailwind CSS v3 |
| **State / Data** | React Query (TanStack) + React `useState` |
| **Validation** | Zod (client-side schema validation) |
| **Deployment** | Vercel Hobby (free tier, global CDN) |
| **Node Version** | 20.x LTS |

**Tier 2 — Backend (Render)**

| Component | Specification |
| :--- | :--- |
| **Runtime** | Python 3.11+ |
| **Web Framework** | FastAPI 0.115.x |
| **ASGI Server** | Uvicorn (1 worker, async) |
| **Data Validation** | Pydantic v2 + pydantic-settings |
| **Image Processing** | OpenCV 4.x (headless) |
| **DICOM Processing** | PyDICOM 2.x |
| **PDF Generation** | ReportLab 4.x |
| **HTTP Client** | httpx (async, for HF Space calls) |
| **Rate Limiting** | SlowAPI |
| **Deployment** | Render Web Service (free tier, 512 MB RAM) |

**Tier 3 — AI Inference Engine (Hugging Face Spaces)**

| Component | Specification |
| :--- | :--- |
| **Runtime** | Python 3.10 (HF container) |
| **SDK** | Gradio (managed by HF environment) |
| **ML Framework** | TensorFlow 2.x / Keras |
| **GPU Acceleration** | ZeroGPU (NVIDIA A10G, via `@spaces.GPU` decorator) |
| **RAM** | Up to 16 GB (free tier) |
| **Deployment** | Hugging Face Spaces (free tier) |

---

## 3. System Architecture

### 3.1 Component Decomposition

**Tier 1 — Next.js Frontend**
```
frontend/
├── app/
│   ├── layout.tsx              # Root layout (metadata, providers)
│   ├── page.tsx                # Workstation UI entry point
│   └── api/
│       ├── predict/route.ts    # Server-side proxy → FastAPI /api/v1/predict
│       └── compare/route.ts    # Server-side proxy → FastAPI /api/v1/compare
├── components/
│   ├── upload/DropZone.tsx     # Drag-and-drop CXR/DICOM upload
│   ├── patient/PatientForm.tsx # Demographics input (Zod-validated)
│   ├── model/ModelSelector.tsx # CNN model selection cards
│   └── results/
│       ├── VerdictBadge.tsx    # NORMAL/PNEUMONIA result display
│       ├── ProbabilityBars.tsx # Animated confidence bars
│       ├── GradCamViewer.tsx   # Tabbed Grad-CAM viewer
│       └── EnsemblePanel.tsx   # Multi-model breakdown table
├── lib/
│   ├── types.ts                # TypeScript: PredictResponse, EnsembleResponse
│   ├── api.ts                  # FastAPI client functions
│   └── validation.ts           # Zod schemas
└── next.config.ts              # Next.js config (rewrites, image domains)
```

**Tier 2 — FastAPI Backend**
```
backend/
├── main.py                     # FastAPI app factory, CORS, rate limiting
├── config.py                   # Pydantic BaseSettings (env vars)
├── requirements.txt            # Python dependencies (NO tensorflow, NO .h5)
├── routers/
│   ├── predict.py              # POST /api/v1/predict
│   ├── compare.py              # POST /api/v1/compare
│   ├── report.py               # GET  /api/v1/report/{filename}
│   ├── samples.py              # GET  /api/v1/samples
│   └── health.py               # GET  /api/v1/health
├── core/
│   ├── dicom_parser.py         # DICOM pixel decoding & metadata extraction
│   ├── report_generator.py     # Clinical PDF report composition (ReportLab)
│   ├── sample_manager.py       # Synthetic sample radiograph generation & catalog
│   ├── hf_client.py            # Async httpx client for HF Spaces inference API
│   └── validator.py            # File upload validation & path sanitization
├── schemas/
│   ├── predict.py              # Pydantic: PredictResponse, ErrorResponse
│   └── compare.py              # Pydantic: EnsembleResponse, ModelResult
└── static/
    ├── samples/                # Pre-generated synthetic CXR samples
    └── reports/                # Generated PDF reports (ephemeral)
```

**Tier 3 — HF Spaces AI Inference Engine**
```
pneumonia-hub/ (HF Space root)
├── app.py                      # Gradio UI + Flask-compat REST API
├── requirements.txt            # TF, pydicom, opencv-headless, spaces
├── Flask Application/core/
│   ├── model_manager.py        # Thread-safe singleton + @spaces.GPU decorator
│   ├── preprocessor.py         # Image normalization & tensor preparation
│   ├── gradcam.py              # Grad-CAM heatmap computation & overlay rendering
│   ├── ensemble.py             # Multi-model soft-voting consensus aggregation
│   └── dicom_parser.py         # DICOM pixel decoding (parse_dicom_bytes)
├── VGG19_model.h5              # VGG19 weights (443 MB — Git LFS)
├── resnet50_model.h5           # ResNet50 weights (101 MB — Git LFS)
├── efficientnet_model.h5       # EfficientNetB0 weights (20 MB — Git LFS)
└── mobilenet_model.h5          # MobileNetV2 weights (11 MB — Git LFS)
```

### 3.2 Request Lifecycle

```
[Browser — Next.js UI]
        │
        │  fetch("/api/predict")  multipart/form-data
        ▼
[Next.js API Route — app/api/predict/route.ts]
        │  (server-side proxy, keeps FASTAPI_URL secret)
        │  POST with X-API-Key header
        ▼
[FastAPI Backend — /api/v1/predict]
        │
        ├── Pydantic validation + file size check
        ├── DICOM detection + parse_dicom_bytes() [if .dcm]
        ├── httpx.AsyncClient.post() ──────────────────────────────────┐
        │                                                               │
        │                                [HF Space — /api/v1/predict]  │
        │                                        │                     │
        │                                  preprocess_image()          │
        │                                  @spaces.GPU inference       │
        │                                  compute_gradcam_heatmap()   │
        │                                  Return JSON + base64 CAM    │
        │                                                               │
        ├── Decode base64 Grad-CAM → save to static/reports/ ◄─────────┘
        ├── generate_clinical_pdf_report()
        └── Return JSON to Next.js → forwarded to browser
```

---

## 4. Functional Requirements

### 4.1 Image Upload & Validation

| ID | Requirement |
| :--- | :--- |
| **FR-001** | The system SHALL accept image file uploads in PNG, JPG, JPEG, and WEBP formats. |
| **FR-002** | The system SHALL accept DICOM (.dcm) medical image file uploads. |
| **FR-003** | The system SHALL reject uploads exceeding 32 MB with a `400 Bad Request` error and descriptive message. |
| **FR-004** | The system SHALL sanitize all uploaded filenames to remove path traversal characters, spaces, and special characters. |
| **FR-005** | The system SHALL assign a UUID-prefixed safe filename to each upload to prevent conflicts. |
| **FR-006** | The system SHALL validate file extension AND MIME type, rejecting mismatched files. |
| **FR-007** | The system SHALL store all uploaded files in the `static/uploads/` directory with write permissions. |

### 4.2 DICOM Processing

| ID | Requirement |
| :--- | :--- |
| **FR-010** | The system SHALL detect DICOM files by examining the `DICM` magic header at byte offset 128–132. |
| **FR-011** | The system SHALL decode DICOM pixel arrays using PyDICOM with `force=True` to handle non-standard clinical headers. |
| **FR-012** | The system SHALL apply Rescale Slope and Rescale Intercept transformations to raw pixel values. |
| **FR-013** | The system SHALL apply VOI LUT windowing for optimal pulmonary anatomy visualization. |
| **FR-014** | The system SHALL handle MONOCHROME1 photometric interpretation by inverting pixel values to standard MONOCHROME2. |
| **FR-015** | The system SHALL extract and return the following DICOM metadata fields: PatientID, PatientName, PatientAge, PatientSex, StudyDate, Modality, BodyPartExamined, Manufacturer, KVP, ExposureTime, PhotometricInterpretation, Rows, Columns. |
| **FR-016** | The system SHALL convert decoded DICOM pixel data to 8-bit 3-channel JPEG format for downstream CNN preprocessing. |

### 4.3 Image Preprocessing

| ID | Requirement |
| :--- | :--- |
| **FR-020** | The system SHALL convert all input images to grayscale before CNN processing, matching the training data pipeline. |
| **FR-021** | The system SHALL resize all images to 128x128 pixels using area interpolation (INTER_AREA). |
| **FR-022** | The system SHALL merge the single grayscale channel into a 3-channel pseudo-RGB tensor for transfer learning backbone compatibility. |
| **FR-023** | The system SHALL normalize pixel values from [0, 255] to [0.0, 1.0] float32. |
| **FR-024** | The system SHALL reshape the tensor to batch format (1, 128, 128, 3) for model input. |
| **FR-025** | The system SHALL accept input from file paths, raw bytes, and existing NumPy arrays. |

### 4.4 Single-Model Inference

| ID | Requirement |
| :--- | :--- |
| **FR-030** | The system SHALL load Keras `.h5` model files with `compile=False` for accelerated inference-only mode. |
| **FR-031** | The system SHALL cache all loaded models in-memory in a thread-safe singleton `ModelManager` to eliminate repeated disk I/O. |
| **FR-032** | The system SHALL pre-warm the default model (MobileNetV2) on application startup. |
| **FR-033** | The system SHALL support runtime selection from 4 registered models: `mobilenet`, `resnet50`, `efficientnet`, `VGG19`. |
| **FR-034** | The system SHALL return the predicted class label (NORMAL or PNEUMONIA), confidence percentage, and both class probability scores. |
| **FR-035** | The system SHALL measure and report inference latency in milliseconds using `time.perf_counter()` high-resolution timer. |
| **FR-036** | The system SHALL return model metadata including name, parameter count, badge label, and target convolutional layer. |

### 4.5 Multi-Model Consensus Engine (Ensemble)

| ID | Requirement |
| :--- | :--- |
| **FR-040** | The system SHALL execute inference across all 4 registered model backbones when the ensemble endpoint is invoked. |
| **FR-041** | The system SHALL compute a weighted soft-voting consensus probability score: `P_consensus(C) = Sum_m [ weight_m * P_m(C) ]`. |
| **FR-042** | The system SHALL apply the following default ensemble weights: MobileNetV2 = 0.45, ResNet50 = 0.25, EfficientNetB0 = 0.20, VGG19 = 0.10. |
| **FR-043** | The system SHALL normalize consensus probabilities by the sum of weights to account for any weight misconfiguration. |
| **FR-044** | The system SHALL determine the consensus verdict as PNEUMONIA if `P_consensus(PNEUMONIA) >= 50.0%`. |
| **FR-045** | The system SHALL count individual model votes and classify agreement as: UNANIMOUS (4/4), STRONG_MAJORITY (3/4), or SPLIT_DECISION (2/4). |
| **FR-046** | The system SHALL return a structured breakdown for each individual model result in the ensemble response. |
| **FR-047** | The system SHALL report total ensemble inference latency (wall-clock time across all 4 model predictions). |

### 4.6 Grad-CAM Explainability Engine

| ID | Requirement |
| :--- | :--- |
| **FR-050** | The system SHALL compute Grad-CAM heatmaps using TensorFlow `GradientTape` to capture gradients of the predicted class score with respect to the feature maps of the target convolutional layer. |
| **FR-051** | The system SHALL apply Global Average Pooling over the gradient tensor to produce per-channel importance weights. |
| **FR-052** | The system SHALL apply ReLU activation to the weighted feature map sum to retain only positively correlated activations. |
| **FR-053** | The system SHALL normalize the resulting heatmap to [0.0, 1.0]. |
| **FR-054** | The system SHALL use the following target layer mapping per model: MobileNetV2 -> `out_relu`, ResNet50 -> `conv5_block3_out`, EfficientNetB0 -> `top_activation`, VGG19 -> `block5_conv4`. |
| **FR-055** | The system SHALL fall back to the last convolutional or activation layer if the named target layer is not found in the model graph. |
| **FR-056** | The system SHALL generate three output visualizations: (1) Grad-CAM overlay (alpha-blended), (2) standalone heatmap (JET colormap), (3) side-by-side 3-panel composite. |
| **FR-057** | The system SHALL save all three Grad-CAM visualizations to `static/uploads/` and return their public URL paths in the API response. |
| **FR-058** | Grad-CAM generation SHALL be configurable per-request via the `explain` parameter (default: `true`). |

### 4.7 PDF Clinical Report Generation

| ID | Requirement |
| :--- | :--- |
| **FR-060** | The system SHALL generate a structured, multi-page PDF report using ReportLab upon completion of each inference request. |
| **FR-061** | The PDF report SHALL include: Report Header & Branding, Patient Demographics & Clinical Indication, AI Diagnostic Findings (verdict, confidence, probabilities), Model Telemetry (architecture, latency, parameters), Ensemble Breakdown Table (if applicable), Visual Evidence (embedded CXR and Grad-CAM overlay), and Medical Disclaimer. |
| **FR-062** | The system SHALL embed the original radiograph and Grad-CAM overlay image (if available) directly into the PDF. |
| **FR-063** | The PDF SHALL be named `report_{scan_id}.pdf` and stored in `static/uploads/`. |
| **FR-064** | The API SHALL expose a `/api/v1/report/{filename}` endpoint for authorized download of generated PDF reports. |
| **FR-065** | PDF generation SHALL be configurable per-request via the `generate_report` parameter (default: `true`). |

### 4.8 Sample Radiograph Library

| ID | Requirement |
| :--- | :--- |
| **FR-070** | The system SHALL include a library of 3 pre-packaged synthetic clinical sample radiographs: Normal Clear CXR, Bacterial Lobar Pneumonia, and Viral Interstitial Pneumonia. |
| **FR-071** | Sample images SHALL be generated programmatically at startup using OpenCV if not already present on disk. |
| **FR-072** | Each synthetic radiograph SHALL depict anatomically realistic features including thoracic rib cage, clavicles, mediastinum, cardiac silhouette, diaphragmatic domes, and lung-specific pathology patterns. |
| **FR-073** | The API SHALL expose a `/api/v1/samples` catalog endpoint listing all available samples with metadata and preview URLs. |
| **FR-074** | Users SHALL be able to initiate inference on any sample without uploading a file by specifying `sample_id` in the request. |

### 4.9 REST API

| ID | Requirement |
| :--- | :--- |
| **FR-080** | The system SHALL expose a versioned REST API under the `/api/v1/` prefix. |
| **FR-081** | All API responses SHALL use `application/json` content type and follow consistent response schema. |
| **FR-082** | The system SHALL expose `GET /api/v1/health` for system status and model availability checks. |
| **FR-083** | The system SHALL expose `GET /api/v1/models` for the full model registry catalog. |
| **FR-084** | The system SHALL expose `POST /api/v1/predict` for single-model inference with optional Grad-CAM and report. |
| **FR-085** | The system SHALL expose `POST /api/v1/compare` for 4-model ensemble consensus inference. |
| **FR-086** | The system SHALL expose `GET /api/v1/samples` for the pre-packaged sample catalog. |
| **FR-087** | The system SHALL expose `GET /api/v1/report/{filename}` for PDF report download. |
| **FR-088** | The FastAPI backend SHALL automatically expose interactive Swagger UI at `GET /docs` and ReDoc at `GET /redoc` via FastAPI's built-in OpenAPI generation. |
| **FR-089** | All API error responses SHALL return a JSON body with `{"success": false, "error": "<descriptive message>"}`. |

### 4.10 Next.js Browser Workstation UI

| ID | Requirement |
| :--- | :--- |
| **FR-090** | The system SHALL provide a Next.js 14 App Router workstation accessible at the Vercel deployment root (`/`). |
| **FR-091** | The `DropZone` React component SHALL support direct drag-and-drop and click-to-browse file upload, including `.dcm` DICOM files. |
| **FR-092** | The `PatientForm` React component SHALL display demographics input fields (Patient ID, Age, Gender, Clinical Indication, Referring Physician) with Zod client-side validation. |
| **FR-093** | The `ModelSelector` component SHALL provide card-based model architecture selection with descriptive labels, parameter counts, and recommendation badges. |
| **FR-094** | The `VerdictBadge` and `ProbabilityBars` components SHALL display inference results including verdict with confidence and animated probability bar charts. |
| **FR-095** | The `GradCamViewer` component SHALL render tabbed Grad-CAM visualization (Original, Heatmap, Composite) and provide inline PDF report download. |
| **FR-096** | The `EnsemblePanel` component SHALL render a per-model breakdown table with agreement indicator chip (UNANIMOUS / STRONG_MAJORITY / SPLIT_DECISION). |
| **FR-097** | The UI SHALL be fully responsive and functional on modern desktop browsers (Chrome, Firefox, Edge, Safari). |
| **FR-098** | All requests from the Next.js UI to the FastAPI backend SHALL be routed via Next.js API Routes (server-side proxy) to keep the `FASTAPI_URL` secret server-side. |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| ID | Requirement | Target |
| :--- | :--- | :--- |
| **NFR-001** | Single-model inference latency (warm model) | <= 250 ms |
| **NFR-002** | Ensemble 4-model inference latency (warm models) | <= 800 ms |
| **NFR-003** | Model loading time (cold start, from disk) | <= 15 s per model |
| **NFR-004** | PDF report generation time | <= 3 s |
| **NFR-005** | DICOM decoding and conversion time | <= 2 s |
| **NFR-006** | Maximum simultaneous Uvicorn workers (FastAPI/Render) | 1 async worker |
| **NFR-007** | Maximum HF Space startup time (model loading) | <= 60 s (all 4 models cold-loaded) |
| **NFR-008** | Next.js page initial load time (Vercel CDN) | <= 2 s (Largest Contentful Paint) |

### 5.2 Reliability

| ID | Requirement |
| :--- | :--- |
| **NFR-010** | The ModelManager SHALL handle concurrent inference requests without race conditions using `threading.Lock`. |
| **NFR-011** | Grad-CAM failures SHALL be gracefully caught and reported without crashing the inference response (`has_gradcam: false`). |
| **NFR-012** | DICOM parsing failures SHALL return descriptive `400` errors without leaking internal stack traces to API clients. |
| **NFR-013** | The application SHALL continue serving requests if PDF generation fails for an individual report. |
| **NFR-014** | The system SHALL log all internal errors with full stack traces to the application logger. |

### 5.3 Scalability

| ID | Requirement |
| :--- | :--- |
| **NFR-020** | The system SHOULD support horizontal scaling via Docker container replication behind a load balancer. |
| **NFR-021** | The ModelManager Singleton pattern SHALL ensure models are loaded once per process, not per request. |
| **NFR-022** | The system SHOULD support model hot-swapping via configuration change without application restart (future roadmap). |

### 5.4 Usability

| ID | Requirement |
| :--- | :--- |
| **NFR-030** | All API error messages SHALL use plain English descriptions without exposing internal implementation details. |
| **NFR-031** | The browser UI SHALL provide loading indicators during inference to communicate system activity. |
| **NFR-032** | The Swagger documentation SHALL include example request bodies and example responses for all endpoints. |

### 5.5 Maintainability

| ID | Requirement |
| :--- | :--- |
| **NFR-040** | All model weights, labels, ensemble weights, and layer names SHALL be configurable via `config.py` on the HF Space without modifying source code. |
| **NFR-041** | The FastAPI backend SHALL follow router-based separation of concerns (`routers/`, `core/`, `schemas/` packages). |
| **NFR-042** | All core modules (backend and HF Space) SHALL have corresponding Pytest unit test coverage. |
| **NFR-043** | Model addition SHALL require only: (1) adding the `.h5` file to the HF Space, and (2) registering metadata in `config.py::AVAILABLE_MODELS`. |
| **NFR-044** | The Next.js frontend TypeScript interfaces (`lib/types.ts`) SHALL be the single source of truth for API response shapes shared between UI components. |

### 5.6 Portability

| ID | Requirement |
| :--- | :--- |
| **NFR-050** | The FastAPI backend SHALL be fully containerizable via Docker for deployment on any OCI-compliant container runtime. |
| **NFR-051** | The HF Space (Tier 3) SHALL run on Linux (Debian/Ubuntu) within the Hugging Face Docker container without code modification. |
| **NFR-052** | The FastAPI server port SHALL be configurable via the `PORT` environment variable (assigned automatically by Render). |
| **NFR-053** | The Next.js frontend SHALL be deployable to any Vercel-compatible static/SSR hosting environment by changing the `FASTAPI_URL` environment variable. |

---

## 6. External Interface Requirements

### 6.1 User Interfaces
- **Next.js Workstation (`/` on Vercel)**: React-based multi-component application built with Next.js 14 App Router. TypeScript + Tailwind CSS. Deployed to Vercel's global CDN edge network.
- **FastAPI Swagger UI (`/docs` on Render)**: Interactive OpenAPI 3.1 documentation UI auto-generated by FastAPI from Pydantic schemas. No manual specification required.
- **FastAPI ReDoc (`/redoc` on Render)**: Alternative API reference documentation UI, also auto-generated.
- **Gradio Research UI (`/` on HF Space)**: Direct-access interactive AI workstation for research and manual testing of the inference engine.

### 6.2 API Interfaces

| Method | Endpoint | Content-Type | Purpose |
| :--- | :--- | :--- | :--- |
| GET | `/api/v1/health` | `application/json` | System health & model catalog |
| GET | `/api/v1/models` | `application/json` | Model registry |
| GET | `/api/v1/samples` | `application/json` | Sample radiograph catalog |
| POST | `/api/v1/predict` | `multipart/form-data` | Single-model inference |
| POST | `/api/v1/compare` | `multipart/form-data` | Ensemble comparison |
| GET | `/api/v1/report/{filename}` | `application/pdf` | Report download |

### 6.3 Hardware Interfaces

| Configuration | Specification |
| :--- | :--- |
| **Minimum (Development)** | 4-core CPU, 8 GB RAM, 5 GB free disk |
| **Recommended (Production)** | 4-core CPU, 16 GB RAM, 10 GB free disk |
| **GPU (Optional)** | CUDA-compatible NVIDIA GPU for accelerated TensorFlow inference |

---

## 7. Data Requirements

### 7.1 Input Data Schema

| Field | Type | Constraints | Required |
| :--- | :--- | :--- | :--- |
| `file` | Binary | PNG/JPG/JPEG/WEBP/DCM, <= 32 MB | Yes* |
| `sample_id` | String | `sample_normal`, `sample_bacterial`, or `sample_viral` | Yes* |
| `model_choice` | String | `mobilenet`, `resnet50`, `efficientnet`, or `VGG19` | No (default: `mobilenet`) |
| `explain` | Boolean string | `true`/`false`, `1`/`0`, `yes` | No (default: `true`) |
| `generate_report` | Boolean string | `true`/`false` | No (default: `true`) |
| `patient_id` | String | Max 50 chars | No |
| `patient_age` | String | Numeric string, 0-120 | No |
| `patient_gender` | String | `Male`, `Female`, `Other`, or `Unspecified` | No |
| `clinical_history` | String | Max 500 chars | No |
| `referring_physician` | String | Max 100 chars | No |

*Either `file` OR `sample_id` must be provided.

### 7.2 Predict Response Schema

```json
{
  "success": true,
  "scan_id": "A3F29C1B",
  "prediction": "PNEUMONIA",
  "confidence": 94.37,
  "probabilities": { "NORMAL": 5.63, "PNEUMONIA": 94.37 },
  "raw_probabilities": { "NORMAL": 0.0563, "PNEUMONIA": 0.9437 },
  "model_id": "mobilenet",
  "model_name": "MobileNetV2",
  "model_parameters": "3.5M",
  "model_badge": "Recommended",
  "target_conv_layer": "out_relu",
  "inference_time_ms": 87.42,
  "has_gradcam": true,
  "gradcam_overlay_url": "/static/uploads/gradcam_overlay_scan.jpg",
  "gradcam_heatmap_url": "/static/uploads/gradcam_heat_scan.jpg",
  "gradcam_composite_url": "/static/uploads/gradcam_comp_scan.jpg",
  "image_url": "/static/uploads/scan.jpg",
  "report_pdf_url": "/api/v1/report/report_A3F29C1B.pdf",
  "filename": "scan.jpg",
  "dicom_metadata": null
}
```

### 7.3 File Storage

| Tier | Path | Contents | Access |
| :--- | :--- | :--- | :--- |
| **Render (Backend)** | `backend/static/reports/` | Generated PDF reports (ephemeral) | Read/Write |
| **Render (Backend)** | `backend/static/samples/` | Pre-generated synthetic CXR samples | Read (generated at startup) |
| **HF Space (AI)** | `pneumonia-hub/*.h5` | TensorFlow model weight files | Read-only (Git LFS) |
| **HF Space (AI)** | `/tmp/` | Ephemeral Grad-CAM image files | Read/Write (cleared between restarts) |

> [!NOTE]
> No uploaded patient images are stored permanently. The Render backend processes images in-memory for PDF embedding and discards them. The HF Space receives JPEG bytes per-request only.

---

## 8. Security Requirements

| ID | Requirement |
| :--- | :--- |
| **SEC-001** | The FastAPI backend SHALL validate all uploaded filenames using `pathlib.Path` safe path construction. Pydantic SHALL enforce type and size constraints before any filesystem operation. |
| **SEC-002** | File path construction SHALL use `pathlib.Path` throughout the backend to prevent path traversal attacks (`../` injection). |
| **SEC-003** | The FastAPI backend SHALL validate both file extension AND content-type header to prevent MIME-type spoofing. |
| **SEC-004** | The `INTERNAL_API_KEY` (shared between Next.js API Routes and FastAPI) SHALL be configured via environment variables only and SHALL NOT be hardcoded in source code. |
| **SEC-005** | FastAPI `HTTPException` handlers SHALL NOT expose internal Python stack traces to API clients in production mode. |
| **SEC-006** | The system SHALL not execute or evaluate any content from uploaded files as code. |
| **SEC-007** | PDF report filenames SHALL use server-generated UUIDs and SHALL NOT reflect user-supplied input. |
| **SEC-008** | All credentials (`HF_API_TOKEN`, `INTERNAL_API_KEY`, `FASTAPI_URL`) SHALL not be committed to version control (enforced by `.gitignore`). |
| **SEC-009** | The `FASTAPI_URL` SHALL be stored as a server-side-only Next.js environment variable (no `NEXT_PUBLIC_` prefix) to prevent Render backend URL exposure to the browser. |
| **SEC-010** | The FastAPI backend SHALL enforce CORS to allow requests only from the configured Vercel frontend origin (`CORS_ORIGINS` env var). |

---

## 9. AI Model Specifications

### 9.1 Registered Model Architectures

| Model | ID | Parameters | Val. Accuracy | Ensemble Weight | Target Layer | File Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | `mobilenet` | 3.5M | 87.5% | 45% | `out_relu` | ~11 MB |
| **ResNet50** | `resnet50` | 25.6M | ~83% | 25% | `conv5_block3_out` | ~101 MB |
| **EfficientNetB0** | `efficientnet` | 5.3M | ~82% | 20% | `top_activation` | ~20 MB |
| **VGG19** | `VGG19` | 63.1M | ~79% | 10% | `block5_conv4` | ~443 MB |

### 9.2 Input Preprocessing Pipeline

```
Raw Image (any supported format)
        |
        v
    Grayscale Conversion (cv2.IMREAD_GRAYSCALE)
        |
        v
    Resize to 128x128 px (cv2.INTER_AREA)
        |
        v
    3-Channel Stack (merge grayscale to RGB)
        |
        v
    Normalize to [0.0, 1.0] float32 (/ 255.0)
        |
        v
    Batch Expansion -> (1, 128, 128, 3) tensor
        |
        v
    Model.predict(verbose=0)
```

### 9.3 Output Label Mapping

| Index | Label | Interpretation |
| :--- | :--- | :--- |
| 0 | `NORMAL` | No radiological evidence of pneumonia detected |
| 1 | `PNEUMONIA` | Radiological patterns consistent with pulmonary infection |

### 9.4 Ensemble Soft-Voting Formula

```
P_consensus(PNEUMONIA) = Sum_m [ weight_m * P_m(PNEUMONIA) ]
                       = 0.45 * P_mobilenet + 0.25 * P_resnet50
                         + 0.20 * P_efficientnet + 0.10 * P_vgg19

Final Verdict: PNEUMONIA if P_consensus(PNEUMONIA) >= 0.50
```

### 9.5 Training Dataset Reference
- **Base Dataset**: Chest X-Ray Images (Pneumonia) — Kaggle (Mooney, 2018)
- **Classes**: NORMAL (1,583 training images) vs. PNEUMONIA (4,273 training images)
- **Augmentation**: Rotation, zoom, horizontal flip, brightness adjustment
- **Split**: 80% train / 10% validation / 10% test

---

## 10. Constraints & Assumptions

### 10.1 Constraints
1. Model weight files (`.h5`) exceed GitHub's 100 MB limit and must be managed via Git LFS.
2. The VGG19 model (443 MB) may cause OOM in environments with less than 4 GB available RAM.
3. TensorFlow CUDA detection runs on startup even without GPU, producing non-fatal warning messages.
4. Hugging Face Spaces ZeroGPU is request-scoped — dedicated GPU availability is not guaranteed between requests.
5. The DICOM parser requires the standard DICM magic header at byte offset 128 for reliable file detection.

### 10.2 Assumptions
1. The four `.h5` model files are present in the Hugging Face Space root directory at runtime (tracked via Git LFS).
2. The HF Spaces container has `libgl1`, `libglib2.0-0`, and `libgomp1` system libraries available for OpenCV headless mode (specified in `packages.txt`).
3. The `static/reports/` directory on the Render backend is writable by the Uvicorn process user.
4. SSL termination is handled automatically by Vercel (frontend) and Render (backend) — no manual reverse proxy configuration is required.
5. The Next.js API Routes (`/api/predict`, `/api/compare`) act as secure proxies. The Render FastAPI URL is never exposed to the browser.
6. All uploaded chest radiographs are PA (Posterior-Anterior) or AP (Anterior-Posterior) views. The CNN models were not trained on lateral projections.
7. The Render backend free tier service may spin down after 15 minutes of inactivity. An external uptime monitor (e.g., UptimeRobot) should ping `/api/v1/health` every 5 minutes to keep it warm.

---

## Appendix A — REST API Contract

### A.1 POST /api/v1/predict

**Request (multipart/form-data):**

| Field | Value |
| :--- | :--- |
| `file` | Binary CXR image or DICOM file |
| `model_choice` | `mobilenet` or `resnet50` or `efficientnet` or `VGG19` |
| `explain` | `true` or `false` |
| `generate_report` | `true` or `false` |
| `patient_id` | String |
| `patient_age` | Numeric string |
| `patient_gender` | `Male` or `Female` or `Other` |
| `clinical_history` | String |

**Response 200 OK:**
```json
{
  "success": true,
  "prediction": "PNEUMONIA",
  "confidence": 94.37,
  "probabilities": { "NORMAL": 5.63, "PNEUMONIA": 94.37 },
  "model_id": "mobilenet",
  "inference_time_ms": 87.42,
  "has_gradcam": true,
  "gradcam_overlay_url": "/static/uploads/gradcam_overlay_scan.jpg",
  "report_pdf_url": "/api/v1/report/report_A3F29C1B.pdf"
}
```

**Response 400 Bad Request:**
```json
{ "success": false, "error": "File size exceeds maximum allowed limit of 32 MB." }
```

### A.2 POST /api/v1/compare

Same request schema as `/predict`. Additional response fields:

```json
{
  "is_ensemble": true,
  "consensus_verdict": "PNEUMONIA",
  "consensus_confidence": 91.22,
  "agreement_level": "UNANIMOUS",
  "agreement_text": "Unanimous Consensus (4/4 Models in Full Agreement)",
  "models_breakdown": [
    {
      "id": "mobilenet",
      "name": "MobileNetV2",
      "weight": 0.45,
      "prediction": "PNEUMONIA",
      "confidence": 94.37,
      "inference_time_ms": 87.42
    }
  ],
  "total_inference_time_ms": 312.5
}
```

### A.3 GET /api/v1/health

**Response 200 OK:**
```json
{
  "status": "healthy",
  "service": "Pneumonia-Diagnostic-Hub-API",
  "version": "2.3.0",
  "features": [
    "multi_model_inference", "gradcam_xai", "ensemble_consensus",
    "dicom_parser", "clinical_pdf_reporting", "sample_radiograph_library"
  ],
  "models_count": 4,
  "available_models": ["mobilenet", "resnet50", "efficientnet", "VGG19"]
}
```

---

## Appendix B — Glossary

| Term | Definition |
| :--- | :--- |
| **CXR** | Chest X-Ray — Standard radiographic imaging technique to visualize pulmonary structures |
| **DICOM** | Digital Imaging and Communications in Medicine — International standard for medical imaging data |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping — XAI technique for visual CNN explanations |
| **Ensemble** | Multi-model aggregation combining predictions from multiple models for improved robustness |
| **Soft Voting** | Ensemble strategy averaging class probability distributions from each model |
| **VOI LUT** | Value of Interest Look-Up Table — DICOM windowing transformation for anatomical visualization |
| **XAI** | Explainable Artificial Intelligence — Methods making AI decisions human-interpretable |
| **FastAPI** | Modern Python async web framework for building REST APIs with automatic OpenAPI documentation |
| **Uvicorn** | ASGI server for Python, used to serve FastAPI in production |
| **Next.js** | React framework supporting App Router, Server Components, and Vercel-native deployment |
| **Pydantic** | Python data validation library using type annotations; used for FastAPI request/response schemas |
| **Zod** | TypeScript-first schema validation library; used for Next.js form validation |
| **ASGI** | Asynchronous Server Gateway Interface — Python async web server standard (FastAPI uses ASGI) |
| **SSR** | Server-Side Rendering — Next.js renders HTML on the server for improved SEO and performance |
| **RSC** | React Server Components — Next.js 14 feature for server-rendered React components |
| **Git LFS** | Large File Storage — Git extension for versioning large binary files |
| **HIS** | Hospital Information System — Enterprise platform managing hospital operations |
| **OOM** | Out Of Memory — Condition where a process exhausts available system RAM |
| **PA / AP** | Posterior-Anterior / Anterior-Posterior — Standard CXR projection orientations |
| **CNN** | Convolutional Neural Network — Deep learning architecture for image recognition tasks |
| **ZeroGPU** | Hugging Face Spaces feature providing on-demand, request-scoped GPU acceleration |
