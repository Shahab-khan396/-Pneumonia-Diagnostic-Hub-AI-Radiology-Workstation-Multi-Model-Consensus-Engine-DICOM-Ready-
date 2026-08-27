---
title: Pneumonia Diagnostic Hub AI Workstation
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Pneumonia Diagnostic Hub

Flask and TensorFlow workstation for experimental chest-radiograph pneumonia screening. It provides single-model inference, a four-model weighted comparison, Grad-CAM visualizations, DICOM conversion, sample studies, and downloadable PDF reports through both a browser UI and a REST API.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black)
![TensorFlow](https://img.shields.io/badge/TensorFlow-model%20inference-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **Medical disclaimer:** This is an educational and research decision-support project. It is not a medical device and must not be used as the sole basis for diagnosis or treatment. A qualified clinician must review every study and the original source data.

## What the application does

- Accepts `PNG`, `JPG`, `JPEG`, `WEBP`, and `DICOM (.dcm)` uploads up to 32 MB.
- Converts DICOM pixel data to an 8-bit three-channel JPEG and extracts selected patient and acquisition metadata.
- Runs one selected CNN or all four registered models with weighted soft voting.
- Generates Grad-CAM heatmaps and overlays when explainability is enabled.
- Produces a PDF report containing the scan identifier, supplied demographics, probabilities, model telemetry, and available visual evidence.
- Includes three generated sample radiographs for testing the workflow without an upload.
- Provides a browser workstation at `/` and interactive Swagger UI at `/docs`.

The default model is MobileNetV2 (`mobilenet`). The ensemble weights are MobileNetV2 45%, ResNet50 25%, EfficientNetB0 20%, and VGG19 10%. These weights and the model metadata are defined in `Flask Application/config.py`; they are not a substitute for clinical validation.

## Repository layout

```text
Flask Application/
  app.py                 Flask application factory and development entry point
  wsgi.py                Production WSGI entry point
  config.py              Upload, model, label, and sample configuration
  core/                  Preprocessing, inference, Grad-CAM, DICOM, and PDF logic
  routes/                Web, API, and OpenAPI/Swagger blueprints
  templates/index.html   Browser workstation
  static/                Generated samples, uploads, overlays, and reports
  *_model.h5             Bundled TensorFlow/Keras model weights
tests/                    Pytest coverage for the core modules and routes
Dockerfile               Gunicorn image definition
docker-compose.yml       Local container orchestration
requirements.txt         Python dependencies
```

## Quickstart

### Prerequisites

- Python 3.11 is the tested container/runtime baseline.
- Docker and Docker Compose are an alternative to a local Python environment.
- The four `.h5` model files must remain in `Flask Application/`.

### Local installation

Run these commands from the repository root. The root `requirements.txt` is the canonical dependency file used by the Dockerfile.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

### Start the development server

```powershell
python "Flask Application/app.py"
```

Open `http://127.0.0.1:5000/` for the workstation or `http://127.0.0.1:5000/docs` for Swagger UI. Startup attempts to preload MobileNetV2; missing or unloadable weights are logged as a warning and will fail when that model is used.

### Start with Docker

```powershell
docker compose up --build
```

The compose file maps host port `5000` to container port `7860`. The Docker image runs Gunicorn with one worker and four threads, and uses `PORT` (default `7860`) and `SECRET_KEY` environment variables.

For Hugging Face Spaces, the included Docker metadata exposes port `7860`. Do not commit real patient data, production secrets, generated reports, or uploaded studies.

## API

All JSON endpoints are versioned under `/api/v1`.

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service status and model availability |
| `GET` | `/api/v1/models` | Registered model catalog and load status |
| `GET` | `/api/v1/samples` | Generated sample radiograph catalog |
| `POST` | `/api/v1/predict` | Single-model prediction, optional Grad-CAM and PDF |
| `POST` | `/api/v1/compare` | Four-model predictions and weighted consensus |
| `GET` | `/api/v1/report/<filename>` | Download a generated PDF report |
| `GET` | `/api/v1/openapi.json` | OpenAPI 3.0.3 document |

`predict` and `compare` use `multipart/form-data`. Provide either a `file` part or a `sample_id`. Optional fields include `model_choice`, `explain`, `generate_report`, `patient_id`, `patient_age`, `patient_gender`, `clinical_history`, and `referring_physician`. `model_choice` defaults to `mobilenet`; `explain` defaults to `true`; `generate_report` defaults to `true` for `predict` and reports are always generated by `compare`.

Example request using the built-in sample:

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/v1/predict `
  -F "sample_id=sample_bacterial" `
  -F "model_choice=mobilenet" `
  -F "explain=true" `
  -F "generate_report=true"
```

The response includes probabilities, inference timing, a scan ID, image/Grad-CAM URLs when available, and a `report_pdf_url`. The comparison response additionally includes per-model results, vote counts, agreement status, and consensus probabilities.

## Testing

Run the suite from the repository root:

```powershell
python -m pytest tests/ -v
```

The tests cover validation, preprocessing, model management, Grad-CAM, ensemble voting, DICOM parsing, PDF creation, sample generation, API routes, and Swagger/OpenAPI routes. Tests load the bundled TensorFlow models, so the first run can be resource-intensive.

## Data and security notes

- Uploaded images, converted DICOM JPEGs, Grad-CAM images, and reports are written under `Flask Application/static/uploads/`.
- Sample images are generated under `Flask Application/static/samples/` when the catalog is first requested.
- The default secret key is for development only. Set `SECRET_KEY` to a strong deployment-specific value.
- DICOM metadata can contain protected health information. Use anonymized fixtures and apply appropriate access controls, retention, and audit practices before any real deployment.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).