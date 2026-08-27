import os
import sys
import uuid
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Configure directory resolution
ROOT_DIR = Path(__file__).resolve().parent
FLASK_APP_DIR = ROOT_DIR / "Flask Application"

if str(FLASK_APP_DIR) not in sys.path:
    sys.path.insert(0, str(FLASK_APP_DIR))

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import gradio as gr

# Hugging Face ZeroGPU compatibility
try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except Exception:
    def GPU_DECORATOR(func):
        return func

from config import AVAILABLE_MODELS, DEFAULT_MODEL, SAMPLES_CATALOG
from core.preprocessor import preprocess_image
from core.model_manager import get_model_manager
from core.ensemble import run_multi_model_comparison
from core.dicom_parser import is_dicom_file, parse_dicom_file, parse_dicom_bytes
from core.report_generator import generate_clinical_pdf_report
from core.sample_manager import ensure_samples_generated, get_sample_info

# Initialize sample catalog
ensure_samples_generated()


@GPU_DECORATOR
def analyze_radiograph(
    image_input,
    model_selection: str,
    enable_gradcam: bool,
    patient_id: str,
    patient_age: str,
    patient_gender: str
):
    """
    Main diagnostic inference pipeline decorated for Hugging Face ZeroGPU execution.
    """
    if image_input is None:
        return (
            "⚠️ Please upload a chest radiograph or select one of the pre-loaded clinical samples.",
            None,
            None,
            {},
            [],
            None
        )

    scan_id = str(uuid.uuid4())[:8].upper()
    temp_dir = Path(tempfile.gettempdir()) / "pneumonia_hub"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Resolve image source and handle DICOM if applicable
    dicom_meta = {}
    if isinstance(image_input, str) and Path(image_input).exists():
        input_path = Path(image_input)
        if input_path.suffix.lower() == ".dcm" or is_dicom_file(input_path):
            img_bgr, dicom_meta, orig_save_path = parse_dicom_file(
                input_path, output_jpg_path=temp_dir / f"{scan_id}_source.jpg"
            )
        else:
            orig_save_path = input_path
            img_bgr = cv2.imread(str(input_path))
    elif isinstance(image_input, np.ndarray):
        img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR) if image_input.ndim == 3 else cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        orig_save_path = temp_dir / f"{scan_id}_source.jpg"
        cv2.imwrite(str(orig_save_path), img_bgr)
    else:
        orig_save_path = temp_dir / f"{scan_id}_source.jpg"
        img_bgr = cv2.imread(str(orig_save_path))

    # Preprocess for CNN
    try:
        img_tensor = preprocess_image(orig_save_path)
    except Exception as e:
        return f"❌ Preprocessing Error: {str(e)}", None, None, {}, [], None

    # Model Execution (Single vs Ensemble)
    model_manager = get_model_manager()
    is_ensemble = (model_selection == "Multi-Model Consensus Ensemble (All 4 Models)")

    if is_ensemble:
        result = run_multi_model_comparison(
            image_tensor=img_tensor,
            original_image_path=orig_save_path,
            base_filename=f"{scan_id}",
            generate_cams=enable_gradcam
        )
        verdict = result["consensus_verdict"]
        confidence = result["consensus_confidence"]
        probs = {
            "NORMAL": result["consensus_probabilities"]["NORMAL"] / 100.0,
            "PNEUMONIA": result["consensus_probabilities"]["PNEUMONIA"] / 100.0,
        }
        status_md = f"""### 🩺 Consensus Verdict: **{verdict}**
**Confidence:** `{confidence:.2f}%` • **Agreement:** `{result['agreement_text']}`
**Latency:** `{result['total_inference_time_ms']} ms` across 4 deep learning backbones
"""
        breakdown_table = [
            [m["name"], m["parameters"], f"{m['weight']*100:.0f}%", m["prediction"], f"{m['confidence']:.2f}%", f"{m['inference_time_ms']} ms"]
            for m in result["models_breakdown"]
        ]
        gradcam_path = result.get("primary_gradcam_overlay_url")
    else:
        # Map choice to model ID
        model_id_map = {
            "MobileNetV2 (Recommended • 87.5% Acc)": "mobilenet",
            "ResNet50 (Deep Residual • 25.6M Params)": "resnet50",
            "EfficientNetB0 (Compound Scaling)": "efficientnet",
            "VGG19 (19-Layer Baseline)": "VGG19",
        }
        model_id = model_id_map.get(model_selection, DEFAULT_MODEL)
        result = model_manager.predict(
            model_id=model_id,
            image_tensor=img_tensor,
            generate_cam=enable_gradcam,
            original_image_path=orig_save_path,
            base_filename=f"{scan_id}_{model_id}"
        )
        verdict = result["prediction"]
        confidence = result["confidence"]
        probs = {
            "NORMAL": result["probabilities"]["NORMAL"] / 100.0,
            "PNEUMONIA": result["probabilities"]["PNEUMONIA"] / 100.0,
        }
        status_md = f"""### 🩺 Diagnostic Verdict: **{verdict}**
**Confidence:** `{confidence:.2f}%` • **Model:** `{result['model_name']}`
**Inference Latency:** `{result['inference_time_ms']} ms`
"""
        breakdown_table = [
            [result["model_name"], result["model_parameters"], "100%", verdict, f"{confidence:.2f}%", f"{result['inference_time_ms']} ms"]
        ]
        gradcam_path = result.get("gradcam_overlay_url")

    # Grad-CAM visualization image
    cam_display = None
    if gradcam_path:
        full_cam_path = FLASK_APP_DIR / gradcam_path.lstrip("/")
        if full_cam_path.exists():
            cam_display = cv2.cvtColor(cv2.imread(str(full_cam_path)), cv2.COLOR_BGR2RGB)

    orig_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Generate Official Clinical PDF Report
    pdf_report_path = None
    try:
        patient_info = {
            "patient_id": patient_id or f"PT-{scan_id}",
            "age": patient_age or "45",
            "gender": patient_gender or "Unspecified",
            "clinical_indication": "Suspected pulmonary consolidation / acute respiratory infection",
            "dicom_metadata": dicom_meta
        }
        full_overlay_p = (FLASK_APP_DIR / gradcam_path.lstrip("/")) if gradcam_path else None
        pdf_path = generate_clinical_pdf_report(
            scan_id=scan_id,
            prediction_data=result,
            original_image_path=orig_save_path,
            gradcam_overlay_path=full_overlay_p if (full_overlay_p and full_overlay_p.exists()) else None,
            patient_metadata=patient_info,
            output_dir=temp_dir
        )
        pdf_report_path = str(pdf_path)
    except Exception as e:
        print(f"[!] PDF generation warning: {e}")

    return status_md, orig_display, cam_display, probs, breakdown_table, pdf_report_path


# Define pre-packaged sample helper
def load_sample(sample_key: str):
    info = get_sample_info(sample_key)
    if info and info["file_path"].exists():
        return str(info["file_path"])
    return None


# Build Professional Gradio Interface
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate"
)

with gr.Blocks(theme=custom_theme, title="Pneumonia Diagnostic Hub • AI Radiology Workstation") as demo:
    gr.Markdown(
        """
        # 🫁 Pneumonia Diagnostic Hub
        ### AI Radiology Workstation & Multi-Model Consensus Engine
        *High-accuracy automated chest radiograph screening, weighted multi-model consensus, Grad-CAM visual explainability, and DICOM integration.*
        
        ---
        """
    )

    with gr.Row():
        # Left Column: Inputs & Study Controls
        with gr.Column(scale=4):
            gr.Markdown("### 📥 Study Intake & Parameters")
            image_input = gr.Image(
                label="Chest Radiograph / DICOM Upload",
                type="filepath",
                height=300
            )

            gr.Markdown("**Quick Clinical Samples:**")
            with gr.Row():
                btn_normal = gr.Button("🟢 Normal CXR", size="sm")
                btn_bacterial = gr.Button("🔴 Bacterial Lobar", size="sm")
                btn_viral = gr.Button("🟡 Viral Interstitial", size="sm")

            model_selector = gr.Dropdown(
                label="Diagnostic Architecture",
                choices=[
                    "Multi-Model Consensus Ensemble (All 4 Models)",
                    "MobileNetV2 (Recommended • 87.5% Acc)",
                    "ResNet50 (Deep Residual • 25.6M Params)",
                    "EfficientNetB0 (Compound Scaling)",
                    "VGG19 (19-Layer Baseline)"
                ],
                value="Multi-Model Consensus Ensemble (All 4 Models)"
            )

            with gr.Accordion("📋 Patient Demographics & Options", open=False):
                with gr.Row():
                    patient_id = gr.Textbox(label="Patient ID", value="PT-2026-891")
                    patient_age = gr.Textbox(label="Age", value="52")
                    patient_gender = gr.Dropdown(label="Gender", choices=["Male", "Female", "Other"], value="Male")
                enable_cam = gr.Checkbox(label="Generate Grad-CAM Heatmap Overlays", value=True)

            run_btn = gr.Button("⚡ Run AI Radiology Diagnostic", variant="primary", size="lg")

        # Right Column: Diagnostic Intelligence & Evidence
        with gr.Column(scale=5):
            gr.Markdown("### 🩺 Diagnostic Intelligence & Visual Evidence")
            verdict_output = gr.Markdown("### *Upload a chest radiograph to begin analysis.*")
            
            with gr.Row():
                orig_image_out = gr.Image(label="Source Radiograph", height=260)
                cam_image_out = gr.Image(label="Grad-CAM Anomaly Overlay", height=260)

            prob_output = gr.Label(label="Class Probability Distribution", num_top_classes=2)

            with gr.Accordion("📊 Multi-Model Comparison Breakdown", open=True):
                breakdown_table_out = gr.Dataframe(
                    headers=["Model", "Parameters", "Weight", "Prediction", "Confidence", "Latency"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    label="Model Consensus Telemetry"
                )

            pdf_download_out = gr.File(label="📄 Official Clinical PDF Diagnostic Report")

    gr.Markdown(
        """
        ---
        > **⚠️ Medical & Research Disclaimer:** This workstation is an educational and research decision-support tool. It is not a certified medical device and must not be used as the sole basis for clinical diagnosis. A certified radiologist or physician must verify all findings.
        """
    )

    # Button Event Handlers
    btn_normal.click(fn=lambda: load_sample("sample_normal"), outputs=image_input)
    btn_bacterial.click(fn=lambda: load_sample("sample_bacterial"), outputs=image_input)
    btn_viral.click(fn=lambda: load_sample("sample_viral"), outputs=image_input)

    run_btn.click(
        fn=analyze_radiograph,
        inputs=[image_input, model_selector, enable_cam, patient_id, patient_age, patient_gender],
        outputs=[verdict_output, orig_image_out, cam_image_out, prob_output, breakdown_table_out, pdf_download_out]
    )


# ─── FastAPI REST API Engine ──────────────────────────────────────────────────
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import base64

api_router = APIRouter()


@api_router.get("/hub_api/health")
def health_api():
    return {
        "status": "healthy",
        "service": "Pneumonia-Diagnostic-Hub-HF-Engine",
        "hf_space_url": "https://shahabkhan396-pneumonia-hub.hf.space"
    }


@api_router.post("/hub_api/predict")
@GPU_DECORATOR
def predict_api_endpoint(
    file: UploadFile = File(...),
    model_choice: str = Form("mobilenet"),
    explain: str = Form("true"),
    generate_report: str = Form("false"),
):
    scan_id = uuid.uuid4().hex[:8].upper()
    temp_dir = Path(tempfile.gettempdir()) / "pneumonia_hub"
    temp_dir.mkdir(parents=True, exist_ok=True)

    raw_bytes = file.file.read()
    temp_file = temp_dir / f"{scan_id}_{file.filename or 'scan.jpg'}"
    temp_file.write_bytes(raw_bytes)

    try:
        dicom_meta = None
        if is_dicom_file(temp_file):
            _, dicom_meta, converted_jpg = parse_dicom_file(temp_file)
            input_path = converted_jpg
        else:
            input_path = temp_file

        img_tensor = preprocess_image(input_path)
        manager = get_model_manager()

        res = manager.predict(
            model_id=model_choice,
            image_tensor=img_tensor,
            generate_cam=(explain.lower() in ["true", "1", "yes"]),
            original_image_path=input_path,
            base_filename=f"{scan_id}_{model_choice}.jpg"
        )

        # Convert Grad-CAM image to base64 if available
        cam_b64 = None
        if res.get("has_gradcam") and res.get("gradcam_overlay_url"):
            cam_path = FLASK_APP_DIR / res["gradcam_overlay_url"].lstrip("/")
            if cam_path.exists():
                cam_b64 = base64.b64encode(cam_path.read_bytes()).decode("utf-8")

        res["scan_id"] = scan_id
        res["gradcam_overlay_b64"] = cam_b64
        res["dicom_metadata"] = dicom_meta
        return res
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})
    finally:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)


@api_router.post("/hub_api/compare")
@GPU_DECORATOR
def compare_api_endpoint(
    file: UploadFile = File(...),
    explain: str = Form("true"),
    generate_report: str = Form("false"),
):
    scan_id = uuid.uuid4().hex[:8].upper()
    temp_dir = Path(tempfile.gettempdir()) / "pneumonia_hub"
    temp_dir.mkdir(parents=True, exist_ok=True)

    raw_bytes = file.file.read()
    temp_file = temp_dir / f"{scan_id}_{file.filename or 'scan.jpg'}"
    temp_file.write_bytes(raw_bytes)

    try:
        dicom_meta = None
        if is_dicom_file(temp_file):
            _, dicom_meta, converted_jpg = parse_dicom_file(temp_file)
            input_path = converted_jpg
        else:
            input_path = temp_file

        img_tensor = preprocess_image(input_path)
        res = run_multi_model_comparison(
            image_tensor=img_tensor,
            original_image_path=input_path,
            base_filename=f"{scan_id}.jpg",
            generate_cams=(explain.lower() in ["true", "1", "yes"])
        )

        cam_b64 = None
        if res.get("primary_gradcam_overlay_url"):
            cam_path = FLASK_APP_DIR / res["primary_gradcam_overlay_url"].lstrip("/")
            if cam_path.exists():
                cam_b64 = base64.b64encode(cam_path.read_bytes()).decode("utf-8")

        res["scan_id"] = scan_id
        res["gradcam_overlay_b64"] = cam_b64
        res["dicom_metadata"] = dicom_meta
        return res
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})
    finally:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)


# Mount REST API router directly onto Gradio's internal FastAPI app
demo.app.include_router(api_router)
demo.queue()

app = demo.app

if __name__ == "__main__":
    demo.launch()



