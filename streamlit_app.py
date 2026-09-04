"""
Pneumonia Diagnostic Hub • AI Radiology Workstation
Production-Grade Streamlit Clinical Decision Support System.
Features:
  - 4-Model Weighted Soft-Voting Consensus Engine (MobileNetV2, ResNet50, EfficientNetB0, VGG19)
  - Explainable AI (Grad-CAM with real-time multi-colormap & opacity blending)
  - Medical DICOM (.dcm) Header Ingestion & VOI LUT Normalization
  - Multi-Architecture Comparison Grid & Inference Latency Metrics
  - Publication-Quality Clinical PDF Diagnostic Report Export
"""

import os
import sys
import time
import uuid
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np
from PIL import Image

import streamlit as st

# Suppress TensorFlow verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import system configuration and core radiology modules
from config import (
    BASE_DIR,
    MODELS_DIR,
    STATIC_DIR,
    UPLOAD_FOLDER,
    SAMPLES_DIR,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    ENSEMBLE_WEIGHTS,
    SAMPLES_CATALOG,
    ALLOWED_EXTENSIONS,
)
from core.preprocessor import preprocess_image
from core.model_manager import get_model_manager
from core.ensemble import run_multi_model_comparison
from core.dicom_parser import is_dicom_file, parse_dicom_file
from core.report_generator import generate_clinical_pdf_report
from core.sample_manager import ensure_samples_generated, list_sample_catalog, get_sample_info
from core.gradcam import compute_gradcam_heatmap, create_gradcam_overlay, save_gradcam_visualizations

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pneumonia Diagnostic Hub • AI Radiology Workstation",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize sample catalog files
ensure_samples_generated()

# ─── Custom CSS Design System ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Radiology Workstation Glassmorphism Container */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0d1527 0%, #080c16 100%);
    }

    /* Workstation Header Banner */
    .hub-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    .hub-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .hub-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 6px;
        font-weight: 400;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .badge-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }

    .badge-emerald {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .badge-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.35);
    }

    /* Clinical Metric Card */
    .metric-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        border-color: rgba(96, 165, 250, 0.4);
        transform: translateY(-2px);
    }

    .verdict-box-normal {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.35) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.15);
    }

    .verdict-box-pneumonia {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.35) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 0 35px rgba(239, 68, 68, 0.18);
    }

    .mono-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }

    /* Subtle custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0B0F19;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Management ──────────────────────────────────────────────────
if "selected_image_path" not in st.session_state:
    st.session_state["selected_image_path"] = None
if "selected_study_name" not in st.session_state:
    st.session_state["selected_study_name"] = None
if "is_dicom" not in st.session_state:
    st.session_state["is_dicom"] = False
if "dicom_metadata" not in st.session_state:
    st.session_state["dicom_metadata"] = None
if "inference_result" not in st.session_state:
    st.session_state["inference_result"] = None
if "patient_notes" not in st.session_state:
    st.session_state["patient_notes"] = ""
if "clinician_name" not in st.session_state:
    st.session_state["clinician_name"] = "Dr. S. Khan, MD"
if "facility_name" not in st.session_state:
    st.session_state["facility_name"] = "Metropolitan Medical Center • AI Radiology Suite"


# ─── Hardware & Model Health Monitor ───────────────────────────────────────────
def check_model_availability() -> Dict[str, Dict[str, Any]]:
    """Inspect local models directory and determine weights health and size."""
    health: Dict[str, Dict[str, Any]] = {}
    for mid, meta in AVAILABLE_MODELS.items():
        fname = meta["filename"]
        mpath = MODELS_DIR / fname
        if not mpath.exists():
            mpath = BASE_DIR / fname
        
        if mpath.exists():
            size_mb = mpath.stat().st_size / (1024 * 1024)
            # Check if Git LFS pointer text file (< 1KB)
            is_lfs_pointer = mpath.stat().st_size < 1024
            health[mid] = {
                "name": meta["name"],
                "exists": True,
                "is_lfs_pointer": is_lfs_pointer,
                "size_mb": round(size_mb, 1),
                "path": mpath,
                "badge": meta["badge"]
            }
        else:
            health[mid] = {
                "name": meta["name"],
                "exists": False,
                "is_lfs_pointer": False,
                "size_mb": 0.0,
                "path": None,
                "badge": meta["badge"]
            }
    return health

models_health = check_model_availability()


# ─── Sidebar Clinical Workstation Controls ─────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
        <span style="font-size: 2rem;">🫁</span>
        <div>
            <div style="font-weight: 800; font-size: 1.15rem; color: #F8FAFC;">Pneumonia Hub</div>
            <div style="font-size: 0.75rem; color: #38BDF8; font-weight: 600; letter-spacing: 0.05em;">AI RADIOLOGY WORKSTATION</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Diagnostic Mode Selector
    st.markdown("##### ⚙️ Screening Architecture")
    op_mode = st.radio(
        "Inference Engine Mode",
        ["Multi-Model Consensus", "Single Architecture Deep-Dive"],
        index=0,
        help="Consensus mode uses soft-voting across MobileNetV2, ResNet50, EfficientNet, and VGG19. Deep-Dive evaluates an isolated model."
    )

    selected_single_model = DEFAULT_MODEL
    fast_consensus_mode = False
    
    if op_mode == "Multi-Model Consensus":
        st.caption("Consensus Engine aggregates 4 deep architectures with validation-calibrated soft-voting weights.")
        fast_consensus_mode = st.checkbox(
            "⚡ Streamlit Cloud Fast Mode",
            value=False,
            help="Excludes the 443MB VGG19 model to ensure rapid inference and minimize RAM footprint under cloud hosting limits."
        )
    else:
        model_options = list(AVAILABLE_MODELS.keys())
        selected_single_model = st.selectbox(
            "Select Deep Learning Backbone",
            options=model_options,
            format_func=lambda x: f"{AVAILABLE_MODELS[x]['name']} ({AVAILABLE_MODELS[x]['badge']})",
            index=model_options.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_options else 0
        )

    st.markdown("---")

    # Explainable AI (Grad-CAM) Visual Controls
    st.markdown("##### 🔬 Explainability (Grad-CAM)")
    enable_gradcam = st.toggle("Compute Grad-CAM Activation Maps", value=True)
    
    colormap_choice = st.selectbox(
        "Heatmap Colormap",
        ["JET (Standard Clinical)", "VIRIDIS (Perceptual)", "INFERNO (High Contrast)", "BONE (Monochrome)", "MAGMA"],
        index=0
    )
    
    cmap_map = {
        "JET (Standard Clinical)": cv2.COLORMAP_JET,
        "VIRIDIS (Perceptual)": cv2.COLORMAP_VIRIDIS,
        "INFERNO (High Contrast)": cv2.COLORMAP_INFERNO,
        "BONE (Monochrome)": cv2.COLORMAP_BONE,
        "MAGMA": cv2.COLORMAP_MAGMA,
    }
    active_cv2_cmap = cmap_map[colormap_choice]

    blend_alpha = st.slider(
        "Parenchymal Overlay Opacity",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Adjust the transparency of the attention heatmap overlaid onto the anatomical chest radiograph."
    )

    st.markdown("---")

    # Session & Clinician Info
    with st.expander("👤 Clinician & Facility Profile", expanded=False):
        clinician_input = st.text_input("Reviewing Radiologist", value=st.session_state["clinician_name"])
        facility_input = st.text_input("Hospital / Facility", value=st.session_state["facility_name"])
        st.session_state["clinician_name"] = clinician_input
        st.session_state["facility_name"] = facility_input

    # Model Weights Telemetry Expander
    with st.expander("🖥️ Hardware & Weights Health", expanded=False):
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                st.success(f"Accelerated GPU: {gpus[0].name}")
            else:
                st.info("Execution Device: CPU (Optimized)")
        except Exception:
            st.info("Execution Device: Standard Host")

        for mid, hinfo in models_health.items():
            if hinfo["exists"]:
                if hinfo["is_lfs_pointer"]:
                    st.warning(f"⚠️ {hinfo['name']}: Git LFS pointer ({hinfo['size_mb']} MB)")
                else:
                    st.write(f"🟢 **{hinfo['name']}**: Ready ({hinfo['size_mb']} MB)")
            else:
                st.error(f"🔴 **{hinfo['name']}**: Weights Missing")

    if st.button("🔄 Reset Workstation Session", use_container_width=True):
        st.session_state["selected_image_path"] = None
        st.session_state["selected_study_name"] = None
        st.session_state["is_dicom"] = False
        st.session_state["dicom_metadata"] = None
        st.session_state["inference_result"] = None
        st.session_state["patient_notes"] = ""
        st.rerun()


# ─── Workstation Header Banner ─────────────────────────────────────────────────
st.markdown("""
<div class="hub-header">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1 class="hub-title">Pneumonia Diagnostic Hub</h1>
            <div class="hub-subtitle">Enterprise Clinical AI Radiology Workstation • Multi-Model Consensus Engine & DICOM Analytics</div>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
            <span class="badge-pill badge-blue">DICOM 3.0 Ready</span>
            <span class="badge-pill badge-emerald">4-Model Consensus</span>
            <span class="badge-pill badge-purple">Grad-CAM XAI</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Step 1: Study Ingestion & DICOM Inspection ────────────────────────────────
st.markdown("### 📥 Step 1: Ingest Chest Radiograph Study")

tab_upload, tab_samples = st.tabs(["📁 Upload Radiograph / DICOM (.dcm)", "🧪 Pre-Loaded Clinical Benchmark Cases"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Select Chest Radiograph Image or DICOM File",
        type=list(ALLOWED_EXTENSIONS),
        help="Upload standard radiographs (.png, .jpg, .jpeg, .webp) or medical DICOM (.dcm) files up to 32MB."
    )
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        temp_dir = Path(UPLOAD_FOLDER)
        temp_dir.mkdir(parents=True, exist_ok=True)
        saved_path = temp_dir / f"uploaded_{uuid.uuid4().hex[:8]}.{file_ext}"

        # Write uploaded bytes to disk
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Check if DICOM file
        if is_dicom_file(saved_path):
            try:
                converted_jpg = temp_dir / f"dicom_converted_{uuid.uuid4().hex[:8]}.jpg"
                _, dcm_meta, dcm_img_path = parse_dicom_file(saved_path, output_jpg_path=converted_jpg)
                st.session_state["selected_image_path"] = str(dcm_img_path)
                st.session_state["selected_study_name"] = f"DICOM: {uploaded_file.name}"
                st.session_state["is_dicom"] = True
                st.session_state["dicom_metadata"] = dcm_meta
                st.success(f"Successfully decoded DICOM medical object: **{uploaded_file.name}**")
            except Exception as e:
                st.error(f"Error parsing DICOM object: {e}")
        else:
            st.session_state["selected_image_path"] = str(saved_path)
            st.session_state["selected_study_name"] = uploaded_file.name
            st.session_state["is_dicom"] = False
            st.session_state["dicom_metadata"] = None

with tab_samples:
    st.caption("Select a pre-synthesized radiological case to evaluate clinical AI consensus:")
    samples_list = list_sample_catalog()
    sample_cols = st.columns(len(samples_list))

    for idx, s_info in enumerate(samples_list):
        with sample_cols[idx]:
            sample_resolved = get_sample_info(s_info["id"])
            if sample_resolved and sample_resolved["path"].exists():
                st.image(str(sample_resolved["path"]), caption=s_info["title"], use_container_width=True)
                st.markdown(f"**{s_info['badge']}**")
                st.caption(s_info["subtitle"])
                if st.button(f"Load {s_info['title']}", key=f"btn_load_{s_info['id']}", use_container_width=True):
                    st.session_state["selected_image_path"] = str(sample_resolved["path"])
                    st.session_state["selected_study_name"] = s_info["title"]
                    st.session_state["is_dicom"] = False
                    st.session_state["dicom_metadata"] = {
                        "patient_id": f"REF-{s_info['category']}-001",
                        "patient_name": f"Anonymous ({s_info['category']})",
                        "patient_age": "48Y",
                        "patient_sex": "M",
                        "study_date": datetime.date.today().strftime("%Y%m%d"),
                        "modality": "CR",
                        "body_part": "CHEST",
                        "manufacturer": "Synthetic Clinical Reference",
                    }
                    st.session_state["inference_result"] = None
                    st.rerun()


# ─── Preview & DICOM Metadata Section ──────────────────────────────────────────
if st.session_state["selected_image_path"]:
    active_path = Path(st.session_state["selected_image_path"])
    if active_path.exists():
        st.markdown("---")
        prev_col1, prev_col2 = st.columns([1, 2])
        
        with prev_col1:
            st.markdown(f"##### 🖼️ Active Study: `{st.session_state['selected_study_name']}`")
            st.image(str(active_path), use_container_width=True)
        
        with prev_col2:
            st.markdown("##### 📋 Acquisition & Clinical Context")
            if st.session_state["is_dicom"] or st.session_state["dicom_metadata"]:
                meta = st.session_state["dicom_metadata"]
                tag_c1, tag_c2, tag_c3 = st.columns(3)
                with tag_c1:
                    st.write(f"**Patient ID:** `{meta.get('patient_id', 'ANON')}`")
                    st.write(f"**Age / Sex:** {meta.get('patient_age', 'N/A')} / {meta.get('patient_sex', 'N/A')}")
                with tag_c2:
                    st.write(f"**Modality:** `{meta.get('modality', 'CR')}`")
                    st.write(f"**Study Date:** {meta.get('study_date', 'N/A')}")
                with tag_c3:
                    st.write(f"**Body Part:** `{meta.get('body_part', 'CHEST')}`")
                    st.write(f"**Photometric:** {meta.get('photometric', 'MONOCHROME2')}")
                
                with st.expander("🔍 View Complete DICOM Technical Header", expanded=False):
                    st.json(meta)
            else:
                st.info("Standard raster radiograph loaded. No DICOM clinical tags present.")
                st.write(f"**File Size:** {round(active_path.stat().st_size / 1024, 1)} KB")
                st.write(f"**Format:** {active_path.suffix.upper()}")

        # ─── Step 2: Diagnostic Screening Run ─────────────────────────────────
        st.markdown("---")
        st.markdown("### ⚡ Step 2: Diagnostic AI Screening")
        
        run_btn_col1, run_btn_col2 = st.columns([2, 3])
        with run_btn_col1:
            run_screening = st.button(
                "🚀 Execute AI Diagnostic Analysis",
                type="primary",
                use_container_width=True,
                help="Run convolutional neural network inference and explainability mapping."
            )

        if run_screening:
            with st.spinner("Processing radiograph, executing deep learning backbones, and synthesizing Grad-CAM activations..."):
                try:
                    # 1. Preprocess image
                    input_tensor = preprocess_image(active_path)
                    base_fname = f"scan_{uuid.uuid4().hex[:8]}.jpg"

                    if op_mode == "Multi-Model Consensus":
                        # Multi-model consensus inference
                        comparison_res = run_multi_model_comparison(
                            image_tensor=input_tensor,
                            original_image_path=active_path,
                            base_filename=base_fname,
                            generate_cams=enable_gradcam,
                            cam_colormap=active_cv2_cmap,
                            cam_alpha=blend_alpha,
                            exclude_vgg=fast_consensus_mode
                        )
                        st.session_state["inference_result"] = comparison_res
                        st.session_state["inference_type"] = "consensus"
                        st.session_state["active_heatmap"] = comparison_res.get("primary_raw_heatmap")
                    else:
                        # Single model inference
                        manager = get_model_manager()
                        pred_res = manager.predict(
                            model_id=selected_single_model,
                            image_tensor=input_tensor,
                            generate_cam=enable_gradcam,
                            original_image_path=active_path,
                            base_filename=base_fname,
                            cam_colormap=active_cv2_cmap,
                            cam_alpha=blend_alpha
                        )
                        st.session_state["inference_result"] = pred_res
                        st.session_state["inference_type"] = "single"
                        st.session_state["active_heatmap"] = pred_res.get("raw_heatmap")

                    st.success("AI Diagnostic Screening Completed!")
                except Exception as infer_err:
                    st.error(f"Inference execution encountered an issue: {infer_err}")


# ─── Step 3: Diagnostic Decision Support & Consensus Metrics ───────────────────
if st.session_state.get("inference_result"):
    res = st.session_state["inference_result"]
    is_consensus = st.session_state.get("inference_type") == "consensus"
    
    st.markdown("---")
    st.markdown("### 📊 Step 3: Clinical Decision Support & Consensus Verdict")

    # Determine verdict and confidence
    if is_consensus:
        verdict = res["consensus_verdict"]
        confidence = res["consensus_confidence"]
        norm_pct = res["consensus_probabilities"]["NORMAL"]
        pneu_pct = res["consensus_probabilities"]["PNEUMONIA"]
        agreement_text = res["agreement_text"]
        disagreement = res["disagreement_warning"]
        latency = res["total_inference_time_ms"]
    else:
        verdict = res["prediction"]
        confidence = res["confidence"]
        norm_pct = res["probabilities"]["NORMAL"]
        pneu_pct = res["probabilities"]["PNEUMONIA"]
        agreement_text = f"Single Architecture: {res['model_name']}"
        disagreement = False
        latency = res["inference_time_ms"]

    # Verdict Presentation Card
    if verdict == "PNEUMONIA":
        verdict_class = "verdict-box-pneumonia"
        badge_color = "#EF4444"
        verdict_icon = "⚠️"
        verdict_title = "PNEUMONIA DETECTED"
        verdict_desc = "Significant radiographic consolidation or interstitial opacities detected. Clinical correlation strongly advised."
    else:
        verdict_class = "verdict-box-normal"
        badge_color = "#10B981"
        verdict_icon = "✅"
        verdict_title = "NORMAL (NO ACUTE FINDINGS)"
        verdict_desc = "Clear bilateral lung fields without focal alveolar consolidation or pleural effusion."

    st.markdown(f"""
    <div class="{verdict_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="font-size: 2.2rem; margin-right: 10px;">{verdict_icon}</span>
                <span style="font-size: 1.8rem; font-weight: 800; color: {badge_color};">{verdict_title}</span>
                <div style="color: #CBD5E1; font-size: 1rem; margin-top: 6px;">{verdict_desc}</div>
            </div>
            <div style="text-align: right; margin-top: 10px;">
                <div style="font-size: 2.4rem; font-weight: 800; color: #F8FAFC;">{confidence:.1f}%</div>
                <div style="color: #94A3B8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">Consensus Confidence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Disagreement Warning Alert
    if disagreement:
        st.warning(f"⚠️ **Inter-Model Discrepancy Detected:** {agreement_text}. Review individual architectural heatmaps below prior to sign-off.")

    # High-level Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown("""<div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">CONSENSUS STATUS</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #60A5FA; margin-top: 4px;">""" + agreement_text + """</div>
        </div>""", unsafe_allow_html=True)
    with m_col2:
        st.markdown("""<div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">PNEUMONIA PROBABILITY</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #F87171; margin-top: 4px;">""" + f"{pneu_pct:.1f}%" + """</div>
        </div>""", unsafe_allow_html=True)
    with m_col3:
        st.markdown("""<div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">NORMAL PROBABILITY</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #34D399; margin-top: 4px;">""" + f"{norm_pct:.1f}%" + """</div>
        </div>""", unsafe_allow_html=True)
    with m_col4:
        st.markdown("""<div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">TOTAL INFERENCE LATENCY</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #A78BFA; margin-top: 4px;">""" + f"{latency} ms" + """</div>
        </div>""", unsafe_allow_html=True)


    # ─── Step 4: Explainable AI (Grad-CAM) Visual Comparison ───────────────────
    st.markdown("---")
    st.markdown("### 🔬 Step 4: Explainable AI (Grad-CAM Visual Attention)")
    st.caption("Class Activation Mapping highlights pulmonary anatomical regions with highest influence on the deep CNN decision.")

    active_image_path = Path(st.session_state["selected_image_path"])

    # Extract primary Grad-CAM image paths and heatmaps
    overlay_path = None
    composite_path = None
    
    if is_consensus:
        overlay_url = res.get("primary_gradcam_overlay_url")
        comp_url = res.get("primary_gradcam_composite_url")
        raw_hm = res.get("primary_raw_heatmap")
    else:
        overlay_url = res.get("gradcam_overlay_url")
        comp_url = res.get("gradcam_composite_url")
        raw_hm = res.get("raw_heatmap")

    if overlay_url:
        overlay_path = STATIC_DIR / overlay_url.replace("/static/", "").lstrip("/")
    if comp_url:
        composite_path = STATIC_DIR / comp_url.replace("/static/", "").lstrip("/")

    # Check if we can dynamically render with current sidebar slider & colormap
    dynamic_rendered = False
    dyn_overlay_rgb = None
    dyn_comp_rgb = None
    dyn_overlay_bgr = None

    if raw_hm is not None:
        try:
            dyn_overlay_bgr, _, dyn_comp_bgr = create_gradcam_overlay(
                original_image_path=active_image_path,
                heatmap=raw_hm,
                colormap=active_cv2_cmap,
                alpha=blend_alpha
            )
            dyn_overlay_rgb = cv2.cvtColor(dyn_overlay_bgr, cv2.COLOR_BGR2RGB)
            dyn_comp_rgb = cv2.cvtColor(dyn_comp_bgr, cv2.COLOR_BGR2RGB)
            dynamic_rendered = True
        except Exception:
            dynamic_rendered = False

    cam_c1, cam_c2 = st.columns(2)
    with cam_c1:
        st.markdown("##### 🫁 Original Radiograph")
        st.image(str(active_image_path), use_container_width=True)

    with cam_c2:
        st.markdown(f"##### 🎯 Grad-CAM Attention Overlay (Alpha: {blend_alpha:.2f})")
        if dynamic_rendered and dyn_overlay_rgb is not None:
            st.image(dyn_overlay_rgb, use_container_width=True, caption=f"Active Attention: {colormap_choice} ({int(blend_alpha * 100)}% Opacity)")
        elif overlay_path and overlay_path.exists():
            st.image(str(overlay_path), use_container_width=True)
        elif res.get("gradcam_error"):
            st.error(f"Grad-CAM error: {res['gradcam_error']}")
        else:
            st.info("Grad-CAM overlay rendering complete.")

    if dynamic_rendered and dyn_comp_rgb is not None:
        with st.expander("🖼️ View High-Resolution Diagnostic Composite Triad", expanded=True):
            st.image(dyn_comp_rgb, caption=f"[ Original Radiograph | Class Activation Heatmap ({colormap_choice}) | Diagnostic Anatomical Overlay ]", use_container_width=True)
    elif composite_path and composite_path.exists():
        with st.expander("🖼️ View High-Resolution Diagnostic Composite Triad", expanded=True):
            st.image(str(composite_path), caption="[ Original Radiograph | Class Activation Heatmap | Diagnostic Anatomical Overlay ]", use_container_width=True)


    # ─── Step 5: Multi-Model Architecture Comparative Grid ─────────────────────
    st.markdown("---")
    st.markdown("### 🏛️ Step 5: Multi-Architecture Comparative Grid")
    if is_consensus and "models_breakdown" in res:
        st.caption("Side-by-side performance, parameters, confidence, and isolated Grad-CAM attention per CNN architecture:")

        breakdown = res["models_breakdown"]
        grid_cols = st.columns(len(breakdown))

        for idx, m_card in enumerate(breakdown):
            with grid_cols[idx]:
                m_verdict = m_card["prediction"]
                m_conf = m_card["confidence"]
                v_color = "#EF4444" if m_verdict == "PNEUMONIA" else "#10B981"

                st.markdown(f"""
                <div class="metric-card" style="border-top: 3px solid {v_color};">
                    <div style="font-weight: 700; font-size: 1rem; color: #F8FAFC;">{m_card['name']}</div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">Weight: {m_card['weight']:.2f} • {m_card['parameters']}</div>
                    <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 8px 0;" />
                    <div style="font-size: 1.15rem; font-weight: 800; color: {v_color};">{m_verdict}</div>
                    <div style="font-size: 0.85rem; color: #CBD5E1;">Confidence: <b>{m_conf:.1f}%</b></div>
                    <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">Latency: {m_card['inference_time_ms']} ms</div>
                </div>
                """, unsafe_allow_html=True)

                if m_card.get("gradcam_overlay_url"):
                    m_over_path = STATIC_DIR / m_card["gradcam_overlay_url"].replace("/static/", "").lstrip("/")
                    if m_over_path.exists():
                        st.image(str(m_over_path), caption=f"{m_card['name']} CAM", use_container_width=True)
    else:
        active_model_name = res.get("model_name", "MobileNetV2")
        st.info(f"💡 **Single Architecture Mode Active**: You evaluated this study using **{active_model_name}**. To view the side-by-side Multi-Architecture Comparative Grid with isolated metrics and Grad-CAM attention across all 4 models (MobileNetV2, ResNet50, EfficientNetB0, VGG19), select **Multi-Model Consensus** in the left sidebar and click *Execute AI Diagnostic Analysis*.")


    # ─── Step 6: Publication-Grade Clinical PDF Report Generator ───────────────
    st.markdown("---")
    st.markdown("### 📄 Step 6: Clinical Report Export & Radiologist Sign-Off")

    rep_col1, rep_col2 = st.columns([2, 1])
    with rep_col1:
        notes_input = st.text_area(
            "Radiologist Clinical Findings & Differential Impressions",
            value=st.session_state["patient_notes"],
            placeholder="Document observed bronchovascular markings, focal consolidation boundaries, or recommendations for repeat high-resolution CT...",
            height=120
        )
        st.session_state["patient_notes"] = notes_input

    with rep_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        gen_pdf_btn = st.button("📑 Compile Clinical PDF Report", type="secondary", use_container_width=True)

        scan_uuid = uuid.uuid4().hex[:8]
        report_output_path = None

        if gen_pdf_btn:
            with st.spinner("Compiling publication-quality PDF report with embedded CXR and Grad-CAM imagery..."):
                try:
                    patient_meta = st.session_state.get("dicom_metadata") or {
                        "patient_id": f"PT-{scan_uuid.upper()}",
                        "patient_name": "Anonymous Patient",
                        "patient_age": "N/A",
                        "patient_sex": "N/A",
                        "study_date": datetime.date.today().strftime("%Y-%m-%d"),
                        "modality": "CR",
                        "body_part": "CHEST",
                        "manufacturer": "Digital Radiography Workstation",
                    }
                    patient_meta["clinician_notes"] = notes_input
                    patient_meta["reviewing_radiologist"] = st.session_state["clinician_name"]
                    patient_meta["facility_name"] = st.session_state["facility_name"]

                    report_cam_path = overlay_path if (overlay_path and overlay_path.exists()) else None
                    if not report_cam_path and dynamic_rendered and dyn_overlay_bgr is not None:
                        temp_report_cam = UPLOAD_FOLDER / f"report_cam_{scan_uuid}.jpg"
                        cv2.imwrite(str(temp_report_cam), dyn_overlay_bgr)
                        report_cam_path = temp_report_cam

                    pdf_file_path = generate_clinical_pdf_report(
                        scan_id=scan_uuid,
                        prediction_data=res,
                        original_image_path=active_image_path,
                        gradcam_overlay_path=report_cam_path,
                        patient_metadata=patient_meta,
                        output_dir=Path(UPLOAD_FOLDER)
                    )

                    if pdf_file_path.exists():
                        st.session_state["pdf_report_path"] = str(pdf_file_path)
                        st.success("Official Clinical Report Compiled!")
                except Exception as pdf_err:
                    st.error(f"Error compiling PDF report: {pdf_err}")

        if st.session_state.get("pdf_report_path"):
            pdf_path_obj = Path(st.session_state["pdf_report_path"])
            if pdf_path_obj.exists():
                with open(pdf_path_obj, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                st.download_button(
                    label="📥 Download Clinical PDF Report",
                    data=pdf_bytes,
                    file_name=f"Clinical_Radiology_Report_{scan_uuid}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )


# ─── Medical Disclaimer Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.8rem; padding: 16px 0;">
    <b>Medical & Clinical Disclaimer:</b> This AI Radiology Workstation is designed as a Clinical Decision Support System (CDSS) for investigational and research purposes.
    Predictions and Grad-CAM activations should be interpreted by board-certified radiologists in conjunction with patient history and clinical presentation.
    <br>Pneumonia Diagnostic Hub v2.0 • DICOM 3.0 Ready • Open-Source Healthcare Initiative.
</div>
""", unsafe_allow_html=True)
