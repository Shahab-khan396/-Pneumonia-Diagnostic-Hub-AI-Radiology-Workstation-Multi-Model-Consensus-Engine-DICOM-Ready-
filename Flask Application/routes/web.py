import uuid
import traceback
from pathlib import Path
from flask import Blueprint, render_template, request, current_app, send_from_directory
from core.validator import validate_image_file, get_safe_filepath
from core.preprocessor import preprocess_image
from core.model_manager import get_model_manager
from core.sample_manager import list_sample_catalog, get_sample_info
from core.ensemble import run_multi_model_comparison
from core.report_generator import generate_clinical_pdf_report
from core.dicom_parser import is_dicom_file, parse_dicom_file
from config import DEFAULT_MODEL, UPLOAD_FOLDER

web_bp = Blueprint("web", __name__)


@web_bp.route("/", methods=["GET"])
def index():
    """Render the radiology workstation interface."""
    manager = get_model_manager()
    models = manager.list_available_models()
    samples = list_sample_catalog()
    
    selected_sample_id = request.args.get("sample")
    selected_sample = get_sample_info(selected_sample_id) if selected_sample_id else None

    return render_template(
        "index.html",
        models=models,
        samples=samples,
        selected_sample=selected_sample,
        default_model=DEFAULT_MODEL
    )


@web_bp.route("/predict", methods=["POST"])
def predict():
    """Handle web form submission for single-model or multi-model ensemble analysis with DICOM & Patient records."""
    manager = get_model_manager()
    models = manager.list_available_models()
    samples = list_sample_catalog()

    analysis_mode = request.form.get("analysis_mode", "single")
    model_choice = request.form.get("model_choice", DEFAULT_MODEL)
    sample_id = request.form.get("sample_id")

    dest_path = None
    safe_name = None
    is_sample = False
    dicom_meta = None

    # Determine image source (Sample vs Upload)
    if sample_id:
        sample_meta = get_sample_info(sample_id)
        if sample_meta and sample_meta["file_path"].exists():
            dest_path = sample_meta["file_path"]
            safe_name = sample_meta["filename"]
            is_sample = True
            
    if not dest_path:
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template(
                "index.html",
                message="Please upload a Chest Radiograph (PNG, JPG, DICOM .dcm) or select a sample radiograph.",
                models=models,
                samples=samples,
                default_model=model_choice
            ), 400

        file_storage = request.files["file"]
        is_valid, err_msg = validate_image_file(file_storage)
        if not is_valid:
            return render_template(
                "index.html",
                message=err_msg,
                models=models,
                samples=samples,
                default_model=model_choice
            ), 400

        safe_name, dest_path = get_safe_filepath(file_storage.filename)
        file_storage.save(str(dest_path))

    # DICOM conversion if applicable
    if is_dicom_file(dest_path):
        try:
            jpg_path = dest_path.with_suffix(".jpg")
            _, dicom_meta, dest_path = parse_dicom_file(dest_path, output_jpg_path=jpg_path)
            safe_name = dest_path.name
        except Exception as dcm_e:
            return render_template(
                "index.html",
                message=f"Failed to parse DICOM medical file: {str(dcm_e)}",
                models=models,
                samples=samples,
                default_model=model_choice
            ), 400

    # Extract Patient Demographics
    patient_metadata = {
        "patient_id": request.form.get("patient_id") or (dicom_meta.get("patient_id") if dicom_meta else None),
        "patient_age": request.form.get("patient_age") or (dicom_meta.get("patient_age") if dicom_meta else None),
        "patient_gender": request.form.get("patient_gender") or (dicom_meta.get("patient_sex") if dicom_meta else None),
        "clinical_history": request.form.get("clinical_history"),
        "referring_physician": request.form.get("referring_physician"),
        "modality": dicom_meta.get("modality") if dicom_meta else "Digital Radiography (CXR)"
    }

    try:
        image_tensor = preprocess_image(dest_path)
        scan_id = uuid.uuid4().hex[:8].upper()
        public_image_url = f"/static/samples/{safe_name}" if is_sample else f"/static/uploads/{safe_name}"

        # MODE 1: MULTI-MODEL ENSEMBLE CONSENSUS
        if analysis_mode == "ensemble":
            comparison = run_multi_model_comparison(
                image_tensor=image_tensor,
                original_image_path=dest_path,
                base_filename=safe_name,
                generate_cams=True
            )
            
            # Generate PDF Report
            overlay_file = Path(UPLOAD_FOLDER) / f"gradcam_overlay_mobilenet_{safe_name}"
            pdf_path = generate_clinical_pdf_report(
                scan_id=scan_id,
                prediction_data=comparison,
                original_image_path=dest_path,
                gradcam_overlay_path=overlay_file if overlay_file.exists() else None,
                patient_metadata=patient_metadata
            )

            return render_template(
                "index.html",
                is_ensemble=True,
                scan_id=scan_id,
                consensus_verdict=comparison["consensus_verdict"],
                consensus_confidence=comparison["consensus_confidence"],
                consensus_probabilities=comparison["consensus_probabilities"],
                agreement_level=comparison["agreement_level"],
                agreement_text=comparison["agreement_text"],
                disagreement_warning=comparison["disagreement_warning"],
                total_inference_time_ms=comparison["total_inference_time_ms"],
                models_breakdown=comparison["models_breakdown"],
                image_path=public_image_url,
                gradcam_overlay_url=comparison.get("primary_gradcam_overlay_url"),
                gradcam_composite_url=comparison.get("primary_gradcam_composite_url"),
                has_gradcam=True,
                patient_metadata=patient_metadata,
                dicom_metadata=dicom_meta,
                report_pdf_url=f"/api/v1/report/{pdf_path.name}",
                report_filename=pdf_path.name,
                analysis_mode="ensemble",
                models=models,
                samples=samples,
                default_model=model_choice
            )

        # MODE 2: SINGLE-MODEL INFERENCE
        else:
            result = manager.predict(
                model_id=model_choice,
                image_tensor=image_tensor,
                generate_cam=True,
                original_image_path=dest_path,
                base_filename=safe_name
            )

            # Generate PDF Report
            overlay_file = Path(UPLOAD_FOLDER) / f"gradcam_overlay_{safe_name}"
            pdf_path = generate_clinical_pdf_report(
                scan_id=scan_id,
                prediction_data=result,
                original_image_path=dest_path,
                gradcam_overlay_path=overlay_file if overlay_file.exists() else None,
                patient_metadata=patient_metadata
            )

            return render_template(
                "index.html",
                is_ensemble=False,
                scan_id=scan_id,
                prediction=result["prediction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                image_path=public_image_url,
                gradcam_overlay_url=result.get("gradcam_overlay_url"),
                gradcam_heatmap_url=result.get("gradcam_heatmap_url"),
                gradcam_composite_url=result.get("gradcam_composite_url"),
                has_gradcam=result.get("has_gradcam", False),
                target_conv_layer=result.get("target_conv_layer", "N/A"),
                model_used=result["model_name"],
                model_badge=result["model_badge"],
                model_parameters=result["model_parameters"],
                inference_time_ms=result["inference_time_ms"],
                patient_metadata=patient_metadata,
                dicom_metadata=dicom_meta,
                report_pdf_url=f"/api/v1/report/{pdf_path.name}",
                report_filename=pdf_path.name,
                selected_model=model_choice,
                analysis_mode="single",
                models=models,
                samples=samples,
                default_model=DEFAULT_MODEL
            )

    except Exception as exc:
        current_app.logger.error(f"Web analysis error: {traceback.format_exc()}")
        return render_template(
            "index.html",
            message=f"Diagnostic analysis error: {str(exc)}",
            models=models,
            samples=samples,
            default_model=model_choice
        ), 500


@web_bp.route("/download-report/<filename>", methods=["GET"])
def download_report_file(filename: str):
    """Download generated PDF report from web interface."""
    upload_dir = Path(UPLOAD_FOLDER)
    if not (upload_dir / filename).exists():
        return "Report file not found.", 404
    return send_from_directory(
        directory=str(upload_dir),
        path=filename,
        as_attachment=True,
        download_name=filename
    )
