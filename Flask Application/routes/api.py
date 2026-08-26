import uuid
import traceback
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app, send_from_directory
from core.validator import validate_image_file, get_safe_filepath
from core.preprocessor import preprocess_image
from core.model_manager import get_model_manager
from core.sample_manager import list_sample_catalog, get_sample_info
from core.ensemble import run_multi_model_comparison
from core.report_generator import generate_clinical_pdf_report
from core.dicom_parser import is_dicom_file, parse_dicom_file
from config import DEFAULT_MODEL, UPLOAD_FOLDER

api_bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")


@api_bp.route("/health", methods=["GET"])
def health_check():
    """System, explainability engine, ensemble, and model catalog health endpoint."""
    manager = get_model_manager()
    models = manager.list_available_models()
    return jsonify({
        "status": "healthy",
        "service": "Pneumonia-Diagnostic-Hub-API",
        "version": "2.3.0",
        "features": [
            "multi_model_inference",
            "gradcam_xai",
            "ensemble_consensus",
            "dicom_parser",
            "clinical_pdf_reporting",
            "sample_radiograph_library"
        ],
        "models_count": len(models),
        "available_models": models
    }), 200


@api_bp.route("/models", methods=["GET"])
def get_models():
    """Retrieve catalog of available deep learning models and specifications."""
    manager = get_model_manager()
    return jsonify({
        "success": True,
        "models": manager.list_available_models()
    }), 200


@api_bp.route("/samples", methods=["GET"])
def get_samples():
    """Retrieve catalog of pre-packaged sample radiographs."""
    samples = list_sample_catalog()
    return jsonify({
        "success": True,
        "samples": samples
    }), 200


@api_bp.route("/predict", methods=["POST"])
def predict_api():
    """
    Single-model inference & Grad-CAM endpoint. Supports standard images & DICOM (.dcm).
    """
    sample_id = request.form.get("sample_id")
    dest_path = None
    safe_name = None
    dicom_meta = None

    if sample_id:
        sample_meta = get_sample_info(sample_id)
        if not sample_meta or not sample_meta["file_path"].exists():
            return jsonify({"success": False, "error": f"Sample ID '{sample_id}' not found."}), 404
        dest_path = sample_meta["file_path"]
        safe_name = sample_meta["filename"]
    elif "file" in request.files:
        file_storage = request.files["file"]
        is_valid, err_msg = validate_image_file(file_storage)
        if not is_valid:
            return jsonify({"success": False, "error": err_msg}), 400
        safe_name, dest_path = get_safe_filepath(file_storage.filename)
        file_storage.save(str(dest_path))
    else:
        return jsonify({"success": False, "error": "Missing 'file' or 'sample_id' in form data."}), 400

    # DICOM conversion if applicable
    if is_dicom_file(dest_path):
        try:
            jpg_path = dest_path.with_suffix(".jpg")
            _, dicom_meta, dest_path = parse_dicom_file(dest_path, output_jpg_path=jpg_path)
            safe_name = dest_path.name
        except Exception as dcm_e:
            return jsonify({"success": False, "error": f"Failed to parse DICOM file: {str(dcm_e)}"}), 400

    model_choice = request.form.get("model_choice", DEFAULT_MODEL)
    explain_param = request.form.get("explain", "true").lower() in ["true", "1", "yes"]
    generate_report = request.form.get("generate_report", "true").lower() in ["true", "1", "yes"]
    
    # Patient Demographics
    patient_metadata = {
        "patient_id": request.form.get("patient_id") or (dicom_meta.get("patient_id") if dicom_meta else None),
        "patient_age": request.form.get("patient_age") or (dicom_meta.get("patient_age") if dicom_meta else None),
        "patient_gender": request.form.get("patient_gender") or (dicom_meta.get("patient_sex") if dicom_meta else None),
        "clinical_history": request.form.get("clinical_history"),
        "referring_physician": request.form.get("referring_physician"),
        "modality": dicom_meta.get("modality") if dicom_meta else "Digital Radiography"
    }

    manager = get_model_manager()

    try:
        image_tensor = preprocess_image(dest_path)
        scan_id = uuid.uuid4().hex[:8].upper()

        result = manager.predict(
            model_id=model_choice,
            image_tensor=image_tensor,
            generate_cam=explain_param,
            original_image_path=dest_path,
            base_filename=safe_name
        )
        
        result["scan_id"] = scan_id
        result["image_url"] = f"/static/uploads/{safe_name}" if "sample" not in str(dest_path) else f"/static/samples/{safe_name}"
        result["filename"] = safe_name
        if dicom_meta:
            result["dicom_metadata"] = dicom_meta

        # Generate Clinical PDF Report
        if generate_report:
            overlay_file = Path(UPLOAD_FOLDER) / f"gradcam_overlay_{safe_name}" if result.get("has_gradcam") else None
            pdf_path = generate_clinical_pdf_report(
                scan_id=scan_id,
                prediction_data=result,
                original_image_path=dest_path,
                gradcam_overlay_path=overlay_file,
                patient_metadata=patient_metadata
            )
            result["report_pdf_url"] = f"/api/v1/report/{pdf_path.name}"

        return jsonify(result), 200

    except Exception as exc:
        current_app.logger.error(f"API Prediction error: {traceback.format_exc()}")
        return jsonify({"success": False, "error": f"Inference failed: {str(exc)}"}), 500


@api_bp.route("/compare", methods=["POST"])
def compare_api():
    """
    Multi-model comparison & weighted consensus endpoint. Supports DICOM & Patient metadata.
    """
    sample_id = request.form.get("sample_id")
    dest_path = None
    safe_name = None
    dicom_meta = None

    if sample_id:
        sample_meta = get_sample_info(sample_id)
        if not sample_meta or not sample_meta["file_path"].exists():
            return jsonify({"success": False, "error": f"Sample ID '{sample_id}' not found."}), 404
        dest_path = sample_meta["file_path"]
        safe_name = sample_meta["filename"]
    elif "file" in request.files:
        file_storage = request.files["file"]
        is_valid, err_msg = validate_image_file(file_storage)
        if not is_valid:
            return jsonify({"success": False, "error": err_msg}), 400
        safe_name, dest_path = get_safe_filepath(file_storage.filename)
        file_storage.save(str(dest_path))
    else:
        return jsonify({"success": False, "error": "Missing 'file' or 'sample_id' in form data."}), 400

    # DICOM conversion if applicable
    if is_dicom_file(dest_path):
        try:
            jpg_path = dest_path.with_suffix(".jpg")
            _, dicom_meta, dest_path = parse_dicom_file(dest_path, output_jpg_path=jpg_path)
            safe_name = dest_path.name
        except Exception as dcm_e:
            return jsonify({"success": False, "error": f"Failed to parse DICOM file: {str(dcm_e)}"}), 400

    explain_param = request.form.get("explain", "true").lower() in ["true", "1", "yes"]
    
    patient_metadata = {
        "patient_id": request.form.get("patient_id") or (dicom_meta.get("patient_id") if dicom_meta else None),
        "patient_age": request.form.get("patient_age") or (dicom_meta.get("patient_age") if dicom_meta else None),
        "patient_gender": request.form.get("patient_gender") or (dicom_meta.get("patient_sex") if dicom_meta else None),
        "clinical_history": request.form.get("clinical_history"),
        "referring_physician": request.form.get("referring_physician"),
        "modality": dicom_meta.get("modality") if dicom_meta else "Digital Radiography"
    }

    try:
        image_tensor = preprocess_image(dest_path)
        scan_id = uuid.uuid4().hex[:8].upper()

        comparison = run_multi_model_comparison(
            image_tensor=image_tensor,
            original_image_path=dest_path,
            base_filename=safe_name,
            generate_cams=explain_param
        )
        
        comparison["scan_id"] = scan_id
        comparison["image_url"] = f"/static/uploads/{safe_name}" if "sample" not in str(dest_path) else f"/static/samples/{safe_name}"
        comparison["filename"] = safe_name
        if dicom_meta:
            comparison["dicom_metadata"] = dicom_meta

        # Generate Clinical PDF Report with Ensemble matrix
        overlay_file = Path(UPLOAD_FOLDER) / f"gradcam_overlay_mobilenet_{safe_name}" if explain_param else None
        pdf_path = generate_clinical_pdf_report(
            scan_id=scan_id,
            prediction_data=comparison,
            original_image_path=dest_path,
            gradcam_overlay_path=overlay_file,
            patient_metadata=patient_metadata
        )
        comparison["report_pdf_url"] = f"/api/v1/report/{pdf_path.name}"

        return jsonify(comparison), 200

    except Exception as exc:
        current_app.logger.error(f"Multi-model comparison error: {traceback.format_exc()}")
        return jsonify({"success": False, "error": f"Comparison failed: {str(exc)}"}), 500


@api_bp.route("/report/<report_filename>", methods=["GET"])
def download_report(report_filename: str):
    """Download generated clinical PDF diagnostic report."""
    upload_dir = Path(UPLOAD_FOLDER)
    if not (upload_dir / report_filename).exists():
        return jsonify({"success": False, "error": "Report file not found."}), 404
    return send_from_directory(
        directory=str(upload_dir),
        path=report_filename,
        as_attachment=True,
        download_name=report_filename
    )
