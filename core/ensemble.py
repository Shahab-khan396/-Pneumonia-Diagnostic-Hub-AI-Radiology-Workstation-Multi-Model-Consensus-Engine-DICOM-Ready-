import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from config import AVAILABLE_MODELS, ENSEMBLE_WEIGHTS, DEFAULT_MODEL
from core.model_manager import get_model_manager


def run_multi_model_comparison(
    image_tensor: np.ndarray,
    original_image_path: Path,
    base_filename: str,
    generate_cams: bool = True
) -> Dict[str, Any]:
    """
    Execute inference across all registered deep learning backbones simultaneously
    and calculate a weighted soft-voting consensus decision.
    
    Consensus Formula:
      P_consensus(C) = sum_m [ weight_m * P_m(C) ]
    """
    manager = get_model_manager()
    model_results: List[Dict[str, Any]] = []
    
    weighted_prob_normal = 0.0
    weighted_prob_pneumonia = 0.0
    total_weight = 0.0
    
    pneumonia_votes = 0
    normal_votes = 0
    
    total_start_time = time.perf_counter()

    for model_id, meta in AVAILABLE_MODELS.items():
        weight = ENSEMBLE_WEIGHTS.get(model_id, 0.25)
        total_weight += weight
        
        # Run inference
        res = manager.predict(
            model_id=model_id,
            image_tensor=image_tensor,
            generate_cam=generate_cams,
            original_image_path=original_image_path,
            base_filename=f"{model_id}_{base_filename}"
        )
        
        pred = res["prediction"]
        raw_p = res["raw_probabilities"]
        
        weighted_prob_normal += weight * raw_p["NORMAL"]
        weighted_prob_pneumonia += weight * raw_p["PNEUMONIA"]
        
        if pred == "PNEUMONIA":
            pneumonia_votes += 1
        else:
            normal_votes += 1
            
        model_card = {
            "id": model_id,
            "name": meta["name"],
            "parameters": meta["parameters"],
            "badge": meta["badge"],
            "weight": weight,
            "prediction": pred,
            "confidence": res["confidence"],
            "probabilities": res["probabilities"],
            "inference_time_ms": res["inference_time_ms"],
            "target_conv_layer": res.get("target_conv_layer", "N/A"),
            "gradcam_overlay_url": res.get("gradcam_overlay_url"),
            "gradcam_composite_url": res.get("gradcam_composite_url"),
        }
        model_results.append(model_card)

    total_end_time = time.perf_counter()
    total_time_ms = round((total_end_time - total_start_time) * 1000.0, 2)

    # Normalize consensus probabilities
    if total_weight > 0:
        norm_consensus_normal = (weighted_prob_normal / total_weight) * 100.0
        norm_consensus_pneumonia = (weighted_prob_pneumonia / total_weight) * 100.0
    else:
        norm_consensus_normal = 50.0
        norm_consensus_pneumonia = 50.0

    consensus_normal_pct = round(norm_consensus_normal, 2)
    consensus_pneumonia_pct = round(norm_consensus_pneumonia, 2)

    if consensus_pneumonia_pct >= 50.0:
        consensus_verdict = "PNEUMONIA"
        consensus_confidence = consensus_pneumonia_pct
    else:
        consensus_verdict = "NORMAL"
        consensus_confidence = consensus_normal_pct

    # Determine agreement status
    total_models = len(AVAILABLE_MODELS)
    winning_votes = max(pneumonia_votes, normal_votes)
    
    if winning_votes == total_models:
        agreement_level = "UNANIMOUS"
        agreement_text = f"Unanimous Consensus ({winning_votes}/{total_models} Models in Full Agreement)"
        disagreement_warning = False
    elif winning_votes >= total_models - 1:
        agreement_level = "STRONG_MAJORITY"
        agreement_text = f"Strong Majority Consensus ({winning_votes}/{total_models} Models Agree)"
        disagreement_warning = False
    else:
        agreement_level = "SPLIT_DECISION"
        agreement_text = f"Inter-Model Discrepancy ({winning_votes}/{total_models} Split Vote)"
        disagreement_warning = True

    # Primary Grad-CAM overlay from the highest-weighted model (MobileNetV2)
    primary_model_res = next((m for m in model_results if m["id"] == DEFAULT_MODEL), model_results[0])

    return {
        "success": True,
        "is_ensemble": True,
        "consensus_verdict": consensus_verdict,
        "consensus_confidence": consensus_confidence,
        "consensus_probabilities": {
            "NORMAL": consensus_normal_pct,
            "PNEUMONIA": consensus_pneumonia_pct
        },
        "agreement_level": agreement_level,
        "agreement_text": agreement_text,
        "disagreement_warning": disagreement_warning,
        "pneumonia_votes": pneumonia_votes,
        "normal_votes": normal_votes,
        "total_models": total_models,
        "total_inference_time_ms": total_time_ms,
        "models_breakdown": model_results,
        "primary_gradcam_overlay_url": primary_model_res.get("gradcam_overlay_url"),
        "primary_gradcam_composite_url": primary_model_res.get("gradcam_composite_url"),
    }
