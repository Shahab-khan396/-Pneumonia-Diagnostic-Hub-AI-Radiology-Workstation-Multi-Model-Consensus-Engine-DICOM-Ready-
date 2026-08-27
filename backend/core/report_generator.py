"""
Clinical PDF report generator — ported from Flask Application/core/report_generator.py.
Adapted for FastAPI: output_dir defaults to settings.upload_dir.
"""
import base64
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import get_settings


def _get_output_dir(output_dir: Optional[Path] = None) -> Path:
    d = output_dir or Path(get_settings().upload_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_b64_image(b64_str: str, dest: Path) -> bool:
    """Decode a base64 image string and save to dest. Returns True on success."""
    try:
        raw = base64.b64decode(b64_str)
        dest.write_bytes(raw)
        return True
    except Exception:
        return False


def generate_clinical_pdf_report(
    scan_id: str,
    prediction_data: Dict[str, Any],
    original_image_bytes: Optional[bytes] = None,
    gradcam_b64: Optional[str] = None,
    patient_metadata: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Generate a publication-quality Clinical AI Diagnostic PDF Report.

    Args:
        scan_id:              Unique scan session identifier.
        prediction_data:      Prediction or ensemble consensus result dict from HF Space.
        original_image_bytes: Raw JPEG bytes of the original CXR (embedded in PDF).
        gradcam_b64:          Base64-encoded Grad-CAM overlay image from HF Space response.
        patient_metadata:     Optional patient demographics dict.
        output_dir:           Output directory. Defaults to settings.upload_dir.

    Returns:
        Path to the generated PDF file.
    """
    out_dir = _get_output_dir(output_dir)
    pdf_path = out_dir / f"report_{scan_id}.pdf"

    # ── Save image bytes to temp files for ReportLab embedding ────────────────
    orig_img_path: Optional[Path] = None
    if original_image_bytes:
        orig_img_path = out_dir / f"orig_{scan_id}.jpg"
        orig_img_path.write_bytes(original_image_bytes)

    cam_img_path: Optional[Path] = None
    if gradcam_b64:
        cam_img_path = out_dir / f"cam_{scan_id}.jpg"
        if not _save_b64_image(gradcam_b64, cam_img_path):
            cam_img_path = None

    # ── Document setup ────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=36, leftMargin=36,
        topMargin=36, bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=18, leading=22,
        textColor=colors.HexColor("#0f172a"),
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=8.5, leading=11,
        textColor=colors.HexColor("#64748b"),
    )
    section_heading = ParagraphStyle(
        "SectionHeading", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=11, leading=15,
        textColor=colors.HexColor("#1e293b"), spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "ReportBody", parent=styles["Normal"],
        fontName="Helvetica", fontSize=8.5, leading=12,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "ReportDisclaimer", parent=styles["Normal"],
        fontName="Helvetica-Oblique", fontSize=7.0, leading=9.5,
        textColor=colors.HexColor("#64748b"),
    )

    story = []

    # ── 1. Header ─────────────────────────────────────────────────────────────
    header_data = [[
        Paragraph(
            "<b>PNEUMONIA DIAGNOSTIC HUB</b><br/>"
            "<font size=7.5 color='#64748b'>AI Radiology Workstation &amp; Decision Support System</font>",
            title_style,
        ),
        Paragraph(
            f"<b>SCAN ID:</b> {scan_id}<br/>"
            f"<b>STUDY DATE:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>"
            "<b>FACILITY:</b> AI Radiology Lab",
            subtitle_style,
        ),
    ]]
    header_tbl = Table(header_data, colWidths=[340, 200])
    header_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5,
                            color=colors.HexColor("#0284c7"), spaceAfter=10))

    # ── 2. Patient demographics ───────────────────────────────────────────────
    pat = patient_metadata or {}
    pat_id      = pat.get("patient_id") or f"PT-{scan_id[:6]}"
    pat_age     = pat.get("patient_age") or "N/A"
    pat_sex     = pat.get("patient_gender") or pat.get("patient_sex") or "N/A"
    physician   = pat.get("referring_physician") or "Dr. On-Duty / Specialist"
    history     = pat.get("clinical_history") or "Routine pulmonary radiograph examination."
    modality    = pat.get("modality") or "Digital Radiography (CXR AP/PA)"

    pat_data = [
        [
            Paragraph(f"<b>Patient ID:</b> {pat_id}", body_style),
            Paragraph(f"<b>Age / Gender:</b> {pat_age} / {pat_sex}", body_style),
            Paragraph(f"<b>Modality:</b> {modality}", body_style),
        ],
        [
            Paragraph(f"<b>Physician:</b> {physician}", body_style),
            Paragraph(f"<b>Indication / History:</b> {history}", body_style),
            Paragraph("<b>View:</b> Anterior-Posterior (AP)", body_style),
        ],
    ]
    pat_tbl = Table(pat_data, colWidths=[180, 220, 140])
    pat_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX",        (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID",  (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING",    (0, 0), (-1, -1), 5),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(pat_tbl)
    story.append(Spacer(1, 10))

    # ── 3. Verdict box ────────────────────────────────────────────────────────
    is_ensemble = prediction_data.get("is_ensemble", False)
    if is_ensemble:
        verdict     = prediction_data.get("consensus_verdict", "UNKNOWN")
        confidence  = prediction_data.get("consensus_confidence", 0.0)
        probs       = prediction_data.get("consensus_probabilities", {})
        engine_lbl  = "Multi-Model Weighted Ensemble (4 Models)"
        agree_note  = prediction_data.get("agreement_text", "")
    else:
        verdict     = prediction_data.get("prediction", "UNKNOWN")
        confidence  = prediction_data.get("confidence", 0.0)
        probs       = prediction_data.get("probabilities", {})
        engine_lbl  = (
            f"{prediction_data.get('model_name', 'CNN')} "
            f"({prediction_data.get('model_parameters', '')})"
        )
        agree_note  = f"Single Backbone: {engine_lbl}"

    p_pneumonia = probs.get("PNEUMONIA", 0.0)
    p_normal    = probs.get("NORMAL", 0.0)

    v_color = colors.HexColor("#b91c1c") if verdict == "PNEUMONIA" else colors.HexColor("#047857")
    v_bg    = colors.HexColor("#fee2e2") if verdict == "PNEUMONIA" else colors.HexColor("#d1fae5")

    verdict_data = [[
        Paragraph(
            f"<font size=10 color='{v_color.hexval()}'><b>DIAGNOSTIC VERDICT:</b></font><br/>"
            f"<font size=16 color='{v_color.hexval()}'><b>{verdict}</b></font><br/>"
            f"<font size=8 color='#475569'>{agree_note}</font>",
            styles["Normal"],
        ),
        Paragraph(
            f"<b>Overall Confidence:</b> {confidence}%<br/>"
            f"<b>Pneumonia Probability:</b> {p_pneumonia}%<br/>"
            f"<b>Normal Probability:</b> {p_normal}%<br/>"
            f"<b>Engine:</b> {engine_lbl}",
            styles["Normal"],
        ),
    ]]
    verdict_tbl = Table(verdict_data, colWidths=[270, 270])
    verdict_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), v_bg),
        ("BOX",        (0, 0), (-1, -1), 1, v_color),
        ("PADDING",    (0, 0), (-1, -1), 8),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(verdict_tbl)
    story.append(Spacer(1, 10))

    # ── 4. Visual evidence ────────────────────────────────────────────────────
    story.append(Paragraph("<b>1. Radiological &amp; Spatial Attention Evidence</b>", section_heading))

    img_w, img_h = 235, 175
    if orig_img_path and orig_img_path.exists():
        orig_flow = Image(str(orig_img_path), width=img_w, height=img_h)
        cam_flow  = (
            Image(str(cam_img_path), width=img_w, height=img_h)
            if cam_img_path and cam_img_path.exists()
            else orig_flow
        )
        img_data = [
            [orig_flow, cam_flow],
            [
                Paragraph("<font size=7.5 color='#64748b'><b>Figure A:</b> Original Chest Radiograph</font>", styles["Normal"]),
                Paragraph("<font size=7.5 color='#64748b'><b>Figure B:</b> Grad-CAM Anatomical Heatmap Overlay</font>", styles["Normal"]),
            ],
        ]
        img_tbl = Table(img_data, colWidths=[270, 270])
        img_tbl.setStyle(TableStyle([
            ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING",(0, 0), (-1, -1), 3),
        ]))
        story.append(img_tbl)
        story.append(Spacer(1, 10))

    # ── 5. Multi-model matrix (ensemble only) ─────────────────────────────────
    if is_ensemble and "models_breakdown" in prediction_data:
        story.append(Paragraph("<b>2. Multi-Model Telemetry &amp; Consensus Matrix</b>", section_heading))
        rows = [["Model Architecture", "Parameters", "Weight", "Prediction", "Confidence", "Latency"]]
        for m in prediction_data["models_breakdown"]:
            pred_clr = "#b91c1c" if m.get("prediction") == "PNEUMONIA" else "#047857"
            rows.append([
                m.get("name", ""),
                m.get("parameters", ""),
                f"{int(m.get('weight', 0) * 100)}%",
                Paragraph(f"<font color='{pred_clr}'><b>{m.get('prediction', '')}</b></font>", styles["Normal"]),
                f"{m.get('confidence', 0)}%",
                f"{m.get('inference_time_ms', 0)} ms",
            ])
        matrix = Table(rows, colWidths=[130, 75, 55, 95, 90, 95])
        matrix.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(matrix)
        story.append(Spacer(1, 10))

    # ── 6. Clinical findings ──────────────────────────────────────────────────
    story.append(Paragraph("<b>3. Interpretive Findings &amp; Anatomical Note</b>", section_heading))
    if verdict == "PNEUMONIA":
        finding = (
            "Neural activation gradients demonstrate focal concentration over the pulmonary parenchyma "
            "consistent with alveolar consolidation, patchy infiltrates, or pleural haziness. "
            "Immediate correlation with clinical signs and laboratory biomarkers is recommended."
        )
    else:
        finding = (
            "No significant radiological patterns of dense consolidation, lobar opacification, or "
            "reticular interstitial infiltrates were detected. Bilateral lung fields demonstrate "
            "standard radiolucency across all evaluated neural architectures."
        )
    story.append(Paragraph(finding, body_style))
    story.append(Spacer(1, 10))

    # ── 7. Disclaimer & signature ─────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#cbd5e1"), spaceAfter=6))
    story.append(Paragraph(
        "<b>LEGAL &amp; CLINICAL DISCLAIMER:</b> This diagnostic report is generated by an artificial "
        "intelligence decision-support tool for research and second-opinion purposes only. It is not an "
        "autonomous diagnostic device. Definitive medical conclusions must be validated by a licensed "
        "physician or radiologist.",
        disclaimer_style,
    ))
    story.append(Spacer(1, 6))
    sig_data = [[
        Paragraph("<font size=7 color='#64748b'><b>AI Workstation:</b> Pneumonia-Diagnostic-Hub v2.4 (FastAPI)</font>", styles["Normal"]),
        Paragraph("<font size=7 color='#64748b'><b>Reviewing Radiologist:</b> ___________________________</font>", styles["Normal"]),
    ]]
    sig_tbl = Table(sig_data, colWidths=[270, 270])
    sig_tbl.setStyle(TableStyle([("ALIGN", (1, 0), (1, 0), "RIGHT")]))
    story.append(sig_tbl)

    doc.build(story)

    # ── Cleanup temp image files ───────────────────────────────────────────────
    for tmp in [orig_img_path, cam_img_path]:
        if tmp and tmp.exists():
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    return pdf_path
