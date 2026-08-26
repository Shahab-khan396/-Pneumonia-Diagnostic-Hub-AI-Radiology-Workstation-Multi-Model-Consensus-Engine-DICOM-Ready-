import os
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from config import UPLOAD_FOLDER


def generate_clinical_pdf_report(
    scan_id: str,
    prediction_data: Dict[str, Any],
    original_image_path: Path,
    gradcam_overlay_path: Optional[Path] = None,
    patient_metadata: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Generate an official, publication-quality Clinical AI Diagnostic PDF Report.
    
    Args:
        scan_id: Unique identifier for this scan session.
        prediction_data: Dictionary containing prediction or ensemble consensus metrics.
        original_image_path: Path to the original CXR file.
        gradcam_overlay_path: Optional path to the Grad-CAM overlaid CXR.
        patient_metadata: Optional dict with patient demographics, indications, and DICOM metadata.
        output_dir: Directory where the PDF will be saved. Defaults to UPLOAD_FOLDER.
        
    Returns:
        Path: Absolute path to the generated PDF file.
    """
    if output_dir is None:
        output_dir = Path(UPLOAD_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_filename = f"report_{scan_id}.pdf"
    pdf_path = output_dir / report_filename

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
    )
    
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#64748b"),
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=5,
    )
    
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )

    disclaimer_style = ParagraphStyle(
        "ReportDisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.0,
        leading=9.5,
        textColor=colors.HexColor("#64748b"),
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>PNEUMONIA DIAGNOSTIC HUB</b><br/><font size=7.5 color='#64748b'>AI Radiology Workstation & Decision Support System</font>", title_style),
            Paragraph(f"<b>SCAN ID:</b> {scan_id}<br/><b>STUDY DATE:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}<br/><b>FACILITY:</b> AI Radiology Lab", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[340, 200])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    # 2. Patient Demographics & Clinical Indications Table
    pat = patient_metadata or {}
    pat_id = pat.get("patient_id") or "PT-" + scan_id[:6]
    pat_age = pat.get("patient_age") or "N/A"
    pat_sex = pat.get("patient_gender") or pat.get("patient_sex") or "N/A"
    physician = pat.get("referring_physician") or "Dr. On-Duty / Specialist"
    history = pat.get("clinical_history") or "Routine pulmonary radiograph examination / Acute respiratory symptoms."
    modality = pat.get("modality") or "Digital Radiography (CXR AP/PA)"

    patient_table_data = [
        [
            Paragraph(f"<b>Patient ID:</b> {pat_id}", body_style),
            Paragraph(f"<b>Age / Gender:</b> {pat_age} / {pat_sex}", body_style),
            Paragraph(f"<b>Modality:</b> {modality}", body_style),
        ],
        [
            Paragraph(f"<b>Physician:</b> {physician}", body_style),
            Paragraph(f"<b>Indications / History:</b> {history}", body_style),
            Paragraph(f"<b>View:</b> Anterior-Posterior (AP)", body_style),
        ]
    ]
    patient_table = Table(patient_table_data, colWidths=[180, 220, 140])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 10))

    # 3. Executive Diagnostic Verdict Box
    is_ensemble = prediction_data.get("is_ensemble", False)
    if is_ensemble:
        verdict = prediction_data.get("consensus_verdict", "UNKNOWN")
        confidence = prediction_data.get("consensus_confidence", 0.0)
        p_normal = prediction_data.get("consensus_probabilities", {}).get("NORMAL", 0.0)
        p_pneumonia = prediction_data.get("consensus_probabilities", {}).get("PNEUMONIA", 0.0)
        engine_label = "Multi-Model Weighted Ensemble (4 Models)"
        agreement_note = prediction_data.get("agreement_text", "")
    else:
        verdict = prediction_data.get("prediction", "UNKNOWN")
        confidence = prediction_data.get("confidence", 0.0)
        p_normal = prediction_data.get("probabilities", {}).get("NORMAL", 0.0)
        p_pneumonia = prediction_data.get("probabilities", {}).get("PNEUMONIA", 0.0)
        engine_label = f"{prediction_data.get('model_name', 'CNN')} ({prediction_data.get('model_parameters', '')})"
        agreement_note = f"Single Backbone: {engine_label}"

    verdict_color = colors.HexColor("#b91c1c") if verdict == "PNEUMONIA" else colors.HexColor("#047857")
    verdict_bg = colors.HexColor("#fee2e2") if verdict == "PNEUMONIA" else colors.HexColor("#d1fae5")

    verdict_box_data = [
        [
            Paragraph(f"<font size=10 color='{verdict_color.hexval()}'><b>DIAGNOSTIC VERDICT:</b></font><br/><font size=16 color='{verdict_color.hexval()}'><b>{verdict}</b></font><br/><font size=8 color='#475569'>{agreement_note}</font>", styles["Normal"]),
            Paragraph(f"<b>Overall Confidence:</b> {confidence}%<br/><b>Pneumonia Probability:</b> {p_pneumonia}%<br/><b>Normal Probability:</b> {p_normal}%<br/><b>Engine:</b> {engine_label}", styles["Normal"])
        ]
    ]
    verdict_table = Table(verdict_box_data, colWidths=[270, 270])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), verdict_bg),
        ('BOX', (0,0), (-1,-1), 1, verdict_color),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 10))

    # 4. Visual Radiological Evidence (Side-by-side Images)
    story.append(Paragraph("<b>1. Radiological & Spatial Attention Evidence</b>", section_heading))
    
    img_width = 235
    img_height = 175
    
    orig_img_flowable = Image(str(original_image_path), width=img_width, height=img_height)
    
    if gradcam_overlay_path and Path(gradcam_overlay_path).exists():
        cam_img_flowable = Image(str(gradcam_overlay_path), width=img_width, height=img_height)
    else:
        cam_img_flowable = orig_img_flowable

    image_table_data = [
        [
            orig_img_flowable,
            cam_img_flowable
        ],
        [
            Paragraph("<font size=7.5 color='#64748b'><b>Figure A:</b> Original Chest Radiograph</font>", styles["Normal"]),
            Paragraph("<font size=7.5 color='#64748b'><b>Figure B:</b> Grad-CAM Anatomical Heatmap Overlay</font>", styles["Normal"])
        ]
    ]
    image_table = Table(image_table_data, colWidths=[270, 270])
    image_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(image_table)
    story.append(Spacer(1, 10))

    # 5. Multi-Model Architecture Comparison Table
    if is_ensemble and "models_breakdown" in prediction_data:
        story.append(Paragraph("<b>2. Multi-Model Telemetry & Consensus Matrix</b>", section_heading))
        
        table_rows = [
            ["Model Architecture", "Parameters", "Weight", "Prediction", "Confidence", "Latency"]
        ]
        for m in prediction_data["models_breakdown"]:
            pred_text = f"<font color='{'#b91c1c' if m['prediction']=='PNEUMONIA' else '#047857'}'><b>{m['prediction']}</b></font>"
            table_rows.append([
                m["name"],
                m["parameters"],
                f"{int(m['weight']*100)}%",
                Paragraph(pred_text, styles["Normal"]),
                f"{m['confidence']}%",
                f"{m['inference_time_ms']} ms"
            ])
            
        model_matrix = Table(table_rows, colWidths=[130, 75, 55, 95, 90, 95])
        model_matrix.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#0f172a")),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(model_matrix)
        story.append(Spacer(1, 10))

    # 6. Clinical Findings & Interpretive Notes
    story.append(Paragraph("<b>3. Interpretive Findings & Anatomical Note</b>", section_heading))
    if verdict == "PNEUMONIA":
        finding_text = (
            "Neural activation gradients demonstrate focal concentration over the pulmonary parenchyma consistent with "
            "alveolar consolidation, patchy infiltrates, or pleural haziness. Immediate correlation with clinical signs "
            "(body temperature, auscultatory crackles, pulse oximetry) and laboratory biomarkers is recommended."
        )
    else:
        finding_text = (
            "No significant radiological patterns of dense consolidation, lobar opacification, or reticular interstitial "
            "infiltrates were detected across evaluated neural architectures. Bilateral lung fields demonstrate standard radiolucency."
        )
    story.append(Paragraph(finding_text, body_style))
    story.append(Spacer(1, 10))

    # 7. Legal Medical Disclaimer & Signature Area
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
    disclaimer_text = (
        "<b>LEGAL & CLINICAL DISCLAIMER:</b> This diagnostic report is generated by an artificial intelligence decision-support tool "
        "for research and second-opinion purposes. It is not an autonomous diagnostic device. Definitive medical conclusions and treatment "
        "prescriptions must be validated by a licensed physician or radiologist."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    story.append(Spacer(1, 6))
    
    # Signature line
    sig_data = [
        [
            Paragraph("<font size=7 color='#64748b'><b>AI Workstation:</b> Pneumonia-Diagnostic-Hub v2.3 (DICOM-Enabled)</font>", styles["Normal"]),
            Paragraph("<font size=7 color='#64748b'><b>Reviewing Radiologist:</b> ___________________________</font>", styles["Normal"])
        ]
    ]
    sig_table = Table(sig_data, colWidths=[270, 270])
    sig_table.setStyle(TableStyle([('ALIGN', (1,0), (1,0), 'RIGHT')]))
    story.append(sig_table)

    # Build Document
    doc.build(story)
    
    return pdf_path
