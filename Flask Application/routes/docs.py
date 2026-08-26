from flask import Blueprint, jsonify, render_template_string

docs_bp = Blueprint("docs", __name__)

OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "Pneumonia Diagnostic Hub API",
        "version": "2.3.0",
        "description": (
            "Enterprise-grade deep learning REST API for automated pneumonia detection, "
            "multi-model consensus comparison, Grad-CAM visual explainability, DICOM parsing, and PDF clinical reporting."
        ),
        "contact": {
            "name": "Pneumonia Diagnostic Hub AI Engineering Team",
            "email": "support@pneumonia-hub.ai"
        },
        "license": {
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        }
    },
    "servers": [
        {"url": "/", "description": "Current Server Host"}
    ],
    "tags": [
        {"name": "Diagnostics", "description": "Inference, Multi-Model Comparison & Grad-CAM XAI"},
        {"name": "Clinical Reports", "description": "PDF Report Generation & Download"},
        {"name": "System & Catalog", "description": "Health check, Model Specifications & Sample Radiographs"}
    ],
    "paths": {
        "/api/v1/health": {
            "get": {
                "tags": ["System & Catalog"],
                "summary": "Health and Telemetry Endpoint",
                "description": "Returns operational health status, API version, and loaded deep learning models.",
                "responses": {
                    "200": {
                        "description": "System is healthy",
                        "content": {
                            "application/json": {
                                "example": {
                                    "status": "healthy",
                                    "service": "Pneumonia-Diagnostic-Hub-API",
                                    "version": "2.3.0",
                                    "models_count": 4
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/v1/models": {
            "get": {
                "tags": ["System & Catalog"],
                "summary": "Model Catalog & Specifications",
                "description": "Returns the registry of all available CNN backbones with parameter sizes and target layers.",
                "responses": {
                    "200": {
                        "description": "List of deep learning models"
                    }
                }
            }
        },
        "/api/v1/samples": {
            "get": {
                "tags": ["System & Catalog"],
                "summary": "Sample Radiograph Library",
                "description": "Returns pre-packaged sample radiographs for instant zero-upload testing.",
                "responses": {
                    "200": {
                        "description": "Catalog of sample radiographs"
                    }
                }
            }
        },
        "/api/v1/predict": {
            "post": {
                "tags": ["Diagnostics"],
                "summary": "Single Model Inference & Grad-CAM",
                "description": "Upload a Chest X-Ray or DICOM file to receive diagnostic classification, confidence probabilities, and Grad-CAM heatmaps.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "file": {"type": "string", "format": "binary", "description": "Chest Radiograph (PNG, JPG, WEBP, or DICOM .dcm)"},
                                    "sample_id": {"type": "string", "description": "Optional ID of a sample radiograph (e.g. sample_bacterial)"},
                                    "model_choice": {"type": "string", "enum": ["mobilenet", "efficientnet", "resnet50", "VGG19"], "default": "mobilenet"},
                                    "explain": {"type": "boolean", "default": True},
                                    "generate_report": {"type": "boolean", "default": True},
                                    "patient_id": {"type": "string", "example": "PT-98210"},
                                    "patient_age": {"type": "string", "example": "48"},
                                    "patient_gender": {"type": "string", "example": "Female"},
                                    "clinical_history": {"type": "string", "example": "3-day cough, fever 38.5C"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Diagnostic prediction & Grad-CAM URLs"},
                    "400": {"description": "Validation error or unreadable image"},
                    "500": {"description": "Internal processing error"}
                }
            }
        },
        "/api/v1/compare": {
            "post": {
                "tags": ["Diagnostics"],
                "summary": "Multi-Model Consensus Comparison",
                "description": "Executes inference across all 4 deep learning models simultaneously and computes a weighted consensus verdict.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "file": {"type": "string", "format": "binary"},
                                    "sample_id": {"type": "string"},
                                    "explain": {"type": "boolean", "default": True},
                                    "patient_id": {"type": "string"},
                                    "clinical_history": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Multi-model comparison scorecard and weighted consensus decision"}
                }
            }
        },
        "/api/v1/report/{report_filename}": {
            "get": {
                "tags": ["Clinical Reports"],
                "summary": "Download Clinical PDF Diagnostic Report",
                "parameters": [
                    {
                        "name": "report_filename",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                "responses": {
                    "200": {"description": "Binary PDF stream (application/pdf)"},
                    "404": {"description": "Report not found"}
                }
            }
        }
    }
}

SWAGGER_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Pneumonia Diagnostic Hub | Swagger API Playground</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css" />
    <link rel="icon" type="image/png" href="https://unpkg.com/swagger-ui-dist@5.11.0/favicon-32x32.png" />
    <style>
        body { margin: 0; background: #fafafa; font-family: sans-serif; }
        .topbar { display: none !important; }
        .swagger-ui .info { margin: 30px 0; }
        .swagger-ui .info .title { color: #0284c7; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js"></script>
    <script>
        window.onload = function() {
            SwaggerUIBundle({
                url: "/api/v1/openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.SwaggerUIStandalonePreset
                ],
                layout: "BaseLayout"
            });
        };
    </script>
</body>
</html>
"""


@docs_bp.route("/api/v1/openapi.json", methods=["GET"])
def get_openapi_spec():
    """Serve OpenAPI 3.0 specification JSON."""
    return jsonify(OPENAPI_SPEC), 200


@docs_bp.route("/docs", methods=["GET"])
@docs_bp.route("/api/docs", methods=["GET"])
def swagger_ui():
    """Render interactive Swagger UI API playground."""
    return render_template_string(SWAGGER_HTML_TEMPLATE)
