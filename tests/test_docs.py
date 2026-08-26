import sys
from pathlib import Path
import pytest

# Ensure Flask Application is in sys.path
flask_app_dir = Path(__file__).resolve().parent.parent / "Flask Application"
if str(flask_app_dir) not in sys.path:
    sys.path.insert(0, str(flask_app_dir))

from app import create_app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_openapi_json_endpoint(client):
    """Verify GET /api/v1/openapi.json returns valid OpenAPI 3.0 schema."""
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    spec = response.get_json()
    assert spec["openapi"] == "3.0.3"
    assert "paths" in spec
    assert "/api/v1/predict" in spec["paths"]
    assert "/api/v1/compare" in spec["paths"]
    assert "/api/v1/samples" in spec["paths"]
    assert "/api/v1/health" in spec["paths"]


def test_swagger_ui_endpoint(client):
    """Verify GET /docs serves the Swagger UI HTML."""
    response = client.get("/docs")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "swagger-ui" in html
    assert "SwaggerUIBundle" in html
