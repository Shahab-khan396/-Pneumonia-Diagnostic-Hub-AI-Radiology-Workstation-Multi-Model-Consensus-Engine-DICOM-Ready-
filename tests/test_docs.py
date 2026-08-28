import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


def test_openapi_json_endpoint(client):
    """Verify GET /openapi.json returns valid OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    assert "openapi" in spec
    assert "paths" in spec
    assert "/predict" in spec["paths"] or "/hub_api/predict" in spec["paths"]
    assert "/compare" in spec["paths"] or "/hub_api/compare" in spec["paths"]
    assert "/samples" in spec["paths"] or "/hub_api/samples" in spec["paths"]
    assert "/health" in spec["paths"] or "/hub_api/health" in spec["paths"]


def test_swagger_ui_endpoint(client):
    """Verify GET /docs serves the Swagger UI HTML."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger-ui" in response.text or "SwaggerUIBundle" in response.text
