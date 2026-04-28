"""
Tests for root and health-check endpoints.
"""


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "1.0.0"
    assert "docs" in data


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_docs_available_in_test_env(client):
    """Swagger UI should be accessible when ENVIRONMENT != production."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Aegis AI - Intelligent Surveillance Platform"
    # All major route groups must be present in paths
    paths = schema["paths"]
    assert any("/auth/login" in p for p in paths)
    assert any("/cameras" in p for p in paths)
    assert any("/detections" in p for p in paths)
    assert any("/analytics" in p for p in paths)
    assert any("/alerts" in p for p in paths)
    assert any("/evidence" in p for p in paths)
