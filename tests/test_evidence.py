"""
Tests for evidence endpoints:
  GET  /api/v1/evidence
  GET  /api/v1/evidence/{id}
  POST /api/v1/evidence/export
"""
import pytest

# Detect pandas / numpy binary incompatibility in the current environment
try:
    import pandas  # noqa: F401
    _PANDAS_OK = True
except (ImportError, ValueError):
    _PANDAS_OK = False

BASE = "/api/v1/evidence"


class TestListEvidence:
    def test_officer_can_list(self, client, officer_headers):
        resp = client.get(BASE, headers=officer_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_viewer_can_list(self, client, viewer_headers):
        resp = client.get(BASE, headers=viewer_headers)
        assert resp.status_code == 200

    def test_requires_auth(self, client):
        assert client.get(BASE).status_code == 401

    def test_filter_by_detection_type(self, client, officer_headers):
        resp = client.get(f"{BASE}?detection_type=weapon", headers=officer_headers)
        assert resp.status_code == 200

    def test_filter_by_threat_level(self, client, officer_headers):
        resp = client.get(f"{BASE}?threat_level=high", headers=officer_headers)
        assert resp.status_code == 200

    def test_pagination_limit(self, client, officer_headers):
        resp = client.get(f"{BASE}?limit=5", headers=officer_headers)
        assert resp.status_code == 200
        assert len(resp.json()) <= 5

    def test_filter_by_min_confidence(self, client, officer_headers):
        resp = client.get(f"{BASE}?min_confidence=0.5", headers=officer_headers)
        assert resp.status_code == 200
        for ev in resp.json():
            assert ev["confidence"] is None or ev["confidence"] >= 0.5


class TestGetEvidence:
    def test_get_nonexistent(self, client, officer_headers):
        resp = client.get(f"{BASE}/999999", headers=officer_headers)
        assert resp.status_code == 404

    def test_requires_auth(self, client):
        assert client.get(f"{BASE}/1").status_code == 401


class TestExportEvidence:
    def test_export_requires_auth(self, client):
        resp = client.post(f"{BASE}/export", json=[1])
        assert resp.status_code == 401

    @pytest.mark.skipif(not _PANDAS_OK, reason="pandas/numpy binary incompatibility in this env")
    def test_export_empty_list(self, client, admin_headers):
        resp = client.post(
            f"{BASE}/export",
            headers=admin_headers,
            json=[],
        )
        assert resp.status_code in (200, 422)

    @pytest.mark.skipif(not _PANDAS_OK, reason="pandas/numpy binary incompatibility in this env")
    def test_export_nonexistent_ids(self, client, admin_headers):
        """Exporting IDs that don't exist should return empty CSV, not 4xx."""
        resp = client.post(
            f"{BASE}/export",
            headers=admin_headers,
            json=[999999, 999998],
        )
        assert resp.status_code == 200
        assert "csv" in resp.headers.get("content-type", "")
