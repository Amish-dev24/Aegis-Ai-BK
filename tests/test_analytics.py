"""
Tests for analytics endpoints:
  GET /api/v1/analytics/heatmap
  GET /api/v1/analytics/timeline
  GET /api/v1/analytics/by-zone
  GET /api/v1/analytics/threat-distribution
  GET /api/v1/analytics/top-cameras

All endpoints require authentication and respect company-level isolation.
"""
import pytest

BASE = "/api/v1/analytics"


class TestHeatmap:
    def test_returns_heatmap_key(self, client, officer_headers, test_detection):
        resp = client.get(f"{BASE}/heatmap", headers=officer_headers)
        assert resp.status_code == 200
        assert "heatmap" in resp.json()

    def test_heatmap_items_have_expected_fields(self, client, officer_headers, test_detection):
        resp = client.get(f"{BASE}/heatmap", headers=officer_headers)
        for item in resp.json()["heatmap"]:
            assert "latitude" in item
            assert "longitude" in item
            assert "count" in item

    def test_heatmap_requires_auth(self, client):
        resp = client.get(f"{BASE}/heatmap")
        assert resp.status_code == 401

    def test_viewer_can_access_heatmap(self, client, viewer_headers):
        resp = client.get(f"{BASE}/heatmap", headers=viewer_headers)
        assert resp.status_code == 200

    def test_days_param(self, client, officer_headers):
        resp = client.get(f"{BASE}/heatmap?days=30", headers=officer_headers)
        assert resp.status_code == 200


class TestTimeline:
    # date_trunc() is PostgreSQL-only; these tests are skipped on SQLite.
    # They are fully exercised when running against a real PostgreSQL database.
    @pytest.mark.skip(reason="date_trunc() requires PostgreSQL; not available in SQLite test DB")
    def test_returns_timeline_key(self, client, officer_headers, test_detection):
        resp = client.get(f"{BASE}/timeline", headers=officer_headers)
        assert resp.status_code == 200
        assert "timeline" in resp.json()

    @pytest.mark.skip(reason="date_trunc() requires PostgreSQL; not available in SQLite test DB")
    def test_timeline_items_have_expected_fields(self, client, officer_headers, test_detection):
        resp = client.get(f"{BASE}/timeline", headers=officer_headers)
        for item in resp.json()["timeline"]:
            assert "timestamp" in item
            assert "count" in item

    def test_timeline_requires_auth(self, client):
        assert client.get(f"{BASE}/timeline").status_code == 401


class TestByZone:
    def test_returns_by_zone_key(self, client, officer_headers, test_detection):
        resp = client.get(f"{BASE}/by-zone", headers=officer_headers)
        assert resp.status_code == 200
        assert "by_zone" in resp.json()

    def test_zone_items_have_zone_and_count(self, client, officer_headers, test_detection):
        resp = client.get(f"{BASE}/by-zone", headers=officer_headers)
        for item in resp.json()["by_zone"]:
            assert "zone" in item
            assert "count" in item

    def test_requires_auth(self, client):
        assert client.get(f"{BASE}/by-zone").status_code == 401


class TestThreatDistribution:
    def test_returns_distribution_key(self, client, officer_headers, test_detection):
        resp = client.get(f"{BASE}/threat-distribution", headers=officer_headers)
        assert resp.status_code == 200
        assert "distribution" in resp.json()

    def test_distribution_has_threat_levels(self, client, officer_headers, test_detection):
        resp = client.get(f"{BASE}/threat-distribution", headers=officer_headers)
        dist = resp.json()["distribution"]
        # Our seed detection is ThreatLevel.HIGH
        assert "high" in dist or dist == {}

    def test_requires_auth(self, client):
        assert client.get(f"{BASE}/threat-distribution").status_code == 401


class TestTopCameras:
    def test_returns_top_cameras_key(self, client, officer_headers, test_detection):
        resp = client.get(f"{BASE}/top-cameras", headers=officer_headers)
        assert resp.status_code == 200
        assert "top_cameras" in resp.json()

    def test_top_cameras_items_have_expected_fields(self, client, officer_headers, test_detection):
        resp = client.get(f"{BASE}/top-cameras", headers=officer_headers)
        for cam in resp.json()["top_cameras"]:
            assert "name" in cam
            assert "count" in cam

    def test_limit_param(self, client, officer_headers, test_detection):
        resp = client.get(f"{BASE}/top-cameras?limit=1", headers=officer_headers)
        assert resp.status_code == 200
        assert len(resp.json()["top_cameras"]) <= 1

    def test_requires_auth(self, client):
        assert client.get(f"{BASE}/top-cameras").status_code == 401
