"""
Tests for alert management endpoints:
  GET  /api/v1/alerts
  GET  /api/v1/alerts/{id}
  POST /api/v1/alerts/{id}/acknowledge
"""

BASE = "/api/v1/alerts"


class TestListAlerts:
    def test_officer_can_list(self, client, officer_headers, test_alert):
        resp = client.get(BASE, headers=officer_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_viewer_can_list(self, client, viewer_headers, test_alert):
        resp = client.get(BASE, headers=viewer_headers)
        assert resp.status_code == 200

    def test_requires_auth(self, client):
        assert client.get(BASE).status_code == 401

    def test_alert_has_expected_fields(self, client, officer_headers, test_alert):
        resp = client.get(BASE, headers=officer_headers)
        alerts = resp.json()
        assert len(alerts) >= 1
        alert = alerts[0]
        assert "id" in alert
        assert "title" in alert
        assert "status" in alert

    def test_filter_by_status(self, client, officer_headers, test_alert):
        resp = client.get(f"{BASE}?status=pending", headers=officer_headers)
        assert resp.status_code == 200
        for a in resp.json():
            assert a["status"] == "pending"


class TestGetAlert:
    def test_get_by_id(self, client, officer_headers, test_alert):
        resp = client.get(f"{BASE}/{test_alert.id}", headers=officer_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == test_alert.id

    def test_get_nonexistent(self, client, officer_headers):
        resp = client.get(f"{BASE}/999999", headers=officer_headers)
        assert resp.status_code == 404

    def test_viewer_can_get(self, client, viewer_headers, test_alert):
        resp = client.get(f"{BASE}/{test_alert.id}", headers=viewer_headers)
        assert resp.status_code == 200


class TestAcknowledgeAlert:
    def test_officer_can_acknowledge(self, client, officer_headers, test_alert):
        resp = client.post(
            f"{BASE}/{test_alert.id}/acknowledge",
            headers=officer_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "acknowledged"

    def test_viewer_cannot_acknowledge(self, client, viewer_headers, test_alert):
        resp = client.post(
            f"{BASE}/{test_alert.id}/acknowledge",
            headers=viewer_headers,
        )
        assert resp.status_code == 403

    def test_acknowledge_nonexistent(self, client, officer_headers):
        resp = client.post(f"{BASE}/999999/acknowledge", headers=officer_headers)
        assert resp.status_code == 404
