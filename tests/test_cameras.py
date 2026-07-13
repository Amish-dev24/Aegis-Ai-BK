"""
Tests for camera management endpoints:
  POST   /api/v1/cameras
  GET    /api/v1/cameras
  GET    /api/v1/cameras/{id}
  PUT    /api/v1/cameras/{id}
  DELETE /api/v1/cameras/{id}

Access rules:
  - Create / update / delete → security_officer+
  - Read → any authenticated user
  - Company users only see their own company's cameras
"""

BASE = "/api/v1/cameras"


# ── Create ────────────────────────────────────────────────────────────────────


class TestCreateCamera:
    def test_officer_can_create(self, client, officer_headers):
        resp = client.post(
            BASE,
            headers=officer_headers,
            json={"name": "Side Gate", "location": "East wing", "zone": "Zone-B"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "Side Gate"
        assert body["zone"] == "Zone-B"
        assert "id" in body

    def test_admin_can_create(self, client, admin_headers):
        resp = client.post(
            BASE,
            headers=admin_headers,
            json={"name": "Lobby Camera", "location": "Lobby", "zone": "Zone-A"},
        )
        assert resp.status_code == 201

    def test_viewer_cannot_create(self, client, viewer_headers):
        resp = client.post(
            BASE,
            headers=viewer_headers,
            json={"name": "Restricted Cam", "location": "X", "zone": "Zone-X"},
        )
        assert resp.status_code == 403

    def test_unauthenticated_cannot_create(self, client):
        resp = client.post(
            BASE,
            json={"name": "Anon Cam", "location": "X", "zone": "Z"},
        )
        assert resp.status_code == 401

    def test_camera_with_coordinates(self, client, officer_headers):
        resp = client.post(
            BASE,
            headers=officer_headers,
            json={
                "name": "Geo Cam",
                "location": "Roof",
                "zone": "Zone-C",
                "latitude": 31.5204,
                "longitude": 74.3587,
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["latitude"] == pytest.approx(31.5204)
        assert body["longitude"] == pytest.approx(74.3587)


# ── List ──────────────────────────────────────────────────────────────────────


class TestListCameras:
    def test_officer_sees_own_company_cameras(self, client, officer_headers, test_camera):
        resp = client.get(BASE, headers=officer_headers)
        assert resp.status_code == 200
        ids = [c["id"] for c in resp.json()]
        assert test_camera.id in ids

    def test_viewer_can_list(self, client, viewer_headers):
        resp = client.get(BASE, headers=viewer_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_unauthenticated_cannot_list(self, client):
        resp = client.get(BASE)
        assert resp.status_code == 401

    def test_active_only_filter(self, client, officer_headers, test_camera):
        resp = client.get(f"{BASE}?active_only=true", headers=officer_headers)
        assert resp.status_code == 200
        payload = resp.json()
        assert any(c["id"] == test_camera.id for c in payload), (
            "active_only must return active cameras (regression: SQLAlchemy `is True` bug)"
        )
        for cam in payload:
            assert cam["is_active"] is True

    def test_pagination(self, client, officer_headers):
        resp = client.get(f"{BASE}?limit=2&offset=0", headers=officer_headers)
        assert resp.status_code == 200
        assert len(resp.json()) <= 2


# ── Get by ID ─────────────────────────────────────────────────────────────────


class TestGetCamera:
    def test_get_existing_camera(self, client, officer_headers, test_camera):
        resp = client.get(f"{BASE}/{test_camera.id}", headers=officer_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == test_camera.id
        assert resp.json()["name"] == test_camera.name

    def test_get_nonexistent_camera(self, client, officer_headers):
        resp = client.get(f"{BASE}/999999", headers=officer_headers)
        assert resp.status_code == 404

    def test_viewer_can_get_camera(self, client, viewer_headers, test_camera):
        resp = client.get(f"{BASE}/{test_camera.id}", headers=viewer_headers)
        assert resp.status_code == 200


# ── Update ────────────────────────────────────────────────────────────────────


class TestUpdateCamera:
    def test_officer_can_update(self, client, officer_headers, test_camera):
        resp = client.put(
            f"{BASE}/{test_camera.id}",
            headers=officer_headers,
            json={"description": "Updated by CI test"},
        )
        assert resp.status_code == 200
        assert resp.json()["description"] == "Updated by CI test"

    def test_viewer_cannot_update(self, client, viewer_headers, test_camera):
        resp = client.put(
            f"{BASE}/{test_camera.id}",
            headers=viewer_headers,
            json={"description": "Should fail"},
        )
        assert resp.status_code == 403

    def test_update_nonexistent_camera(self, client, officer_headers):
        resp = client.put(
            f"{BASE}/999999",
            headers=officer_headers,
            json={"name": "Ghost"},
        )
        assert resp.status_code == 404


# ── Delete ────────────────────────────────────────────────────────────────────


class TestDeleteCamera:
    def test_officer_can_delete(self, client, officer_headers):
        # Create a disposable camera first
        create_resp = client.post(
            BASE,
            headers=officer_headers,
            json={"name": "Temp Cam", "location": "Temp", "zone": "Zone-T"},
        )
        cam_id = create_resp.json()["id"]

        del_resp = client.delete(f"{BASE}/{cam_id}", headers=officer_headers)
        assert del_resp.status_code == 204

        get_resp = client.get(f"{BASE}/{cam_id}", headers=officer_headers)
        assert get_resp.status_code == 404

    def test_viewer_cannot_delete(self, client, viewer_headers, test_camera):
        resp = client.delete(f"{BASE}/{test_camera.id}", headers=viewer_headers)
        assert resp.status_code == 403

    def test_delete_nonexistent(self, client, officer_headers):
        resp = client.delete(f"{BASE}/999999", headers=officer_headers)
        assert resp.status_code == 404


import pytest  # noqa: E402 (kept at bottom to avoid shadowing fixtures)
