"""
Tests for authentication endpoints:
  POST /api/v1/auth/login
  POST /api/v1/auth/refresh
  GET  /api/v1/auth/me
  POST /api/v1/auth/signup
  POST /api/v1/auth/register
"""
import pytest


BASE = "/api/v1/auth"


# ── Login ─────────────────────────────────────────────────────────────────────

class TestLogin:
    def test_login_success_officer(self, client, officer):
        resp = client.post(
            f"{BASE}/login",
            data={"username": officer.username, "password": "Officer1!"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"

    def test_login_success_admin(self, client, company_admin):
        resp = client.post(
            f"{BASE}/login",
            data={"username": company_admin.username, "password": "CompanyAdmin1!"},
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_login_wrong_password(self, client, officer):
        resp = client.post(
            f"{BASE}/login",
            data={"username": officer.username, "password": "WrongPass!"},
        )
        assert resp.status_code == 401
        assert "Incorrect" in resp.json()["detail"]

    def test_login_nonexistent_user(self, client):
        resp = client.post(
            f"{BASE}/login",
            data={"username": "no_such_user", "password": "Pass1!"},
        )
        assert resp.status_code == 401

    def test_login_inactive_user(self, client, db):
        """Inactive users must be rejected."""
        from app.models.user import User, Role
        from app.core.security import get_password_hash
        user = User(
            username="inactive_ci",
            email="inactive@example.com",
            hashed_password=get_password_hash("Pass1!"),
            role=Role.VIEWER,
            is_active=False,
        )
        db.add(user)
        db.commit()

        resp = client.post(
            f"{BASE}/login",
            data={"username": "inactive_ci", "password": "Pass1!"},
        )
        assert resp.status_code == 401


# ── Refresh token ─────────────────────────────────────────────────────────────

class TestRefreshToken:
    def test_refresh_with_valid_token(self, client, officer):
        login = client.post(
            f"{BASE}/login",
            data={"username": officer.username, "password": "Officer1!"},
        )
        refresh_tok = login.json()["refresh_token"]

        # refresh_token is a query parameter, not a JSON body
        resp = client.post(f"{BASE}/refresh", params={"refresh_token": refresh_tok})
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["refresh_token"] == refresh_tok

    def test_refresh_with_access_token_fails(self, client, officer_headers):
        """Passing an access token where a refresh token is expected must fail."""
        access = officer_headers["Authorization"].split(" ")[1]
        resp = client.post(f"{BASE}/refresh", params={"refresh_token": access})
        assert resp.status_code == 401

    def test_refresh_with_garbage_token(self, client):
        resp = client.post(f"{BASE}/refresh", params={"refresh_token": "not.a.token"})
        assert resp.status_code == 401


# ── /me ───────────────────────────────────────────────────────────────────────

class TestGetMe:
    def test_me_returns_current_user(self, client, officer_headers, officer):
        resp = client.get(f"{BASE}/me", headers=officer_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["username"] == officer.username
        assert body["role"] == "security_officer"

    def test_me_unauthenticated(self, client):
        resp = client.get(f"{BASE}/me")
        assert resp.status_code == 401

    def test_me_for_viewer(self, client, viewer_headers, viewer):
        resp = client.get(f"{BASE}/me", headers=viewer_headers)
        assert resp.status_code == 200
        assert resp.json()["role"] == "viewer"

    def test_me_for_aegis_admin(self, client, aegis_admin_headers):
        resp = client.get(f"{BASE}/me", headers=aegis_admin_headers)
        assert resp.status_code == 200
        assert resp.json()["role"] == "aegis_admin"


# ── Signup (public) ───────────────────────────────────────────────────────────

class TestSignup:
    def test_signup_creates_inactive_viewer(self, client, test_company):
        resp = client.post(
            f"{BASE}/signup",
            json={
                "username": "signup_test_user",
                "email": "signup_test@acmesecurity.example",
                "password": "SignupPass1!",
                "full_name": "Signup User",
                "company_id": test_company.id,
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["username"] == "signup_test_user"
        assert body["role"] == "viewer"
        assert body["is_active"] is False  # must be activated by admin

    def test_signup_duplicate_username(self, client, test_company):
        payload = {
            "username": "dup_signup_user",
            "email": "dup1@example.com",
            "password": "Dup1!",
            "full_name": "Dup",
            "company_id": test_company.id,
        }
        client.post(f"{BASE}/signup", json=payload)
        payload["email"] = "dup2@example.com"
        resp = client.post(f"{BASE}/signup", json=payload)
        assert resp.status_code == 400
        assert "already registered" in resp.json()["detail"]

    def test_signup_duplicate_email(self, client, test_company):
        payload = {
            "username": "email_dup_user1",
            "email": "same_email@example.com",
            "password": "Pass1!",
            "full_name": "User",
            "company_id": test_company.id,
        }
        client.post(f"{BASE}/signup", json=payload)
        payload["username"] = "email_dup_user2"
        resp = client.post(f"{BASE}/signup", json=payload)
        assert resp.status_code == 400
        assert "already registered" in resp.json()["detail"]


# ── Register (admin only) ─────────────────────────────────────────────────────

class TestRegister:
    def test_register_by_admin(self, client, admin_headers, test_company):
        resp = client.post(
            f"{BASE}/register",
            headers=admin_headers,
            json={
                "username": "reg_by_admin",
                "email": "reg_by_admin@acmesecurity.example",
                "password": "RegPass1!",
                "full_name": "Registered User",
                "role": "viewer",
                "company_id": test_company.id,
            },
        )
        assert resp.status_code == 201
        assert resp.json()["username"] == "reg_by_admin"

    def test_register_by_viewer_forbidden(self, client, viewer_headers, test_company):
        resp = client.post(
            f"{BASE}/register",
            headers=viewer_headers,
            json={
                "username": "should_not_be_created",
                "email": "denied@example.com",
                "password": "Pass1!",
                "full_name": "Denied",
                "role": "viewer",
                "company_id": test_company.id,
            },
        )
        assert resp.status_code == 403

    def test_register_requires_auth(self, client, test_company):
        resp = client.post(
            f"{BASE}/register",
            json={
                "username": "no_auth_reg",
                "email": "noauth@example.com",
                "password": "Pass1!",
                "full_name": "NoAuth",
                "role": "viewer",
                "company_id": test_company.id,
            },
        )
        assert resp.status_code == 401
