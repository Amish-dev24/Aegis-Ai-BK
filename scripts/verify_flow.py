import logging
import sys

import httpx

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# User A (Company Admin)
USER_A_USERNAME = "user_a"
USER_A_EMAIL = "usera@example.com"
USER_A_PASSWORD = "password123"
USER_A_PHONE = "1234567890"

# Company
COMPANY_NAME = "Aegis Corp"
COMPANY_DOMAIN = "aegis.com"

# User B (Security Officer)
USER_B_USERNAME = "officer_b"
USER_B_EMAIL = "officer_b@aegis.com"
USER_B_PASSWORD = "password123"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def get_token(username, password):
    try:
        response = httpx.post(
            f"{BASE_URL}/auth/login",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            return response.json()["access_token"]
        logger.error(f"Login failed for {username}: {response.text}")
        return None
    except Exception as e:
        logger.error(f"Connection error: {e}")
        return None

def run_verification():
    logger.info("--- Starting End-to-End Flow Verification ---")

    # 1. Aegis Admin Login
    logger.info("1. Logging in as Aegis Admin...")
    admin_token = get_token(ADMIN_USERNAME, ADMIN_PASSWORD)
    if not admin_token:
        logger.error("Failed to login as admin. Exiting.")
        return
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    logger.info("   Admin logged in successfully.")

    # 2. User A Registration
    logger.info("2. Registering User A (Company Owner)...")
    user_a_data = {
        "username": USER_A_USERNAME,
        "email": USER_A_EMAIL,
        "password": USER_A_PASSWORD,
        "full_name": "User A",
        "phone_number": USER_A_PHONE
    }
    response = httpx.post(f"{BASE_URL}/auth/register", json=user_a_data)
    if response.status_code == 201:
        user_a_id = response.json()["id"]
        logger.info(f"   User A registered (ID: {user_a_id}).")
    else:
        logger.error(f"   Registration failed: {response.text}")
        # If already exists, try to find ID? assuming clean DB for now.
        return

    # 3. Aegis Admin Verify User A
    logger.info("3. Aegis Admin Verifying User A...")
    response = httpx.post(
        f"{BASE_URL}/users/{user_a_id}/verify",
        headers=admin_headers
    )
    if response.status_code == 200:
        logger.info("   User A verified by Admin.")
    else:
        logger.error(f"   Verification failed: {response.text}")
        return

    # 4. User A Login
    logger.info("4. User A Logging in...")
    user_a_token = get_token(USER_A_USERNAME, USER_A_PASSWORD)
    if not user_a_token:
        logger.error("Failed to login as User A.")
        return
    user_a_headers = {"Authorization": f"Bearer {user_a_token}"}
    logger.info("   User A logged in.")

    # 5. User A Creates Company
    logger.info("5. User A Creating Company...")
    company_data = {
        "name": COMPANY_NAME,
        "domain": COMPANY_DOMAIN,
        "contact_email": USER_A_EMAIL
    }
    response = httpx.post(
        f"{BASE_URL}/companies",
        json=company_data,
        headers=user_a_headers
    )
    if response.status_code == 201:
        company_id = response.json()["id"]
        logger.info(f"   Company created (ID: {company_id}). User A should be linked.")
    else:
        logger.error(f"   Company creation failed: {response.text}")
        return

    # 6. Aegis Admin Verifies Company
    logger.info("6. Aegis Admin Verifying Company...")
    response = httpx.post(
        f"{BASE_URL}/companies/{company_id}/verify",
        headers=admin_headers
    )
    if response.status_code == 200:
        logger.info("   Company verified. User A should be promoted to ADMIN.")
    else:
        logger.error(f"   Company verification failed: {response.text}")
        return

    # 7. Refresh User A Token (to get new role/company claims if any, though role update might need re-login depending on implementation)
    # The current implementation of get_current_user fetches user from DB, so role is up to date,
    # but the token claims 'scopes' might be old if we used them. We verify DB state.

    # 8. User A (now Admin) Creates Security Officer (User B)
    logger.info("8. User A (Admin) Creating Security Officer (User B)...")
    user_b_data = {
        "username": USER_B_USERNAME,
        "email": USER_B_EMAIL,
        "password": USER_B_PASSWORD,
        "full_name": "Officer B",
        "role": "security_officer",
        "company_id": company_id # Company admin can explicitly set this to their own company
    }
    response = httpx.post(
        f"{BASE_URL}/users",
        json=user_b_data,
        headers=user_a_headers
    )
    if response.status_code == 201:
        user_b_id = response.json()["id"]
        logger.info(f"   Security Officer created (ID: {user_b_id}).")
    else:
        logger.error(f"   Security Officer creation failed: {response.text}")
        return

    logger.info("--- Flow Verification Completed Successfully ---")

if __name__ == "__main__":
    # Ensure server is running
    try:
        httpx.get(f"{BASE_URL}/docs")
    except:
        print("Error: Server not running at http://localhost:8000")
        sys.exit(1)

    run_verification()
