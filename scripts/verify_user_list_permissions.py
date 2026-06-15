import time

import httpx

BASE_URL = "http://localhost:8000/api/v1"

def verify_user_list_permissions():
    print("=== Verifying User List Permissions ===")
    ts = int(time.time())

    # 1. Login as Aegis Admin
    print("\n1. Login as Aegis Admin...")
    resp = httpx.post(f"{BASE_URL}/auth/login", data={"username": "admin", "password": "admin123"})
    if resp.status_code != 200:
        print(f"Failed to login as Aegis Admin: {resp.text}")
        return
    aegis_token = resp.json()["access_token"]
    aegis_headers = {"Authorization": f"Bearer {aegis_token}"}

    # 2. Create Users who will be Company Owners
    print("\n2. Creating Future Company Owners...")
    # Owner A
    owner_a_data = {"username": f"owner_a_{ts}", "password": "password123", "full_name": "Owner A", "email": f"owner_a_{ts}@test.com", "role": "admin"}
    # Note: We create them as 'admin' role but without company initially (or verify them immediately)
    # Using the create_user endpoint as admin allows creating them verified.
    resp = httpx.post(f"{BASE_URL}/users", json=owner_a_data, headers=aegis_headers)
    if resp.status_code != 201:
        print(f"Failed to create Owner A: {resp.text}")
        return
    owner_a_created = resp.json()
    print(f"Owner A created: ID={owner_a_created.get('id')}, CompanyID={owner_a_created.get('company_id')}")

    # Owner B
    owner_b_data = {"username": f"owner_b_{ts}", "password": "password123", "full_name": "Owner B", "email": f"owner_b_{ts}@test.com", "role": "admin"}
    resp = httpx.post(f"{BASE_URL}/users", json=owner_b_data, headers=aegis_headers)
    if resp.status_code != 201:
        print(f"Failed to create Owner B: {resp.text}")
        return
    owner_b_created = resp.json()
    print(f"Owner B created: ID={owner_b_created.get('id')}, CompanyID={owner_b_created.get('company_id')}")

    # 3. Owners Create Companies
    print("\n3. Owners Creating Companies...")

    # Login as Owner A
    resp = httpx.post(f"{BASE_URL}/auth/login", data={"username": owner_a_data["username"], "password": "password123"})
    owner_a_token = resp.json()["access_token"]
    owner_a_headers = {"Authorization": f"Bearer {owner_a_token}"}

    resp = httpx.post(f"{BASE_URL}/companies", json={"name": f"CompA_{ts}", "email": f"a_{ts}@t.com", "phone_number": "111"}, headers=owner_a_headers)
    if resp.status_code != 201:
        print(f"Failed to create Company A: {resp.text}")
        return
    comp_a_id = resp.json()["id"]

    # Login as Owner B
    resp = httpx.post(f"{BASE_URL}/auth/login", data={"username": owner_b_data["username"], "password": "password123"})
    owner_b_token = resp.json()["access_token"]
    owner_b_headers = {"Authorization": f"Bearer {owner_b_token}"}

    resp = httpx.post(f"{BASE_URL}/companies", json={"name": f"CompB_{ts}", "email": f"b_{ts}@t.com", "phone_number": "222"}, headers=owner_b_headers)
    if resp.status_code != 201:
        print(f"Failed to create Company B: {resp.text}")
        return
    comp_b_id = resp.json()["id"]

    # Verify companies (Aegis Admin)
    httpx.post(f"{BASE_URL}/companies/{comp_a_id}/verify", headers=aegis_headers)
    httpx.post(f"{BASE_URL}/companies/{comp_b_id}/verify", headers=aegis_headers)

    # 4. Owners Add Employees
    print("\n4. Owners Adding Employees...")
    # Owner A adds Employee A1
    emp_a1_data = {"username": f"emp_a1_{ts}", "password": "password123", "full_name": "Emp A1", "role": "security_officer", "company_id": comp_a_id, "email": f"emp_a1_{ts}@test.com"}
    httpx.post(f"{BASE_URL}/users", json=emp_a1_data, headers=owner_a_headers)

    # Owner B adds Employee B1
    emp_b1_data = {"username": f"emp_b1_{ts}", "password": "password123", "full_name": "Emp B1", "role": "security_officer", "company_id": comp_b_id, "email": f"emp_b1_{ts}@test.com"}
    httpx.post(f"{BASE_URL}/users", json=emp_b1_data, headers=owner_b_headers)

    # 5. TEST: Aegis Admin List Users
    print("\n5. TEST: Aegis Admin getting ALL users...")
    resp = httpx.get(f"{BASE_URL}/users", headers=aegis_headers)
    all_users = resp.json()
    usernames = [u['username'] for u in all_users]

    print(f"Aegis Admin sees {len(all_users)} users.")
    if owner_a_data['username'] in usernames and emp_b1_data['username'] in usernames:
        print("✅ SUCCESS: Aegis Admin sees users from BOTH companies.")
    else:
        print("❌ FAIL: Aegis Admin missing users.")

    # 6. TEST: Company Admin List Users
    print("\n6. TEST: Company Admin A getting users...")
    resp = httpx.get(f"{BASE_URL}/users", headers=owner_a_headers)
    comp_users = resp.json()
    comp_usernames = [u['username'] for u in comp_users]

    print(f"Company Admin A sees {len(comp_users)} users.")

    sees_self = owner_a_data['username'] in comp_usernames
    sees_emp = emp_a1_data['username'] in comp_usernames
    sees_other = owner_b_data['username'] in comp_usernames

    if sees_self and sees_emp and not sees_other:
        print("✅ SUCCESS: Company Admin A sees only their own company users.")
    else:
        print("❌ FAIL: Isolation check failed.")
        print(f"Sees self? {sees_self}")
        print(f"Sees own employee? {sees_emp}")
        print(f"Sees other company user? {sees_other} (Should be False)")

if __name__ == "__main__":
    verify_user_list_permissions()
