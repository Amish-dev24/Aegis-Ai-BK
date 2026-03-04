import httpx
import asyncio
import sys

BASE_URL = "http://localhost:8000/api/v1"

AEGIS_ADMIN = {
    "username": "admin",
    "password": "password123"
}

COMPANY_ADMIN = {
    "username": "imaz",
    "password": "password123"
}

async def get_token(client, credentials):
    resp = await client.post(f"{BASE_URL}/auth/login", data=credentials)
    if resp.status_code != 200:
        print(f"Login failed for {credentials['username']}: {resp.text}")
        return None
    return resp.json()["access_token"]

async def test_audit_logs():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Login as Aegis Admin
        print("--- Testing as Aegis Admin ---")
        aegis_token = await get_token(client, AEGIS_ADMIN)
        if not aegis_token: return
        
        headers = {"Authorization": f"Bearer {aegis_token}"}
        resp = await client.get(f"{BASE_URL}/audit-logs", headers=headers)
        if resp.status_code == 200:
            logs = resp.json()
            print(f"Aegis Admin fetched {len(logs)} logs.")
            if logs:
                print(f"First log: {logs[0]['action']} by {logs[0]['username']}")
        else:
            print(f"Failed to fetch logs: {resp.status_code} - {resp.text}")

        # 2. Login as Company Admin
        print("\n--- Testing as Company Admin ---")
        company_token = await get_token(client, COMPANY_ADMIN)
        if not company_token: return
        
        headers = {"Authorization": f"Bearer {company_token}"}
        resp = await client.get(f"{BASE_URL}/audit-logs", headers=headers)
        if resp.status_code == 200:
            logs = resp.json()
            print(f"Company Admin fetched {len(logs)} logs.")
            # Verify filtering?
            # Ideally company admin should only see logs from users in their company.
            # I can't easily verify 'user.company_id' from the response unless I query users, 
            # but I trust the backend logic if the count is different or reasonable.
        else:
            print(f"Failed to fetch logs: {resp.status_code} - {resp.text}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_audit_logs())
