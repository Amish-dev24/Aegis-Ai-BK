import sys
import httpx
import argparse
import json

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"  # Default password from create_admin.py

def login():
    """Login as admin and return headers with token."""
    print(f"Logging in as {ADMIN_USERNAME}...")
    try:
        response = httpx.post(
            f"{BASE_URL}/auth/login", 
            data={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD}
        )
        if response.status_code != 200:
            print(f"Error: Login failed! {response.status_code}")
            print(response.text)
            sys.exit(1)
            
        token = response.json()["access_token"]
        print("Login successful!")
        return {"Authorization": f"Bearer {token}"}
    except Exception as e:
        print(f"Connection error: {e}")
        print("Make sure the server is running on localhost:8000")
        sys.exit(1)

def verify_user(username_or_id):
    """Verify a user by ID or username."""
    headers = login()
    
    # Resolve ID if username provided (simple check)
    user_id = username_or_id
    if not str(username_or_id).isdigit():
        print(f"Resolving username '{username_or_id}'...")
        # We need to list users to find the ID (since there's no get-by-username endpoint exposed to admin yet easily)
        # Actually, let's just assume they provide ID for now or implement a search.
        # Admin list users has filtering? No.
        # Let's list all and find.
        response = httpx.get(f"{BASE_URL}/users", headers=headers)
        if response.status_code == 200:
            users = response.json()
            found = False
            for u in users:
                if u["username"] == username_or_id:
                    user_id = u["id"]
                    found = True
                    break
            if not found:
                print(f"User '{username_or_id}' not found.")
                return
    
    print(f"Verifying user ID {user_id}...")
    response = httpx.post(f"{BASE_URL}/users/{user_id}/verify", headers=headers)
    
    if response.status_code == 200:
        print(f"Success! User {user_id} verified.")
        print(json.dumps(response.json(), indent=2))
    elif response.status_code == 400 and "already verified" in response.text:
        print(f"User {user_id} is already verified.")
    else:
        print(f"Failed to verify user: {response.status_code}")
        print(response.text)

def create_security_officer(username, password, email, company_id=None):
    """Create a security officer."""
    headers = login()
    
    data = {
        "username": username,
        "password": password,
        "email": email,
        "role": "security_officer",
        "company_id": company_id
    }
    
    print(f"Creating Security Officer '{username}'...")
    response = httpx.post(f"{BASE_URL}/users", json=data, headers=headers)
    
    if response.status_code == 201:
        print("Success! Security Officer created.")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Failed to create user: {response.status_code}")
        print(response.text)

def verify_email(username_or_id):
    """Verify a user's email by ID or username."""
    headers = login()
    
    # Resolve ID if username provided
    user_id = username_or_id
    if not str(username_or_id).isdigit():
        print(f"Resolving username '{username_or_id}'...")
        response = httpx.get(f"{BASE_URL}/users", headers=headers)
        if response.status_code == 200:
            users = response.json()
            found = False
            for u in users:
                if u["username"] == username_or_id:
                    user_id = u["id"]
                    found = True
                    break
            if not found:
                print(f"User '{username_or_id}' not found.")
                return
    
    print(f"Verifying email for user ID {user_id}...")
    response = httpx.post(f"{BASE_URL}/auth/verify-email/{user_id}", headers=headers)
    
    if response.status_code == 200:
        print(f"Success! Email for user {user_id} verified.")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Failed to verify email: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aegis AI Admin Actions")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Verify User Command (Admin Verification)
    verify_parser = subparsers.add_parser("verify", help="Verify a user (Admin Verify)")
    verify_parser.add_argument("user", help="User ID or Username to verify")
    
    # Verify Email Command
    email_parser = subparsers.add_parser("verify_email", help="Verify a user's email")
    email_parser.add_argument("user", help="User ID or Username to verify")
    
    # Create User Command
    create_parser = subparsers.add_parser("create_so", help="Create a Security Officer")
    create_parser.add_argument("username", help="Username")
    create_parser.add_argument("password", help="Password")
    create_parser.add_argument("email", help="Email")
    create_parser.add_argument("--company-id", type=int, help="Company ID (optional if Admin belongs to company)")

    args = parser.parse_args()
    
    if args.command == "verify":
        verify_user(args.user)
    elif args.command == "verify_email":
        verify_email(args.user)
    elif args.command == "create_so":
        create_security_officer(args.username, args.password, args.email, args.company_id)
    else:
        parser.print_help()
