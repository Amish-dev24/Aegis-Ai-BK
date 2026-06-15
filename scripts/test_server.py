"""
Test if the server is running.
"""
import sys
import time

import requests


def test_server():
    """Test if the server is responding."""
    url = "http://localhost:8000"

    print("Testing server connection...")
    print(f"URL: {url}")
    print()

    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                print("SUCCESS: Server is running!")
                print(f"Response: {response.json()}")
                print()
                print("API Documentation:")
                print(f"  Swagger UI: {url}/docs")
                print(f"  ReDoc: {url}/redoc")
                return True
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                print(f"Waiting for server to start... ({i+1}/{max_retries})")
                time.sleep(2)
            else:
                print("ERROR: Server is not responding")
                print()
                print("Make sure the server is running:")
                print("  python run.py")
                return False
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    return False

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("Installing requests library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests

    success = test_server()
    sys.exit(0 if success else 1)


