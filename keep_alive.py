#!/usr/bin/env python3
"""
Keep NIM4CC Space alive by pinging it periodically.
Run this script with a cron job to prevent the Space from sleeping.
"""

import requests
import time
from datetime import datetime

SPACE_URL = "https://gregcho-nim4cc.hf.space"
PING_ENDPOINTS = [
    "/v1",
    "/api/dashboard",
    "/v1/models"
]

def ping_space():
    """Ping multiple endpoints to keep the Space active."""
    results = []
    for endpoint in PING_ENDPOINTS:
        url = f"{SPACE_URL}{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            status = "OK" if response.status_code == 200 else "FAIL"
            results.append(f"  [{status}] {endpoint} ({response.status_code})")
        except Exception as e:
            results.append(f"  [FAIL] {endpoint} (Error: {e})")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Ping results:")
    for result in results:
        print(result)

    return all("[OK]" in r for r in results)

if __name__ == "__main__":
    print("Pinging NIM4CC Space to keep it alive...")
    success = ping_space()
    print(f"\n{'All endpoints responding' if success else 'Some endpoints failed'}")
