import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Bypass ngrok browser interstitial for programmatic requests
HEADERS = {"ngrok-skip-browser-warning": "true"}
