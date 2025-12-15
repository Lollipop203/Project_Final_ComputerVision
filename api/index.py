import sys
import os
from pathlib import Path

# Add project root to python path needed for Vercel
# This allows importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import the FastAPI app
try:
    from src.api.app import app
except ImportError as e:
    # Fallback/Debug if import fails
    from fastapi import FastAPI
    app = FastAPI()
    @app.get("/")
    def error():
        return {"error": str(e), "path": sys.path}
