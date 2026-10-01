"""
Entry point to run the AI Document Assistant FastAPI backend server.
Usage:
    python run_api.py
"""

import sys
import uvicorn
from app.config import API_HOST, API_PORT

if __name__ == "__main__":
    # Ensure proper encoding on Windows console
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print(f"Starting AI Document Assistant API on http://{API_HOST}:{API_PORT}")
    print(f"Interactive Swagger documentation available at: http://{API_HOST}:{API_PORT}/docs")
    uvicorn.run("app.api:app", host=API_HOST, port=API_PORT, reload=False)
