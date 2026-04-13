"""
Enterprise Vector Store — main entry point.

Starts the FastAPI server.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Or simply:
    python main.py
"""

import uvicorn
import config
from api.assistant import app  # noqa: F401 — re-exported for uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info",
    )
