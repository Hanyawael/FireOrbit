"""Vercel serverless entrypoint.

Exposes the existing FastAPI app (backend/main.py) unchanged.
vercel.json rewrites /api/* and /health to this function.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app  # noqa: E402,F401
