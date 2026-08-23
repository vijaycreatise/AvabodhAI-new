"""
api/routes/health.py
--------------------
Health check endpoints for monitoring.
"""

from fastapi import APIRouter
from sqlalchemy import text

from db.database import engine
from config.settings import get_settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()


@router.get("/", summary="Basic health check")
async def health():
    return {"status": "ok", "service": "Avabodh API"}


@router.get("/db", summary="Database health check")
async def health_db():
    """Check if PostgreSQL is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        logger.error("DB health check failed: %s", e)
        return {"status": "error", "database": str(e)}


@router.get("/qdrant", summary="Qdrant health check")
async def health_qdrant():
    """Check if Qdrant (the chunk/vector store) is reachable."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
        client.get_collections()
        return {"status": "ok", "qdrant": "connected"}
    except Exception as e:
        logger.error("Qdrant health check failed: %s", e)
        return {"status": "error", "qdrant": str(e)}