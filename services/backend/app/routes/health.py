"""Health-Check Endpunkte."""

from fastapi import APIRouter, Request
from datetime import datetime, timezone

router = APIRouter()


@router.get("/health")
async def health():
    """Einfacher Health-Check für Docker."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health/detail")
async def health_detail(request: Request):
    """Detaillierter Health-Check aller Services."""
    checks = {
        "backend": "ok",
        "embedding": "unknown",
        "vectorstore": "unknown",
        "stt": "unknown",
    }

    if hasattr(request.app.state, "embedding"):
        checks["embedding"] = (
            "ok" if request.app.state.embedding.is_ready() else "error"
        )
    if hasattr(request.app.state, "vectorstore"):
        checks["vectorstore"] = (
            "ok" if await request.app.state.vectorstore.is_ready() else "error"
        )
    if hasattr(request.app.state, "stt"):
        checks["stt"] = "ok" if request.app.state.stt.is_ready() else "error"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
