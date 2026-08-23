"""Liveness, readiness, and monitoring endpoints."""
from __future__ import annotations

import platform
import sys

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from backend.core import metrics as app_metrics
from backend.core.config import settings
from backend.registry import store
from backend.training import jobs

router = APIRouter(tags=["monitoring"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.VERSION}


@router.get("/ready")
def ready() -> dict:
    """Readiness also proves the storage layer is reachable and writable."""
    checks = {}
    try:
        stats = store.storage_stats()
        checks["storage"] = "ok"
        checks["storage_stats"] = stats
    except Exception as exc:
        checks["storage"] = f"error: {exc}"
    checks["active_runs"] = jobs.active_runs()
    healthy = checks["storage"] == "ok"
    return {"status": "ready" if healthy else "degraded", "checks": checks}


@router.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    return app_metrics.render()


@router.get("/system")
def system_info() -> dict:
    from backend.training.model_zoo import HAS_XGBOOST, available_models
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "xgboost_available": HAS_XGBOOST,
        "storage_backend": settings.STORAGE_BACKEND,
        "limits": {
            "max_upload_mb": settings.MAX_UPLOAD_MB,
            "max_train_rows": settings.MAX_TRAIN_ROWS,
            "max_features": settings.MAX_FEATURES,
            "train_workers": settings.TRAIN_WORKERS,
        },
        "models": {
            "classification": available_models("classification"),
            "regression": available_models("regression"),
        },
        "metrics": app_metrics.snapshot(),
        "storage": store.storage_stats(),
    }
