"""Persistence for datasets, training runs, and fitted models.

Backed by the local filesystem. Every access goes through the functions here,
so pointing at MongoDB later means reimplementing this module and nothing else.
"""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backend.core.config import settings
from backend.core.exceptions import NotFoundError
from backend.core.logging_config import get_logger

log = get_logger(__name__)

_lock = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _write_json(path: Path, payload: dict) -> None:
    """Write via a temp file so a crash mid-write cannot corrupt the record."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- datasets

def save_dataset(df: pd.DataFrame, name: str, source: str,
                 source_detail: dict | None = None) -> dict:
    dataset_id = new_id("ds")
    data_path = settings.DATASET_DIR / f"{dataset_id}.parquet"
    try:
        df.to_parquet(data_path, index=False)
        fmt = "parquet"
    except Exception:  # object columns pyarrow cannot type; pickle keeps them
        data_path = settings.DATASET_DIR / f"{dataset_id}.pkl"
        df.to_pickle(data_path)
        fmt = "pickle"

    meta = {
        "dataset_id": dataset_id,
        "name": name,
        "source": source,
        "source_detail": source_detail or {},
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "format": fmt,
        "path": str(data_path),
        "created_at": _now(),
    }
    with _lock:
        _write_json(settings.DATASET_DIR / f"{dataset_id}.meta.json", meta)
    log.info("Saved dataset %s (%d x %d) from %s",
             dataset_id, df.shape[0], df.shape[1], source)
    return meta


def load_dataset(dataset_id: str) -> pd.DataFrame:
    meta = get_dataset_meta(dataset_id)
    path = Path(meta["path"])
    if not path.exists():
        raise NotFoundError(f"Dataset file for '{dataset_id}' is missing")
    return pd.read_parquet(path) if meta["format"] == "parquet" else pd.read_pickle(path)


def get_dataset_meta(dataset_id: str) -> dict:
    path = settings.DATASET_DIR / f"{dataset_id}.meta.json"
    if not path.exists():
        raise NotFoundError(f"Unknown dataset '{dataset_id}'")
    return _read_json(path)


def list_datasets() -> list[dict]:
    items = [_read_json(p) for p in settings.DATASET_DIR.glob("*.meta.json")]
    return sorted(items, key=lambda m: m["created_at"], reverse=True)


def delete_dataset(dataset_id: str) -> None:
    meta = get_dataset_meta(dataset_id)
    Path(meta["path"]).unlink(missing_ok=True)
    (settings.DATASET_DIR / f"{dataset_id}.meta.json").unlink(missing_ok=True)


# ------------------------------------------------------------------------------- runs

def create_run(payload: dict) -> dict:
    run_id = new_id("run")
    record = {
        "run_id": run_id,
        "status": "queued",
        "progress": 0.0,
        "stage": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "leaderboard": [],
        "error": None,
        **payload,
    }
    with _lock:
        _write_json(settings.RUN_DIR / f"{run_id}.json", record)
    return record


def update_run(run_id: str, **fields: Any) -> dict:
    with _lock:
        record = get_run(run_id)
        record.update(fields)
        record["updated_at"] = _now()
        _write_json(settings.RUN_DIR / f"{run_id}.json", record)
    return record


def get_run(run_id: str) -> dict:
    path = settings.RUN_DIR / f"{run_id}.json"
    if not path.exists():
        raise NotFoundError(f"Unknown run '{run_id}'")
    return _read_json(path)


def list_runs(limit: int = 50) -> list[dict]:
    items = [_read_json(p) for p in settings.RUN_DIR.glob("run_*.json")]
    items.sort(key=lambda r: r["created_at"], reverse=True)
    return items[:limit]


# ----------------------------------------------------------------------------- models

def save_model(pipeline: Any, meta: dict) -> dict:
    model_id = new_id("mdl")
    model_path = settings.MODEL_DIR / f"{model_id}.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    record = {
        "model_id": model_id,
        "path": str(model_path),
        "created_at": _now(),
        "size_kb": round(model_path.stat().st_size / 1024, 1),
        **meta,
    }
    with _lock:
        _write_json(settings.MODEL_DIR / f"{model_id}.meta.json", record)
    log.info("Registered model %s (%s, %s)",
             model_id, record.get("algorithm"), record.get("task"))
    return record


def load_model(model_id: str):
    meta = get_model_meta(model_id)
    path = Path(meta["path"])
    if not path.exists():
        raise NotFoundError(f"Model artifact for '{model_id}' is missing")
    return joblib.load(path), meta


def get_model_meta(model_id: str) -> dict:
    path = settings.MODEL_DIR / f"{model_id}.meta.json"
    if not path.exists():
        raise NotFoundError(f"Unknown model '{model_id}'")
    return _read_json(path)


def list_models() -> list[dict]:
    items = [_read_json(p) for p in settings.MODEL_DIR.glob("*.meta.json")]
    return sorted(items, key=lambda m: m["created_at"], reverse=True)


def delete_model(model_id: str) -> None:
    meta = get_model_meta(model_id)
    Path(meta["path"]).unlink(missing_ok=True)
    (settings.MODEL_DIR / f"{model_id}.meta.json").unlink(missing_ok=True)


def storage_stats() -> dict:
    total = sum(f.stat().st_size for f in settings.STORAGE_DIR.rglob("*") if f.is_file())
    return {
        "datasets": len(list(settings.DATASET_DIR.glob("*.meta.json"))),
        "models": len(list(settings.MODEL_DIR.glob("*.meta.json"))),
        "runs": len(list(settings.RUN_DIR.glob("run_*.json"))),
        "disk_mb": round(total / 1_048_576, 2),
        "free_mb": round(shutil.disk_usage(settings.STORAGE_DIR).free / 1_048_576, 1),
    }
