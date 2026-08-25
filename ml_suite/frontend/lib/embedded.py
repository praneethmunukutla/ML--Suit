"""In-process backend, for when no FastAPI server is available.

Streamlit Community Cloud runs a single process, so there is nowhere for the
API to live. This module exposes the same surface as the HTTP client by calling
the backend directly. The route handlers are reused wherever they are plain
sync functions, so behaviour cannot drift between the two modes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# The backend package sits two levels up from frontend/lib.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from backend.core.exceptions import MLSuiteError  # noqa: E402
from backend.core.logging_config import configure_logging  # noqa: E402

configure_logging()


def _translate(exc: MLSuiteError):
    """Backend errors must reach the pages as the same ApiError they expect."""
    from lib.api_client import ApiError
    return ApiError(exc.message, exc.detail, exc.status_code)


def _call(fn, *args, **kwargs) -> Any:
    try:
        return fn(*args, **kwargs)
    except MLSuiteError as exc:
        raise _translate(exc) from exc


# ------------------------------------------------------------------ monitoring

def health() -> dict:
    from backend.api.routes import health as route
    return _call(route.health)


def system_info() -> dict:
    from backend.api.routes import health as route
    return _call(route.system_info)


# -------------------------------------------------------------------- datasets

def upload_dataset(file_bytes: bytes, filename: str, name: str | None = None) -> dict:
    """The HTTP route is async and takes an UploadFile, so this path calls the
    ingestion services directly instead."""
    from backend.core import metrics as app_metrics
    from backend.core.exceptions import IngestionError
    from backend.ingestion import loaders, sanitize
    from backend.registry import store

    def _run():
        if not file_bytes:
            raise IngestionError("The uploaded file is empty")
        df = sanitize.prepare(loaders.load_bytes(file_bytes, filename))
        meta = store.save_dataset(
            df, name or filename, source="file",
            source_detail={"filename": filename,
                           "size_kb": round(len(file_bytes) / 1024, 1)})
        app_metrics.inc("mlsuite_datasets_ingested_total", {"source": "file"})
        return meta

    return _call(_run)


def dataset_from_sql(connection_uri: str, query: str, name: str | None,
                     limit: int | None) -> dict:
    from backend.api.routes import datasets as route
    from backend.schemas.models import SQLSourceRequest
    return _call(route.dataset_from_sql, SQLSourceRequest(
        connection_uri=connection_uri, query=query, name=name, limit=limit))


def dataset_from_mongo(uri: str, database: str, collection: str,
                       query: dict, limit: int, name: str | None) -> dict:
    from backend.api.routes import datasets as route
    from backend.schemas.models import MongoSourceRequest
    return _call(route.dataset_from_mongo, MongoSourceRequest(
        uri=uri, database=database, collection=collection,
        query=query, limit=limit, name=name))


def list_datasets() -> list[dict]:
    from backend.api.routes import datasets as route
    return _call(route.list_datasets)["datasets"]


def get_profile(dataset_id: str, preview_rows: int = 20) -> dict:
    from backend.api.routes import datasets as route
    return _call(route.get_profile, dataset_id, preview_rows)


def delete_dataset(dataset_id: str) -> dict:
    from backend.api.routes import datasets as route
    return _call(route.delete_dataset, dataset_id)


# -------------------------------------------------------------------- training

def task_preview(dataset_id: str, target_column: str) -> dict:
    from backend.api.routes import training as route
    return _call(route.preview_task, dataset_id, target_column)


def start_training(payload: dict) -> dict:
    from backend.api.routes import training as route
    from backend.schemas.models import TrainRequest
    return _call(route.start_training, TrainRequest(**payload))


def get_run(run_id: str) -> dict:
    from backend.api.routes import training as route
    return _call(route.get_run, run_id)


def list_runs(limit: int = 50) -> list[dict]:
    from backend.api.routes import training as route
    return _call(route.list_runs, limit)["runs"]


# ---------------------------------------------------------------------- models

def list_models() -> list[dict]:
    from backend.api.routes import models as route
    return _call(route.list_models)["models"]


def get_model(model_id: str) -> dict:
    from backend.api.routes import models as route
    return _call(route.get_model, model_id)


def delete_model(model_id: str) -> dict:
    from backend.api.routes import models as route
    return _call(route.delete_model, model_id)


# ------------------------------------------------------------------ prediction

def predict_rows(model_id: str, rows: list[dict]) -> dict:
    from backend.api.routes import predict as route
    from backend.schemas.models import PredictRequest
    return _call(route.predict_rows, model_id,
                 PredictRequest(rows=rows, include_probabilities=True))


def predict_file(model_id: str, file_bytes: bytes, filename: str) -> dict:
    from backend.ingestion import loaders, sanitize
    from backend.training import predict as service

    def _run():
        df = sanitize.clean_columns(loaders.load_bytes(file_bytes, filename))
        result = service.predict(model_id, df)
        result["input_rows"] = int(len(df))
        return result

    return _call(_run)


def predict_file_csv(model_id: str, file_bytes: bytes, filename: str) -> bytes:
    """Returns CSV bytes, matching what the download endpoint streams."""
    from backend.ingestion import loaders, sanitize
    from backend.training import predict as service

    def _run() -> bytes:
        df = sanitize.clean_columns(loaders.load_bytes(file_bytes, filename))
        return service.predict_frame(model_id, df).to_csv(index=False).encode()

    return _call(_run)
