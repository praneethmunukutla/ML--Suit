"""Client the pages call. Two interchangeable backends sit behind it.

- **http**: talk to the FastAPI service (the local `./run.sh` setup, or a
  separately hosted API).
- **embedded**: call the backend in this same process, for single-process hosts
  like Streamlit Community Cloud where no API server can run.

Mode is resolved once per process. `MLSUITE_MODE` forces one; the default
`auto` uses the API when it answers and falls back to embedded when it does
not, so the same code deploys either way.
"""
from __future__ import annotations

import os
from typing import Any

import requests

API_BASE = os.environ.get("MLSUITE_API", "http://127.0.0.1:8000")
TIMEOUT = 120
PROBE_TIMEOUT = 2.0

_mode: str | None = None


class ApiError(Exception):
    """A structured failure, raised identically by both backends."""

    def __init__(self, message: str, detail: Any = None, status: int | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.status = status


def mode() -> str:
    """Resolve the backend once and remember it for the process lifetime."""
    global _mode
    if _mode is not None:
        return _mode
    forced = os.environ.get("MLSUITE_MODE", "auto").strip().lower()
    if forced in ("http", "embedded"):
        _mode = forced
        return _mode
    try:
        requests.get(f"{API_BASE}/health", timeout=PROBE_TIMEOUT).raise_for_status()
        _mode = "http"
    except Exception:
        # No API reachable — run the backend in-process instead of failing.
        _mode = "embedded"
    return _mode


def _embedded():
    from lib import embedded as impl
    return impl


def describe_mode() -> str:
    return ("in-process (no API server)" if mode() == "embedded"
            else f"API server at {API_BASE}")


def _request(method: str, path: str, **kwargs) -> Any:
    url = f"{API_BASE}{path}"
    try:
        response = requests.request(method, url, timeout=TIMEOUT, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            "Cannot reach the API server",
            detail=f"Is it running at {API_BASE}?  Start it with ./run.sh api",
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError(f"The request to {path} timed out after {TIMEOUT}s") from exc

    if response.status_code >= 400:
        try:
            body = response.json()
            raise ApiError(body.get("message", response.text),
                           body.get("detail"), response.status_code)
        except ValueError:
            raise ApiError(f"HTTP {response.status_code}", response.text[:500],
                           response.status_code) from None
    if response.headers.get("content-type", "").startswith("text/csv"):
        return response.content
    return response.json()


# ------------------------------------------------------------------ monitoring

def health() -> dict:
    if mode() == "embedded":
        return _embedded().health()
    return _request("GET", "/health")


def system_info() -> dict:
    if mode() == "embedded":
        return _embedded().system_info()
    return _request("GET", "/system")


# -------------------------------------------------------------------- datasets

def upload_dataset(file_bytes: bytes, filename: str, name: str | None = None) -> dict:
    if mode() == "embedded":
        return _embedded().upload_dataset(file_bytes, filename, name)
    return _request("POST", "/api/datasets/upload",
                    files={"file": (filename, file_bytes)},
                    data={"name": name} if name else None)


def dataset_from_sql(connection_uri: str, query: str, name: str | None,
                     limit: int | None) -> dict:
    if mode() == "embedded":
        return _embedded().dataset_from_sql(connection_uri, query, name, limit)
    return _request("POST", "/api/datasets/from-sql", json={
        "connection_uri": connection_uri, "query": query,
        "name": name, "limit": limit})


def dataset_from_mongo(uri: str, database: str, collection: str,
                       query: dict, limit: int, name: str | None) -> dict:
    if mode() == "embedded":
        return _embedded().dataset_from_mongo(uri, database, collection,
                                              query, limit, name)
    return _request("POST", "/api/datasets/from-mongo", json={
        "uri": uri, "database": database, "collection": collection,
        "query": query, "limit": limit, "name": name})


def list_datasets() -> list[dict]:
    if mode() == "embedded":
        return _embedded().list_datasets()
    return _request("GET", "/api/datasets")["datasets"]


def get_profile(dataset_id: str, preview_rows: int = 20) -> dict:
    if mode() == "embedded":
        return _embedded().get_profile(dataset_id, preview_rows)
    return _request("GET", f"/api/datasets/{dataset_id}/profile",
                    params={"preview_rows": preview_rows})


def delete_dataset(dataset_id: str) -> dict:
    if mode() == "embedded":
        return _embedded().delete_dataset(dataset_id)
    return _request("DELETE", f"/api/datasets/{dataset_id}")


# -------------------------------------------------------------------- training

def task_preview(dataset_id: str, target_column: str) -> dict:
    if mode() == "embedded":
        return _embedded().task_preview(dataset_id, target_column)
    return _request("GET", f"/api/task-preview/{dataset_id}",
                    params={"target_column": target_column})


def start_training(payload: dict) -> dict:
    if mode() == "embedded":
        return _embedded().start_training(payload)
    return _request("POST", "/api/train", json=payload)


def get_run(run_id: str) -> dict:
    if mode() == "embedded":
        return _embedded().get_run(run_id)
    return _request("GET", f"/api/runs/{run_id}")


def list_runs(limit: int = 50) -> list[dict]:
    if mode() == "embedded":
        return _embedded().list_runs(limit)
    return _request("GET", "/api/runs", params={"limit": limit})["runs"]


# ---------------------------------------------------------------------- models

def list_models() -> list[dict]:
    if mode() == "embedded":
        return _embedded().list_models()
    return _request("GET", "/api/models")["models"]


def get_model(model_id: str) -> dict:
    if mode() == "embedded":
        return _embedded().get_model(model_id)
    return _request("GET", f"/api/models/{model_id}")


def delete_model(model_id: str) -> dict:
    if mode() == "embedded":
        return _embedded().delete_model(model_id)
    return _request("DELETE", f"/api/models/{model_id}")


# ------------------------------------------------------------------ prediction

def predict_rows(model_id: str, rows: list[dict]) -> dict:
    if mode() == "embedded":
        return _embedded().predict_rows(model_id, rows)
    return _request("POST", f"/api/predict/{model_id}",
                    json={"rows": rows, "include_probabilities": True})


def predict_file(model_id: str, file_bytes: bytes, filename: str) -> dict:
    if mode() == "embedded":
        return _embedded().predict_file(model_id, file_bytes, filename)
    return _request("POST", f"/api/predict/{model_id}/file",
                    files={"file": (filename, file_bytes)})


def predict_file_csv(model_id: str, file_bytes: bytes, filename: str) -> bytes:
    if mode() == "embedded":
        return _embedded().predict_file_csv(model_id, file_bytes, filename)
    return _request("POST", f"/api/predict/{model_id}/file/download",
                    files={"file": (filename, file_bytes)})
