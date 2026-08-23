"""Thin HTTP client for the backend. All error translation happens here so the
pages can call these functions and show a message without try/except noise."""
from __future__ import annotations

import os
from typing import Any

import requests

API_BASE = os.environ.get("MLSUITE_API", "http://127.0.0.1:8000")
TIMEOUT = 120


class ApiError(Exception):
    """A structured failure returned by the backend."""

    def __init__(self, message: str, detail: Any = None, status: int | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.status = status


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


def health() -> dict:
    return _request("GET", "/health")


def system_info() -> dict:
    return _request("GET", "/system")


def upload_dataset(file_bytes: bytes, filename: str, name: str | None = None) -> dict:
    return _request("POST", "/api/datasets/upload",
                    files={"file": (filename, file_bytes)},
                    data={"name": name} if name else None)


def dataset_from_sql(connection_uri: str, query: str, name: str | None,
                     limit: int | None) -> dict:
    return _request("POST", "/api/datasets/from-sql", json={
        "connection_uri": connection_uri, "query": query,
        "name": name, "limit": limit})


def dataset_from_mongo(uri: str, database: str, collection: str,
                       query: dict, limit: int, name: str | None) -> dict:
    return _request("POST", "/api/datasets/from-mongo", json={
        "uri": uri, "database": database, "collection": collection,
        "query": query, "limit": limit, "name": name})


def list_datasets() -> list[dict]:
    return _request("GET", "/api/datasets")["datasets"]


def get_profile(dataset_id: str, preview_rows: int = 20) -> dict:
    return _request("GET", f"/api/datasets/{dataset_id}/profile",
                    params={"preview_rows": preview_rows})


def delete_dataset(dataset_id: str) -> dict:
    return _request("DELETE", f"/api/datasets/{dataset_id}")


def task_preview(dataset_id: str, target_column: str) -> dict:
    return _request("GET", f"/api/task-preview/{dataset_id}",
                    params={"target_column": target_column})


def start_training(payload: dict) -> dict:
    return _request("POST", "/api/train", json=payload)


def get_run(run_id: str) -> dict:
    return _request("GET", f"/api/runs/{run_id}")


def list_runs(limit: int = 50) -> list[dict]:
    return _request("GET", "/api/runs", params={"limit": limit})["runs"]


def list_models() -> list[dict]:
    return _request("GET", "/api/models")["models"]


def get_model(model_id: str) -> dict:
    return _request("GET", f"/api/models/{model_id}")


def delete_model(model_id: str) -> dict:
    return _request("DELETE", f"/api/models/{model_id}")


def predict_rows(model_id: str, rows: list[dict]) -> dict:
    return _request("POST", f"/api/predict/{model_id}",
                    json={"rows": rows, "include_probabilities": True})


def predict_file(model_id: str, file_bytes: bytes, filename: str) -> dict:
    return _request("POST", f"/api/predict/{model_id}/file",
                    files={"file": (filename, file_bytes)})


def predict_file_csv(model_id: str, file_bytes: bytes, filename: str) -> bytes:
    return _request("POST", f"/api/predict/{model_id}/file/download",
                    files={"file": (filename, file_bytes)})
