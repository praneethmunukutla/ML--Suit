"""Dataset ingestion and inspection."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from backend.core import metrics as app_metrics
from backend.core.exceptions import IngestionError, ValidationError
from backend.core.logging_config import get_logger
from backend.ingestion import loaders, sanitize
from backend.preprocessing.profiler import profile_dataset
from backend.registry import store
from backend.schemas.models import MongoSourceRequest, SQLSourceRequest

router = APIRouter(prefix="/api/datasets", tags=["datasets"])
log = get_logger(__name__)


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...),
                         name: str | None = Form(default=None)) -> dict:
    """Ingest a CSV, Excel, JSON, or Parquet upload."""
    raw = await file.read()
    if not raw:
        raise IngestionError("The uploaded file is empty")
    df = sanitize.prepare(loaders.load_bytes(raw, file.filename or "upload.csv"))
    meta = store.save_dataset(df, name or (file.filename or "upload"), source="file",
                              source_detail={"filename": file.filename,
                                             "size_kb": round(len(raw) / 1024, 1)})
    app_metrics.inc("mlsuite_datasets_ingested_total", {"source": "file"})
    return meta


@router.post("/from-sql")
def dataset_from_sql(request: SQLSourceRequest) -> dict:
    """Ingest the result of a read-only SQL query (Neon/Postgres, MySQL, SQLite)."""
    df = sanitize.prepare(
        loaders.load_sql(request.connection_uri, request.query, request.limit))
    meta = store.save_dataset(
        df, request.name or "sql_query", source="sql",
        # The connection URI holds credentials and is deliberately not stored.
        source_detail={"query": request.query[:500]})
    app_metrics.inc("mlsuite_datasets_ingested_total", {"source": "sql"})
    return meta


@router.post("/from-mongo")
def dataset_from_mongo(request: MongoSourceRequest) -> dict:
    """Ingest documents from a MongoDB collection."""
    df = sanitize.prepare(loaders.load_mongo(
        request.uri, request.database, request.collection,
        request.query, request.limit))
    meta = store.save_dataset(
        df, request.name or f"{request.database}.{request.collection}",
        source="mongodb",
        source_detail={"database": request.database,
                       "collection": request.collection,
                       "query": request.query})
    app_metrics.inc("mlsuite_datasets_ingested_total", {"source": "mongodb"})
    return meta


@router.get("")
def list_datasets() -> dict:
    return {"datasets": store.list_datasets()}


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    return store.get_dataset_meta(dataset_id)


@router.get("/{dataset_id}/profile")
def get_profile(dataset_id: str, preview_rows: int = 20) -> dict:
    """Column types, missing-value counts, and a preview for the configure page."""
    if not 1 <= preview_rows <= 200:
        raise ValidationError("preview_rows must be between 1 and 200")
    df = store.load_dataset(dataset_id)
    profile = profile_dataset(df, preview_rows)
    profile["dataset_id"] = dataset_id
    profile["name"] = store.get_dataset_meta(dataset_id)["name"]
    return profile


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict:
    store.delete_dataset(dataset_id)
    return {"deleted": dataset_id}
