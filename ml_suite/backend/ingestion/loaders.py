"""Data ingestion. Every source funnels into one pandas DataFrame so the rest
of the pipeline never learns where the data came from."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from backend.core.config import settings
from backend.core.exceptions import IngestionError
from backend.core.logging_config import get_logger

log = get_logger(__name__)

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"}


def load_file(path: str | Path, filename: str | None = None) -> pd.DataFrame:
    """Read a local file, dispatching on extension."""
    path = Path(path)
    suffix = Path(filename or path.name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestionError(
            f"Unsupported file type '{suffix}'",
            detail=f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        if suffix == ".json":
            return pd.read_json(path)
        return pd.read_parquet(path)
    except Exception as exc:
        raise IngestionError(f"Could not parse {suffix} file", detail=str(exc)) from exc


def load_bytes(raw: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded file already held in memory."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestionError(
            f"Unsupported file type '{suffix}'",
            detail=f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )
    size_mb = len(raw) / 1_048_576
    if size_mb > settings.MAX_UPLOAD_MB:
        raise IngestionError(
            f"File is {size_mb:.0f} MB, limit is {settings.MAX_UPLOAD_MB} MB"
        )
    buf = io.BytesIO(raw)
    try:
        if suffix == ".csv":
            return pd.read_csv(buf)
        if suffix == ".tsv":
            return pd.read_csv(buf, sep="\t")
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(buf)
        if suffix == ".json":
            return pd.read_json(buf)
        return pd.read_parquet(buf)
    except Exception as exc:
        raise IngestionError(f"Could not parse {filename}", detail=str(exc)) from exc


def load_sql(connection_uri: str, query: str, limit: int | None = None) -> pd.DataFrame:
    """Run a read-only query against any SQLAlchemy-supported database
    (Postgres/Neon, MySQL, SQLite, ...)."""
    _reject_write_statements(query)
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:  # pragma: no cover
        raise IngestionError("SQLAlchemy is not installed") from exc

    sql = query.strip().rstrip(";")
    if limit:
        sql = f"SELECT * FROM ({sql}) AS _mlsuite_sub LIMIT {int(limit)}"
    try:
        engine = create_engine(connection_uri, pool_pre_ping=True)
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        engine.dispose()
    except Exception as exc:
        raise IngestionError("SQL query failed", detail=str(exc)) from exc
    log.info("Loaded %d rows from SQL source", len(df))
    return df


def load_mongo(uri: str, database: str, collection: str,
               query: dict | None = None, limit: int = 50_000) -> pd.DataFrame:
    """Pull documents from MongoDB and flatten nested fields into columns."""
    try:
        from pymongo import MongoClient
    except ImportError as exc:  # pragma: no cover
        raise IngestionError("pymongo is not installed") from exc
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        cursor = client[database][collection].find(query or {}).limit(int(limit))
        docs = list(cursor)
        client.close()
    except Exception as exc:
        raise IngestionError("MongoDB query failed", detail=str(exc)) from exc
    if not docs:
        raise IngestionError("Query returned no documents")
    df = pd.json_normalize(docs)
    if "_id" in df.columns:
        df["_id"] = df["_id"].astype(str)
    log.info("Loaded %d documents from MongoDB", len(df))
    return df


_WRITE_KEYWORDS = (
    "insert ", "update ", "delete ", "drop ", "truncate ",
    "alter ", "create ", "grant ", "revoke ",
)


def _reject_write_statements(query: str) -> None:
    """Ingestion is read-only; refuse anything that could mutate the source."""
    lowered = " " + " ".join(query.lower().split()) + " "
    for keyword in _WRITE_KEYWORDS:
        if f" {keyword}" in lowered:
            raise IngestionError(
                f"Only read queries are allowed here (found '{keyword.strip()}')"
            )
