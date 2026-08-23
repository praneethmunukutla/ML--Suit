"""Normalisation applied once at ingestion so downstream stages can assume
clean column names, real dtypes, and no all-empty columns."""
from __future__ import annotations

import re

import pandas as pd

from backend.core.exceptions import IngestionError
from backend.core.logging_config import get_logger

log = get_logger(__name__)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names unique, stripped, and safe to round-trip through JSON."""
    seen: dict[str, int] = {}
    names = []
    for i, col in enumerate(df.columns):
        name = re.sub(r"\s+", " ", str(col)).strip() or f"column_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    df.columns = names
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Recover numeric and datetime columns that arrived as strings."""
    for col in df.columns:
        if df[col].dtype != object:
            continue
        sample = df[col].dropna()
        if sample.empty:
            continue
        converted = pd.to_numeric(sample, errors="coerce")
        if converted.notna().mean() > 0.95:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            continue
        if _looks_like_dates(sample):
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.90:
                df[col] = parsed
    return df


def _looks_like_dates(sample: pd.Series) -> bool:
    """Cheap pre-check so we only attempt date parsing on plausible columns."""
    head = sample.astype(str).head(50)
    pattern = re.compile(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}")
    return head.str.contains(pattern).mean() > 0.8


def drop_empty(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns and rows that carry no information at all."""
    before = df.shape
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant:
        log.info("Dropping %d constant column(s): %s", len(constant), constant[:10])
        df = df.drop(columns=constant)
    if df.shape != before:
        log.info("Sanitised shape %s -> %s", before, df.shape)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Full ingestion clean-up, applied to every source."""
    if df is None or df.empty:
        raise IngestionError("The source returned an empty table")
    df = clean_columns(df.copy())
    df = df.loc[:, ~df.columns.duplicated()]
    df = coerce_types(df)
    df = drop_empty(df)
    if df.empty or df.shape[1] == 0:
        raise IngestionError("No usable columns remained after cleaning")
    return df.reset_index(drop=True)
