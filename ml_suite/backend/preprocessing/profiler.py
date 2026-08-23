"""Dataset profiling: classify every column and surface the problems a user
needs to see before choosing features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.core.config import settings

NUMERIC = "numeric"
CATEGORICAL = "categorical"
HIGH_CARD = "high_cardinality"
DATETIME = "datetime"
BOOLEAN = "boolean"
IDENTIFIER = "identifier"
EMPTY = "empty"


def classify_column(series: pd.Series, n_rows: int) -> str:
    """Decide how a column should be treated by the pipeline."""
    non_null = series.dropna()
    if non_null.empty:
        return EMPTY
    nunique = non_null.nunique()

    if pd.api.types.is_bool_dtype(series):
        return BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME
    if pd.api.types.is_numeric_dtype(series):
        # A numeric column that is almost entirely distinct is usually a key.
        if nunique == n_rows and n_rows > 20 and _is_integral(non_null):
            return IDENTIFIER
        return NUMERIC
    # Object / category from here on.
    if n_rows and nunique / max(len(non_null), 1) > settings.DROP_CARDINALITY_RATIO:
        return IDENTIFIER
    if nunique > settings.ONEHOT_MAX_CARDINALITY:
        return HIGH_CARD
    return CATEGORICAL


def _is_integral(series: pd.Series) -> bool:
    try:
        return bool(np.all(np.equal(np.mod(series.to_numpy(dtype=float), 1), 0)))
    except (TypeError, ValueError):
        return False


def profile_column(series: pd.Series, n_rows: int) -> dict:
    kind = classify_column(series, n_rows)
    missing = int(series.isna().sum())
    info = {
        "name": series.name,
        "dtype": str(series.dtype),
        "kind": kind,
        "missing": missing,
        "missing_pct": round(100 * missing / n_rows, 2) if n_rows else 0.0,
        "unique": int(series.nunique(dropna=True)),
        "sample": [_jsonable(v) for v in series.dropna().head(3).tolist()],
    }
    if kind == NUMERIC:
        described = series.describe()
        info.update({
            "mean": _jsonable(described.get("mean")),
            "std": _jsonable(described.get("std")),
            "min": _jsonable(described.get("min")),
            "max": _jsonable(described.get("max")),
            "skew": _jsonable(series.skew()),
        })
    elif kind in (CATEGORICAL, BOOLEAN):
        counts = series.value_counts().head(10)
        info["top_values"] = {str(k): int(v) for k, v in counts.items()}
    return info


def _jsonable(value):
    """numpy scalars and NaN do not survive JSON encoding untouched."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else round(float(value), 6)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def profile_dataset(df: pd.DataFrame, preview_rows: int = 20) -> dict:
    n_rows = len(df)
    columns = [profile_column(df[c], n_rows) for c in df.columns]
    preview = df.head(preview_rows).copy()
    for col in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[col]):
            preview[col] = preview[col].astype(str)
    return {
        "rows": n_rows,
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing": int(df.isna().sum().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 2),
        "column_profiles": columns,
        "preview": preview.where(pd.notna(preview), None).to_dict(orient="records"),
        "suggested_targets": _suggest_targets(columns, n_rows),
    }


def _suggest_targets(columns: list[dict], n_rows: int) -> list[str]:
    """Columns that plausibly serve as a label: complete, and either
    low-cardinality (classification) or continuous (regression)."""
    picks = []
    for col in columns:
        if col["kind"] in (IDENTIFIER, EMPTY) or col["missing_pct"] > 10:
            continue
        if col["kind"] in (CATEGORICAL, BOOLEAN) and 2 <= col["unique"] <= 20:
            picks.append(col["name"])
        elif col["kind"] == NUMERIC:
            picks.append(col["name"])
    return picks[:15]
