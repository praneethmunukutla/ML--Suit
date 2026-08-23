"""Decide whether a target column implies classification or regression."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.core.exceptions import ValidationError

CLASSIFICATION = "classification"
REGRESSION = "regression"


def detect_task(y: pd.Series) -> tuple[str, str]:
    """Return (task, human-readable reason)."""
    clean = y.dropna()
    if clean.empty:
        raise ValidationError("The target column is entirely empty")
    nunique = clean.nunique()
    if nunique < 2:
        raise ValidationError(
            f"Target '{y.name}' has only one distinct value — nothing to learn"
        )

    if pd.api.types.is_bool_dtype(clean):
        return CLASSIFICATION, "target is boolean"
    if not pd.api.types.is_numeric_dtype(clean):
        return CLASSIFICATION, f"target is non-numeric with {nunique} classes"
    if pd.api.types.is_float_dtype(clean) and not _is_integral(clean):
        return REGRESSION, "target is continuous"
    # Integer targets are ambiguous: few distinct values reads as class labels.
    if nunique <= 20 and nunique / len(clean) < 0.05:
        return CLASSIFICATION, f"target is integer with only {nunique} distinct values"
    return REGRESSION, f"target is numeric with {nunique} distinct values"


def _is_integral(series: pd.Series) -> bool:
    try:
        return bool(np.all(np.equal(np.mod(series.to_numpy(dtype=float), 1), 0)))
    except (TypeError, ValueError):
        return False


def validate_for_task(y: pd.Series, task: str) -> None:
    """Catch target/task mismatches before an opaque sklearn error does."""
    clean = y.dropna()
    if task == CLASSIFICATION:
        counts = clean.value_counts()
        if len(counts) > 100:
            raise ValidationError(
                f"{len(counts)} distinct classes is too many for classification",
                detail="Did you mean regression?",
            )
        rare = counts[counts < 2]
        if len(rare):
            raise ValidationError(
                f"{len(rare)} class(es) appear only once and cannot be split",
                detail=f"Rare classes: {list(rare.index[:10])}",
            )
    elif not pd.api.types.is_numeric_dtype(clean):
        raise ValidationError(
            f"Regression needs a numeric target, but '{y.name}' is {clean.dtype}"
        )
