"""Inference against a registered model.

The saved pipeline holds its own preprocessing, so callers pass raw rows in
the original schema and nothing else.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.core import metrics as app_metrics
from backend.core.config import settings
from backend.core.exceptions import PredictionError
from backend.core.logging_config import get_logger
from backend.registry import store
from backend.training.task_detect import CLASSIFICATION

log = get_logger(__name__)


def _align(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Reindex incoming rows onto the schema the model was trained on."""
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise PredictionError(
            f"Input is missing {len(missing)} required column(s)",
            detail={"missing": missing[:20], "expected": expected},
        )
    return df[expected]


def predict(model_id: str, rows: pd.DataFrame, include_proba: bool = True) -> dict:
    pipeline, meta = store.load_model(model_id)
    expected = meta["feature_columns"]
    frame = _align(rows, expected)

    try:
        raw = pipeline.predict(frame)
    except Exception as exc:
        raise PredictionError("Model could not score these rows",
                              detail=str(exc)) from exc

    result: dict = {
        "model_id": model_id,
        "algorithm": meta.get("algorithm"),
        "task": meta.get("task"),
        "target_column": meta.get("target_column"),
        "n_rows": int(len(frame)),
    }

    if meta.get("task") == CLASSIFICATION:
        labels = _decode_labels(model_id, raw, meta)
        result["predictions"] = labels
        if include_proba and hasattr(pipeline.named_steps.get("model"), "predict_proba"):
            proba = pipeline.predict_proba(frame)
            classes = meta.get("class_labels") or [str(i) for i in range(proba.shape[1])]
            result["probabilities"] = [
                {cls: round(float(p), 6) for cls, p in zip(classes, row)}
                for row in proba
            ]
            result["confidence"] = [round(float(row.max()), 6) for row in proba]
    else:
        result["predictions"] = [round(float(v), 6) for v in np.asarray(raw, dtype=float)]

    app_metrics.inc("mlsuite_predictions_total",
                    {"model": meta.get("algorithm", "unknown")}, len(frame))
    log.info("Scored %d row(s) with %s (%s)", len(frame), model_id, meta.get("algorithm"))
    return result


def _decode_labels(model_id: str, raw, meta: dict) -> list:
    """Turn encoded integers back into the labels the user supplied."""
    encoder_path = Path(settings.MODEL_DIR) / f"{model_id}.labels.joblib"
    if encoder_path.exists():
        try:
            encoder = joblib.load(encoder_path)
            return [str(v) for v in encoder.inverse_transform(
                np.asarray(raw).astype(int))]
        except Exception as exc:
            log.warning("Label decode failed for %s, returning raw: %s", model_id, exc)
    class_labels = meta.get("class_labels")
    if class_labels:
        return [class_labels[int(v)] if 0 <= int(v) < len(class_labels) else str(v)
                for v in raw]
    return [str(v) for v in raw]


def predict_frame(model_id: str, rows: pd.DataFrame) -> pd.DataFrame:
    """Same as predict(), returned as the input frame plus prediction columns."""
    result = predict(model_id, rows)
    out = rows.copy()
    target = result.get("target_column") or "target"
    out[f"predicted_{target}"] = result["predictions"]
    if "confidence" in result:
        out["confidence"] = result["confidence"]
    return out
