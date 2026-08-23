"""Metric computation and the extra artifacts the dashboard renders."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from backend.training.task_detect import CLASSIFICATION

CLASSIFICATION_METRICS = ["accuracy", "balanced_accuracy", "f1", "precision",
                          "recall", "roc_auc"]
REGRESSION_METRICS = ["r2", "neg_rmse", "neg_mae", "explained_variance"]

# What sklearn's scoring API should optimise during the hyperparameter search.
SCORER_NAMES = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1_weighted",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "roc_auc": "roc_auc_ovr_weighted",
    "r2": "r2",
    "neg_rmse": "neg_root_mean_squared_error",
    "neg_mae": "neg_mean_absolute_error",
    "explained_variance": "explained_variance",
}

# Metrics where a larger number is better once reported (RMSE/MAE are
# reported as positive errors, so they invert).
HIGHER_IS_BETTER = {
    "accuracy": True, "balanced_accuracy": True, "f1": True, "precision": True,
    "recall": True, "roc_auc": True, "r2": True, "explained_variance": True,
    "rmse": False, "mae": False, "neg_rmse": True, "neg_mae": True,
}


def default_metric(task: str) -> str:
    return "accuracy" if task == CLASSIFICATION else "r2"


def valid_metrics(task: str) -> list[str]:
    return CLASSIFICATION_METRICS if task == CLASSIFICATION else REGRESSION_METRICS


def scorer_for(metric: str) -> str:
    return SCORER_NAMES.get(metric, metric)


def _safe(value) -> float | None:
    try:
        out = float(value)
        return None if np.isnan(out) or np.isinf(out) else round(out, 6)
    except (TypeError, ValueError):
        return None


def evaluate_classification(y_true, y_pred, y_proba=None, n_classes: int = 2) -> dict:
    scores = {
        "accuracy": _safe(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _safe(balanced_accuracy_score(y_true, y_pred)),
        "f1": _safe(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": _safe(precision_score(y_true, y_pred, average="weighted",
                                           zero_division=0)),
        "recall": _safe(recall_score(y_true, y_pred, average="weighted",
                                     zero_division=0)),
    }
    if y_proba is not None:
        try:
            if n_classes == 2:
                scores["roc_auc"] = _safe(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                scores["roc_auc"] = _safe(roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"))
        except ValueError:
            # A class missing from the test split makes AUC undefined.
            scores["roc_auc"] = None
    else:
        scores["roc_auc"] = None
    return scores


def evaluate_regression(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": _safe(r2_score(y_true, y_pred)),
        "rmse": _safe(rmse),
        "mae": _safe(mean_absolute_error(y_true, y_pred)),
        "explained_variance": _safe(explained_variance_score(y_true, y_pred)),
        # Reported as negatives too, so ranking logic stays uniform.
        "neg_rmse": _safe(-rmse),
        "neg_mae": _safe(-mean_absolute_error(y_true, y_pred)),
    }


def diagnostics(task: str, y_true, y_pred, class_labels=None, sample: int = 500) -> dict:
    """Chart-ready artifacts for the report page."""
    if task == CLASSIFICATION:
        matrix = confusion_matrix(y_true, y_pred)
        return {
            "confusion_matrix": matrix.tolist(),
            "class_labels": [str(c) for c in (class_labels
                                              if class_labels is not None
                                              else sorted(set(y_true)))],
        }
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) > sample:  # keep the payload small for the browser
        idx = np.random.RandomState(0).choice(len(y_true), sample, replace=False)
        y_true, y_pred = y_true[idx], y_pred[idx]
    return {
        "actual": [round(float(v), 6) for v in y_true],
        "predicted": [round(float(v), 6) for v in y_pred],
        "residuals": [round(float(a - p), 6) for a, p in zip(y_true, y_pred)],
    }


def extract_importance(pipeline, feature_names: list[str], top_n: int = 25) -> list[dict]:
    """Pull feature weights from whichever estimator ended up winning."""
    model = pipeline.named_steps.get("model")
    if model is None:
        return []
    # A scale-sensitive regressor sits inside a TransformedTargetRegressor.
    model = getattr(model, "regressor_", model)
    values = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    if values is None or not len(values):
        return []
    if len(feature_names) != len(values):
        feature_names = [f"feature_{i}" for i in range(len(values))]
    order = np.argsort(values)[::-1][:top_n]
    return [{"feature": feature_names[i], "importance": round(float(values[i]), 6)}
            for i in order]
