"""Training orchestration: split, search hyperparameters for every candidate
model, rank them on a held-out test set, and register the winner.

The saved artifact is the whole Pipeline — preprocessor and estimator
together — so prediction later needs nothing but raw rows.
"""
from __future__ import annotations

import time
import traceback
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from backend.core import metrics as app_metrics
from backend.core.config import settings
from backend.core.exceptions import TrainingError, ValidationError
from backend.core.logging_config import get_logger
from backend.preprocessing.builder import build_preprocessor, feature_names
from backend.registry import store
from backend.training import evaluation, model_zoo
from backend.training.task_detect import CLASSIFICATION, detect_task, validate_for_task

log = get_logger(__name__)

ProgressFn = Callable[[float, str], None]


def _noop(progress: float, stage: str) -> None:
    return None


def prepare_frame(df: pd.DataFrame, feature_cols: list[str], target_col: str
                  ) -> tuple[pd.DataFrame, pd.Series]:
    """Validate the column selection and drop rows with no label."""
    missing = [c for c in [*feature_cols, target_col] if c not in df.columns]
    if missing:
        raise ValidationError(f"Columns not in dataset: {missing}")
    if target_col in feature_cols:
        raise ValidationError(f"Target '{target_col}' cannot also be a feature")
    if not feature_cols:
        raise ValidationError("Select at least one feature column")
    if len(feature_cols) > settings.MAX_FEATURES:
        raise ValidationError(
            f"{len(feature_cols)} features exceeds the limit of {settings.MAX_FEATURES}")

    frame = df[[*feature_cols, target_col]].copy()
    before = len(frame)
    frame = frame.dropna(subset=[target_col])
    if len(frame) < before:
        log.info("Dropped %d row(s) with a missing target", before - len(frame))
    if len(frame) < 20:
        raise ValidationError(
            f"Only {len(frame)} labelled rows remain — need at least 20 to train")
    if len(frame) > settings.MAX_TRAIN_ROWS:
        log.info("Sampling %d of %d rows", settings.MAX_TRAIN_ROWS, len(frame))
        frame = frame.sample(settings.MAX_TRAIN_ROWS,
                             random_state=settings.RANDOM_STATE)
    return frame[feature_cols], frame[target_col]


def train(dataset_id: str, feature_cols: list[str], target_col: str,
          task: str | None = None, metric: str | None = None,
          models: list[str] | None = None, test_size: float = 0.2,
          cv_folds: int | None = None, search_iter: int | None = None,
          run_id: str | None = None, progress: ProgressFn = _noop) -> dict:
    """Run the full comparison. Returns the summary written to the run record."""
    started = time.time()
    cv_folds = cv_folds or settings.DEFAULT_CV_FOLDS
    search_iter = search_iter or settings.DEFAULT_SEARCH_ITER

    progress(0.02, "loading dataset")
    df = store.load_dataset(dataset_id)
    X, y = prepare_frame(df, feature_cols, target_col)

    progress(0.06, "detecting task")
    if task:
        detected_reason = "specified by user"
    else:
        task, detected_reason = detect_task(y)
    validate_for_task(y, task)
    metric = metric or evaluation.default_metric(task)
    if metric not in evaluation.valid_metrics(task):
        raise ValidationError(
            f"Metric '{metric}' is not valid for {task}",
            detail=f"Choose from: {evaluation.valid_metrics(task)}")

    # Classifiers get integer labels so every estimator (XGBoost included)
    # sees the same encoding; the encoder travels with the saved model.
    label_encoder = None
    class_labels = None
    if task == CLASSIFICATION:
        label_encoder = LabelEncoder()
        y_encoded = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index)
        class_labels = [str(c) for c in label_encoder.classes_]
    else:
        y_encoded = pd.to_numeric(y, errors="coerce")
        if y_encoded.isna().any():
            raise ValidationError(
                f"Target '{target_col}' has non-numeric values that regression cannot use")

    progress(0.10, "building preprocessor")
    preprocessor, prep_report = build_preprocessor(X, feature_cols)

    stratify = y_encoded if (task == CLASSIFICATION
                             and y_encoded.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size,
        random_state=settings.RANDOM_STATE, stratify=stratify)

    candidates = model_zoo.select_models(task, models, len(X_train))
    if not candidates:
        raise TrainingError("No candidate models available for this dataset")
    log.info("Training %d candidate(s) for %s on %d rows",
             len(candidates), task, len(X_train))

    splitter = (StratifiedKFold(n_splits=_safe_folds(cv_folds, y_train),
                                shuffle=True, random_state=settings.RANDOM_STATE)
                if task == CLASSIFICATION
                else KFold(n_splits=cv_folds, shuffle=True,
                           random_state=settings.RANDOM_STATE))
    scoring = evaluation.scorer_for(metric)

    leaderboard: list[dict] = []
    fitted: dict[str, Pipeline] = {}
    span = 0.85 - 0.12

    for i, (name, spec) in enumerate(candidates.items()):
        progress(0.12 + span * i / len(candidates), f"training {name}")
        entry = _train_one(name, spec, preprocessor, X_train, y_train, X_test, y_test,
                           task, splitter, scoring, search_iter, class_labels)
        leaderboard.append(entry["summary"])
        if entry["pipeline"] is not None:
            fitted[name] = entry["pipeline"]
        if run_id:  # stream partial results to the dashboard
            store.update_run(run_id, leaderboard=_rank(leaderboard, metric))

    leaderboard = _rank(leaderboard, metric)
    successful = [e for e in leaderboard if e["status"] == "ok"]
    if not successful:
        raise TrainingError(
            "Every candidate model failed",
            detail=[e.get("error") for e in leaderboard[:3]])

    progress(0.90, "registering best model")
    best = successful[0]
    best_pipeline = fitted[best["model"]]

    names = feature_names(best_pipeline.named_steps["preprocessor"])
    best["feature_importance"] = evaluation.extract_importance(best_pipeline, names)
    best["diagnostics"] = _diagnostics_for(best_pipeline, X_test, y_test,
                                           task, class_labels)

    model_meta = store.save_model(best_pipeline, {
        "algorithm": best["model"],
        "task": task,
        "dataset_id": dataset_id,
        "dataset_name": store.get_dataset_meta(dataset_id)["name"],
        "target_column": target_col,
        "feature_columns": feature_cols,
        "used_features": prep_report["used_features"],
        "dropped_features": prep_report["dropped"],
        "class_labels": class_labels,
        "primary_metric": metric,
        "metrics": best["metrics"],
        "best_params": best["best_params"],
        "run_id": run_id,
    })
    if label_encoder is not None:
        # Stored beside the pipeline so predictions come back as original labels.
        import joblib
        joblib.dump(label_encoder,
                    settings.MODEL_DIR / f"{model_meta['model_id']}.labels.joblib")

    elapsed = round(time.time() - started, 2)
    app_metrics.inc("mlsuite_training_runs_total", {"task": task, "status": "success"})
    app_metrics.observe("mlsuite_training_seconds", elapsed, {"task": task})
    progress(1.0, "complete")

    return {
        "task": task,
        "task_reason": detected_reason,
        "primary_metric": metric,
        "leaderboard": leaderboard,
        "best_model": best["model"],
        "best_score": best["primary_score"],
        "model_id": model_meta["model_id"],
        "class_labels": class_labels,
        "preprocessing": prep_report,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "elapsed_seconds": elapsed,
    }


def _train_one(name, spec, preprocessor, X_train, y_train, X_test, y_test,
               task, splitter, scoring, search_iter, class_labels) -> dict:
    """Fit one candidate. A failure here is recorded, not raised — one bad
    model must not abort the comparison."""
    started = time.time()
    try:
        from sklearn.base import clone
        estimator, params = _wrap_estimator(spec, task)
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", estimator),
        ])
        n_iter = min(search_iter, _grid_size(params))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            search = RandomizedSearchCV(
                pipeline, params, n_iter=n_iter, cv=splitter,
                scoring=scoring, n_jobs=-1, random_state=settings.RANDOM_STATE,
                refit=True, error_score="raise",
            )
            search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_
        y_pred = best_pipeline.predict(X_test)
        if task == CLASSIFICATION:
            proba = (best_pipeline.predict_proba(X_test)
                     if hasattr(best_pipeline.named_steps["model"], "predict_proba")
                     else None)
            scores = evaluation.evaluate_classification(
                y_test, y_pred, proba, len(class_labels or []) or 2)
        else:
            scores = evaluation.evaluate_regression(y_test, y_pred)

        elapsed = round(time.time() - started, 2)
        log.info("%-18s %s", name, _fmt_scores(scores))
        return {
            "pipeline": best_pipeline,
            "summary": {
                "model": name,
                "status": "ok",
                "metrics": scores,
                "cv_score": round(float(search.best_score_), 6),
                "best_params": {k.replace("model__regressor__", "").replace("model__", ""):
                                _jsonable(v) for k, v in search.best_params_.items()},
                "fit_seconds": elapsed,
                "n_candidates": n_iter,
            },
        }
    except Exception as exc:
        log.warning("%s failed: %s", name, exc)
        log.debug(traceback.format_exc())
        app_metrics.inc("mlsuite_model_failures_total", {"model": name})
        return {
            "pipeline": None,
            "summary": {
                "model": name, "status": "failed", "metrics": {},
                "error": f"{type(exc).__name__}: {exc}",
                "fit_seconds": round(time.time() - started, 2),
            },
        }


def _wrap_estimator(spec: dict, task: str):
    """Put a scaler around the target for estimators that assume unit scale.

    Returns the estimator to fit and the parameter grid with keys rewritten to
    match, since wrapping adds a nesting level to the pipeline path.
    """
    params = dict(spec["params"])
    if task == CLASSIFICATION or not spec.get("scale_target"):
        return spec["estimator"], params
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler
    wrapped = TransformedTargetRegressor(
        regressor=spec["estimator"], transformer=StandardScaler())
    rewritten = {k.replace("model__", "model__regressor__"): v
                 for k, v in params.items()}
    return wrapped, rewritten


def _diagnostics_for(pipeline, X_test, y_test, task, class_labels) -> dict:
    y_pred = pipeline.predict(X_test)
    return evaluation.diagnostics(task, y_test, y_pred, class_labels)


def _rank(entries: list[dict], metric: str) -> list[dict]:
    """Order by the chosen metric, failures last."""
    def key(entry):
        if entry["status"] != "ok":
            return (1, 0.0)
        score = entry["metrics"].get(metric)
        if score is None:
            return (1, 0.0)
        return (0, -score if evaluation.HIGHER_IS_BETTER.get(metric, True) else score)

    ordered = sorted(entries, key=key)
    for rank, entry in enumerate(ordered, start=1):
        entry["rank"] = rank if entry["status"] == "ok" else None
        entry["primary_score"] = entry["metrics"].get(metric) if entry["status"] == "ok" else None
    return ordered


def _grid_size(params: dict) -> int:
    """Cap n_iter at the real size of a fully discrete search space."""
    total = 1
    for values in params.values():
        if hasattr(values, "rvs"):  # a scipy distribution is unbounded
            return 10_000
        total *= max(len(values), 1)
    return total


def _safe_folds(folds: int, y: pd.Series) -> int:
    """Stratified CV cannot use more folds than the rarest class has members."""
    smallest = int(y.value_counts().min())
    return max(2, min(folds, smallest))


def _fmt_scores(scores: dict) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in scores.items()
                     if v is not None and not k.startswith("neg_"))


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
