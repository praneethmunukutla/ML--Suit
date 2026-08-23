"""Training runs: submit, poll, list."""
from __future__ import annotations

from fastapi import APIRouter

from backend.core.logging_config import get_logger
from backend.registry import store
from backend.schemas.models import TrainRequest
from backend.training import evaluation, jobs, model_zoo
from backend.training.task_detect import detect_task

router = APIRouter(prefix="/api", tags=["training"])
log = get_logger(__name__)


@router.post("/train")
def start_training(request: TrainRequest) -> dict:
    """Queue a training run. Returns immediately with a run_id to poll."""
    # Fail fast on a bad column selection rather than inside a worker thread.
    meta = store.get_dataset_meta(request.dataset_id)
    known = set(meta["column_names"])
    unknown = [c for c in [*request.feature_columns, request.target_column]
               if c not in known]
    if unknown:
        from backend.core.exceptions import ValidationError
        raise ValidationError(f"Columns not in dataset: {unknown}")
    if request.target_column in request.feature_columns:
        from backend.core.exceptions import ValidationError
        raise ValidationError(
            f"Target '{request.target_column}' cannot also be a feature",
            detail="Leaking the target into X makes the scores meaningless.")

    run = store.create_run({
        "dataset_id": request.dataset_id,
        "dataset_name": meta["name"],
        "feature_columns": request.feature_columns,
        "target_column": request.target_column,
        "requested_task": request.task,
        "requested_metric": request.metric,
        "requested_models": request.models,
        "test_size": request.test_size,
        "cv_folds": request.cv_folds,
        "search_iter": request.search_iter,
    })
    jobs.submit(run["run_id"], {
        "dataset_id": request.dataset_id,
        "feature_cols": request.feature_columns,
        "target_col": request.target_column,
        "task": request.task,
        "metric": request.metric,
        "models": request.models,
        "test_size": request.test_size,
        "cv_folds": request.cv_folds,
        "search_iter": request.search_iter,
    })
    return {"run_id": run["run_id"], "status": "queued"}


@router.get("/runs")
def list_runs(limit: int = 50) -> dict:
    return {"runs": store.list_runs(limit)}


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return store.get_run(run_id)


@router.get("/task-preview/{dataset_id}")
def preview_task(dataset_id: str, target_column: str) -> dict:
    """What task the system would infer from this target, shown before training."""
    df = store.load_dataset(dataset_id)
    if target_column not in df.columns:
        from backend.core.exceptions import ValidationError
        raise ValidationError(f"Column '{target_column}' is not in this dataset")
    task, reason = detect_task(df[target_column])
    return {
        "task": task,
        "reason": reason,
        "metrics": evaluation.valid_metrics(task),
        "default_metric": evaluation.default_metric(task),
        "models": model_zoo.available_models(task),
        "n_classes": int(df[target_column].nunique()) if task == "classification" else None,
    }
