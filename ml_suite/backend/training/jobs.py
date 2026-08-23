"""Background execution for training runs.

Training takes minutes, which is far longer than an HTTP request should live.
Runs go onto a small thread pool and the client polls the run record.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from backend.core import metrics as app_metrics
from backend.core.config import settings
from backend.core.exceptions import MLSuiteError
from backend.core.logging_config import get_logger
from backend.registry import store
from backend.training.trainer import train

log = get_logger(__name__)

_executor = ThreadPoolExecutor(
    max_workers=settings.TRAIN_WORKERS, thread_name_prefix="trainer"
)
_active: set[str] = set()


def submit(run_id: str, params: dict) -> None:
    _active.add(run_id)
    app_metrics.gauge("mlsuite_active_runs", len(_active))
    _executor.submit(_run, run_id, params)
    log.info("Queued run %s", run_id)


def _run(run_id: str, params: dict) -> None:
    def progress(fraction: float, stage: str) -> None:
        store.update_run(run_id, progress=round(fraction, 3), stage=stage)

    store.update_run(run_id, status="running", stage="starting")
    try:
        result = train(run_id=run_id, progress=progress, **params)
        store.update_run(run_id, status="succeeded", progress=1.0,
                         stage="complete", **result)
        log.info("Run %s finished: best=%s score=%s",
                 run_id, result["best_model"], result["best_score"])
    except MLSuiteError as exc:
        log.warning("Run %s failed: %s", run_id, exc.message)
        store.update_run(run_id, status="failed", stage="failed",
                         error=exc.message, error_detail=exc.detail)
        app_metrics.inc("mlsuite_training_runs_total", {"status": "failed"})
    except Exception as exc:  # noqa: BLE001 - a worker thread must never die silently
        log.exception("Run %s crashed", run_id)
        store.update_run(run_id, status="failed", stage="failed",
                         error=f"{type(exc).__name__}: {exc}")
        app_metrics.inc("mlsuite_training_runs_total", {"status": "crashed"})
    finally:
        _active.discard(run_id)
        app_metrics.gauge("mlsuite_active_runs", len(_active))


def active_runs() -> list[str]:
    return sorted(_active)


def shutdown() -> None:
    _executor.shutdown(wait=False, cancel_futures=True)
