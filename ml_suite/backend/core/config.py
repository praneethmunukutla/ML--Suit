"""Central configuration. Everything overridable by env var so the same code
runs locally on files today and against Mongo/Neon later."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


class Settings:
    APP_NAME = "ML Suite"
    VERSION = "0.1.0"

    HOST = _env("MLSUITE_HOST", "127.0.0.1")
    PORT = _env_int("MLSUITE_PORT", 8000)

    # Storage backend: "local" today, "mongo" once a URI is supplied.
    STORAGE_BACKEND = _env("MLSUITE_STORAGE", "local")
    STORAGE_DIR = Path(_env("MLSUITE_STORAGE_DIR", str(BASE_DIR / "storage")))
    DATASET_DIR = STORAGE_DIR / "datasets"
    MODEL_DIR = STORAGE_DIR / "models"
    RUN_DIR = STORAGE_DIR / "runs"
    UPLOAD_DIR = STORAGE_DIR / "uploads"

    LOG_DIR = Path(_env("MLSUITE_LOG_DIR", str(BASE_DIR / "logs")))
    LOG_LEVEL = _env("MLSUITE_LOG_LEVEL", "INFO")
    LOG_JSON = _env("MLSUITE_LOG_JSON", "0") == "1"

    # Guardrails: training runs in-process, so cap the work it can take on.
    MAX_UPLOAD_MB = _env_int("MLSUITE_MAX_UPLOAD_MB", 200)
    MAX_TRAIN_ROWS = _env_int("MLSUITE_MAX_TRAIN_ROWS", 200_000)
    MAX_FEATURES = _env_int("MLSUITE_MAX_FEATURES", 500)
    DEFAULT_CV_FOLDS = _env_int("MLSUITE_CV_FOLDS", 3)
    DEFAULT_SEARCH_ITER = _env_int("MLSUITE_SEARCH_ITER", 12)
    TRAIN_WORKERS = _env_int("MLSUITE_TRAIN_WORKERS", 2)
    RANDOM_STATE = 42

    # A column with more distinct values than this is treated as high-cardinality
    # and ordinal-encoded rather than one-hot expanded.
    ONEHOT_MAX_CARDINALITY = _env_int("MLSUITE_ONEHOT_MAX_CARD", 20)
    # Above this ratio of unique values to rows, a categorical column is
    # assumed to be an identifier and dropped.
    DROP_CARDINALITY_RATIO = 0.9

    def ensure_dirs(self) -> None:
        for d in (self.DATASET_DIR, self.MODEL_DIR, self.RUN_DIR,
                  self.UPLOAD_DIR, self.LOG_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
