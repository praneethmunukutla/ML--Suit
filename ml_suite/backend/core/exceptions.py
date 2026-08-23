"""Domain exceptions. Each carries the HTTP status the API should surface,
so route handlers never build error responses by hand."""
from __future__ import annotations

from typing import Any


class MLSuiteError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, detail: Any = None):
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict:
        return {"error": self.code, "message": self.message, "detail": self.detail}


class NotFoundError(MLSuiteError):
    status_code = 404
    code = "not_found"


class ValidationError(MLSuiteError):
    status_code = 400
    code = "validation_error"


class IngestionError(MLSuiteError):
    status_code = 400
    code = "ingestion_error"


class TrainingError(MLSuiteError):
    status_code = 500
    code = "training_error"


class PredictionError(MLSuiteError):
    status_code = 400
    code = "prediction_error"
