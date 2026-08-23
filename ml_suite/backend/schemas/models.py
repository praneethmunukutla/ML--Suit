"""Request/response contracts. Validation lives here so route handlers stay
thin and the OpenAPI docs describe the real API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class SQLSourceRequest(BaseModel):
    connection_uri: str = Field(..., description="SQLAlchemy URI, e.g. postgresql+psycopg2://...")
    query: str = Field(..., min_length=1)
    name: str | None = None
    limit: int | None = Field(default=None, ge=1, le=1_000_000)


class MongoSourceRequest(BaseModel):
    uri: str
    database: str
    collection: str
    query: dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=50_000, ge=1, le=1_000_000)
    name: str | None = None


class TrainRequest(BaseModel):
    dataset_id: str
    feature_columns: list[str] = Field(..., min_length=1)
    target_column: str
    task: Literal["classification", "regression"] | None = None
    metric: str | None = None
    models: list[str] | None = None
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    cv_folds: int = Field(default=3, ge=2, le=10)
    search_iter: int = Field(default=12, ge=1, le=200)

    @field_validator("feature_columns")
    @classmethod
    def _unique_features(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("feature_columns contains duplicates")
        return value


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1)
    include_probabilities: bool = True


class DatasetSummary(BaseModel):
    dataset_id: str
    name: str
    source: str
    rows: int
    columns: int
    created_at: str


class RunSummary(BaseModel):
    run_id: str
    status: str
    stage: str
    progress: float
    created_at: str
    best_model: str | None = None
    best_score: float | None = None
    error: str | None = None


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: Any = None
