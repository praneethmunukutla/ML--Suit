"""Scoring endpoints — JSON rows or an uploaded file."""
from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse

from backend.core.exceptions import PredictionError
from backend.ingestion import loaders, sanitize
from backend.schemas.models import PredictRequest
from backend.training import predict as predict_service

router = APIRouter(prefix="/api/predict", tags=["prediction"])


@router.post("/{model_id}")
def predict_rows(model_id: str, request: PredictRequest) -> dict:
    """Score rows supplied as JSON."""
    frame = pd.DataFrame(request.rows)
    if frame.empty:
        raise PredictionError("No rows were supplied")
    return predict_service.predict(model_id, frame, request.include_probabilities)


@router.post("/{model_id}/file")
async def predict_file(model_id: str, file: UploadFile = File(...)) -> dict:
    """Score every row of an uploaded file."""
    raw = await file.read()
    df = loaders.load_bytes(raw, file.filename or "input.csv")
    df = sanitize.clean_columns(df)
    result = predict_service.predict(model_id, df)
    result["input_rows"] = int(len(df))
    return result


@router.post("/{model_id}/file/download")
async def predict_file_download(model_id: str, file: UploadFile = File(...)):
    """Score an uploaded file and stream back the original rows plus predictions."""
    raw = await file.read()
    df = sanitize.clean_columns(loaders.load_bytes(raw, file.filename or "input.csv"))
    out = predict_service.predict_frame(model_id, df)
    buffer = io.StringIO()
    out.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{model_id}.csv"},
    )
