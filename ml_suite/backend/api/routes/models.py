"""The model registry."""
from __future__ import annotations

from fastapi import APIRouter

from backend.registry import store

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def list_models() -> dict:
    return {"models": store.list_models()}


@router.get("/{model_id}")
def get_model(model_id: str) -> dict:
    return store.get_model_meta(model_id)


@router.delete("/{model_id}")
def delete_model(model_id: str) -> dict:
    store.delete_model(model_id)
    return {"deleted": model_id}
