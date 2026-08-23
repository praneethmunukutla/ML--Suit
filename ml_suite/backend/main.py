"""FastAPI application: middleware, error handling, and route wiring."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core import metrics as app_metrics
from backend.core.config import settings
from backend.core.exceptions import MLSuiteError
from backend.core.logging_config import (
    configure_logging,
    get_logger,
    new_request_id,
    request_id_var,
)

configure_logging()
log = get_logger("mlsuite.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("%s v%s starting (storage=%s)",
             settings.APP_NAME, settings.VERSION, settings.STORAGE_DIR)
    yield
    from backend.training import jobs
    jobs.shutdown()
    log.info("Shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Upload any dataset, pick X and y, and get a tuned, compared, "
                "reusable model.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Local-only deployment; tighten to explicit origins before exposing this.
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tag every request with an id, time it, and record the outcome."""
    request_id_var.set(new_request_id())
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.perf_counter() - started
        log.exception("Unhandled error on %s %s after %.0fms",
                      request.method, request.url.path, elapsed * 1000)
        app_metrics.inc("mlsuite_requests_total",
                        {"path": request.url.path, "status": "500"})
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error",
                     "message": "Something went wrong handling this request",
                     "request_id": request_id_var.get()},
        )
    elapsed = time.perf_counter() - started
    app_metrics.inc("mlsuite_requests_total",
                    {"path": request.url.path, "status": str(response.status_code)})
    app_metrics.observe("mlsuite_request_seconds", elapsed, {"path": request.url.path})
    response.headers["X-Request-ID"] = request_id_var.get()
    if elapsed > 2.0:
        log.warning("Slow request %s %s took %.1fs",
                    request.method, request.url.path, elapsed)
    return response


@app.exception_handler(MLSuiteError)
async def handle_domain_error(request: Request, exc: MLSuiteError):
    """Domain failures are expected outcomes, logged at warning, not traced."""
    log.warning("%s on %s: %s", exc.code, request.url.path, exc.message)
    return JSONResponse(status_code=exc.status_code,
                        content={**exc.to_dict(), "request_id": request_id_var.get()})


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "request_validation_error",
                 "message": "The request body did not match the expected schema",
                 "detail": exc.errors()[:10],
                 "request_id": request_id_var.get()},
    )


from backend.api.routes import datasets, health, models, predict, training  # noqa: E402

app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(predict.router)


@app.get("/", tags=["monitoring"])
def root() -> dict:
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
    }
