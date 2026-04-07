from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from src.api.dependencies import get_inference_service
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.api.service import AMLInferenceService
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AML Detection API",
    description="REST API для выявления сомнительных финансовых операций",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health(
    service: AMLInferenceService = Depends(get_inference_service),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        is_model_loaded=True,
        loaded_model_type=service.model_type,
    )


@app.get("/ready", response_model=HealthResponse)
def ready(
    service: AMLInferenceService = Depends(get_inference_service),
) -> HealthResponse:
    return HealthResponse(
        status="ready",
        is_model_loaded=True,
        loaded_model_type=service.model_type,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: TransactionRequest,
    service: AMLInferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    try:
        result = service.predict_one(request.model_dump())
        return PredictionResponse(**result)
    except Exception as exc:
        logger.exception("Ошибка при выполнении /predict")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
    service: AMLInferenceService = Depends(get_inference_service),
) -> BatchPredictionResponse:
    try:
        results = service.predict_batch([item.model_dump() for item in request.transactions])
        return BatchPredictionResponse(items=results)
    except Exception as exc:
        logger.exception("Ошибка при выполнении /predict_batch")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
