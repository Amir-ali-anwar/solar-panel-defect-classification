"""API route definitions."""
import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.config import settings
from app.inference import InvalidImageError, ModelService
from app.schemas import ClassesResponse, HealthResponse, PredictionResponse

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/gif"}


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


@router.get("/health", response_model=HealthResponse)
def health(model_service: ModelService = Depends(get_model_service)):
    return HealthResponse(status="ok", model_loaded=model_service.is_loaded)


@router.get("/classes", response_model=ClassesResponse)
def classes(model_service: ModelService = Depends(get_model_service)):
    if not model_service.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    return ClassesResponse(classes=model_service.class_names)


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_service: ModelService = Depends(get_model_service),
):
    if not model_service.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds max upload size of {settings.max_upload_mb} MB",
        )

    try:
        result = model_service.predict(contents)
    except InvalidImageError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed")

    return PredictionResponse(**result)
