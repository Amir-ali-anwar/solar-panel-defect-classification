"""Pydantic request/response models."""
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ClassesResponse(BaseModel):
    classes: list[str]


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class ErrorResponse(BaseModel):
    detail: str
