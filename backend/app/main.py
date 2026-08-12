"""FastAPI application entrypoint."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.inference import ModelService
from app.routes import router

logging.basicConfig(level=settings.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service = ModelService(settings.model_path, settings.class_names_path)
    try:
        model_service.load()
    except FileNotFoundError as exc:
        logger.warning("Starting without a loaded model: %s", exc)
    app.state.model_service = model_service
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health", include_in_schema=False)
def root_health():
    return {"status": "ok", "model_loaded": app.state.model_service.is_loaded}
