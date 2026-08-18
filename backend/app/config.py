"""Application settings, loaded from environment variables / .env."""
import json
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Solar Panel Classification API"
    model_path: Path = BACKEND_DIR / "models" / "solar_panel_classifier.keras"
    class_names_path: Path = BACKEND_DIR / "models" / "class_names.json"
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173", "http://localhost:3000"]
    max_upload_mb: float = 8.0
    log_level: str = "INFO"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                return json.loads(value)
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


settings = Settings()
