"""Application settings, loaded from environment variables / .env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Solar Panel Classification API"
    model_path: Path = DEFAULT_PROJECT_ROOT / "models" / "solar_panel_classifier.keras"
    class_names_path: Path = DEFAULT_PROJECT_ROOT / "models" / "class_names.json"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    max_upload_mb: float = 8.0
    log_level: str = "INFO"


settings = Settings()
