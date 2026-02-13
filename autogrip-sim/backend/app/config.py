"""Application configuration using pydantic-settings."""

from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096

    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "OPENAI_API_KEY must be set. "
                "Provide it via environment variable or .env file."
            )
        return v

    # Isaac Sim
    isaac_sim_path: str = "/isaac-sim"
    isaac_sim_headless: bool = True
    isaac_sim_docker_image: str = "nvcr.io/nvidia/isaac-sim:4.2.0"

    # CORS
    cors_origins: list[str] = ["http://localhost:8000", "http://localhost:3000"]

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 100
    max_loop_iterations: int = 20
    success_threshold: int = 3

    # Vector Store
    chroma_persist_dir: str = "./chroma_db"

    # Logging
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)

    @property
    def cad_upload_path(self) -> Path:
        return self.upload_path / "cad"

    @property
    def manual_upload_path(self) -> Path:
        return self.upload_path / "manuals"

    @property
    def results_path(self) -> Path:
        return self.upload_path / "results"


settings = Settings()
