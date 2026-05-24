"""
config.py — single source of truth for all environment-driven settings.

Every other module imports from here. No module reads os.environ directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from pathlib import Path


class Settings(BaseSettings):
    """
    Reads from .env automatically (via pydantic-settings).
    All fields are typed; missing required fields raise a clear error at startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars silently
    )

    # ── AI provider (passed through to the ai/ package) ──────────────────────
    llm_provider: str = Field(default="gemini", description="gemini")
    llm_model: str = Field(default="gemini-2.0-flash")
    embedding_provider: str = Field(default="gemini")
    embedding_model: str = Field(default="text-embedding-004")

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/lostfound",
        description="Full asyncpg-compatible connection string",
    )

    # ── File storage ──────────────────────────────────────────────────────────
    image_storage_dir: Path = Field(
        default=Path("./data/uploads"),
        description="Directory where uploaded images are stored on disk",
    )
    max_image_size_mb: int = Field(default=5, ge=1, le=50)

    # ── Concurrency & rate limiting ───────────────────────────────────────────
    semaphore_limit: int = Field(
        default=5,
        ge=1,
        description="Max parallel AI calls at once — keeps us under provider rate limits",
    )

    # ── Caching ───────────────────────────────────────────────────────────────
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=0,
        description="How long to cache embedding results (0 = disabled)",
    )

    # ── HTTP server ───────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    # ── Retry policy ─────────────────────────────────────────────────────────
    retry_max_attempts: int = Field(default=3, ge=1)
    retry_wait_seconds: float = Field(default=1.0, ge=0.1)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return upper

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        allowed = {"gemini"}
        lower = v.lower()
        if lower not in allowed:
            raise ValueError(f"llm_provider must be 'gemini', got {v!r}")
        return lower

    @field_validator("image_storage_dir")
    @classmethod
    def ensure_storage_dir_exists(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this everywhere:  from src.config import settings
settings = Settings()
