"""
Centralised runtime configuration for the pipeline handling credentials/environment requirements
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/pipeline/settings.py -> src/pipeline -> src -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """
    Values are read from process environment variables first, 
    falling back to a `.env` file at the repository root. 
    """

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #Field names are snake_case for normal Python use
    #`alias` pins each one to the exact environment variable name already used in `.env` (e.g. `GEE_PROJECT`)
    gee_project: str | None = Field(default=None, alias="GEE_PROJECT")
    opensky_username: str | None = Field(default=None, alias="OPENSKY_USERNAME")

    # Root directory for all pipeline data (masks, chips, caches, etc).
    # Defaults to <repo_root>/data; override DATA_DIR in .env to store data elsewhere.
    data_dir: Path = Field(default=_REPO_ROOT / "data", alias="DATA_DIR")

    # Individual Postgres connection parts: 
    # mirrors what docker-compose.yml reads to initialise the container, so changing
    # e.g. POSTGRES_PORT in .env updates settings here too
    postgres_host: str = Field(default="127.0.0.1", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="parallax", alias="POSTGRES_DB")
    postgres_user: str = Field(default="parallax", alias="POSTGRES_USER")
    
    # Optional password and database URL override can be set in .env
    postgres_password: str | None = Field(default=None, alias="POSTGRES_PASSWORD")
    database_url_override: str | None = Field(default=None, alias="DATABASE_URL")

    @property
    def database_url(self) -> str | None:
        if self.database_url_override:
            return self.database_url_override
        if not self.postgres_password:
            return None
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def run_dir(self, run_id: int) -> Path:
        """
        Root directory for one pipeline run's on-disk artifacts 
        - for files too large to store in postgres
        - E.g. masks, chips, per-run caches

        Functionality: 
        - Every run gets own subtree under DATA_DIR so on-disk files
        also remain run:id scoped 
        """
        return self.data_dir / "runs" / f"run_{run_id}"

    def run_subdir(self, run_id: int, name: str) -> Path:
        """A named subdirectory under a run's root, created if missing."""
        path = self.run_dir(run_id) / name
        path.mkdir(parents=True, exist_ok=True)
        return path


def get_settings() -> Settings:
    """
    Returns a fresh Settings instance, re-reading `.env`/env vars each call.
    """
    return Settings()
