"""
SQLAlchemy engine/session setup
"""

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from pipeline.settings import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """
    Returns SQLAlchemy Engine:
    - created on first call, settings read once at that point
    - Loaded from cache on subsequent calls (see get_settings())
    - Connection is tested eagerly here (once), so a Docker/Postgres
      container that isn't running fails with a clear message now
      rather than a raw psycopg2 error the first time some query happens
      to run.
    """
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env (or add "
            "the DATABASE_URL line to your existing .env), e.g.\n"
            "  DATABASE_URL=postgresql+psycopg2://parallax:parallax@localhost:5432/parallax"
        )

    engine = create_engine(settings.database_url, future=True)
    try:
        with engine.connect():
            pass
    except OperationalError as e:
        url = make_url(settings.database_url)
        raise RuntimeError(
            f"Could not connect to Postgres at {url.host}:{url.port}.\n"
            f"The database container is probably not running. Start it with:\n"
            f"  docker compose up -d\n"
            f"Then confirm it's actually healthy (not just started) before retrying:\n"
            f"  docker compose ps\n"
            f"(look for 'Up (healthy)', not just 'Up')"
        ) from e

    return engine


def get_session() -> Session:
    """
    Returns a new Session bound to process-wide engine
    """
    Session_ = sessionmaker(bind=get_engine(), future=True)
    return Session_()
