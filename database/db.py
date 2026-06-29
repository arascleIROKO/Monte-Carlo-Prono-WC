"""Database engine, session factory, and initialization."""
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.loader import load_config
from database.models import Base


def _resolve_db_path(db_path: str | None) -> Path:
    """Resolve relative database paths from the project root."""
    if db_path is None:
        config = load_config()
        db_path = config["database"]["path"]
    root = Path(__file__).parent.parent
    resolved = root / db_path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def get_engine(db_path: str | None = None) -> Engine:
    """Create and return a SQLAlchemy engine."""
    path = _resolve_db_path(db_path)
    return create_engine(f"sqlite:///{path}", echo=False)


def get_session_factory(db_path: str | None = None) -> sessionmaker:
    """Return a session factory bound to the database."""
    engine = get_engine(db_path)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db(db_path: str | None = None) -> None:
    """Create all tables if they do not already exist."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)


def get_session(db_path: str | None = None) -> Session:
    """Return a new database session. Caller is responsible for closing it."""
    factory = get_session_factory(db_path)
    return factory()
