"""Shared pytest fixtures."""
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import Base, Match, Team


@pytest.fixture
def in_memory_engine():
    """SQLite in-memory engine with schema created."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(in_memory_engine):
    """Scoped session for a single test; rolled back after each test."""
    Session = sessionmaker(bind=in_memory_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def two_teams(db_session):
    """Return two persisted Team objects with known Elo ratings."""
    home = Team(id=1, name="Brazil", elo=1500.0)
    away = Team(id=2, name="Japan", elo=1200.0)
    db_session.add_all([home, away])
    db_session.commit()
    return home, away
