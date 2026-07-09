"""SQLAlchemy ORM models for the World Cup prediction engine."""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    """Represents a national football team with its Elo rating and stats."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    elo: Mapped[float] = mapped_column(Float, nullable=False, default=1000.0)
    goals_for: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    goals_against: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    matches_played: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    form: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    last_update: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    home_matches: Mapped[list["Match"]] = relationship(
        "Match", foreign_keys="Match.home_team_id", back_populates="home_team"
    )
    away_matches: Mapped[list["Match"]] = relationship(
        "Match", foreign_keys="Match.away_team_id", back_populates="away_team"
    )


class Match(Base):
    """Represents a football match."""

    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    competition: Mapped[str] = mapped_column(String(50), nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    home_team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False
    )
    away_team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False
    )
    home_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="SCHEDULED"
    )
    stage: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    neutral: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    home_team: Mapped["Team"] = relationship(
        "Team", foreign_keys=[home_team_id], back_populates="home_matches"
    )
    away_team: Mapped["Team"] = relationship(
        "Team", foreign_keys=[away_team_id], back_populates="away_matches"
    )
    prediction: Mapped[Optional["Prediction"]] = relationship(
        "Prediction", back_populates="match", uselist=False
    )
    result: Mapped[Optional["Result"]] = relationship(
        "Result", back_populates="match", uselist=False
    )


class Prediction(Base):
    """Stores the model's score prediction and associated metrics for a match."""

    __tablename__ = "predictions"

    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), primary_key=True
    )
    predicted_home_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_away_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_value: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )

    match: Mapped["Match"] = relationship("Match", back_populates="prediction")


class Result(Base):
    """Records actual match result and points earned by the prediction."""

    __tablename__ = "results"

    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), primary_key=True
    )
    real_home_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    real_away_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    points: Mapped[int] = mapped_column(Integer, nullable=False)
    correct_winner: Mapped[bool] = mapped_column(Boolean, nullable=False)
    correct_goal_difference: Mapped[bool] = mapped_column(Boolean, nullable=False)
    exact_score: Mapped[bool] = mapped_column(Boolean, nullable=False)

    match: Mapped["Match"] = relationship("Match", back_populates="result")
