"""Tests for database models and schema."""
from datetime import datetime

from database.models import Match, Prediction, Result, Team


def test_team_creation(db_session):
    team = Team(id=10, name="France", elo=1600.0)
    db_session.add(team)
    db_session.commit()

    fetched = db_session.query(Team).filter_by(name="France").first()
    assert fetched is not None
    assert fetched.elo == 1600.0
    assert fetched.goals_for == 0
    assert fetched.matches_played == 0


def test_match_creation(db_session, two_teams):
    home, away = two_teams
    match = Match(
        id=1,
        competition="WC",
        date=datetime(2026, 6, 14),
        home_team_id=home.id,
        away_team_id=away.id,
        status="SCHEDULED",
    )
    db_session.add(match)
    db_session.commit()

    fetched = db_session.query(Match).filter_by(id=1).first()
    assert fetched.home_team.name == "Brazil"
    assert fetched.away_team.name == "Japan"
    assert fetched.home_goals is None


def test_prediction_creation(db_session, two_teams):
    home, away = two_teams
    match = Match(
        id=1,
        competition="WC",
        date=datetime(2026, 6, 14),
        home_team_id=home.id,
        away_team_id=away.id,
        status="SCHEDULED",
    )
    db_session.add(match)
    db_session.commit()

    pred = Prediction(
        match_id=1,
        predicted_home_goals=2,
        predicted_away_goals=0,
        expected_value=3.14,
        confidence=0.85,
    )
    db_session.add(pred)
    db_session.commit()

    fetched = db_session.query(Prediction).filter_by(match_id=1).first()
    assert fetched.predicted_home_goals == 2
    assert abs(fetched.expected_value - 3.14) < 1e-6


def test_result_creation(db_session, two_teams):
    home, away = two_teams
    match = Match(
        id=1,
        competition="WC",
        date=datetime(2026, 6, 14),
        home_team_id=home.id,
        away_team_id=away.id,
        home_goals=2,
        away_goals=1,
        status="FINISHED",
    )
    db_session.add(match)
    db_session.commit()

    result = Result(
        match_id=1,
        real_home_goals=2,
        real_away_goals=1,
        points=6,
        correct_winner=True,
        correct_goal_difference=True,
        exact_score=True,
    )
    db_session.add(result)
    db_session.commit()

    fetched = db_session.query(Result).filter_by(match_id=1).first()
    assert fetched.points == 6
    assert fetched.exact_score is True
