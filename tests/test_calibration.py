"""Tests for the walk-forward calibration back-test."""
from datetime import datetime

from database.models import Match, Team
from models.calibration import walk_forward_backtest


def _seed(session):
    session.add_all([
        Team(id=1, name="A", elo=1000.0),
        Team(id=2, name="B", elo=1000.0),
        Team(id=3, name="C", elo=1000.0),
    ])
    fixtures = [
        (10, datetime(2026, 6, 1), 1, 2, 2, 0),
        (11, datetime(2026, 6, 5), 2, 3, 1, 1),
        (12, datetime(2026, 6, 9), 3, 1, 0, 2),
        (13, datetime(2026, 6, 13), 1, 2, 3, 1),
        (14, datetime(2026, 6, 17), 2, 3, 2, 0),
    ]
    for mid, date, h, a, hg, ag in fixtures:
        session.add(Match(
            id=mid, competition="WC", date=date,
            home_team_id=h, away_team_id=a,
            home_goals=hg, away_goals=ag, status="FINISHED",
        ))
    session.commit()


def test_backtest_reports_expected_keys(db_session):
    _seed(db_session)
    report = walk_forward_backtest(db_session)
    for key in ("n_matches", "brier", "log_loss", "accuracy", "avg_points", "reliability"):
        assert key in report
    assert report["n_matches"] == 5


def test_backtest_metric_ranges(db_session):
    _seed(db_session)
    report = walk_forward_backtest(db_session)
    assert 0.0 <= report["brier"] <= 2.0
    assert report["log_loss"] >= 0.0
    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["avg_points"] >= 0.0


def test_reliability_counts_sum_to_three_per_match(db_session):
    _seed(db_session)
    report = walk_forward_backtest(db_session)
    total = sum(b["count"] for b in report["reliability"])
    # One (prob, hit) pair per outcome class per match.
    assert total == 3 * report["n_matches"]


def test_empty_database_is_safe(db_session):
    report = walk_forward_backtest(db_session)
    assert report["n_matches"] == 0
    assert report["brier"] is None
