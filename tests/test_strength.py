"""Tests for recency-/competition-weighted team strength."""
from datetime import datetime

from database.models import Match, Team
from models.strength import weighted_team_stats


def _add_match(session, mid, date, home_id, away_id, hg, ag, comp="WC"):
    session.add(Match(
        id=mid, competition=comp, date=date,
        home_team_id=home_id, away_team_id=away_id,
        home_goals=hg, away_goals=ag, status="FINISHED",
    ))


def test_only_prior_finished_matches_count(db_session):
    db_session.add_all([Team(id=1, name="A"), Team(id=2, name="B")])
    _add_match(db_session, 10, datetime(2026, 6, 1), 1, 2, 3, 0)
    # A future match must not leak into an as-of-June prediction.
    _add_match(db_session, 11, datetime(2026, 7, 1), 1, 2, 0, 4)
    db_session.commit()

    gf, ga, n = weighted_team_stats(db_session, 1, datetime(2026, 6, 15))
    assert n > 0
    # Only the 3-0 win counts → scored 3, conceded 0 (before weighting ratio).
    assert gf > ga


def test_orientation_home_and_away(db_session):
    db_session.add_all([Team(id=1, name="A"), Team(id=2, name="B")])
    _add_match(db_session, 10, datetime(2026, 6, 1), 1, 2, 2, 1)  # team 1 home
    _add_match(db_session, 11, datetime(2026, 6, 5), 2, 1, 0, 3)  # team 1 away, scored 3
    db_session.commit()

    gf, ga, n = weighted_team_stats(db_session, 1, datetime(2026, 6, 10))
    # Team 1 scored 2 + 3 = 5 (before decay), conceded 1 + 0 = 1.
    assert gf > ga
    assert n > 0


def test_recency_downweights_old_matches(db_session):
    db_session.add_all([Team(id=1, name="A"), Team(id=2, name="B")])
    # Old high-scoring match vs recent low-scoring one.
    _add_match(db_session, 10, datetime(2024, 6, 1), 1, 2, 5, 0)
    _add_match(db_session, 11, datetime(2026, 6, 1), 1, 2, 1, 0)
    db_session.commit()

    as_of = datetime(2026, 6, 15)
    gf, ga, n = weighted_team_stats(db_session, 1, as_of)
    # With a 365-day half-life the 2024 match is heavily discounted, so the
    # weighted average goals-for should sit well below the raw mean of 3.0.
    assert gf / n < 3.0


def test_competition_weight_applied(db_session):
    db_session.add_all([Team(id=1, name="A"), Team(id=2, name="B")])
    _add_match(db_session, 10, datetime(2026, 6, 1), 1, 2, 2, 0, comp="EC")
    db_session.commit()
    as_of = datetime(2026, 6, 2)
    gf, ga, n = weighted_team_stats(db_session, 1, as_of)
    # EC weight is 0.6 and the match is ~1 day old (decay ≈ 1), so n ≈ 0.6.
    assert 0.5 < n < 0.65
