"""Tests for pipeline helper logic (no network / live DB)."""
from pipeline.update import _compute_points, _is_neutral


def test_is_neutral_group_stage_is_not_neutral():
    assert _is_neutral("GROUP_STAGE") is False
    assert _is_neutral("REGULAR_SEASON") is False


def test_is_neutral_knockout_stages():
    assert _is_neutral("QUARTER_FINALS") is True
    assert _is_neutral("LAST_16") is True
    assert _is_neutral("FINAL") is True


def test_is_neutral_none_defaults_false():
    assert _is_neutral(None) is False


def test_compute_points_exact():
    pts, winner, diff, exact = _compute_points(2, 1, 2, 1)
    assert exact and diff and winner
    assert pts == 6


def test_compute_points_goal_difference():
    pts, winner, diff, exact = _compute_points(2, 1, 3, 2)
    assert not exact and diff and winner
    assert pts == 4


def test_compute_points_winner_only():
    pts, winner, diff, exact = _compute_points(2, 1, 4, 1)
    assert winner and not diff and not exact
    assert pts == 2


def test_compute_points_wrong():
    pts, winner, diff, exact = _compute_points(2, 1, 0, 2)
    assert pts == 0
    assert not winner and not diff and not exact
