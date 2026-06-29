"""Tests for the Elo rating model."""
import pytest

from models.elo import expected_goals, expected_result, update_elo, win_probabilities


def test_expected_result_equal_teams():
    # Equal teams with no home advantage → 50/50.
    p = expected_result(1000.0, 1000.0, home_advantage=0.0)
    assert abs(p - 0.5) < 1e-6


def test_expected_result_home_advantage():
    # With home advantage the home team should be favoured.
    p = expected_result(1000.0, 1000.0, home_advantage=65.0)
    assert p > 0.5


def test_expected_result_stronger_home():
    p_stronger = expected_result(1200.0, 1000.0, home_advantage=0.0)
    p_equal = expected_result(1000.0, 1000.0, home_advantage=0.0)
    assert p_stronger > p_equal


def test_elo_update_home_win():
    result = update_elo(1000.0, 1000.0, home_goals=2, away_goals=0, k_factor=30, home_advantage=0)
    # Home wins → home Elo increases, away decreases.
    assert result.new_home_elo > 1000.0
    assert result.new_away_elo < 1000.0


def test_elo_update_draw():
    result = update_elo(1000.0, 1000.0, home_goals=1, away_goals=1, k_factor=30, home_advantage=0)
    # Draw between equals → no change.
    assert abs(result.new_home_elo - 1000.0) < 1e-6
    assert abs(result.new_away_elo - 1000.0) < 1e-6


def test_elo_update_away_win():
    result = update_elo(1000.0, 1000.0, home_goals=0, away_goals=2, k_factor=30, home_advantage=0)
    assert result.new_home_elo < 1000.0
    assert result.new_away_elo > 1000.0


def test_elo_update_conservation():
    result = update_elo(1200.0, 1000.0, home_goals=1, away_goals=0, k_factor=30, home_advantage=0)
    total_before = 1200.0 + 1000.0
    total_after = result.new_home_elo + result.new_away_elo
    assert abs(total_before - total_after) < 1e-6


def test_win_probabilities_sum_to_one():
    probs = win_probabilities(1200.0, 1000.0)
    total = probs["home"] + probs["draw"] + probs["away"]
    assert abs(total - 1.0) < 1e-6


def test_win_probabilities_favourite_has_highest():
    probs = win_probabilities(1400.0, 1000.0)
    assert probs["home"] > probs["draw"]
    assert probs["home"] > probs["away"]


def test_expected_goals_stronger_scores_more():
    lam_home, lam_away = expected_goals(1400.0, 1000.0, home_advantage=0)
    assert lam_home > lam_away


def test_expected_goals_equal_teams_symmetric():
    lam_home, lam_away = expected_goals(1000.0, 1000.0, home_advantage=0)
    assert abs(lam_home - lam_away) < 1e-6


def test_expected_goals_home_advantage_boosts_home():
    lam_no_ha, _ = expected_goals(1000.0, 1000.0, home_advantage=0)
    lam_with_ha, _ = expected_goals(1000.0, 1000.0, home_advantage=65)
    assert lam_with_ha > lam_no_ha
