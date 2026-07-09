"""Tests for the blended expected-goals model, MoV scaling and neutral venue."""
from models.elo import (
    expected_goals_blended,
    margin_of_victory_factor,
    update_elo,
)


def test_mov_factor_values():
    assert margin_of_victory_factor(1, 0) == 1.0
    assert margin_of_victory_factor(0, 1) == 1.0
    assert margin_of_victory_factor(3, 1) == 1.5          # diff 2
    assert margin_of_victory_factor(4, 0) == (11 + 4) / 8  # diff 4
    assert margin_of_victory_factor(2, 2) == 1.0           # draw


def test_mov_amplifies_rating_change():
    small = update_elo(1000, 1000, 1, 0, k_factor=40, home_advantage=0, mov_enabled=True)
    big = update_elo(1000, 1000, 5, 0, k_factor=40, home_advantage=0, mov_enabled=True)
    assert (big.new_home_elo - 1000) > (small.new_home_elo - 1000)


def test_mov_disabled_matches_plain_k():
    with_mov = update_elo(1000, 1000, 4, 0, k_factor=40, home_advantage=0, mov_enabled=True)
    without = update_elo(1000, 1000, 4, 0, k_factor=40, home_advantage=0, mov_enabled=False)
    assert with_mov.new_home_elo > without.new_home_elo


def test_blended_neutral_symmetric_for_equal_teams():
    lam_h, lam_a = expected_goals_blended(1000, 1000, 0, 0, 0, 0, 0, 0, neutral=True)
    assert abs(lam_h - lam_a) < 1e-9


def test_blended_home_advantage_when_not_neutral():
    lam_h, lam_a = expected_goals_blended(1000, 1000, 0, 0, 0, 0, 0, 0, neutral=False)
    assert lam_h > lam_a


def test_blended_stats_pull_toward_scoring_rate():
    # A high-scoring, leaky home team vs an average away team: sample large
    # enough to give the stats model full weight.
    lam_h, lam_a = expected_goals_blended(
        1000, 1000,
        goals_for_home=18, goals_against_home=3, matches_home=6,
        goals_for_away=6, goals_against_away=6, matches_away=6,
        neutral=True,
    )
    # Home attacks far above base and away defends averagely → home lambda high.
    assert lam_h > lam_a
    assert lam_h > 1.4


def test_blended_clamped_to_config_bounds():
    lam_h, lam_a = expected_goals_blended(
        3000, 500,
        goals_for_home=50, goals_against_home=0, matches_home=10,
        goals_for_away=0, goals_against_away=50, matches_away=10,
        neutral=False,
    )
    assert 0.3 <= lam_h <= 6.0
    assert 0.3 <= lam_a <= 6.0
