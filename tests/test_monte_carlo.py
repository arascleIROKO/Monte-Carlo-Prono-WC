"""Tests for the Monte-Carlo simulator."""
from models.monte_carlo import simulate_match
from models.poisson import outcome_probabilities, probability_matrix


def test_regulation_probs_sum_to_one():
    r = simulate_match(1.6, 1.1, iterations=50000, seed=1)
    assert abs(r["p_home"] + r["p_draw"] + r["p_away"] - 1.0) < 1e-9


def test_advance_probs_sum_to_one():
    r = simulate_match(1.6, 1.1, iterations=50000, knockout=True, seed=1)
    assert abs(r["p_advance_home"] + r["p_advance_away"] - 1.0) < 1e-9


def test_no_advance_keys_without_knockout():
    r = simulate_match(1.5, 1.5, iterations=10000, seed=1)
    assert "p_advance_home" not in r


def test_stronger_team_advances_more_often():
    r = simulate_match(2.2, 0.7, iterations=50000, knockout=True, seed=2)
    assert r["p_advance_home"] > r["p_advance_away"]


def test_reproducible_with_seed():
    a = simulate_match(1.4, 1.4, iterations=20000, knockout=True, seed=7)
    b = simulate_match(1.4, 1.4, iterations=20000, knockout=True, seed=7)
    assert a == b


def test_advance_favours_regulation_winner_side():
    # Advance prob for the stronger side should exceed its regulation win prob
    # (it also wins some of the drawn ties in ET/penalties).
    lam_h, lam_a = 1.8, 1.0
    matrix = probability_matrix(lam_h, lam_a)
    reg = outcome_probabilities(matrix)
    r = simulate_match(lam_h, lam_a, iterations=60000, knockout=True, seed=3)
    assert r["p_advance_home"] > reg["home"]
