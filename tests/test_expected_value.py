"""Tests for the Expected Value engine."""
import numpy as np
import pytest

from models.expected_value import best_predictions, recommend, score_ev
from models.poisson import probability_matrix


def test_exact_score_gives_6_points():
    # If the outcome is certain (probability = 1.0 for score 1-0),
    # then predicting 1-0 should give EV = 6.
    matrix = np.zeros((7, 7))
    matrix[1, 0] = 1.0
    ev = score_ev(matrix, pred_home=1, pred_away=0)
    assert abs(ev - 6.0) < 1e-9


def test_correct_goal_difference_gives_4_points():
    # All probability on 2-1; predicting 3-2 has same diff (+1) → EV = 4.
    matrix = np.zeros((7, 7))
    matrix[2, 1] = 1.0
    ev = score_ev(matrix, pred_home=3, pred_away=2)
    assert abs(ev - 4.0) < 1e-9


def test_correct_winner_only_gives_2_points():
    # All probability on 3-0; predicting 1-0 shares winner but wrong diff → EV = 2.
    matrix = np.zeros((7, 7))
    matrix[3, 0] = 1.0
    ev = score_ev(matrix, pred_home=1, pred_away=0)
    assert abs(ev - 2.0) < 1e-9


def test_wrong_prediction_gives_0_points():
    matrix = np.zeros((7, 7))
    matrix[0, 2] = 1.0  # away win 0-2
    ev = score_ev(matrix, pred_home=2, pred_away=0)  # home win prediction
    assert abs(ev - 0.0) < 1e-9


def test_best_predictions_length():
    matrix = probability_matrix(1.2, 1.0, max_goals=6)
    preds = best_predictions(matrix, top_n=5)
    assert len(preds) == 5


def test_best_predictions_descending_ev():
    matrix = probability_matrix(1.5, 0.8, max_goals=6)
    preds = best_predictions(matrix, top_n=10)
    evs = [p["ev"] for p in preds]
    assert evs == sorted(evs, reverse=True)


def test_recommend_returns_highest_ev():
    matrix = probability_matrix(1.5, 0.8, max_goals=6)
    all_preds = best_predictions(matrix, top_n=49)
    best = recommend(matrix)
    assert best["ev"] == max(p["ev"] for p in all_preds)


def test_ev_favoured_team_recommended():
    # For a very dominant home team, the recommendation should be a home win.
    matrix = probability_matrix(3.0, 0.5, max_goals=6)
    rec = recommend(matrix)
    assert rec["home"] > rec["away"]


def test_draw_recommended_for_equal_teams():
    # When teams are equal and draws are likely, the model should not strongly
    # recommend a lopsided scoreline.
    matrix = probability_matrix(1.2, 1.2, max_goals=6)
    rec = recommend(matrix)
    # Both teams expected to score similarly; goal difference should be small.
    assert abs(rec["home"] - rec["away"]) <= 1
