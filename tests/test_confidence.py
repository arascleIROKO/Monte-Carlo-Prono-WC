"""Tests for the confidence module."""
import numpy as np

from models.confidence import calculate_confidence
from models.poisson import outcome_probabilities, probability_matrix


def test_confidence_home_pred_equals_home_prob():
    matrix = probability_matrix(2.0, 0.8)
    o = outcome_probabilities(matrix)
    # A home-win prediction's confidence is the probability of a home win.
    conf = calculate_confidence(matrix, 2, 1)
    assert abs(conf - o["home"]) < 1e-9


def test_confidence_away_pred_equals_away_prob():
    matrix = probability_matrix(0.7, 2.1)
    o = outcome_probabilities(matrix)
    conf = calculate_confidence(matrix, 0, 2)
    assert abs(conf - o["away"]) < 1e-9


def test_confidence_draw_pred_equals_draw_prob():
    matrix = probability_matrix(1.2, 1.2)
    o = outcome_probabilities(matrix)
    conf = calculate_confidence(matrix, 1, 1)
    assert abs(conf - o["draw"]) < 1e-9


def test_confidence_in_unit_interval():
    matrix = probability_matrix(1.6, 1.3)
    for ph, pa in [(2, 1), (0, 0), (1, 3)]:
        c = calculate_confidence(matrix, ph, pa)
        assert 0.0 <= c <= 1.0


def test_draw_does_not_inflate_home_confidence():
    # The old definition (1 - P(away)) counted draws as confident; the new one
    # must not exceed the home-win probability for a home tip.
    matrix = probability_matrix(1.1, 1.0)
    o = outcome_probabilities(matrix)
    conf = calculate_confidence(matrix, 2, 1)
    assert conf <= o["home"] + 1e-9
    assert conf < 1.0 - o["away"]  # strictly less than the old measure
