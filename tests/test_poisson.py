"""Tests for the Poisson distribution model."""
import numpy as np
import pytest

from models.poisson import outcome_probabilities, probability_matrix, score_probability, top_scores


def test_probability_matrix_shape():
    matrix = probability_matrix(1.2, 1.0, max_goals=6)
    assert matrix.shape == (7, 7)


def test_probability_matrix_sums_to_one():
    matrix = probability_matrix(1.2, 1.0, max_goals=6)
    assert abs(matrix.sum() - 1.0) < 0.01  # small tail loss beyond max_goals is acceptable


def test_probability_matrix_values_non_negative():
    matrix = probability_matrix(1.5, 0.8, max_goals=6)
    assert np.all(matrix >= 0)


def test_score_probability_matches_matrix_up_to_normalisation():
    # The matrix is renormalised after truncation, so cells are the raw pmf
    # products scaled by 1/total.  Renormalisation preserves relative ratios.
    matrix = probability_matrix(1.2, 1.0, max_goals=6)
    p10 = score_probability(1.2, 1.0, 1, 0)
    p21 = score_probability(1.2, 1.0, 2, 1)
    assert float(matrix[1, 0]) >= p10  # normalisation only scales up
    assert abs(matrix[1, 0] / matrix[2, 1] - p10 / p21) < 1e-9


def test_dixon_coles_lifts_draw_mass():
    from models.poisson import outcome_probabilities
    indep = outcome_probabilities(probability_matrix(1.3, 1.3, max_goals=6, rho=0.0))
    corrected = outcome_probabilities(probability_matrix(1.3, 1.3, max_goals=6, rho=-0.1))
    assert corrected["draw"] > indep["draw"]


def test_outcome_probabilities_sum_to_one():
    matrix = probability_matrix(1.5, 1.0, max_goals=6)
    probs = outcome_probabilities(matrix)
    total = probs["home"] + probs["draw"] + probs["away"]
    assert abs(total - matrix.sum()) < 1e-9


def test_top_scores_length():
    matrix = probability_matrix(1.2, 1.0, max_goals=6)
    scores = top_scores(matrix, top_n=5)
    assert len(scores) == 5


def test_top_scores_descending_probability():
    matrix = probability_matrix(1.2, 1.0, max_goals=6)
    scores = top_scores(matrix, top_n=10)
    probs = [s["probability"] for s in scores]
    assert probs == sorted(probs, reverse=True)


def test_stronger_team_favoured():
    # A much stronger home team should have higher P(home win).
    matrix_dominant = probability_matrix(3.0, 0.5, max_goals=6)
    matrix_equal = probability_matrix(1.2, 1.2, max_goals=6)
    probs_dominant = outcome_probabilities(matrix_dominant)
    probs_equal = outcome_probabilities(matrix_equal)
    assert probs_dominant["home"] > probs_equal["home"]
