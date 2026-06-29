"""Poisson distribution model for football score prediction.

Given expected goals (lambda) for each team, computes the probability
of every possible scoreline up to max_goals.
"""
import numpy as np
from scipy.stats import poisson

from config.loader import load_config


def probability_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int | None = None,
) -> np.ndarray:
    """Build a (max_goals+1) x (max_goals+1) matrix of score probabilities.

    Element [i][j] is P(home scores i goals, away scores j goals).

    Args:
        lambda_home: Expected goals for the home team.
        lambda_away: Expected goals for the away team.
        max_goals: Maximum goals to consider per team. Defaults to config value.

    Returns:
        2-D NumPy array with shape (max_goals+1, max_goals+1).
    """
    if max_goals is None:
        max_goals = load_config()["poisson"]["max_goals"]

    home_probs = np.array([poisson.pmf(g, lambda_home) for g in range(max_goals + 1)])
    away_probs = np.array([poisson.pmf(g, lambda_away) for g in range(max_goals + 1)])

    # Outer product: P(home=i, away=j) = P(home=i) * P(away=j)
    return np.outer(home_probs, away_probs)


def score_probability(
    lambda_home: float,
    lambda_away: float,
    home_goals: int,
    away_goals: int,
) -> float:
    """Return the probability of a specific scoreline.

    Args:
        lambda_home: Expected goals for the home team.
        lambda_away: Expected goals for the away team.
        home_goals: Target home goals.
        away_goals: Target away goals.
    """
    return float(poisson.pmf(home_goals, lambda_home) * poisson.pmf(away_goals, lambda_away))


def outcome_probabilities(matrix: np.ndarray) -> dict[str, float]:
    """Aggregate a probability matrix into home-win / draw / away-win probabilities.

    Args:
        matrix: Score probability matrix from :func:`probability_matrix`.

    Returns:
        Dict with keys "home", "draw", "away".
    """
    p_home = float(np.sum(np.tril(matrix, -1)))
    p_draw = float(np.sum(np.diag(matrix)))
    p_away = float(np.sum(np.triu(matrix, 1)))
    return {"home": p_home, "draw": p_draw, "away": p_away}


def top_scores(
    matrix: np.ndarray,
    top_n: int | None = None,
) -> list[dict]:
    """Return the most probable scorelines in descending probability order.

    Args:
        matrix: Score probability matrix from :func:`probability_matrix`.
        top_n: Number of scores to return. Defaults to config recommendations.top_n.

    Returns:
        List of dicts with keys "home", "away", "probability".
    """
    if top_n is None:
        top_n = load_config()["recommendations"]["top_n"]

    rows, cols = matrix.shape
    scores = [
        {"home": i, "away": j, "probability": float(matrix[i, j])}
        for i in range(rows)
        for j in range(cols)
    ]
    scores.sort(key=lambda x: x["probability"], reverse=True)
    return scores[:top_n]
