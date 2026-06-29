"""Confidence scoring for match predictions.

Confidence reflects how certain the model is that the recommended prediction
will earn points.  Specifically it answers: "if the recommended score's winner
(or draw) is correct, what fraction of outcomes are covered?"

Formula: confidence = 1 - P(opponent wins outright)

For a home-win prediction, confidence = 1 - P(away win).
For an away-win prediction, confidence = 1 - P(home win).
For a draw prediction,      confidence = P(draw).
"""
from models.poisson import outcome_probabilities
import numpy as np


def calculate_confidence(matrix: np.ndarray, pred_home: int, pred_away: int) -> float:
    """Return a confidence score in [0, 1] for the recommended prediction.

    Args:
        matrix: Score probability matrix from models.poisson.
        pred_home: Recommended home goals.
        pred_away: Recommended away goals.

    Returns:
        Confidence as a float between 0 and 1.
    """
    outcomes = outcome_probabilities(matrix)

    if pred_home > pred_away:
        # Predicting home win → confidence = 1 - P(away win)
        return 1.0 - outcomes["away"]

    if pred_away > pred_home:
        # Predicting away win → confidence = 1 - P(home win)
        return 1.0 - outcomes["home"]

    # Predicting draw → confidence = P(draw)
    return outcomes["draw"]
