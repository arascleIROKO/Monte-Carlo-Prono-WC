"""Confidence scoring for match predictions.

Confidence answers a single, honest question: **how likely is the recommended
prediction to earn any points at all?**  Under the competition rules a
prediction scores only when its *outcome class* (home win / draw / away win) is
correct — a home-win tip earns nothing on a draw.  So confidence is simply the
probability of the predicted outcome class:

    home-win prediction → P(home win)
    away-win prediction → P(away win)
    draw prediction     → P(draw)

The previous definition (``1 - P(opponent wins)``) counted draws as
"confident" for a home/away tip even though a draw earns zero points there,
which overstated confidence in tight matches.
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
        Probability that the predicted outcome class occurs (so the tip earns
        at least the winner points), as a float between 0 and 1.
    """
    outcomes = outcome_probabilities(matrix)

    if pred_home > pred_away:
        return outcomes["home"]
    if pred_away > pred_home:
        return outcomes["away"]
    return outcomes["draw"]
