"""Expected Value (EV) engine — the core of the prediction system.

For each possible predicted score, the EV is the weighted sum of points the
model expects to earn given the full Poisson probability matrix:

    EV = P(exact score)       * 6
       + P(same goal diff, not exact) * 4
       + P(same winner, wrong diff)  * 2

The recommended score is the one that maximises this EV.
"""
import numpy as np

from config.loader import load_config


def _outcome_class(pred_home: int, pred_away: int, real_home: int, real_away: int) -> int:
    """Return the points earned when predicting (pred_home, pred_away) and the
    actual result is (real_home, real_away).

    Points:
        6 — exact score
        4 — correct goal difference (implies correct winner)
        2 — correct winner only
        0 — wrong winner
    """
    cfg = load_config()["competition"]
    points_exact = cfg["exact_score_points"]
    points_diff = cfg["goal_difference_points"]
    points_winner = cfg["winner_points"]

    if real_home == pred_home and real_away == pred_away:
        return points_exact

    if (real_home - real_away) == (pred_home - pred_away):
        return points_diff

    pred_winner = (pred_home > pred_away) - (pred_home < pred_away)
    real_winner = (real_home > real_away) - (real_home < real_away)
    if real_winner == pred_winner:
        return points_winner

    return 0


def score_ev(
    matrix: np.ndarray,
    pred_home: int,
    pred_away: int,
) -> float:
    """Calculate the expected value of predicting a specific score.

    Args:
        matrix: Score probability matrix (rows=home goals, cols=away goals).
        pred_home: Predicted home goals.
        pred_away: Predicted away goals.

    Returns:
        Expected points as a float.
    """
    rows, cols = matrix.shape
    ev = 0.0
    for h in range(rows):
        for a in range(cols):
            pts = _outcome_class(pred_home, pred_away, h, a)
            if pts > 0:
                ev += matrix[h, a] * pts
    return ev


def best_predictions(
    matrix: np.ndarray,
    top_n: int | None = None,
) -> list[dict]:
    """Return the top-N predictions ranked by expected value.

    Args:
        matrix: Score probability matrix from models.poisson.
        top_n: Number of predictions to return. Defaults to config value.

    Returns:
        List of dicts with keys "home", "away", "ev", "probability".
    """
    if top_n is None:
        top_n = load_config()["recommendations"]["top_n"]

    rows, cols = matrix.shape
    candidates = [
        {
            "home": h,
            "away": a,
            "ev": score_ev(matrix, h, a),
            "probability": float(matrix[h, a]),
        }
        for h in range(rows)
        for a in range(cols)
    ]
    candidates.sort(key=lambda x: x["ev"], reverse=True)
    return candidates[:top_n]


def recommend(matrix: np.ndarray) -> dict:
    """Return the single best score prediction (highest EV).

    Args:
        matrix: Score probability matrix from models.poisson.

    Returns:
        Dict with keys "home", "away", "ev", "probability".
    """
    return best_predictions(matrix, top_n=1)[0]
