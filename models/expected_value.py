"""Expected Value (EV) engine — the core of the prediction system.

For each possible predicted score, the EV is the weighted sum of points the
model expects to earn given the full Poisson probability matrix:

    EV = P(exact score)       * 6
       + P(same goal diff, not exact) * 4
       + P(same winner, wrong diff)  * 2

The recommended score is the one that maximises this EV.

The scoring is hierarchical (exact ⊂ same-diff ⊂ same-winner), so the whole EV
surface is computed from three cheap mass aggregates — one pass over the
matrix — instead of an O(n⁴) cell-by-cell double loop.
"""
import numpy as np

from config.loader import load_config


def _points_config() -> tuple[int, int, int]:
    cfg = load_config()["competition"]
    return cfg["exact_score_points"], cfg["goal_difference_points"], cfg["winner_points"]


def _outcome_class(pred_home: int, pred_away: int, real_home: int, real_away: int) -> int:
    """Return the points earned when predicting (pred_home, pred_away) and the
    actual result is (real_home, real_away).

    Points:
        6 — exact score
        4 — correct goal difference (implies correct winner)
        2 — correct winner only
        0 — wrong winner
    """
    points_exact, points_diff, points_winner = _points_config()

    if real_home == pred_home and real_away == pred_away:
        return points_exact

    if (real_home - real_away) == (pred_home - pred_away):
        return points_diff

    pred_winner = (pred_home > pred_away) - (pred_home < pred_away)
    real_winner = (real_home > real_away) - (real_home < real_away)
    if real_winner == pred_winner:
        return points_winner

    return 0


def _mass_aggregates(matrix: np.ndarray) -> tuple[dict, float, float, float]:
    """Return per-goal-difference mass and the home/draw/away masses."""
    rows, cols = matrix.shape
    i = np.arange(rows)[:, None]
    j = np.arange(cols)[None, :]
    diff = i - j

    diff_mass: dict[int, float] = {}
    for d in range(-(cols - 1), rows):
        diff_mass[d] = float(matrix[diff == d].sum())

    home_mass = float(matrix[diff > 0].sum())
    draw_mass = float(matrix[diff == 0].sum())
    away_mass = float(matrix[diff < 0].sum())
    return diff_mass, home_mass, draw_mass, away_mass


def score_ev(matrix: np.ndarray, pred_home: int, pred_away: int) -> float:
    """Calculate the expected value (expected points) of predicting a score.

    Args:
        matrix: Score probability matrix (rows=home goals, cols=away goals).
        pred_home: Predicted home goals.
        pred_away: Predicted away goals.

    Returns:
        Expected points as a float.
    """
    p_exact, p_diff, p_win = _points_config()
    diff_mass, home_mass, draw_mass, away_mass = _mass_aggregates(matrix)

    pred_diff = pred_home - pred_away
    pred_sign = (pred_home > pred_away) - (pred_home < pred_away)
    win_mass = home_mass if pred_sign > 0 else away_mass if pred_sign < 0 else draw_mass

    exact = float(matrix[pred_home, pred_away])
    same_diff = diff_mass.get(pred_diff, 0.0)
    return (
        p_exact * exact
        + p_diff * (same_diff - exact)
        + p_win * (win_mass - same_diff)
    )


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

    p_exact, p_diff, p_win = _points_config()
    diff_mass, home_mass, draw_mass, away_mass = _mass_aggregates(matrix)
    rows, cols = matrix.shape

    candidates = []
    for h in range(rows):
        for a in range(cols):
            pred_diff = h - a
            pred_sign = (h > a) - (h < a)
            win_mass = home_mass if pred_sign > 0 else away_mass if pred_sign < 0 else draw_mass
            exact = float(matrix[h, a])
            same_diff = diff_mass.get(pred_diff, 0.0)
            ev = p_exact * exact + p_diff * (same_diff - exact) + p_win * (win_mass - same_diff)
            candidates.append({"home": h, "away": a, "ev": ev, "probability": exact})

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
