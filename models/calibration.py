"""Walk-forward calibration back-test.

Measures how *accurate* and *well-calibrated* the model is, honestly, with no
look-ahead: matches are replayed in chronological order, and each match is
predicted using only the Elo ratings and team strengths available *before* it
kicked off — then the ratings are updated and we move on.

Metrics produced:
    * ``brier``    — multiclass Brier score for 1X2 (lower is better).
    * ``log_loss`` — mean negative log-likelihood of the actual outcome.
    * ``accuracy`` — share of matches whose most-likely outcome was correct.
    * ``avg_points`` — mean competition points the EV recommendation would earn.
    * ``reliability`` — (mean_predicted, empirical_freq, count) per probability
      bin, pooled over the three outcome classes, for a reliability curve.

This is the number to watch when tuning ``base_lambda``, ``elo_sensitivity``,
``k_factor`` and ``dixon_coles_rho``.
"""
from __future__ import annotations

import math
from collections import defaultdict

from sqlalchemy.orm import Session

from config.loader import load_config
from database.models import Match
from models.elo import expected_goals_blended, update_elo
from models.expected_value import recommend, score_ev
from models.poisson import outcome_probabilities, probability_matrix
from models.strength import weighted_team_stats

_CLASSES = ("home", "draw", "away")


def _actual_class(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if away_goals > home_goals:
        return "away"
    return "draw"


def _points(pred_home: int, pred_away: int, real_home: int, real_away: int, cfg: dict) -> int:
    comp = cfg["competition"]
    if pred_home == real_home and pred_away == real_away:
        return comp["exact_score_points"]
    if (pred_home - pred_away) == (real_home - real_away):
        return comp["goal_difference_points"]
    pw = (pred_home > pred_away) - (pred_home < pred_away)
    rw = (real_home > real_away) - (real_home < real_away)
    return comp["winner_points"] if pw == rw else 0


def walk_forward_backtest(
    session: Session,
    competitions: list[str] | None = None,
    cfg: dict | None = None,
    n_bins: int = 10,
) -> dict:
    """Replay finished matches in order, scoring pre-match predictions.

    Args:
        session: Open SQLAlchemy session.
        competitions: Restrict to these competition codes (default: all).
        cfg: Pre-loaded config (default: ``load_config()``).
        n_bins: Number of reliability bins over [0, 1].

    Returns:
        Dict of aggregate metrics plus ``reliability`` and ``n_matches``.
    """
    cfg = cfg or load_config()
    elo_cfg = cfg["elo"]
    initial = float(elo_cfg["initial_rating"])
    base_ha = elo_cfg["home_advantage"]
    mov = elo_cfg.get("mov_enabled", False)
    max_goals = cfg["poisson"]["max_goals"]

    q = (
        session.query(Match)
        .filter(Match.status == "FINISHED")
        .filter(Match.home_goals.isnot(None))
        .order_by(Match.date)
    )
    if competitions:
        q = q.filter(Match.competition.in_(competitions))
    matches = q.all()

    elos: dict[int, float] = defaultdict(lambda: initial)

    brier_sum = log_sum = 0.0
    correct = 0
    points_sum = 0
    n = 0
    # reliability bins pooled over the 3 classes
    bin_pred = [0.0] * n_bins
    bin_hit = [0.0] * n_bins
    bin_cnt = [0] * n_bins

    for m in matches:
        eh, ea = elos[m.home_team_id], elos[m.away_team_id]
        neutral = bool(getattr(m, "neutral", False))

        gfh, gah, nh = weighted_team_stats(session, m.home_team_id, m.date, cfg)
        gfa, gaa, na = weighted_team_stats(session, m.away_team_id, m.date, cfg)

        lam_h, lam_a = expected_goals_blended(
            eh, ea, gfh, gah, nh, gfa, gaa, na, neutral=neutral
        )
        matrix = probability_matrix(lam_h, lam_a, max_goals)
        probs = outcome_probabilities(matrix)
        actual = _actual_class(m.home_goals, m.away_goals)

        # Metrics
        brier_sum += sum((probs[c] - (1.0 if c == actual else 0.0)) ** 2 for c in _CLASSES)
        log_sum += -math.log(max(probs[actual], 1e-12))
        if max(_CLASSES, key=lambda c: probs[c]) == actual:
            correct += 1
        rec = recommend(matrix)
        points_sum += _points(rec["home"], rec["away"], m.home_goals, m.away_goals, cfg)
        n += 1

        # Reliability: one (prob, hit) pair per class
        for c in _CLASSES:
            p = probs[c]
            idx = min(n_bins - 1, int(p * n_bins))
            bin_pred[idx] += p
            bin_hit[idx] += 1.0 if c == actual else 0.0
            bin_cnt[idx] += 1

        # Walk forward: update ratings with the observed result.
        ha = elo_cfg.get("home_advantage_neutral", 0) if neutral else base_ha
        upd = update_elo(
            eh, ea, m.home_goals, m.away_goals, home_advantage=ha, mov_enabled=mov
        )
        elos[m.home_team_id] = upd.new_home_elo
        elos[m.away_team_id] = upd.new_away_elo

    reliability = [
        {
            "bin": i,
            "mean_predicted": (bin_pred[i] / bin_cnt[i]) if bin_cnt[i] else None,
            "empirical_freq": (bin_hit[i] / bin_cnt[i]) if bin_cnt[i] else None,
            "count": bin_cnt[i],
        }
        for i in range(n_bins)
    ]

    return {
        "n_matches": n,
        "brier": (brier_sum / n) if n else None,
        "log_loss": (log_sum / n) if n else None,
        "accuracy": (correct / n) if n else None,
        "avg_points": (points_sum / n) if n else None,
        "reliability": reliability,
    }
