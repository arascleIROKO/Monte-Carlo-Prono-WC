"""Recency- and competition-weighted team strength.

The blended expected-goals model needs attack/defense rates that reflect a
team's *current* form, not a flat average over every match ever stored.  The
raw ``teams.goals_for / matches_played`` aggregates mix competitions years
apart (e.g. Euro 2024 with World Cup 2026) and give teams that simply played
more games a larger, staler sample.

``weighted_team_stats`` recomputes those rates from the match history with:

* a **competition weight** (down-weight matches from other tournaments), and
* an **exponential recency decay** (a configurable half-life in days).

It returns *weighted* totals so the existing attack/defense arithmetic in
``expected_goals_blended`` (goals / matches) keeps working unchanged — the
"count" is now an effective sample size rather than an integer.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import or_
from sqlalchemy.orm import Session

from config.loader import load_config
from database.models import Match


def _competition_weight(competition: str, weights: dict, default: float) -> float:
    return float(weights.get(competition, default))


def weighted_team_stats(
    session: Session,
    team_id: int,
    as_of: datetime,
    cfg: dict | None = None,
) -> tuple[float, float, float]:
    """Return (weighted_goals_for, weighted_goals_against, weighted_matches).

    Only FINISHED matches with a recorded score that kicked off strictly
    before ``as_of`` and involve ``team_id`` are counted.  Each match is
    weighted by ``competition_weight * 0.5 ** (age_days / half_life_days)``.

    Args:
        session: An open SQLAlchemy session.
        team_id: The team whose strength we want.
        as_of: Reference instant; matches on/after it are ignored (no leakage).
        cfg: Optional pre-loaded config (defaults to ``load_config()``).
    """
    cfg = cfg or load_config()
    xg = cfg["expected_goals"]
    weights = xg.get("competition_weights", {}) or {}
    default_weight = min(weights.values()) if weights else 1.0
    half_life = float(xg.get("half_life_days", 0) or 0)

    matches = (
        session.query(Match)
        .filter(Match.status == "FINISHED")
        .filter(Match.home_goals.isnot(None))
        .filter(Match.date < as_of)
        .filter(or_(Match.home_team_id == team_id, Match.away_team_id == team_id))
        .all()
    )

    wgf = wga = wn = 0.0
    for m in matches:
        age_days = max(0.0, (as_of - m.date).total_seconds() / 86400.0)
        decay = 0.5 ** (age_days / half_life) if half_life > 0 else 1.0
        w = _competition_weight(m.competition, weights, default_weight) * decay
        if w <= 0:
            continue
        if m.home_team_id == team_id:
            gf, ga = m.home_goals, m.away_goals
        else:
            gf, ga = m.away_goals, m.home_goals
        wgf += w * gf
        wga += w * ga
        wn += w

    return wgf, wga, wn
