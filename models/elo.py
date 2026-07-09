"""Elo rating system for football teams.

Handles rating updates after matches and derives expected goals / win probabilities
from the Elo difference between two teams.
"""
import math
from dataclasses import dataclass

from config.loader import load_config


@dataclass
class EloUpdate:
    """Result of updating Elo ratings after a match."""

    new_home_elo: float
    new_away_elo: float
    home_expected: float
    away_expected: float


def expected_result(elo_home: float, elo_away: float, home_advantage: float) -> float:
    """Return the Elo-based expected score (0–1) for the home team.

    A score of 1 means a win, 0.5 a draw, 0 a loss.
    """
    return 1.0 / (1.0 + 10.0 ** ((elo_away - elo_home - home_advantage) / 400.0))


def _match_score(home_goals: int, away_goals: int) -> tuple[float, float]:
    """Convert a match result into Elo scores (home, away)."""
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals == away_goals:
        return 0.5, 0.5
    return 0.0, 1.0


def margin_of_victory_factor(home_goals: int, away_goals: int) -> float:
    """Return the goal-difference multiplier applied to K (World-Football-Elo).

    A 1-goal win moves ratings by the base amount; larger margins move them
    more, with diminishing returns.  A draw returns 1.0 (no scaling).
    """
    diff = abs(home_goals - away_goals)
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    return (11.0 + diff) / 8.0


def update_elo(
    elo_home: float,
    elo_away: float,
    home_goals: int,
    away_goals: int,
    k_factor: float | None = None,
    home_advantage: float | None = None,
    mov_enabled: bool | None = None,
) -> EloUpdate:
    """Update Elo ratings after a completed match.

    Args:
        elo_home: Current Elo of the home team.
        elo_away: Current Elo of the away team.
        home_goals: Goals scored by the home team.
        away_goals: Goals scored by the away team.
        k_factor: Learning rate. Defaults to config value.
        home_advantage: Elo bonus for the home team. Defaults to config value.
            Pass 0 for neutral-venue (knockout) matches.
        mov_enabled: Scale K by the margin of victory. Defaults to config value.

    Returns:
        EloUpdate with new ratings and expected scores.
    """
    cfg = load_config()["elo"]
    k = k_factor if k_factor is not None else cfg["k_factor"]
    ha = home_advantage if home_advantage is not None else cfg["home_advantage"]
    use_mov = mov_enabled if mov_enabled is not None else cfg.get("mov_enabled", False)

    home_exp = expected_result(elo_home, elo_away, ha)
    away_exp = 1.0 - home_exp

    home_score, away_score = _match_score(home_goals, away_goals)

    if use_mov:
        k *= margin_of_victory_factor(home_goals, away_goals)

    return EloUpdate(
        new_home_elo=elo_home + k * (home_score - home_exp),
        new_away_elo=elo_away + k * (away_score - away_exp),
        home_expected=home_exp,
        away_expected=away_exp,
    )


def win_probabilities(
    elo_home: float,
    elo_away: float,
    home_advantage: float | None = None,
) -> dict[str, float]:
    """Estimate home-win / draw / away-win probabilities from Elo ratings.

    Draw probability is estimated by a formula that increases when teams are
    evenly matched.  Win probabilities are scaled to the remaining probability
    mass after allocating for draws.

    Returns:
        Dict with keys "home", "draw", "away".
    """
    cfg = load_config()["elo"]
    ha = home_advantage if home_advantage is not None else cfg["home_advantage"]

    p_home_no_draw = expected_result(elo_home, elo_away, ha)
    p_away_no_draw = 1.0 - p_home_no_draw

    # Draw rate peaks for equal teams and shrinks as the Elo gap grows.
    draw_base = cfg.get("draw_base", 0.30)
    draw_slope = cfg.get("draw_slope", 0.0002)
    draw_floor = cfg.get("draw_floor", 0.05)
    elo_diff = abs(elo_home + ha - elo_away)
    p_draw = max(draw_floor, draw_base - elo_diff * draw_slope)

    remaining = 1.0 - p_draw
    return {
        "home": p_home_no_draw * remaining,
        "draw": p_draw,
        "away": p_away_no_draw * remaining,
    }


def expected_goals(
    elo_home: float,
    elo_away: float,
    home_advantage: float | None = None,
    base_lambda: float | None = None,
    elo_sensitivity: float | None = None,
) -> tuple[float, float]:
    """Derive expected goals (lambda) for each team from their Elo ratings.

    Stronger teams are expected to score more and concede less.  The formula is:

        lambda_team = base * exp(effective_elo_diff * sensitivity)

    Args:
        elo_home: Current Elo of the home team.
        elo_away: Current Elo of the away team.
        home_advantage: Elo bonus added to home team. Defaults to config value.
        base_lambda: Average goals per team per match. Defaults to config value.
        elo_sensitivity: Scaling factor for Elo difference. Defaults to config value.

    Returns:
        (lambda_home, lambda_away)
    """
    cfg = load_config()
    elo_cfg = cfg["elo"]
    xg_cfg = cfg["expected_goals"]

    ha = home_advantage if home_advantage is not None else elo_cfg["home_advantage"]
    base = base_lambda if base_lambda is not None else xg_cfg["base_lambda"]
    sens = elo_sensitivity if elo_sensitivity is not None else xg_cfg["elo_sensitivity"]

    diff_home = (elo_home + ha) - elo_away
    diff_away = elo_away - (elo_home + ha)

    return base * math.exp(diff_home * sens), base * math.exp(diff_away * sens)


def expected_goals_blended(
    elo_home: float,
    elo_away: float,
    goals_for_home: float,
    goals_against_home: float,
    matches_home: float,
    goals_for_away: float,
    goals_against_away: float,
    matches_away: float,
    home_advantage: float | None = None,
    base_lambda: float | None = None,
    neutral: bool = False,
) -> tuple[float, float]:
    """Blend Elo-based and stats-based expected goals for a more realistic estimate.

    Once teams have enough match data the Dixon-Coles-style attack/defense model
    dominates; Elo fills in when data is sparse.

    Attack/defense model:
        attack_strength  = team_avg_goals_scored   / global_avg
        defense_strength = team_avg_goals_conceded / global_avg
        lambda_home = global_avg * home_attack * away_defense
        lambda_away = global_avg * away_attack * home_defense

    Blend weight:  w_stats = min(1, min(matches_home, matches_away) / full_stats)
                  w_elo   = 1 - w_stats

    Args:
        goals_for_home / goals_against_home / matches_home: home team weighted
            (recency- and competition-adjusted) totals — may be fractional.
        goals_for_away / goals_against_away / matches_away: away team totals.
        home_advantage: Elo bonus for the home side; ignored when ``neutral``.
        neutral: True for neutral-venue (knockout) matches — no home advantage.

    Returns:
        (lambda_home, lambda_away) clamped to config [lambda_min, lambda_max].
    """
    cfg = load_config()
    xg = cfg["expected_goals"]
    base = base_lambda if base_lambda is not None else xg["base_lambda"]
    min_m: float = xg.get("min_matches_for_stats", 2)
    full_stats: float = xg.get("full_stats_matches", 5.0)
    lam_min: float = xg.get("lambda_min", 0.3)
    lam_max: float = xg.get("lambda_max", 6.0)
    def_floor: float = xg.get("defense_floor", 0.1)

    # Neutral venue (all World Cup knockouts) → no home advantage.
    ha = _neutral_home_advantage(home_advantage) if neutral else home_advantage

    # ── Elo-based baseline ───────────────────────────────────────────
    lam_elo_home, lam_elo_away = expected_goals(
        elo_home, elo_away, home_advantage=ha, base_lambda=base
    )

    enough = matches_home >= min_m and matches_away >= min_m
    if not enough:
        return lam_elo_home, lam_elo_away

    # ── Stats-based model ────────────────────────────────────────────
    home_attack  = (goals_for_home   / matches_home) / base
    home_defense = (goals_against_home / matches_home) / base
    away_attack  = (goals_for_away   / matches_away) / base
    away_defense = (goals_against_away / matches_away) / base

    # Guard against zero (a perfect defense should still concede something)
    home_defense = max(home_defense, def_floor)
    away_defense = max(away_defense, def_floor)

    lam_stats_home = base * home_attack * away_defense
    lam_stats_away = base * away_attack * home_defense

    # ── Weighted blend ───────────────────────────────────────────────
    w_stats = min(1.0, min(matches_home, matches_away) / full_stats)
    w_elo   = 1.0 - w_stats

    lam_home = w_elo * lam_elo_home + w_stats * lam_stats_home
    lam_away = w_elo * lam_elo_away + w_stats * lam_stats_away

    return max(lam_min, min(lam_max, lam_home)), max(lam_min, min(lam_max, lam_away))


def _neutral_home_advantage(home_advantage: float | None) -> float:
    """Return the home-advantage value to use on a neutral venue.

    Defaults to config ``elo.home_advantage_neutral`` (0) unless the caller
    explicitly passed a value.
    """
    if home_advantage is not None:
        return home_advantage
    return load_config()["elo"].get("home_advantage_neutral", 0)
