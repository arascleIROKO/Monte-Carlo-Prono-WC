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


def update_elo(
    elo_home: float,
    elo_away: float,
    home_goals: int,
    away_goals: int,
    k_factor: float | None = None,
    home_advantage: float | None = None,
) -> EloUpdate:
    """Update Elo ratings after a completed match.

    Args:
        elo_home: Current Elo of the home team.
        elo_away: Current Elo of the away team.
        home_goals: Goals scored by the home team.
        away_goals: Goals scored by the away team.
        k_factor: Learning rate. Defaults to config value.
        home_advantage: Elo bonus for the home team. Defaults to config value.

    Returns:
        EloUpdate with new ratings and expected scores.
    """
    cfg = load_config()["elo"]
    k = k_factor if k_factor is not None else cfg["k_factor"]
    ha = home_advantage if home_advantage is not None else cfg["home_advantage"]

    home_exp = expected_result(elo_home, elo_away, ha)
    away_exp = 1.0 - home_exp

    home_score, away_score = _match_score(home_goals, away_goals)

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

    # Draw rate peaks at 30% for equal teams and shrinks as Elo gap grows.
    elo_diff = abs(elo_home + ha - elo_away)
    p_draw = max(0.05, 0.30 - elo_diff * 0.0002)

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
    home_gf: int,
    home_ga: int,
    home_mp: int,
    away_gf: int,
    away_ga: int,
    away_mp: int,
) -> tuple[float, float]:
    """Expected goals blending the Elo-only estimate with observed scoring stats.

    Early in a tournament there is little data, so we lean on the Elo-derived
    expectation.  Once both teams have played at least ``min_matches_for_stats``
    matches, we blend in each team's actual attacking output against the
    opponent's defensive record.

    Stats-based xG uses the standard independent-Poisson normalisation: a team's
    scoring expectation is (its goals/match) x (opponent's conceded/match) scaled
    by the league-average ``base_lambda`` so that an average attack facing an
    average defence yields ``base_lambda``.

    Args:
        elo_home: Current Elo of the home team.
        elo_away: Current Elo of the away team.
        home_gf: Home team's total goals scored so far.
        home_ga: Home team's total goals conceded so far.
        home_mp: Home team's matches played so far.
        away_gf: Away team's total goals scored so far.
        away_ga: Away team's total goals conceded so far.
        away_mp: Away team's matches played so far.

    Returns:
        (lambda_home, lambda_away)
    """
    cfg = load_config()
    xg_cfg = cfg["expected_goals"]
    base = xg_cfg["base_lambda"]
    min_mp = xg_cfg["min_matches_for_stats"]

    elo_h, elo_a = expected_goals(elo_home, elo_away)

    # Not enough data for either team -> fall back to the Elo-only estimate.
    if home_mp < min_mp or away_mp < min_mp:
        return elo_h, elo_a

    home_attack = home_gf / home_mp
    home_defense = home_ga / home_mp
    away_attack = away_gf / away_mp
    away_defense = away_ga / away_mp

    stats_h = (home_attack * away_defense) / base if base else home_attack
    stats_a = (away_attack * home_defense) / base if base else away_attack

    # Equal blend of the Elo signal and the observed stats.
    lam_home = 0.5 * elo_h + 0.5 * stats_h
    lam_away = 0.5 * elo_a + 0.5 * stats_a
    return lam_home, lam_away
