"""Monte-Carlo match simulator.

The analytic Poisson matrix already gives exact 1X2 / scoreline probabilities
for 90 minutes.  What it *cannot* express is knockout progression, where a
regulation draw is decided by extra time and then penalties.  This module
samples ``simulation.iterations`` matches from the two expected-goal rates and,
for knockout ties, plays out extra time (goals at a reduced rate) and a 50/50
shoot-out to produce a **probability that each side advances**.

All sampling is vectorised with NumPy; a ``seed`` makes runs reproducible.
"""
from __future__ import annotations

import numpy as np

from config.loader import load_config


def simulate_match(
    lambda_home: float,
    lambda_away: float,
    iterations: int | None = None,
    knockout: bool = False,
    seed: int | None = None,
    extra_time_fraction: float | None = None,
) -> dict:
    """Simulate a match many times and summarise the outcome distribution.

    Args:
        lambda_home: Expected home goals over 90 minutes.
        lambda_away: Expected away goals over 90 minutes.
        iterations: Number of simulated matches. Defaults to config value.
        knockout: If True, also resolve draws via extra time + penalties and
            report advance probabilities.
        seed: Optional RNG seed for reproducibility.
        extra_time_fraction: Scoring rate of the 30' of extra time relative to
            90 minutes. Defaults to config simulation.extra_time_fraction.

    Returns:
        Dict with regulation outcome probabilities (``p_home``/``p_draw``/
        ``p_away``), mean goals, and — when ``knockout`` — ``p_advance_home`` /
        ``p_advance_away`` (which sum to 1).
    """
    cfg = load_config()["simulation"]
    n = int(iterations if iterations is not None else cfg["iterations"])
    et_frac = (
        extra_time_fraction
        if extra_time_fraction is not None
        else cfg.get("extra_time_fraction", 0.37)
    )

    rng = np.random.default_rng(seed)
    hg = rng.poisson(lambda_home, n)
    ag = rng.poisson(lambda_away, n)

    home_reg = hg > ag
    away_reg = ag > hg
    draw_reg = ~home_reg & ~away_reg

    result = {
        "iterations": n,
        "p_home": float(home_reg.mean()),
        "p_draw": float(draw_reg.mean()),
        "p_away": float(away_reg.mean()),
        "expected_home_goals": float(hg.mean()),
        "expected_away_goals": float(ag.mean()),
    }

    if not knockout:
        return result

    advance_home = home_reg.copy()
    draw_idx = np.flatnonzero(draw_reg)
    if draw_idx.size:
        # Extra time: goals at a reduced rate.
        eth = rng.poisson(lambda_home * et_frac, draw_idx.size)
        eta = rng.poisson(lambda_away * et_frac, draw_idx.size)
        et_diff = eth - eta
        advance_home[draw_idx[et_diff > 0]] = True
        # Penalties for ties still level after extra time: modelled as 50/50.
        still_level = draw_idx[et_diff == 0]
        if still_level.size:
            coin = rng.random(still_level.size) < 0.5
            advance_home[still_level[coin]] = True

    result["p_advance_home"] = float(advance_home.mean())
    result["p_advance_away"] = float(1.0 - advance_home.mean())
    return result
