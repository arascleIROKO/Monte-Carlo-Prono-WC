"""Streamlit dashboard for the World Cup 2026 Prediction Engine.

Run with:  streamlit run dashboard/app.py
"""
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sqlalchemy.orm import selectinload

from config.loader import load_config
from database.db import get_session, init_db
from database.models import Match, Result, Team
from models.confidence import calculate_confidence
from models.elo import expected_goals_blended, win_probabilities
from models.expected_value import best_predictions, score_ev
from models.poisson import probability_matrix

st.set_page_config(
    page_title="Prono d'Anto",
    page_icon="⚽",
    layout="wide",
)

init_db()

UTC_OFFSET_HOURS = 2  # GMT+2
_COMP_CFG = load_config()["competition"]

_APP_THEME_CSS = """
<style>
:root {
    --mc-ink: #121722;
    --mc-ink-2: #1f2a44;
    --mc-muted: #667085;
    --mc-paper: #fbfaf5;
    --mc-card: rgba(255, 255, 255, 0.86);
    --mc-line: rgba(18, 23, 34, 0.11);
    --mc-indigo: #1f2a44;
    --mc-vermilion: #d83a2e;
    --mc-matcha: #5c7f67;
    --mc-gold: #b8892f;
}
.stApp {
    color: var(--mc-ink);
    background:
        linear-gradient(90deg, rgba(31, 42, 68, 0.035) 1px, transparent 1px),
        linear-gradient(180deg, rgba(31, 42, 68, 0.028) 1px, transparent 1px),
        linear-gradient(135deg, rgba(216, 58, 46, 0.06), transparent 34%),
        linear-gradient(180deg, #fbfaf5 0%, #f2f5f1 48%, #eef3f7 100%);
    background-size: 38px 38px, 38px 38px, auto, auto;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.block-container {
    max-width: 1240px;
    padding-top: 1.45rem;
    padding-bottom: 4rem;
}
.mc-hero {
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    gap: 22px;
    min-height: 178px;
    padding: 28px 32px;
    margin: 0 0 14px;
    border: 1px solid var(--mc-line);
    border-radius: 8px;
    background:
        linear-gradient(90deg, rgba(255, 255, 255, 0.93), rgba(255, 255, 255, 0.72)),
        linear-gradient(135deg, rgba(31, 42, 68, 0.06), rgba(216, 58, 46, 0.035));
    box-shadow: 0 22px 60px rgba(31, 42, 68, 0.13);
    backdrop-filter: blur(18px);
}
.mc-hero::before {
    content: "";
    position: absolute;
    right: 42px;
    top: 22px;
    width: 104px;
    height: 104px;
    border-radius: 50%;
    background: var(--mc-vermilion);
    opacity: 0.12;
}
.mc-hero::after {
    content: "";
    position: absolute;
    right: -56px;
    bottom: -84px;
    width: 340px;
    height: 210px;
    border-top: 1px solid rgba(31, 42, 68, 0.15);
    transform: rotate(-11deg);
}
.mc-mon {
    position: relative;
    z-index: 1;
    display: grid;
    place-items: center;
    flex: 0 0 92px;
    width: 92px;
    height: 92px;
    border: 1px solid rgba(216, 58, 46, 0.28);
    border-radius: 50%;
    background:
        linear-gradient(180deg, rgba(216, 58, 46, 0.10), rgba(255, 255, 255, 0.72)),
        #fffdf8;
    color: var(--mc-vermilion);
    font-size: 2.35rem;
    font-weight: 900;
    box-shadow: inset 0 0 0 8px rgba(216, 58, 46, 0.045);
}
.mc-hero-copy {
    position: relative;
    z-index: 1;
    max-width: 760px;
}
.mc-kicker {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    color: var(--mc-vermilion);
    font-size: 0.78rem;
    font-weight: 850;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}
.mc-kicker::before {
    content: "";
    width: 34px;
    height: 2px;
    background: var(--mc-vermilion);
}
.mc-hero h1 {
    margin: 0;
    color: var(--mc-indigo);
    font-size: 2.35rem;
    line-height: 1;
    letter-spacing: 0;
}
.mc-hero p {
    max-width: 680px;
    margin: 11px 0 0;
    color: var(--mc-muted);
    font-size: 1rem;
    line-height: 1.55;
}
.mc-status-row {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 18px;
}
.mc-chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    min-height: 32px;
    padding: 6px 11px;
    border: 1px solid var(--mc-line);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.78);
    color: #3f4858;
    font-size: 0.82rem;
    font-weight: 760;
    box-shadow: 0 6px 18px rgba(31, 42, 68, 0.05);
}
.mc-chip strong {
    color: var(--mc-indigo);
    font-weight: 900;
}
.mc-strip {
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 46px;
    padding: 11px 15px;
    margin: 0 0 18px;
    border: 1px solid rgba(31, 42, 68, 0.10);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.62);
    color: #495466;
    font-size: 0.88rem;
    font-weight: 680;
}
.mc-strip strong {
    color: var(--mc-vermilion);
}
section[data-testid="stSidebar"] {
    background: #111827;
}
div[data-testid="stTabs"] {
    margin-top: 4px;
}
div[data-testid="stTabs"] [role="tablist"] {
    gap: 8px;
    padding: 6px;
    border: 1px solid var(--mc-line);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.58);
}
div[data-testid="stTabs"] button {
    min-height: 42px;
    border-radius: 6px;
    font-weight: 700;
    color: #475467;
    transition: background 120ms ease, color 120ms ease, transform 120ms ease;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    background: var(--mc-indigo);
    color: #ffffff;
    box-shadow: 0 8px 18px rgba(31, 42, 68, 0.18);
}
div[data-testid="stTabs"] button:hover {
    background: rgba(31, 42, 68, 0.08);
    color: var(--mc-indigo);
}
h2, h3 {
    color: var(--mc-indigo);
}
h2 {
    padding-left: 13px;
    border-left: 4px solid var(--mc-vermilion);
}
h3 {
    letter-spacing: 0;
}
div[data-testid="stMetric"] {
    padding: 16px 17px;
    border: 1px solid var(--mc-line);
    border-radius: 8px;
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(255, 255, 255, 0.72)),
        var(--mc-paper);
    box-shadow: 0 12px 32px rgba(31, 42, 68, 0.07);
}
div[data-testid="stMetric"] label {
    color: var(--mc-muted);
    font-weight: 780;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--mc-line);
    border-radius: 8px;
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(255, 255, 255, 0.78)),
        var(--mc-paper);
    box-shadow: 0 16px 42px rgba(31, 42, 68, 0.08);
}
div[data-testid="stExpander"] {
    border: 1px solid var(--mc-line);
    border-radius: 8px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.74);
}
.mc-match-title {
    margin: 0;
    color: var(--mc-indigo);
    font-size: 1.28rem;
    font-weight: 900;
    line-height: 1.2;
}
.mc-kickoff {
    margin: 0;
    padding-top: 4px;
    color: var(--mc-vermilion);
    font-size: 1rem;
    font-weight: 900;
    text-align: right;
}
.mc-small-label {
    color: var(--mc-muted);
    font-size: 0.78rem;
    font-weight: 850;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.mc-score-pick {
    margin: 8px 0 4px;
    padding: 14px 8px;
    border: 1px solid rgba(216, 58, 46, 0.16);
    border-radius: 8px;
    background:
        linear-gradient(135deg, rgba(216, 58, 46, 0.08), rgba(31, 42, 68, 0.04)),
        rgba(255, 255, 255, 0.72);
    color: var(--mc-indigo);
    font-size: 2.65rem;
    font-weight: 950;
    text-align: center;
}
.mc-meta-line {
    color: #465266;
    text-align: center;
    font-size: 0.92rem;
}
.mc-winprob {
    margin: 12px 0 6px;
}
.mc-wp-track {
    display: flex;
    width: 100%;
    height: 30px;
    border-radius: 7px;
    overflow: hidden;
    border: 1px solid var(--mc-line);
    box-shadow: 0 6px 16px rgba(31, 42, 68, 0.08);
}
.mc-wp-seg {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-size: 0.82rem;
    font-weight: 800;
    white-space: nowrap;
    overflow: hidden;
}
.mc-wp-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-top: 8px;
    color: #475467;
    font-size: 0.84rem;
    font-weight: 650;
}
.mc-wp-legend span {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.mc-wp-legend i {
    width: 11px;
    height: 11px;
    border-radius: 3px;
}
.mc-ev {
    display: flex;
    flex-direction: column;
    gap: 9px;
    margin: 4px 0 2px;
}
.mc-ev-row {
    display: flex;
    align-items: center;
    gap: 12px;
}
.mc-ev-score {
    flex: 0 0 54px;
    color: var(--mc-indigo);
    font-weight: 800;
    font-variant-numeric: tabular-nums;
}
.mc-ev-bar {
    flex: 1 1 auto;
    height: 22px;
    border-radius: 6px;
    background: rgba(31, 42, 68, 0.06);
    overflow: hidden;
}
.mc-ev-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, var(--mc-matcha), #6f9a7c);
}
.mc-ev-val {
    flex: 0 0 auto;
    color: #344054;
    font-size: 0.82rem;
    font-weight: 650;
    font-variant-numeric: tabular-nums;
}
.stButton > button {
    width: 100%;
    min-height: 50px;
    border: 0;
    border-radius: 8px;
    background:
        linear-gradient(135deg, var(--mc-vermilion) 0%, #a9261f 46%, var(--mc-indigo) 100%);
    color: #ffffff;
    font-weight: 900;
    box-shadow: 0 14px 28px rgba(216, 58, 46, 0.22);
    transition: transform 120ms ease, box-shadow 120ms ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    color: #ffffff;
    box-shadow: 0 18px 36px rgba(216, 58, 46, 0.30);
}
.stButton > button:active {
    transform: translateY(0);
}
div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--mc-line);
    box-shadow: 0 12px 30px rgba(31, 42, 68, 0.07);
}
.stAlert {
    border-radius: 8px;
    border: 1px solid var(--mc-line);
}
hr {
    border-color: var(--mc-line);
}
@media (max-width: 780px) {
    .block-container { padding-top: 1rem; }
    .mc-hero {
        align-items: flex-start;
        padding: 18px;
        min-height: auto;
    }
    .mc-mon {
        width: 60px;
        height: 60px;
        flex-basis: 60px;
        font-size: 1.55rem;
    }
    .mc-hero h1 { font-size: 1.65rem; }
}
</style>
"""

_ROAMING_CAT_HTML = """
<style>
@keyframes mc-cat-roam {
    0%   { left: -104px; top: 72vh; transform: translateY(0) scaleX(1) rotate(0deg); }
    12%  { left: 12vw;   top: 78vh; transform: translateY(-2px) scaleX(1) rotate(-1deg); }
    24%  { left: 30vw;   top: 18vh; transform: translateY(2px) scaleX(1) rotate(2deg); }
    38%  { left: 56vw;   top: 25vh; transform: translateY(-3px) scaleX(1) rotate(-1deg); }
    49%  { left: calc(100vw - 78px); top: 62vh; transform: translateY(1px) scaleX(1) rotate(0deg); }
    50%  { left: calc(100vw - 78px); top: 62vh; transform: translateY(1px) scaleX(-1) rotate(0deg); }
    62%  { left: 76vw;   top: 80vh; transform: translateY(-2px) scaleX(-1) rotate(1deg); }
    74%  { left: 47vw;   top: 38vh; transform: translateY(2px) scaleX(-1) rotate(-2deg); }
    88%  { left: 18vw;   top: 86vh; transform: translateY(-3px) scaleX(-1) rotate(1deg); }
    99%  { left: -104px; top: 72vh; transform: translateY(0) scaleX(-1) rotate(0deg); }
    100% { left: -104px; top: 72vh; transform: translateY(0) scaleX(1) rotate(0deg); }
}
@keyframes mc-cat-tail {
    0%, 100% { transform: rotate(18deg); }
    50%      { transform: rotate(38deg); }
}
@keyframes mc-cat-step {
    0%, 100% { transform: rotate(12deg); }
    50%      { transform: rotate(-16deg); }
}
.mc-roaming-cat {
    position: fixed;
    width: 68px;
    height: 42px;
    z-index: 999999;
    pointer-events: none;
    animation: mc-cat-roam 30s linear infinite;
    filter: drop-shadow(0 8px 12px rgba(0, 0, 0, 0.22));
}
.mc-roaming-cat .body {
    position: absolute;
    left: 13px;
    bottom: 8px;
    width: 43px;
    height: 24px;
    border-radius: 18px 18px 12px 12px;
    background: #262a31;
}
.mc-roaming-cat .head {
    position: absolute;
    left: 0;
    bottom: 18px;
    width: 28px;
    height: 24px;
    border-radius: 50%;
    background: #262a31;
}
.mc-roaming-cat .head::before,
.mc-roaming-cat .head::after {
    content: "";
    position: absolute;
    top: -7px;
    width: 0;
    height: 0;
    border-left: 7px solid transparent;
    border-right: 7px solid transparent;
    border-bottom: 13px solid #262a31;
}
.mc-roaming-cat .head::before { left: 2px; transform: rotate(-22deg); }
.mc-roaming-cat .head::after  { right: 1px; transform: rotate(22deg); }
.mc-roaming-cat .eye {
    position: absolute;
    top: 9px;
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: #f6d45f;
}
.mc-roaming-cat .eye.left { left: 7px; }
.mc-roaming-cat .eye.right { right: 7px; }
.mc-roaming-cat .collar {
    position: absolute;
    left: 4px;
    bottom: 18px;
    width: 22px;
    height: 4px;
    border-radius: 999px;
    background: #d83a2e;
}
.mc-roaming-cat .collar::after {
    content: "";
    position: absolute;
    left: 9px;
    top: 3px;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #b8892f;
}
.mc-roaming-cat .tail {
    position: absolute;
    right: 0;
    bottom: 23px;
    width: 24px;
    height: 9px;
    border-radius: 999px;
    background: #262a31;
    transform-origin: 2px 6px;
    animation: mc-cat-tail 1.1s ease-in-out infinite;
}
.mc-roaming-cat .paw {
    position: absolute;
    bottom: 0;
    width: 7px;
    height: 13px;
    border-radius: 999px;
    background: #262a31;
    transform-origin: top center;
    animation: mc-cat-step 0.48s ease-in-out infinite;
}
.mc-roaming-cat .paw.one { left: 21px; }
.mc-roaming-cat .paw.two { left: 44px; animation-delay: 0.24s; }
@media (prefers-reduced-motion: reduce) {
    .mc-roaming-cat,
    .mc-roaming-cat .tail,
    .mc-roaming-cat .paw {
        animation: none;
    }
    .mc-roaming-cat {
        left: auto;
        right: 18px;
        top: auto;
        bottom: 18px;
    }
}
</style>
<div class="mc-roaming-cat" aria-hidden="true">
  <span class="tail"></span>
  <span class="body"></span>
  <span class="head"><span class="eye left"></span><span class="eye right"></span><span class="collar"></span></span>
  <span class="paw one"></span>
  <span class="paw two"></span>
</div>
"""


def _inject_app_theme() -> None:
    st.markdown(_APP_THEME_CSS, unsafe_allow_html=True)


def _render_roaming_cat() -> None:
    st.markdown(_ROAMING_CAT_HTML, unsafe_allow_html=True)


_inject_app_theme()
_render_roaming_cat()

# ------------------------------------------------------------------ #
# Data loaders — return plain dicts, no SQLAlchemy objects outside    #
# the session context.                                                #
# ------------------------------------------------------------------ #


@st.cache_data(ttl=600, show_spinner=False)
def _load_upcoming() -> list[dict]:
    with get_session() as s:
        rows = (
            s.query(Match)
            .options(
                selectinload(Match.home_team),
                selectinload(Match.away_team),
                selectinload(Match.prediction),
            )
            .filter(Match.status.in_(["TIMED", "SCHEDULED"]))
            .order_by(Match.date)
            .all()
        )
        return [
            {
                "id": m.id,
                "date": m.date,
                "home_name": m.home_team.name,
                "home_elo": m.home_team.elo,
                "home_gf": m.home_team.goals_for,
                "home_ga": m.home_team.goals_against,
                "home_mp": m.home_team.matches_played,
                "away_name": m.away_team.name,
                "away_elo": m.away_team.elo,
                "away_gf": m.away_team.goals_for,
                "away_ga": m.away_team.goals_against,
                "away_mp": m.away_team.matches_played,
            }
            for m in rows
        ]


@st.cache_data(ttl=600, show_spinner=False)
def _load_history() -> tuple[list[dict], list[dict]]:
    """Return (wc_matches, all_results) as plain dicts."""
    manual_forecasts = _load_manual_forecasts()

    with get_session() as s:
        matches = (
            s.query(Match)
            .options(
                selectinload(Match.home_team),
                selectinload(Match.away_team),
                selectinload(Match.prediction),
                selectinload(Match.result),
            )
            .filter(Match.status == "FINISHED", Match.competition == "WC")
            .order_by(Match.date.desc())
            .all()
        )
        results = s.query(Result).all()

        match_dicts = [
            {
                "id": m.id,
                "date": m.date,
                "home_name": m.home_team.name,
                "away_name": m.away_team.name,
                "home_goals": m.home_goals,
                "away_goals": m.away_goals,
                "pred_home": m.prediction.predicted_home_goals if m.prediction else None,
                "pred_away": m.prediction.predicted_away_goals if m.prediction else None,
                "manual_forecast": (manual := manual_forecasts.get(_manual_forecast_key(
                    m.date,
                    m.home_team.name,
                    m.away_team.name,
                ))) and manual["forecast"],
                "knockout": bool(manual and manual["knockout"]),
                "retroactive": m.prediction is not None and m.result is None,
                "points": m.result.points if m.result else None,
                "exact": m.result.exact_score if m.result else False,
                "diff_ok": m.result.correct_goal_difference if m.result else False,
                "winner_ok": m.result.correct_winner if m.result else False,
            }
            for m in matches
        ]
        result_dicts = [
            {
                "points": r.points,
                "exact": r.exact_score,
                "diff_ok": r.correct_goal_difference and not r.exact_score,
                "winner_ok": r.correct_winner and not r.correct_goal_difference,
            }
            for r in results
        ]
        return match_dicts, result_dicts


@st.cache_data(ttl=600, show_spinner=False)
def _load_teams() -> list[dict]:
    with get_session() as s:
        teams = s.query(Team).order_by(Team.elo.desc()).all()
        return [
            {
                "name": t.name,
                "elo": t.elo,
                "matches_played": t.matches_played,
                "goals_for": t.goals_for,
                "goals_against": t.goals_against,
            }
            for t in teams
        ]


# ------------------------------------------------------------------ #
# Shared helpers                                                       #
# ------------------------------------------------------------------ #


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _parse_score(text: str | None) -> tuple[int, int] | None:
    """Parse a 'H-A' forecast string into (home, away) goals."""
    if not text:
        return None
    m = re.match(r"^\s*(\d+)\s*[-–]\s*(\d+)\s*$", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _match_points(pred: tuple[int, int], real: tuple[int, int], knockout: bool) -> int:
    """Phase-aware points for a predicted score vs the real result.

    Group phase: 1 (winner) / 2 (goal difference) / 3 (exact).
    Knockout phase: doubled via knockout_multiplier -> 2 / 4 / 6.
    """
    win = _COMP_CFG.get("group_stage_winner_points", 1)
    spread = _COMP_CFG.get("group_stage_goal_difference_points", 2)
    exact = _COMP_CFG.get("group_stage_exact_score_points", 3)
    mult = _COMP_CFG.get("knockout_multiplier", 2) if knockout else 1

    ph, pa = pred
    rh, ra = real
    if ph == rh and pa == ra:
        return exact * mult
    if (ph - pa) == (rh - ra):
        return spread * mult
    if ((ph > pa) - (ph < pa)) == ((rh > ra) - (rh < ra)):
        return win * mult
    return 0


def _points_badge(
    pred: tuple[int, int] | None,
    real: tuple[int, int] | None,
    knockout: bool,
) -> tuple[str, int]:
    """Return (html_badge, points) coloured by outcome tier."""
    if pred is None or real is None or real[0] is None:
        return "<span style='color:#aaa'>—</span>", 0
    pts = _match_points(pred, real, knockout)
    ph, pa = pred
    rh, ra = real
    if ph == rh and pa == ra:
        color = "#2ecc71"
    elif (ph - pa) == (rh - ra):
        color = "#3498db"
    elif ((ph > pa) - (ph < pa)) == ((rh > ra) - (rh < ra)):
        color = "#f39c12"
    else:
        color = "#e74c3c"
    badge = (
        f"<span style='background:{color};color:white;padding:2px 9px;"
        f"border-radius:4px;font-weight:bold'>+{pts}</span>"
    )
    return badge, pts


def _local_dt(dt: datetime) -> datetime:
    return dt + timedelta(hours=UTC_OFFSET_HOURS)


_TEAM_NAME_ALIASES = {
    "bosnia and herzegovina": "bosnia herzegovina",
    "cabo verde": "cape verde islands",
    "cote d ivoire": "ivory coast",
    "cote divoire": "ivory coast",
    "czech republic": "czechia",
    "ir iran": "iran",
    "korea republic": "south korea",
    "turkiye": "turkey",
}


def _normalize_team_name(name: str) -> str:
    ascii_name = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    cleaned = re.sub(r"[^a-z0-9]+", " ", ascii_name).strip()
    compact = " ".join(cleaned.split())
    return _TEAM_NAME_ALIASES.get(compact, compact)


def _manual_forecast_key(dt: datetime, home_name: str, away_name: str) -> tuple:
    local = _local_dt(dt)
    return (
        local.day,
        local.hour,
        local.minute,
        _normalize_team_name(home_name),
        _normalize_team_name(away_name),
    )


_STAGE_RE = re.compile(
    r"^(group stage|round of \d+|quarter[- ]?finals?|semi[- ]?finals?|final|third[- ]place.*)$",
    re.IGNORECASE,
)


def _stage_is_knockout(stage: str | None) -> bool:
    """Group/qualifying matches score single; everything else doubles."""
    return bool(stage) and "group" not in stage.lower()


@st.cache_data(ttl=600, show_spinner=False)
def _load_manual_forecasts() -> dict[tuple, dict]:
    """Parse score.md forecasts keyed by local date/time and teams.

    Returns a dict mapping match key -> {"forecast": str, "knockout": bool}.
    The stage label that precedes each match in score.md drives the knockout
    flag (used for phase-aware, doubled scoring).
    """
    score_path = Path(__file__).parent.parent / "score.md"
    if not score_path.exists():
        return {}

    lines = [
        line.strip()
        for line in score_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    forecasts: dict[tuple, dict] = {}
    date_re = re.compile(r"^(\d{1,2}) Jun \| (\d{1,2}):(\d{2})$")
    score_re = re.compile(r"^\d+\-\d+$")
    current_stage: str | None = None

    for idx, line in enumerate(lines):
        if _STAGE_RE.match(line):
            current_stage = line
            continue

        if line != "Your forecast" or idx < 3 or idx + 3 >= len(lines):
            continue

        forecast = lines[idx + 1]
        if forecast != "---" and not score_re.match(forecast):
            continue

        home_name = lines[idx - 2]
        away_name = lines[idx + 2]

        date_match = None
        for prev in reversed(lines[max(0, idx - 12):idx]):
            date_match = date_re.match(prev)
            if date_match:
                break
        if not date_match:
            continue

        day, hour, minute = map(int, date_match.groups())
        key = (
            day,
            hour,
            minute,
            _normalize_team_name(home_name),
            _normalize_team_name(away_name),
        )
        forecasts[key] = {
            "forecast": forecast,
            "knockout": _stage_is_knockout(current_stage),
        }

    return forecasts


@st.cache_data(ttl=600, show_spinner=False)
def _load_manual_points_total() -> int | None:
    """Return the total points shown in score.md, including group questions."""
    score_path = Path(__file__).parent.parent / "score.md"
    if not score_path.exists():
        return None

    text = score_path.read_text(encoding="utf-8")
    points = [
        int(match.group(1) or match.group(2))
        for match in re.finditer(r"(?:\+(\d+)|(\d+))/\d+", text)
    ]
    return sum(points)


@st.cache_data(ttl=600, show_spinner=False)
def _compute_match(m: dict) -> dict:
    """Run the full model for one match dict and return display-ready values."""
    lam_h, lam_a = expected_goals_blended(
        m["home_elo"], m["away_elo"],
        m["home_gf"], m["home_ga"], m["home_mp"],
        m["away_gf"], m["away_ga"], m["away_mp"],
    )
    matrix = probability_matrix(lam_h, lam_a)
    probs = win_probabilities(m["home_elo"], m["away_elo"])
    top = best_predictions(matrix)
    rec = top[0]
    conf = calculate_confidence(matrix, rec["home"], rec["away"])
    return {
        "probs": probs,
        "top": top,
        "rec": rec,
        "conf": conf,
        "lam_h": lam_h,
        "lam_a": lam_a,
        "matrix": matrix,
    }


def _compute_ev_matrix(matrix: np.ndarray) -> np.ndarray:
    """Build a (n×n) array where ev_mat[ph][pa] = EV of predicting ph-pa."""
    rows, cols = matrix.shape
    ev_mat = np.zeros((rows, cols))
    for ph in range(rows):
        for pa in range(cols):
            ev_mat[ph, pa] = score_ev(matrix, ph, pa)
    return ev_mat


def _refresh_data_pipeline() -> None:
    """Run the data pipeline from the dashboard refresh action."""
    from pipeline.update import run_pipeline

    with st.spinner("Mise à jour des matchs, ratings Elo et prédictions..."):
        run_pipeline()

    # Drop cached DB queries / computations so the new data is picked up.
    st.cache_data.clear()
    st.session_state["last_data_refresh"] = datetime.now().strftime("%d/%m/%Y %H:%M")
    st.success("Données rafraîchies. Le moteur est à jour.")


def _render_app_header() -> None:
    last_refresh = st.session_state.get("last_data_refresh", "En attente de refresh")
    st.markdown(
        f"""
<div class="mc-hero">
  <div class="mc-mon">予</div>
  <div class="mc-hero-copy">
    <div class="mc-kicker">Clean forecast desk</div>
    <h1>Prono d'Anto</h1>
    <p>Des maths, du flair, et une confiance totalement raisonnable dans des scores impossibles.</p>
    <div class="mc-status-row">
      <span class="mc-chip">Model <strong>EV-first</strong></span>
      <span class="mc-chip">League <strong>World Cup 2026</strong></span>
      <span class="mc-chip">Timezone <strong>GMT+2</strong></span>
      <span class="mc-chip">Refresh <strong>{last_refresh}</strong></span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([4, 1.15])
    with left:
        st.markdown(
            """
<div class="mc-strip">
  <strong>勝率</strong>
  <span>Tonight board, prediction history and Elo simulator are ready.</span>
</div>
""",
            unsafe_allow_html=True,
        )
    with right:
        if st.button("Refresh data", help="Lance la pipeline de données complète."):
            try:
                _refresh_data_pipeline()
            except Exception as exc:
                st.error(f"Refresh impossible : {exc}")


# ------------------------------------------------------------------ #
# Matplotlib chart helpers                                             #
# ------------------------------------------------------------------ #

_COLORS = {
    "home": "#1f2a44",
    "draw": "#b8892f",
    "away": "#d83a2e",
    "bar": "#5c7f67",
}


def _display_fig(fig: plt.Figure) -> None:
    fig.patch.set_alpha(0)
    for ax in fig.axes:
        ax.set_facecolor((1, 1, 1, 0))
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _win_prob_bar(probs: dict, home_name: str, away_name: str, key: str = "") -> None:
    """Crisp horizontal stacked bar showing win/draw/loss probabilities (HTML/CSS)."""
    segments = [
        (probs["home"], _COLORS["home"]),
        (probs["draw"], _COLORS["draw"]),
        (probs["away"], _COLORS["away"]),
    ]
    bar = "".join(
        f'<div class="mc-wp-seg" style="width:{value * 100:.2f}%;background:{color}">'
        f'{_pct(value) if value >= 0.07 else ""}</div>'
        for value, color in segments
    )
    legend = (
        f'<span><i style="background:{_COLORS["home"]}"></i>🏠 {home_name}</span>'
        f'<span><i style="background:{_COLORS["draw"]}"></i>🤝 Draw</span>'
        f'<span><i style="background:{_COLORS["away"]}"></i>✈️ {away_name}</span>'
    )
    st.markdown(
        f'<div class="mc-winprob"><div class="mc-wp-track">{bar}</div>'
        f'<div class="mc-wp-legend">{legend}</div></div>',
        unsafe_allow_html=True,
    )


def _top5_ev_bar(top: list[dict], key: str = "") -> None:
    """Horizontal bars of top-5 predicted scores ranked by Expected Value (HTML/CSS)."""
    max_ev = max((s["ev"] for s in top), default=0)
    scale = max_ev if max_ev > 0 else 1.0
    rows = "".join(
        '<div class="mc-ev-row">'
        f'<span class="mc-ev-score">{s["home"]}–{s["away"]}</span>'
        '<div class="mc-ev-bar">'
        f'<div class="mc-ev-fill" style="width:{max(s["ev"] / scale * 100, 1.5) if s["ev"] > 0 else 1.5:.1f}%"></div>'
        '</div>'
        f'<span class="mc-ev-val">EV {s["ev"]:.2f} · {_pct(s["probability"])}</span>'
        '</div>'
        for s in top
    )
    st.markdown(f'<div class="mc-ev">{rows}</div>', unsafe_allow_html=True)


def _render_heatmaps(
    matrix: np.ndarray,
    rec: dict,
    home_name: str,
    away_name: str,
    key_prefix: str = "",
) -> None:
    """Dual heatmap expander: score probability surface + EV landscape."""
    with st.expander("Score probability & EV landscape", expanded=False):
        ev_mat = _compute_ev_matrix(matrix)
        n = matrix.shape[0]
        axis_labels = list(range(n))
        rh, ra = rec["home"], rec["away"]

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Score probability surface**")
            st.caption("Where the game is likely to land. Wide spread = volatile match.")
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(matrix, origin="lower", cmap="YlOrRd", aspect="auto")
            ax.add_patch(plt.Rectangle((rh - 0.5, ra - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=2))
            ax.set_xlabel(f"{home_name} goals")
            ax.set_ylabel(f"{away_name} goals")
            ax.set_xticks(axis_labels)
            ax.set_yticks(axis_labels)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _display_fig(fig)

        with col_b:
            st.markdown("**EV landscape**")
            st.caption("Expected points for each possible prediction. Peak = best score to submit.")
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(ev_mat, origin="lower", cmap="RdYlGn", aspect="auto")
            ax.add_patch(plt.Rectangle((rh - 0.5, ra - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=2))
            ax.set_xlabel(f"{home_name} goals (predicted)")
            ax.set_ylabel(f"{away_name} goals (predicted)")
            ax.set_xticks(axis_labels)
            ax.set_yticks(axis_labels)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _display_fig(fig)


# ------------------------------------------------------------------ #
# Tab 1 — Tonight & Upcoming                                          #
# ------------------------------------------------------------------ #


def _match_card(m: dict, calc: dict) -> None:
    local = _local_dt(m["date"])
    probs = calc["probs"]
    top = calc["top"]
    rec = calc["rec"]
    conf = calc["conf"]
    mid = m["id"]

    with st.container(border=True):
        h1, h2 = st.columns([5, 1])
        h1.markdown(
            f"<p class='mc-match-title'>{m['home_name']}  vs  {m['away_name']}</p>",
            unsafe_allow_html=True,
        )
        h2.markdown(
            f"<p class='mc-kickoff'>Kickoff {local.strftime('%H:%M')}</p>",
            unsafe_allow_html=True,
        )

        _win_prob_bar(probs, m["home_name"], m["away_name"], key=f"prob_{mid}")

        left, right = st.columns([1, 2])

        with left:
            st.markdown("<div class='mc-small-label'>Recommendation</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='mc-score-pick'>{rec['home']} - {rec['away']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='mc-meta-line'>EV <b>{rec['ev']:.2f}</b>"
                f"&nbsp; / &nbsp;Confidence <b>{_pct(conf)}</b></div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"xG: {m['home_name']} {calc['lam_h']:.2f} — "
                f"{m['away_name']} {calc['lam_a']:.2f}"
            )

        with right:
            st.markdown("**Top 5 scores by Expected Value**")
            _top5_ev_bar(top, key=f"top_{mid}")

        _render_heatmaps(
            calc["matrix"], rec,
            m["home_name"], m["away_name"],
            key_prefix=str(mid),
        )


@st.fragment
def render_upcoming() -> None:
    matches = _load_upcoming()

    if not matches:
        st.info("No upcoming matches. Use the Refresh data button to update the pipeline.")
        return

    today_utc = datetime.now(tz=timezone.utc).date()

    groups: dict = defaultdict(list)
    for m in matches:
        groups[m["date"].date()].append(m)

    for day in sorted(groups.keys()):
        if day == today_utc:
            st.markdown(
                "<h2 style='color:#e74c3c;margin-bottom:4px'>🔴 Tonight</h2>",
                unsafe_allow_html=True,
            )
        elif day == today_utc + timedelta(days=1):
            st.markdown("## Tomorrow")
        else:
            st.markdown(f"## {day.strftime('%A %d %B')}")

        for m in groups[day]:
            calc = _compute_match(m)
            _match_card(m, calc)

        st.divider()


# ------------------------------------------------------------------ #
# Tab 2 — Results & Predictions                                       #
# ------------------------------------------------------------------ #


@st.fragment
def render_history() -> None:
    matches, results = _load_history()
    manual_total_pts = _load_manual_points_total()

    scored = [r for r in results]
    n_exact = sum(1 for r in scored if r["exact"])
    n_diff  = sum(1 for r in scored if r["diff_ok"])
    n_win   = sum(1 for r in scored if r["winner_ok"])
    n_wrong = len(scored) - n_exact - n_diff - n_win

    # ── Phase-aware tallies (group 1/2/3, knockout doubled 2/4/6) ────
    # Pre-seed each match's editable forecast from score.md once per session.
    for m in matches:
        skey = f"fc_{m['id']}"
        if skey not in st.session_state:
            st.session_state[skey] = m["manual_forecast"] or ""

    model_total = 0
    your_total = 0
    for m in matches:
        real = (m["home_goals"], m["away_goals"])
        if real[0] is None:
            continue
        if m["pred_home"] is not None:
            model_total += _match_points(
                (m["pred_home"], m["pred_away"]), real, m["knockout"]
            )
        your = _parse_score(st.session_state.get(f"fc_{m['id']}", ""))
        if your is not None:
            your_total += _match_points(your, real, m["knockout"])

    # ── Summary metrics ──────────────────────────────────────────────
    st.markdown("### Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("WC Matches Played", len(matches))
    c2.metric("Model Predictions", len(scored))
    c3.metric("Model Points", model_total)
    c4.metric("Your Points", your_total)
    c5.metric("Exact Scores", n_exact)
    c6.metric("Points Anto (score.md)", manual_total_pts if manual_total_pts is not None else "—")

    if scored:
        # ── Breakdown bar ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 3))
        labels = ["Exact 6pt", "Diff 4pt", "Winner 2pt", "Wrong 0pt"]
        values = [n_exact, n_diff, n_win, n_wrong]
        colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
        ax.bar(labels, values, color=colors)
        for idx, value in enumerate(values):
            ax.text(idx, value + 0.2, str(value), ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Predictions")
        ax.set_title("Prediction breakdown")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _display_fig(fig)

        # ── Cumulative points line ────────────────────────────────────
        timeline_pts: list[dict] = []
        cumulative = 0
        for m_row in reversed(matches):
            if m_row["points"] is not None:
                cumulative += m_row["points"]
                timeline_pts.append({
                    "label": f"{m_row['home_name']} v {m_row['away_name']}",
                    "cumulative": cumulative,
                    "pts": m_row["points"],
                })

        if len(timeline_pts) > 1:
            fig, ax = plt.subplots(figsize=(8, 3.2))
            x = list(range(1, len(timeline_pts) + 1))
            y = [t["cumulative"] for t in timeline_pts]
            ax.plot(x, y, marker="o", color=_COLORS["bar"], linewidth=2)
            for xi, yi, item in zip(x, y, timeline_pts):
                ax.scatter([xi], [yi], s=60, color={0: "#e74c3c", 2: "#f39c12", 4: "#3498db", 6: "#2ecc71"}.get(item["pts"], "#5c7f67"))
            ax.set_title("Cumulative points — WC 2026")
            ax.set_xlabel("Scored prediction #")
            ax.set_ylabel("Total points")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
            _display_fig(fig)

    st.divider()

    # ── Full match timeline ───────────────────────────────────────────
    st.markdown("### WC 2026 — Real vs Model vs You")
    st.caption(
        "Type your forecast (e.g. `2-1`) per match — it pre-fills from `score.md`. "
        "Points are phase-aware: group 1 / 2 / 3, knockout doubles to 2 / 4 / 6 (🏆)."
    )

    widths = [1.1, 2.6, 0.8, 0.9, 0.8, 1.2, 0.8]
    header = st.columns(widths)
    for h, label in zip(
        header,
        ["Date", "Match", "Real", "Model", "Model Pts", "Your Forecast", "Your Pts"],
    ):
        h.markdown(f"**{label}**")
    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

    for m in matches:
        real = (m["home_goals"], m["away_goals"])
        real_str = f"{m['home_goals']}–{m['away_goals']}"
        pred = (m["pred_home"], m["pred_away"]) if m["pred_home"] is not None else None
        pred_str = f"{m['pred_home']}–{m['pred_away']}" if pred else "—"
        model_badge, _ = _points_badge(pred, real, m["knockout"])

        row = st.columns(widths, vertical_alignment="center")
        row[0].markdown(_local_dt(m["date"]).strftime("%d %b %H:%M"))
        tag = " 🏆" if m["knockout"] else ""
        row[1].markdown(f"**{m['home_name']}** vs **{m['away_name']}**{tag}")
        row[2].markdown(f"`{real_str}`")
        row[3].markdown(f"`{pred_str}`")
        row[4].markdown(model_badge, unsafe_allow_html=True)
        row[5].text_input(
            "forecast",
            key=f"fc_{m['id']}",
            label_visibility="collapsed",
            placeholder="2-1",
        )
        your = _parse_score(st.session_state.get(f"fc_{m['id']}", ""))
        your_badge, _ = _points_badge(your, real, m["knockout"])
        row[6].markdown(your_badge, unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Tab 3 — Elo Rankings                                                #
# ------------------------------------------------------------------ #


@st.fragment
def render_elo() -> None:
    teams = _load_teams()

    if not teams:
        st.info("No teams. Run the pipeline first.")
        return

    st.markdown("### Elo Rankings — WC 2026 Teams")

    # ── Top-20 Elo horizontal bar ─────────────────────────────────────
    top20 = teams[:20]
    fig, ax = plt.subplots(figsize=(9, 7))
    ordered = list(reversed(top20))
    values = [t["elo"] for t in ordered]
    names = [t["name"] for t in ordered]
    ax.barh(names, values, color=plt.cm.Blues(np.linspace(0.35, 0.85, len(ordered))))
    for idx, value in enumerate(values):
        ax.text(value + 4, idx, f"{value:.0f}", va="center", ha="left", fontsize=9)
    ax.set_title(f"Top {len(top20)} teams by Elo rating")
    ax.set_xlabel("Elo")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _display_fig(fig)

    st.dataframe(
        [
            {
                "Rank": i + 1,
                "Team": t["name"],
                "Elo": f"{t['elo']:.0f}",
                "Played": t["matches_played"],
                "GF": t["goals_for"],
                "GA": t["goals_against"],
                "GD": t["goals_for"] - t["goals_against"],
            }
            for i, t in enumerate(teams)
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("### Head-to-Head Simulator")

    names = [t["name"] for t in teams]
    c1, c2 = st.columns(2)
    home_name = c1.selectbox("Home Team", names, index=0)
    away_name = c2.selectbox("Away Team", names, index=min(1, len(names) - 1))

    if home_name == away_name:
        st.warning("Select two different teams.")
        return

    ht = next(t for t in teams if t["name"] == home_name)
    at = next(t for t in teams if t["name"] == away_name)

    m = {
        "home_name": ht["name"], "home_elo": ht["elo"],
        "home_gf": ht["goals_for"], "home_ga": ht["goals_against"], "home_mp": ht["matches_played"],
        "away_name": at["name"], "away_elo": at["elo"],
        "away_gf": at["goals_for"], "away_ga": at["goals_against"], "away_mp": at["matches_played"],
    }
    calc = _compute_match(m)
    probs = calc["probs"]
    rec   = calc["rec"]
    conf  = calc["conf"]
    top   = calc["top"]

    _win_prob_bar(probs, home_name, away_name, key="h2h_winprob")

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("<div class='mc-small-label'>Recommendation</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='mc-score-pick'>{rec['home']} - {rec['away']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='mc-meta-line'>EV <b>{rec['ev']:.3f}</b>"
            f"&nbsp; / &nbsp;Confidence <b>{_pct(conf)}</b></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"xG: {home_name} {calc['lam_h']:.2f} — {away_name} {calc['lam_a']:.2f}"
        )
    with rc2:
        st.markdown("**Top 5 Scores by EV**")
        _top5_ev_bar(top, key="h2h_top5")

    _render_heatmaps(calc["matrix"], rec, home_name, away_name, key_prefix="h2h")


# ------------------------------------------------------------------ #
# Layout                                                               #
# ------------------------------------------------------------------ #

_render_app_header()

tab1, tab2, tab3 = st.tabs(
    ["🔴 Tonight & Upcoming", "📊 Results & Predictions", "📈 Elo Rankings"]
)

with tab1:
    render_upcoming()

with tab2:
    render_history()

with tab3:
    render_elo()
