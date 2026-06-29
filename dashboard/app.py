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

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy.orm import selectinload

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
# Startup loader — shown once per session                             #
# ------------------------------------------------------------------ #

_LOADER_STEPS = [
    ("🔍", "Analyse des données WC 2026..."),
    ("📊", "Recalcul des ratings Elo..."),
    ("⚽", "Distribution de Poisson en cours..."),
    ("💡", "Calcul des Expected Values..."),
    ("✅", "Prêt !"),
]

_LOADER_CSS = """
<style>
@keyframes spin {
    0%   { transform: rotate(0deg);   }
    100% { transform: rotate(360deg); }
}
@keyframes fadein {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0);   }
}
.cat  { display:inline-block; animation: spin 0.7s linear infinite; font-size:5em; }
.step { animation: fadein 0.4s ease; font-size:1.1em; margin:6px 0; }
.wrap { display:flex; flex-direction:column; align-items:center;
        justify-content:center; height:70vh; text-align:center; }
</style>
"""


def _loader_html(step_icon: str, step_text: str, done: list[str]) -> str:
    done_html = "".join(
        f"<p class='step' style='color:#2ecc71'>✔ {s}</p>" for s in done
    )
    return f"""
{_LOADER_CSS}
<div class='wrap'>
  <div class='cat'>🐱</div>
  <h2 style='margin:20px 0 4px'>World Cup 2026 — Prediction Engine</h2>
  {done_html}
  <p class='step' style='color:#f39c12'>{step_icon} {step_text}</p>
</div>
"""


if "loader_shown" not in st.session_state:
    placeholder = st.empty()
    done: list[str] = []
    delay = 10 / len(_LOADER_STEPS)
    for icon, text in _LOADER_STEPS:
        placeholder.markdown(_loader_html(icon, text, done), unsafe_allow_html=True)
        time.sleep(delay)
        done.append(text)
    placeholder.empty()
    st.session_state["loader_shown"] = True


# ------------------------------------------------------------------ #
# Data loaders — return plain dicts, no SQLAlchemy objects outside    #
# the session context.                                                #
# ------------------------------------------------------------------ #


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
                "manual_forecast": manual_forecasts.get(_manual_forecast_key(
                    m.date,
                    m.home_team.name,
                    m.away_team.name,
                )),
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


def _load_manual_forecasts() -> dict[tuple, str]:
    """Parse score.md forecasts keyed by local date/time and teams."""
    score_path = Path(__file__).parent.parent / "score.md"
    if not score_path.exists():
        return {}

    lines = [
        line.strip()
        for line in score_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    forecasts: dict[tuple, str] = {}
    date_re = re.compile(r"^(\d{1,2}) Jun \| (\d{1,2}):(\d{2})$")
    score_re = re.compile(r"^\d+\-\d+$")

    for idx, line in enumerate(lines):
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
        forecasts[key] = forecast

    return forecasts


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
# Plotly chart helpers                                                 #
# ------------------------------------------------------------------ #

_TRANSPARENT = "rgba(0,0,0,0)"
_COLORS = {
    "home": "#1f2a44",
    "draw": "#b8892f",
    "away": "#d83a2e",
    "bar": "#5c7f67",
}


def _win_prob_bar(probs: dict, home_name: str, away_name: str, key: str = "") -> None:
    """Compact horizontal stacked bar showing win/draw/loss probabilities."""
    fig = go.Figure(go.Bar(
        x=[probs["home"], probs["draw"], probs["away"]],
        y=[f"🏠 {home_name}", "🤝 Draw", f"✈️ {away_name}"],
        orientation="h",
        marker_color=[_COLORS["home"], _COLORS["draw"], _COLORS["away"]],
        text=[_pct(probs["home"]), _pct(probs["draw"]), _pct(probs["away"])],
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        height=120,
        margin=dict(l=0, r=0, t=4, b=4),
        xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
        yaxis=dict(showgrid=False),
        showlegend=False,
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
    )
    st.plotly_chart(fig, use_container_width=True, key=key or None)


def _top5_ev_bar(top: list[dict], key: str = "") -> None:
    """Horizontal bar of top-5 predicted scores ranked by Expected Value."""
    fig = go.Figure(go.Bar(
        x=[s["ev"] for s in top],
        y=[f"{s['home']}–{s['away']}" for s in top],
        orientation="h",
        marker_color=_COLORS["bar"],
        text=[f"EV {s['ev']:.2f}  ·  {_pct(s['probability'])}" for s in top],
        textposition="outside",
        hovertemplate="Score %{y}<br>EV: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=210,
        margin=dict(l=0, r=80, t=4, b=4),
        xaxis_title="Expected Value",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(autorange="reversed", showgrid=False),
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
    )
    st.plotly_chart(fig, use_container_width=True, key=key or None)


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

        col_a, col_b = st.columns(2)

        rh, ra = rec["home"], rec["away"]

        with col_a:
            st.markdown("**Score probability surface**")
            st.caption("Where the game is likely to land. Wide spread = volatile match.")
            fig_prob = go.Figure(go.Heatmap(
                z=matrix,
                x=axis_labels,
                y=axis_labels,
                colorscale="YlOrRd",
                hovertemplate=(
                    f"{home_name}: %{{x}}<br>"
                    f"{away_name}: %{{y}}<br>"
                    "P = %{z:.3f}<extra></extra>"
                ),
            ))
            fig_prob.add_shape(
                type="rect",
                x0=rh - 0.5, x1=rh + 0.5,
                y0=ra - 0.5, y1=ra + 0.5,
                line=dict(color="white", width=2),
            )
            fig_prob.update_layout(
                xaxis_title=f"{home_name} goals",
                yaxis_title=f"{away_name} goals",
                height=340,
                margin=dict(l=50, r=20, t=20, b=50),
                paper_bgcolor=_TRANSPARENT,
                plot_bgcolor=_TRANSPARENT,
            )
            st.plotly_chart(
                fig_prob,
                use_container_width=True,
                key=f"{key_prefix}_prob" or None,
            )

        with col_b:
            st.markdown("**EV landscape**")
            st.caption("Expected points for each possible prediction. Peak = best score to submit.")
            fig_ev = go.Figure(go.Heatmap(
                z=ev_mat,
                x=axis_labels,
                y=axis_labels,
                colorscale="RdYlGn",
                hovertemplate=(
                    f"Predict {home_name}: %{{x}}<br>"
                    f"Predict {away_name}: %{{y}}<br>"
                    "EV = %{z:.3f} pts<extra></extra>"
                ),
            ))
            fig_ev.add_shape(
                type="rect",
                x0=rh - 0.5, x1=rh + 0.5,
                y0=ra - 0.5, y1=ra + 0.5,
                line=dict(color="white", width=2),
            )
            fig_ev.update_layout(
                xaxis_title=f"{home_name} goals (predicted)",
                yaxis_title=f"{away_name} goals (predicted)",
                height=340,
                margin=dict(l=50, r=20, t=20, b=50),
                paper_bgcolor=_TRANSPARENT,
                plot_bgcolor=_TRANSPARENT,
            )
            st.plotly_chart(
                fig_ev,
                use_container_width=True,
                key=f"{key_prefix}_ev" or None,
            )


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


def _pts_badge(
    pts: int | None,
    exact: bool = False,
    diff_ok: bool = False,
    winner_ok: bool = False,
    retroactive: bool = False,
) -> str:
    if pts is None:
        return "<span style='color:#aaa'>—</span>"
    if exact:
        color = "#2ecc71"
    elif diff_ok:
        color = "#3498db"
    elif winner_ok:
        color = "#f39c12"
    else:
        color = "#e74c3c"
    label = f"+{pts}" if pts else "0"
    suffix = "&nbsp;<sup style='font-size:0.7em;color:#aaa'>retro</sup>" if retroactive else ""
    return (
        f"<span style='background:{color};color:white;padding:2px 9px;"
        f"border-radius:4px;font-weight:bold'>{label}</span>{suffix}"
    )


def render_history() -> None:
    matches, results = _load_history()
    manual_total_pts = _load_manual_points_total()

    scored = [r for r in results]
    total_pts = sum(r["points"] for r in scored)
    n_exact = sum(1 for r in scored if r["exact"])
    n_diff  = sum(1 for r in scored if r["diff_ok"])
    n_win   = sum(1 for r in scored if r["winner_ok"])
    n_wrong = len(scored) - n_exact - n_diff - n_win

    # ── Summary metrics ──────────────────────────────────────────────
    st.markdown("### Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("WC Matches Played", len(matches))
    c2.metric("Model Predictions", len(scored))
    c3.metric("Model Points", total_pts)
    c4.metric("Avg pts / match", f"{total_pts/len(scored):.2f}" if scored else "—")
    c5.metric("Exact Scores", n_exact)
    c6.metric("Points Anto :)", manual_total_pts if manual_total_pts is not None else "—")

    if scored:
        # ── Breakdown bar ────────────────────────────────────────────
        fig_bd = go.Figure(go.Bar(
            x=["Exact 6pt", "Diff 4pt", "Winner 2pt", "Wrong 0pt"],
            y=[n_exact, n_diff, n_win, n_wrong],
            marker_color=["#2ecc71", "#3498db", "#f39c12", "#e74c3c"],
            text=[n_exact, n_diff, n_win, n_wrong],
            textposition="outside",
            hovertemplate="%{x}: %{y} predictions<extra></extra>",
        ))
        fig_bd.update_layout(
            title="Prediction breakdown",
            yaxis_title="Predictions",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor=_TRANSPARENT,
            plot_bgcolor=_TRANSPARENT,
        )
        st.plotly_chart(fig_bd, use_container_width=True)

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
            fig_cum = go.Figure(go.Scatter(
                x=list(range(1, len(timeline_pts) + 1)),
                y=[t["cumulative"] for t in timeline_pts],
                mode="lines+markers",
                line=dict(color=_COLORS["bar"], width=2),
                marker=dict(
                    size=10,
                    color=[t["pts"] for t in timeline_pts],
                    colorscale=[
                        [0.0,  "#e74c3c"],
                        [0.33, "#f39c12"],
                        [0.67, "#3498db"],
                        [1.0,  "#2ecc71"],
                    ],
                    cmin=0, cmax=6,
                    showscale=True,
                    colorbar=dict(
                        title="Points",
                        tickvals=[0, 2, 4, 6],
                        ticktext=["0 — Wrong", "2 — Winner", "4 — Diff", "6 — Exact"],
                        len=0.7,
                    ),
                ),
                text=[t["label"] for t in timeline_pts],
                hovertemplate=(
                    "#%{x}: %{text}<br>"
                    "Points: %{marker.color}<br>"
                    "Cumulative: %{y}<extra></extra>"
                ),
            ))
            fig_cum.update_layout(
                title="Cumulative points — WC 2026",
                xaxis_title="Scored prediction #",
                yaxis_title="Total points",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor=_TRANSPARENT,
                plot_bgcolor=_TRANSPARENT,
            )
            st.plotly_chart(fig_cum, use_container_width=True)

    st.divider()

    # ── Full match timeline ───────────────────────────────────────────
    st.markdown("### WC 2026 — All Matches: Real Score vs Prediction")
    st.caption(
        "Model Prediction comes from the app database. My Forecast is parsed from `score.md`."
    )

    cols = st.columns([1, 3, 1, 1, 1, 0.9])
    for h, label in zip(
        cols,
        ["Date", "Match", "Real Score", "Model Prediction", "My Forecast", "Model Pts"],
    ):
        h.markdown(f"**{label}**")
    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

    for m in matches:
        real_str = f"{m['home_goals']}–{m['away_goals']}"
        pred_str = (
            f"{m['pred_home']}–{m['pred_away']}"
            if m["pred_home"] is not None else "—"
        )
        manual_forecast = m["manual_forecast"] or "—"
        is_retro = m["pred_home"] is not None and m["points"] is None
        pts_html = _pts_badge(m["points"], retroactive=is_retro)

        row = st.columns([1, 3, 1, 1, 1, 0.9])
        row[0].markdown(_local_dt(m["date"]).strftime("%d %b %H:%M"))
        row[1].markdown(f"**{m['home_name']}** vs **{m['away_name']}**")
        row[2].markdown(f"`{real_str}`")
        row[3].markdown(f"`{pred_str}`")
        row[4].markdown(f"`{manual_forecast}`")
        row[5].markdown(pts_html, unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Tab 3 — Elo Rankings                                                #
# ------------------------------------------------------------------ #


def render_elo() -> None:
    teams = _load_teams()

    if not teams:
        st.info("No teams. Run the pipeline first.")
        return

    st.markdown("### Elo Rankings — WC 2026 Teams")

    # ── Top-20 Elo horizontal bar ─────────────────────────────────────
    top20 = teams[:20]
    fig_elo = go.Figure(go.Bar(
        x=[t["elo"] for t in reversed(top20)],
        y=[t["name"] for t in reversed(top20)],
        orientation="h",
        marker=dict(
            color=[t["elo"] for t in reversed(top20)],
            colorscale="Blues",
            showscale=False,
        ),
        text=[f"{t['elo']:.0f}" for t in reversed(top20)],
        textposition="outside",
        hovertemplate="%{y}: %{x:.0f} Elo<extra></extra>",
    ))
    fig_elo.update_layout(
        title=f"Top {len(top20)} teams by Elo rating",
        xaxis_title="Elo",
        height=540,
        margin=dict(l=130, r=70, t=40, b=20),
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
    )
    st.plotly_chart(fig_elo, use_container_width=True)

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
