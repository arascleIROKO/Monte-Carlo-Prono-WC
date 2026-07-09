"""One-time seed script — imports Anthony's real Phase 1 predictions.

Run with:  python pipeline/seed_phase1.py

Matches the user's actual competition predictions against DB records,
overwrites any retroactive model predictions, and stores the real points.

Scoring used by this competition:
  Group Stage  → exact=3, diff=2, winner=1, wrong=0  (max 3 pts)
  Round of 32  → exact=6, diff=4, winner=2, wrong=0  (max 6 pts)
  Group winner → +10 pts each  (12 groups × 10 = 120 pts, stored as bonus)

Total Phase 1: 80 pts from matches + 120 pts from group winners = 200 pts.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db import get_session, init_db
from database.models import Match, Prediction, Result, Team

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Map names the user typed → names stored by the football-data.org API
_NAME = {
    "Côte d'Ivoire":        "Ivory Coast",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Türkiye":              "Türkiye",      # same in DB
    "Korea Republic":       "Korea Republic",
    "Cape Verde Islands":   "Cape Verde Islands",
}

def _n(name: str) -> str:
    return _NAME.get(name, name)


# (home, away, real_h, real_a, pred_h, pred_a, points)
# pred = None  →  user made no prediction for that match
_DATA = [
    # ── Round of 32 (Jun 28) ─────────────────────────────────────────
    ("South Africa",  "Canada",            0, 1,   0, 1,   6),  # exact R32

    # ── Group Stage (Jun 28) ─────────────────────────────────────────
    ("Algeria",       "Austria",           3, 3,   1, 1,   2),  # correct diff
    ("Jordan",        "Argentina",         1, 3,   0, 3,   1),  # correct winner
    ("Colombia",      "Portugal",          0, 0,   2, 1,   0),
    ("Congo DR",      "Uzbekistan",        3, 1,   1, 1,   0),

    # ── Group Stage (Jun 27) ─────────────────────────────────────────
    ("Croatia",       "Ghana",             2, 1,   2, 1,   3),  # exact
    ("Panama",        "England",           0, 2,   0, 2,   3),  # exact
    ("New Zealand",   "Belgium",           1, 5,   0, 2,   1),
    ("Egypt",         "Iran",              1, 1,   None, None, 0),
    ("Uruguay",       "Spain",             0, 1,   1, 2,   2),  # correct diff
    ("Cape Verde Islands", "Saudi Arabia", 0, 0,   None, None, 0),

    # ── Group Stage (Jun 26) ─────────────────────────────────────────
    ("Norway",        "France",            1, 4,   1, 3,   1),
    ("Senegal",       "Iraq",              5, 0,   2, 0,   1),
    ("Paraguay",      "Australia",         0, 0,   1, 0,   0),
    ("Türkiye",       "United States",     3, 2,   0, 1,   0),
    ("Tunisia",       "Netherlands",       1, 3,   0, 1,   1),
    ("Japan",         "Sweden",            1, 1,   0, 1,   0),

    # ── Group Stage (Jun 25) ─────────────────────────────────────────
    ("Ecuador",       "Germany",           2, 1,   0, 2,   0),
    ("Curacao",       "Côte d'Ivoire",     0, 2,   0, 2,   3),  # exact
    ("South Africa",  "Korea Republic",    1, 0,   None, None, 0),
    ("Czech Republic","Mexico",            0, 3,   0, 2,   1),
    ("Scotland",      "Brazil",            0, 3,   1, 2,   1),
    ("Morocco",       "Haiti",             4, 2,   2, 0,   2),  # correct diff

    # ── Group Stage (Jun 24) ─────────────────────────────────────────
    ("Switzerland",   "Canada",            2, 1,   2, 1,   3),  # exact
    ("Bosnia and Herzegovina", "Qatar",    3, 1,   2, 0,   2),  # correct diff
    ("Colombia",      "Congo DR",          1, 0,   2, 1,   2),  # correct diff
    ("Panama",        "Croatia",           0, 1,   1, 2,   2),  # correct diff

    # ── Group Stage (Jun 23) ─────────────────────────────────────────
    ("England",       "Ghana",             0, 0,   2, 0,   0),
    ("Portugal",      "Uzbekistan",        5, 0,   2, 1,   1),
    ("Jordan",        "Algeria",           1, 2,   0, 1,   2),  # correct diff
    ("Norway",        "Senegal",           3, 2,   1, 0,   2),  # correct diff

    # ── Group Stage (Jun 22) ─────────────────────────────────────────
    ("France",        "Iraq",              3, 0,   2, 0,   1),
    ("Argentina",     "Austria",           2, 0,   3, 0,   1),
    ("New Zealand",   "Egypt",             1, 3,   2, 2,   0),
    ("Uruguay",       "Cape Verde Islands",2, 2,   1, 0,   0),

    # ── Group Stage (Jun 21) ─────────────────────────────────────────
    ("Belgium",       "Iran",              0, 0,   2, 0,   0),
    ("Spain",         "Saudi Arabia",      4, 0,   2, 0,   1),
    ("Tunisia",       "Japan",             0, 4,   1, 1,   0),
    ("Ecuador",       "Curacao",           0, 0,   None, None, 0),

    # ── Group Stage (Jun 20) ─────────────────────────────────────────
    ("Germany",       "Côte d'Ivoire",     2, 1,   2, 1,   3),  # exact
    ("Netherlands",   "Sweden",            5, 1,   1, 1,   0),
    ("Türkiye",       "Paraguay",          0, 1,   None, None, 0),
    ("Brazil",        "Haiti",             3, 0,   3, 0,   3),  # exact
    ("Scotland",      "Morocco",           0, 1,   0, 2,   1),

    # ── Group Stage (Jun 19) ─────────────────────────────────────────
    ("United States", "Australia",         2, 0,   1, 0,   1),
    ("Mexico",        "Korea Republic",    1, 0,   2, 0,   1),
    ("Canada",        "Qatar",             6, 0,   None, None, 0),

    # ── Group Stage (Jun 18) ─────────────────────────────────────────
    ("Switzerland",   "Bosnia and Herzegovina", 4, 1, 2, 0, 1),
    ("Czech Republic","South Africa",      1, 1,   2, 1,   0),
    ("Uzbekistan",    "Colombia",          1, 3,   0, 2,   2),  # correct diff
    ("Ghana",         "Panama",            1, 0,   1, 1,   0),

    # ── Group Stage (Jun 17) ─────────────────────────────────────────
    ("England",       "Croatia",           4, 2,   3, 1,   2),  # correct diff
    ("Portugal",      "Congo DR",          1, 1,   3, 0,   0),
    ("Austria",       "Jordan",            3, 1,   2, 1,   1),
    ("Argentina",     "Algeria",           3, 0,   3, 0,   3),  # exact
    ("Iraq",          "Norway",            1, 4,   0, 2,   1),

    # ── Group Stage (Jun 16) ─────────────────────────────────────────
    ("France",        "Senegal",           3, 1,   3, 0,   1),
    ("Iran",          "New Zealand",       2, 2,   2, 0,   0),
    ("Saudi Arabia",  "Uruguay",           1, 1,   1, 1,   3),  # exact

    # ── Group Stage (Jun 15) ─────────────────────────────────────────
    ("Belgium",       "Egypt",             1, 1,   2, 1,   0),
    ("Spain",         "Cape Verde Islands",0, 0,   3, 0,   0),
    ("Sweden",        "Tunisia",           5, 1,   2, 1,   1),
    ("Côte d'Ivoire", "Ecuador",           1, 0,   2, 1,   2),  # correct diff

    # ── Group Stage (Jun 14) ─────────────────────────────────────────
    ("Netherlands",   "Japan",             2, 2,   2, 0,   0),
    ("Germany",       "Curacao",           7, 1,   2, 0,   1),
    ("Australia",     "Türkiye",           2, 0,   3, 2,   1),
    ("Haiti",         "Scotland",          0, 1,   1, 1,   0),
    ("Brazil",        "Morocco",           1, 1,   2, 1,   0),

    # ── Group Stage (Jun 13) ─────────────────────────────────────────
    ("Qatar",         "Switzerland",       1, 1,   0, 1,   0),
    ("United States", "Paraguay",          4, 1,   2, 0,   1),

    # ── Group Stage (Jun 12) ─────────────────────────────────────────
    ("Canada",        "Bosnia and Herzegovina", 1, 1, 1, 0, 0),
    ("Korea Republic","Czech Republic",    2, 1,   2, 1,   3),  # exact

    # ── Group Stage (Jun 11) ─────────────────────────────────────────
    ("Mexico",        "South Africa",      2, 0,   2, 0,   3),  # exact
]

# Group winner bonus (not tied to match records).
PHASE1_GROUP_BONUS = 120  # 12 groups × 10 pts


def _build_match_index(session) -> dict[tuple, int]:
    """Build a lookup: (home_name, away_name, home_goals, away_goals) → match_id."""
    index: dict[tuple, int] = {}
    for m in session.query(Match).filter(Match.status == "FINISHED").all():
        h = m.home_team.name
        a = m.away_team.name
        key = (h, a, m.home_goals, m.away_goals)
        index[key] = m.id
    return index


def seed() -> None:
    init_db()

    with get_session() as session:
        # Pre-load all team names for lookup.
        team_by_name = {t.name: t.id for t in session.query(Team).all()}
        match_index = _build_match_index(session)

        inserted = updated = skipped = no_pred = 0

        for row in _DATA:
            home_raw, away_raw, rh, ra, ph, pa, pts = row
            home = _n(home_raw)
            away = _n(away_raw)

            # Find match by normalised names + real score.
            match_id = match_index.get((home, away, rh, ra))
            if match_id is None:
                # Try swapped — API may list teams in opposite order.
                match_id = match_index.get((away, home, ra, rh))
                if match_id is not None:
                    ph, pa = pa, ph  # swap prediction to match actual home/away

            if match_id is None:
                logger.warning("Match not found: %s vs %s (%d-%d)", home, away, rh, ra)
                skipped += 1
                continue

            # ── Prediction ───────────────────────────────────────────
            existing_pred = session.query(Prediction).filter_by(match_id=match_id).first()

            if ph is None:
                # User had no prediction — remove any retroactive model prediction.
                if existing_pred:
                    session.delete(existing_pred)
                no_pred += 1
            else:
                if existing_pred:
                    existing_pred.predicted_home_goals = ph
                    existing_pred.predicted_away_goals = pa
                    existing_pred.expected_value = float(pts)
                    existing_pred.confidence = 1.0
                    updated += 1
                else:
                    session.add(Prediction(
                        match_id=match_id,
                        predicted_home_goals=ph,
                        predicted_away_goals=pa,
                        expected_value=float(pts),
                        confidence=1.0,
                    ))
                    inserted += 1

                # ── Result ───────────────────────────────────────────
                exact  = (ph == rh) and (pa == ra)
                diff   = (ph - pa) == (rh - ra)
                p_win  = (ph > pa) - (ph < pa)
                r_win  = (rh > ra) - (rh < ra)
                winner = p_win == r_win

                existing_res = session.query(Result).filter_by(match_id=match_id).first()
                if existing_res:
                    existing_res.real_home_goals = rh
                    existing_res.real_away_goals = ra
                    existing_res.points = pts
                    existing_res.correct_winner = winner
                    existing_res.correct_goal_difference = diff
                    existing_res.exact_score = exact
                else:
                    session.add(Result(
                        match_id=match_id,
                        real_home_goals=rh,
                        real_away_goals=ra,
                        points=pts,
                        correct_winner=winner,
                        correct_goal_difference=diff,
                        exact_score=exact,
                    ))

        session.commit()

    total_match_pts = sum(r[6] for r in _DATA if r[4] is not None)
    logger.info(
        "Done — inserted=%d updated=%d no_pred=%d skipped=%d",
        inserted, updated, no_pred, skipped,
    )
    logger.info(
        "Match points: %d | Group bonus: %d | Total Phase 1: %d",
        total_match_pts, PHASE1_GROUP_BONUS, total_match_pts + PHASE1_GROUP_BONUS,
    )


if __name__ == "__main__":
    seed()
