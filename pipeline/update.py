"""Daily update pipeline.

Run with:  python pipeline/update.py

Steps:
    1. Download new matches from the API
    2. Store teams and matches in SQLite
    3. Update Elo ratings from finished matches
    4. Generate predictions (xG → Poisson → EV) for upcoming matches
    5. Score past predictions against actual results
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on the path when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.football_data import FootballDataClient, FootballDataError
from config.loader import load_config
from database.db import get_session, init_db
from database.models import Match, Prediction, Result, Team
from models.confidence import calculate_confidence
from models.elo import expected_goals_blended, update_elo
from models.expected_value import recommend
from models.poisson import probability_matrix
from models.strength import weighted_team_stats

# Stages played at a fixed host venue; everything else is neutral turf.
_HOME_VENUE_STAGES = {"GROUP_STAGE", "REGULAR_SEASON", "LEAGUE_STAGE", "PLAY_OFFS"}


def _is_neutral(stage: str | None) -> bool:
    """World Cup / Euro knockout rounds are played on neutral ground."""
    return stage is not None and stage not in _HOME_VENUE_STAGES


def _predict_match(session, match, max_goals: int, cfg: dict) -> tuple[dict, float]:
    """Build the EV recommendation + confidence for one match.

    Uses recency-/competition-weighted team strengths and disables the home
    advantage on neutral-venue (knockout) matches.  Shared by the upcoming and
    retroactive prediction paths.
    """
    home: Team = match.home_team
    away: Team = match.away_team
    as_of = match.date

    gfh, gah, nh = weighted_team_stats(session, home.id, as_of, cfg)
    gfa, gaa, na = weighted_team_stats(session, away.id, as_of, cfg)

    lam_home, lam_away = expected_goals_blended(
        home.elo, away.elo,
        gfh, gah, nh,
        gfa, gaa, na,
        neutral=bool(match.neutral),
    )
    matrix = probability_matrix(lam_home, lam_away, max_goals)
    rec = recommend(matrix)
    conf = calculate_confidence(matrix, rec["home"], rec["away"])
    return rec, conf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Step 1 & 2 — Sync teams and matches from the API                    #
# ------------------------------------------------------------------ #


def _upsert_team(session, team_data: dict, initial_elo: float) -> Team:
    """Insert a team if it doesn't exist, otherwise return the existing record."""
    team = session.query(Team).filter_by(id=team_data["id"]).first()
    if team is None:
        team = Team(
            id=team_data["id"],
            name=team_data["name"],
            elo=initial_elo,
        )
        session.add(team)
        logger.debug("Added team: %s", team.name)
    return team


def _parse_date(date_str: str) -> datetime:
    """Parse ISO-8601 date string to a timezone-naive UTC datetime."""
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    return dt.replace(tzinfo=None)


def sync_competition(competition_code: str, season: int | None = None) -> None:
    """Download and store all matches and teams for a competition."""
    cfg = load_config()
    client = FootballDataClient()
    initial_elo = cfg["elo"]["initial_rating"]

    logger.info("Syncing competition %s (season=%s)", competition_code, season)

    try:
        raw_matches = client.get_competition_matches(competition_code, season=season)
    except FootballDataError as exc:
        logger.error("Failed to fetch matches: %s", exc)
        return

    with get_session() as session:
        skipped = 0
        for raw in raw_matches:
            home_data = raw["homeTeam"]
            away_data = raw["awayTeam"]

            # Skip knockout-stage matches where opponents are not yet determined.
            if not home_data.get("id") or not away_data.get("id"):
                skipped += 1
                continue

            home = _upsert_team(session, home_data, initial_elo)
            away = _upsert_team(session, away_data, initial_elo)
            session.flush()

            stage = raw.get("stage")
            neutral = _is_neutral(stage)

            existing = session.query(Match).filter_by(id=raw["id"]).first()
            if existing:
                # Update status, stage and score if the match has finished.
                existing.status = raw["status"]
                existing.stage = stage
                existing.neutral = neutral
                score = raw.get("score", {})
                full_time = score.get("fullTime", {})
                if full_time.get("home") is not None:
                    existing.home_goals = full_time["home"]
                    existing.away_goals = full_time["away"]
            else:
                score = raw.get("score", {})
                full_time = score.get("fullTime", {})
                match = Match(
                    id=raw["id"],
                    competition=competition_code,
                    date=_parse_date(raw["utcDate"]),
                    home_team_id=home.id,
                    away_team_id=away.id,
                    home_goals=full_time.get("home"),
                    away_goals=full_time.get("away"),
                    status=raw["status"],
                    stage=stage,
                    neutral=neutral,
                )
                session.add(match)

        session.commit()
    logger.info("Sync complete for %s (%d TBD matches skipped)", competition_code, skipped)


# ------------------------------------------------------------------ #
# Step 3 — Elo update from finished matches                           #
# ------------------------------------------------------------------ #


def update_elo_ratings() -> None:
    """Recompute Elo ratings from scratch using all finished matches.

    Resets every team to the initial rating before replaying matches in
    chronological order, so the pipeline is fully idempotent.
    """
    cfg = load_config()["elo"]
    initial_elo = cfg["initial_rating"]
    base_ha = cfg["home_advantage"]
    neutral_ha = cfg.get("home_advantage_neutral", 0)
    mov = cfg.get("mov_enabled", False)

    with get_session() as session:
        # Reset all team stats before replaying history.
        for team in session.query(Team).all():
            team.elo = float(initial_elo)
            team.goals_for = 0
            team.goals_against = 0
            team.matches_played = 0
            team.last_update = None

        matches = (
            session.query(Match)
            .filter(Match.status == "FINISHED")
            .filter(Match.home_goals.isnot(None))
            .order_by(Match.date)
            .all()
        )

        for match in matches:
            home: Team = match.home_team
            away: Team = match.away_team

            match_ha = neutral_ha if match.neutral else base_ha
            result = update_elo(
                home.elo,
                away.elo,
                match.home_goals,
                match.away_goals,
                k_factor=cfg["k_factor"],
                home_advantage=match_ha,
                mov_enabled=mov,
            )

            home.elo = result.new_home_elo
            away.elo = result.new_away_elo
            home.goals_for += match.home_goals
            home.goals_against += match.away_goals
            home.matches_played += 1
            away.goals_for += match.away_goals
            away.goals_against += match.home_goals
            away.matches_played += 1
            now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
            home.last_update = now
            away.last_update = now

        session.commit()
    logger.info("Elo ratings recomputed from %d finished matches", len(matches))


# ------------------------------------------------------------------ #
# Step 4 — Generate predictions for upcoming matches                  #
# ------------------------------------------------------------------ #


def generate_predictions() -> None:
    """Generate EV-maximising predictions for all upcoming matches.

    Handles both SCHEDULED and TIMED statuses (football-data.org uses TIMED
    for confirmed kick-off times and SCHEDULED for unconfirmed ones).
    """
    cfg = load_config()
    max_goals = cfg["poisson"]["max_goals"]

    with get_session() as session:
        matches = (
            session.query(Match)
            .filter(Match.status.in_(["SCHEDULED", "TIMED"]))
            .all()
        )

        count = 0
        for match in matches:
            rec, conf = _predict_match(session, match, max_goals, cfg)

            existing = session.query(Prediction).filter_by(match_id=match.id).first()
            if existing:
                existing.predicted_home_goals = rec["home"]
                existing.predicted_away_goals = rec["away"]
                existing.expected_value = rec["ev"]
                existing.confidence = conf
                existing.created_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
            else:
                pred = Prediction(
                    match_id=match.id,
                    predicted_home_goals=rec["home"],
                    predicted_away_goals=rec["away"],
                    expected_value=rec["ev"],
                    confidence=conf,
                )
                session.add(pred)
            count += 1

        session.commit()
    logger.info("Generated predictions for %d upcoming matches", count)


# ------------------------------------------------------------------ #
# Step 5 — Score past predictions                                     #
# ------------------------------------------------------------------ #


def _compute_points(
    pred_home: int,
    pred_away: int,
    real_home: int,
    real_away: int,
) -> tuple[int, bool, bool, bool]:
    """Return (points, correct_winner, correct_diff, exact)."""
    cfg = load_config()["competition"]

    exact = (pred_home == real_home) and (pred_away == real_away)
    same_diff = (pred_home - pred_away) == (real_home - real_away)
    pred_winner = (pred_home > pred_away) - (pred_home < pred_away)
    real_winner = (real_home > real_away) - (real_home < real_away)
    same_winner = pred_winner == real_winner

    if exact:
        return cfg["exact_score_points"], True, True, True
    if same_diff:
        return cfg["goal_difference_points"], True, True, False
    if same_winner:
        return cfg["winner_points"], True, False, False
    return 0, False, False, False


def score_predictions() -> None:
    """Record points earned for predictions whose matches have now finished."""
    with get_session() as session:
        finished = (
            session.query(Match)
            .filter(Match.status == "FINISHED")
            .filter(Match.home_goals.isnot(None))
            .all()
        )

        count = 0
        for match in finished:
            pred = match.prediction
            if pred is None:
                continue
            if match.result is not None:
                continue  # already scored

            pts, winner, diff, exact = _compute_points(
                pred.predicted_home_goals,
                pred.predicted_away_goals,
                match.home_goals,
                match.away_goals,
            )
            result = Result(
                match_id=match.id,
                real_home_goals=match.home_goals,
                real_away_goals=match.away_goals,
                points=pts,
                correct_winner=winner,
                correct_goal_difference=diff,
                exact_score=exact,
            )
            session.add(result)
            count += 1

        session.commit()
    logger.info("Scored %d new results", count)


# ------------------------------------------------------------------ #
# Step 6 — Retroactive backfill                                       #
# ------------------------------------------------------------------ #


def backfill_predictions() -> None:
    """Generate retroactive predictions for finished matches that have none.

    Uses current Elo + stats (post-group-stage) — predictions are labelled
    retroactive in the dashboard.  Honest caveat: these were not made before
    the match; they reflect what the model would say with today's data.
    """
    cfg = load_config()
    max_goals = cfg["poisson"]["max_goals"]

    with get_session() as session:
        matches = (
            session.query(Match)
            .filter(Match.status == "FINISHED")
            .filter(Match.home_goals.isnot(None))
            .all()
        )

        count = 0
        for match in matches:
            if match.prediction is not None:
                continue  # already has a prediction

            rec, conf = _predict_match(session, match, max_goals, cfg)

            session.add(Prediction(
                match_id=match.id,
                predicted_home_goals=rec["home"],
                predicted_away_goals=rec["away"],
                expected_value=rec["ev"],
                confidence=conf,
            ))
            count += 1

        session.commit()
    logger.info("Backfilled retroactive predictions for %d matches", count)


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #


def run_pipeline() -> None:
    """Execute the full daily update pipeline."""
    cfg = load_config()
    init_db()

    for comp in cfg.get("competitions", []):
        sync_competition(comp["code"])

    update_elo_ratings()
    generate_predictions()
    backfill_predictions()
    score_predictions()
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
