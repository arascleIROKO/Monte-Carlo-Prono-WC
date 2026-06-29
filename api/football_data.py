"""Client for the football-data.org API v4."""
import logging
import time
from typing import Any, Optional

import requests

from config.loader import load_config

logger = logging.getLogger(__name__)


class FootballDataError(Exception):
    """Raised when the API returns an error."""


class FootballDataClient:
    """HTTP client for football-data.org with rate limiting and error handling.

    All data is returned as raw dicts — callers are responsible for persistence.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        config = load_config()
        self._api_key = api_key or config["api"]["key"]
        self._base_url = config["api"]["base_url"]
        self._rate_limit_seconds = config["api"]["rate_limit_seconds"]
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _headers(self) -> dict[str, str]:
        return {"X-Auth-Token": self._api_key}

    def _enforce_rate_limit(self) -> None:
        """Sleep if necessary to stay within the API rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_seconds:
            time.sleep(self._rate_limit_seconds - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Make a GET request and return the parsed JSON body."""
        self._enforce_rate_limit()
        url = f"{self._base_url}/{endpoint}"
        logger.debug("GET %s params=%s", url, params)

        try:
            response = requests.get(
                url, headers=self._headers(), params=params, timeout=30
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise FootballDataError(
                f"API request failed: {exc.response.status_code} {exc.response.text}"
            ) from exc
        except requests.RequestException as exc:
            raise FootballDataError(f"Network error: {exc}") from exc

        return response.json()  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_competition_matches(
        self,
        competition_code: str,
        season: Optional[int] = None,
        status: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Fetch all matches for a competition.

        Args:
            competition_code: E.g. "WC", "EC", "PL".
            season: Four-digit year (e.g. 2026). Defaults to current season.
            status: Filter by match status: "SCHEDULED", "FINISHED", etc.
        """
        params: dict[str, Any] = {}
        if season:
            params["season"] = season
        if status:
            params["status"] = status

        data = self._get(f"competitions/{competition_code}/matches", params)
        matches: list[dict[str, Any]] = data.get("matches", [])
        logger.info("Fetched %d matches for %s", len(matches), competition_code)
        return matches

    def get_competition_teams(
        self,
        competition_code: str,
        season: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Fetch all teams registered in a competition."""
        params: dict[str, Any] = {}
        if season:
            params["season"] = season

        data = self._get(f"competitions/{competition_code}/teams", params)
        teams: list[dict[str, Any]] = data.get("teams", [])
        logger.info("Fetched %d teams for %s", len(teams), competition_code)
        return teams

    def get_team_matches(
        self,
        team_id: int,
        limit: int = 20,
        status: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Fetch recent matches for a specific team."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status

        data = self._get(f"teams/{team_id}/matches", params)
        return data.get("matches", [])

    def get_match(self, match_id: int) -> dict[str, Any]:
        """Fetch a single match by ID."""
        return self._get(f"matches/{match_id}")
