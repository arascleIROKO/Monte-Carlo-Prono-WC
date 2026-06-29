"""Configuration loader — reads settings.yaml from the project root."""
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=None)
def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from settings.yaml.

    The result is memoized: settings.yaml is static during a run, and this
    function is called in hot loops (e.g. the EV engine), so parsing the YAML
    once avoids thousands of redundant reads. Callers must treat the returned
    dict as read-only.

    Overrides api.key with the FOOTBALL_API_KEY environment variable if set.
    """
    if config_path is None:
        root = Path(__file__).parent.parent
        config_path = str(root / "config" / "settings.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    env_key = os.environ.get("FOOTBALL_API_KEY")
    if env_key:
        config["api"]["key"] = env_key

    return config
