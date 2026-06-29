"""Configuration loader — reads settings.yaml from the project root."""
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from settings.yaml.

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
