"""Configuration loader — reads settings.yaml from the project root."""
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = str(Path(__file__).parent.parent / "config" / "settings.yaml")


@lru_cache(maxsize=1)
def _read_yaml(path: str) -> dict[str, Any]:
    """Read and parse settings.yaml once; subsequent calls return the cached result."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Return the merged configuration (YAML + env-var overrides).

    The YAML file is read from disk only on the first call; all subsequent
    calls return the cached dict (O(1)).  Pass a custom path only in tests.
    """
    path = config_path or _CONFIG_PATH
    config: dict[str, Any] = _read_yaml(path)

    env_key = os.environ.get("FOOTBALL_API_KEY")
    if env_key:
        config["api"]["key"] = env_key

    return config
