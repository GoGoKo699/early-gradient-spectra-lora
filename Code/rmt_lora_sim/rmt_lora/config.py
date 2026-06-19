from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension: {p.suffix}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping, got {type(data)!r}")
    return data


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Save a config as YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(dict(config), sort_keys=False))


def deep_get(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Get a dotted key from a nested dict."""
    cur: Any = config
    for part in key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur
