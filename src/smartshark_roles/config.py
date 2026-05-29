from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration used by every pipeline step."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    config = deepcopy(loaded)
    config["_config_path"] = str(path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def project_path(value: str | Path, root: Path | None = None) -> Path:
    """Resolve a path relative to the project root."""
    base = root or PROJECT_ROOT
    path = Path(value)
    return path if path.is_absolute() else base / path


def get_config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def ensure_project_directories(config: dict[str, Any]) -> dict[str, Path]:
    """Create the expected output directories and return their resolved paths."""
    root = Path(config["_project_root"])
    paths = {
        "configs": root / "configs",
        "downloads": root / "data" / "downloads",
        "mongodb": root / "data" / "mongodb",
        "raw": root / "data" / "raw",
        "db_audit": root / "data" / "interim" / "db_audit",
        "events": root / "data" / "interim" / "events",
        "features": root / "data" / "interim" / "features",
        "clusters": root / "data" / "processed" / "clusters",
        "role_profiles": root / "data" / "processed" / "role_profiles",
        "docs": root / "docs",
        "reports_data_audit": root / "reports" / "data_audit",
        "reports_build_events": root / "reports" / "build_events",
        "reports_cluster_profiles": root / "reports" / "cluster_profiles",
        "reports_clustering": root / "reports" / "clustering",
        "reports_figures": root / "reports" / "figures",
        "reports_manual_validation": root / "reports" / "manual_validation",
        "reports_results": root / "reports" / "results",
        "scripts": root / "scripts",
        "package": root / "src" / "smartshark_roles",
        "replication_package": root / "artifact" / "replication_package",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
