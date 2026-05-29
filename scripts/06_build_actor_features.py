from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartshark_roles.config import ensure_project_directories, load_config, project_path
from smartshark_roles.features import run_actor_feature_build
from smartshark_roles.io import utc_now_slug
from smartshark_roles.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build actor-level feature table from normalized SmartSHARK events.")
    parser.add_argument("--events", required=True, help="Input normalized events CSV or Parquet file.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    logger = setup_logging(ROOT / "reports" / "features" / "logs" / f"actor_features_{utc_now_slug()}.log", level=logging.INFO)

    try:
        _, metrics, paths = run_actor_feature_build(
            events_path=project_path(args.events),
            project_root=ROOT,
            logger=logger,
        )
    except Exception:
        logger.exception("Actor feature build failed")
        return 1

    print("Actor feature build succeeded")
    print(f"- input_events: {metrics.input_events}")
    print(f"- actor_unknown_events: {metrics.actor_unknown_events}")
    print(f"- events_used: {metrics.events_used}")
    print(f"- actor_feature_rows: {metrics.actor_feature_rows}")
    print(f"- unique_actors: {metrics.unique_actors}")
    print(f"- bot_actors: {metrics.bot_actors}")
    print(f"- low_activity_actors: {metrics.low_activity_actors}")
    print(f"- feature_columns_including_actor_id: {metrics.feature_columns_including_actor_id}")
    print(f"- feature_columns_excluding_actor_id: {metrics.feature_columns_excluding_actor_id}")
    print("Generated:")
    for path in paths.__dict__.values():
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
