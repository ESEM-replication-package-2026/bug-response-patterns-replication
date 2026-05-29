from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartshark_roles.anonymize import run_actor_normalization
from smartshark_roles.config import ensure_project_directories, load_config, project_path
from smartshark_roles.io import utc_now_slug
from smartshark_roles.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize SmartSHARK event actors and flag automation actors.")
    parser.add_argument("--events", required=True, help="Input events CSV or Parquet file.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    log_path = ROOT / "reports" / "build_events" / "logs" / f"actor_normalization_{utc_now_slug()}.log"
    logger = setup_logging(log_path, level=logging.INFO)

    try:
        metrics, paths = run_actor_normalization(
            events_path=project_path(args.events),
            project_root=ROOT,
            logger=logger,
        )
    except Exception:
        logger.exception("Actor normalization failed")
        return 1

    logger.info("Actor normalization succeeded")
    print("Actor normalization succeeded")
    print(f"- input_events: {metrics.input_events}")
    print(f"- output_events: {metrics.output_events}")
    print(f"- missing_actor_id_raw: {metrics.missing_actor_id_raw}")
    print(f"- unique_raw_actors: {metrics.unique_raw_actors}")
    print(f"- unique_normalized_actors: {metrics.unique_normalized_actors}")
    print(f"- bot_actors: {metrics.bot_actors}")
    print(f"- bot_events: {metrics.bot_events}")
    print(f"- bot_event_rate: {metrics.bot_event_rate:.6f}")
    print("Generated:")
    for path in paths.__dict__.values():
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
