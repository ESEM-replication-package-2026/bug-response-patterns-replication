from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartshark_roles.config import ensure_project_directories, load_config
from smartshark_roles.io import utc_now_slug
from smartshark_roles.logging_utils import setup_logging
from smartshark_roles.project_availability import run_project_availability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize project-level data availability.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument(
        "--include-two-hop",
        action="store_true",
        help="Also compute issue_event and pull_request_event project counts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    if args.include_two_hop:
        config["project_availability"]["include_two_hop_counts"] = True

    log_path = ROOT / "reports" / "data_audit" / "logs" / f"project_availability_{utc_now_slug()}.log"
    logger = setup_logging(log_path, level=logging.INFO)
    try:
        paths = run_project_availability(config, logger)
    except Exception:
        logger.exception("Project availability failed")
        return 1

    logger.info("Generated files: %s", {key: str(value) for key, value in paths.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
