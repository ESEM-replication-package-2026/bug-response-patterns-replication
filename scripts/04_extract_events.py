from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartshark_roles.config import ensure_project_directories, load_config, project_path
from smartshark_roles.extraction import run_event_extraction
from smartshark_roles.io import utc_now_slug
from smartshark_roles.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SmartSHARK bug-fixing lifecycle events.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--selected", default="configs/selected_projects.yaml", help="Selected projects YAML.")
    parser.add_argument(
        "--schema-map",
        default="reports/build_events/extraction_schema_map.json",
        help="Preflight extraction schema map JSON.",
    )
    parser.add_argument("--smoke", action="store_true", help="Limit extraction for smoke testing.")
    parser.add_argument("--max-issues-per-project", type=int, default=50, help="Max closed bug issues per project in smoke mode.")
    parser.add_argument("--max-project-level-prs", type=int, default=200, help="Max project-level PRs per project.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    log_path = ROOT / "reports" / "build_events" / "logs" / f"event_extraction_{utc_now_slug()}.log"
    logger = setup_logging(log_path, level=logging.INFO)

    try:
        paths = run_event_extraction(
            config,
            selected_path=project_path(args.selected),
            schema_map_path=project_path(args.schema_map),
            smoke=args.smoke,
            max_issues_per_project=args.max_issues_per_project,
            max_project_level_prs=args.max_project_level_prs,
            logger=logger,
        )
    except Exception:
        logger.exception("Event extraction failed")
        return 1

    logger.info("Generated files: %s", {key: str(value) for key, value in paths.items()})
    print("Generated:")
    for path in paths.values():
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
