from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartshark_roles.audit import run_db_audit
from smartshark_roles.config import ensure_project_directories, load_config
from smartshark_roles.io import utc_now_slug
from smartshark_roles.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SmartSHARK MongoDB audit step.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--exact", action="store_true", help="Audit field coverage over every document.")
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Override sample size per collection when not using --exact.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    if args.exact:
        config["audit"]["exact_field_coverage"] = True
    if args.max_documents is not None:
        config["audit"]["max_documents_per_collection"] = args.max_documents

    log_path = ROOT / "reports" / "data_audit" / "logs" / f"db_audit_{utc_now_slug()}.log"
    logger = setup_logging(log_path, level=logging.INFO)
    try:
        paths = run_db_audit(config, logger)
    except Exception:
        logger.exception("DB audit failed")
        return 1

    logger.info("Generated files: %s", {key: str(value) for key, value in paths.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
