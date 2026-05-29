from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartshark_roles.config import ensure_project_directories, load_config, project_path
from smartshark_roles.io import utc_now_slug
from smartshark_roles.logging_utils import setup_logging
from smartshark_roles.preprocessing import run_preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess actor features for clustering model input.")
    parser.add_argument("--features", required=True, help="Input actor_features_raw CSV or Parquet file.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    logger = setup_logging(ROOT / "reports" / "features" / "logs" / f"feature_preprocessing_{utc_now_slug()}.log", level=logging.INFO)

    try:
        _, metrics, paths = run_preprocessing(
            features_path=project_path(args.features),
            project_root=ROOT,
            logger=logger,
        )
    except Exception:
        logger.exception("Feature preprocessing failed")
        return 1

    print("Feature preprocessing succeeded")
    print(f"- raw_actors: {metrics.raw_actors}")
    print(f"- bot_excluded: {metrics.bot_excluded}")
    print(f"- low_activity_excluded: {metrics.low_activity_excluded}")
    print(f"- bot_and_low_activity: {metrics.bot_and_low_activity}")
    print(f"- main_clustering_actors: {metrics.main_clustering_actors}")
    print(f"- input_feature_count: {metrics.input_feature_count}")
    print(f"- excluded_feature_count: {metrics.excluded_feature_count}")
    print(f"- final_feature_count: {metrics.final_feature_count}")
    print(f"- nan_after_preprocessing: {metrics.nan_after_preprocessing}")
    print("Generated:")
    for path in paths.__dict__.values():
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
