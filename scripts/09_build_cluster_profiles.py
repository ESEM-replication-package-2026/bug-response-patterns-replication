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
from smartshark_roles.role_profiles import run_cluster_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cluster profiles for draft role interpretation.")
    parser.add_argument("--features", required=True, help="Raw actor features CSV or Parquet file.")
    parser.add_argument("--labels", required=True, help="Actor cluster labels CSV.")
    parser.add_argument("--events", required=True, help="Normalized events CSV or Parquet file.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    logger = setup_logging(ROOT / "reports" / "cluster_profiles" / "logs" / f"cluster_profiles_{utc_now_slug()}.log", level=logging.INFO)

    try:
        profiles, paths = run_cluster_profiles(
            features_path=project_path(args.features),
            labels_path=project_path(args.labels),
            events_path=project_path(args.events),
            project_root=ROOT,
            logger=logger,
        )
    except Exception:
        logger.exception("Cluster profile build failed")
        return 1

    print("Cluster profile build succeeded")
    print(f"- clusters_including_noise: {len(profiles)}")
    for row in profiles.sort_values("cluster_label").itertuples():
        print(f"- cluster {row.cluster_label}: actors={row.n_actors} candidate_role={row.candidate_role_name}")
    print("Generated:")
    for path in paths.__dict__.values():
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
