from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartshark_roles.clustering import run_role_clustering
from smartshark_roles.config import ensure_project_directories, get_config_value, load_config, project_path
from smartshark_roles.io import utc_now_slug
from smartshark_roles.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster actor role features with PCA + HDBSCAN.")
    parser.add_argument("--features", required=True, help="Preprocessed actor feature matrix CSV or Parquet.")
    parser.add_argument("--metadata", required=True, help="Actor metadata CSV aligned to the feature matrix.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    logger = setup_logging(ROOT / "reports" / "clustering" / "logs" / f"role_clustering_{utc_now_slug()}.log", level=logging.INFO)
    random_seed = int(get_config_value(config, "clustering", "random_seed", default=get_config_value(config, "experiment", "random_seed", default=42)))

    try:
        metrics, paths = run_role_clustering(
            features_path=project_path(args.features),
            metadata_path=project_path(args.metadata),
            project_root=ROOT,
            random_seed=random_seed,
            logger=logger,
        )
    except Exception:
        logger.exception("Role clustering failed")
        return 1

    print("Role clustering succeeded")
    print(f"- input_actors: {metrics.input_actors}")
    print(f"- input_features: {metrics.input_features}")
    print(f"- best_pca_components: {metrics.best_pca_components}")
    print(f"- best_min_cluster_size: {metrics.best_min_cluster_size}")
    print(f"- best_min_samples: {metrics.best_min_samples}")
    print(f"- n_clusters: {metrics.n_clusters}")
    print(f"- noise_count: {metrics.noise_count}")
    print(f"- noise_ratio: {metrics.noise_ratio:.6f}")
    print(f"- largest_cluster_size: {metrics.largest_cluster_size}")
    print(f"- largest_cluster_ratio: {metrics.largest_cluster_ratio:.6f}")
    print("Generated:")
    for path in paths.__dict__.values():
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
