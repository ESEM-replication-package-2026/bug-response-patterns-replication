from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartshark_roles.config import ensure_project_directories, get_config_value, load_config, project_path
from smartshark_roles.io import utc_now_slug
from smartshark_roles.logging_utils import setup_logging
from smartshark_roles.validation import StabilityConfig, run_stability_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bootstrap stability analysis and method comparison for role clusters.")
    parser.add_argument("--features", required=True, help="Preprocessed actor feature matrix CSV or Parquet.")
    parser.add_argument("--labels", required=True, help="Actor cluster labels CSV.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--n-bootstrap", type=int, default=30, help="Number of bootstrap repetitions.")
    parser.add_argument("--sample-fraction", type=float, default=0.80, help="Actor sampling fraction per bootstrap.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_yaml = load_config(args.config)
    ensure_project_directories(config_yaml)
    logger = setup_logging(ROOT / "reports" / "results" / "logs" / f"stability_analysis_{utc_now_slug()}.log", level=logging.INFO)
    random_seed = int(get_config_value(config_yaml, "clustering", "random_seed", default=get_config_value(config_yaml, "experiment", "random_seed", default=42)))
    stability_config = StabilityConfig(
        n_bootstrap=args.n_bootstrap,
        sample_fraction=args.sample_fraction,
        random_seed=random_seed,
        pca_components=12,
        min_cluster_size=20,
        min_samples=10,
    )

    try:
        bootstrap, cluster_summary, noise_summary, methods, paths = run_stability_analysis(
            features_path=project_path(args.features),
            labels_path=project_path(args.labels),
            project_root=ROOT,
            config=stability_config,
            logger=logger,
        )
    except Exception:
        logger.exception("Stability analysis failed")
        return 1

    print("Stability analysis succeeded")
    print(f"- bootstrap_runs: {len(bootstrap)}")
    print(f"- ari_include_noise_mean: {bootstrap['ari_include_noise'].mean():.6f}")
    print(f"- nmi_include_noise_mean: {bootstrap['nmi_include_noise'].mean():.6f}")
    print(f"- v_measure_include_noise_mean: {bootstrap['v_measure_include_noise'].mean():.6f}")
    print(f"- ari_exclude_noise_mean: {bootstrap['ari_exclude_noise'].mean():.6f}")
    print(f"- noise_ratio_mean: {bootstrap['noise_ratio'].mean():.6f}")
    print(f"- n_clusters_mean: {bootstrap['n_clusters'].mean():.6f}")
    print("Generated:")
    for path in paths.__dict__.values():
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
