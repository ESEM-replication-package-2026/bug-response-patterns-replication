"""Extended k sensitivity analysis for RQ3 role-composition clustering.

This script does not modify existing RQ3 outputs. It reads the already-built
bug-level role-composition vectors and evaluates k-means for a wider k range.

Claim type: model-selection sensitivity for a descriptive taxonomy.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTORS = (
    ROOT
    / "reports"
    / "results"
    / "rq3_role_composition_taxonomy"
    / "issue_role_composition_vectors.csv"
)
DEFAULT_OUTPUT = ROOT / "reports" / "results" / "rq3_role_composition_taxonomy"
DEFAULT_FIGURES = ROOT / "reports" / "figures" / "rq3_role_composition_taxonomy"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_FIGURES / "matplotlib_cache"))

ROLE_FEATURES = [
    "n_fixer",
    "n_issue_side_participant",
    "n_pr_integrator",
    "n_review_integration_hybrid",
    "n_review_focused_participant",
    "n_issue_side_boundary_participant",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-cluster-size-threshold", type=int, default=100)
    parser.add_argument("--max-largest-cluster-ratio", type=float, default=0.50)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def load_vectors(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in ROLE_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing role feature columns: {missing}")
    return df


def evaluate_k_range(
    vectors: pd.DataFrame,
    k_min: int,
    k_max: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    x = np.log1p(vectors[ROLE_FEATURES].astype(float).to_numpy())
    x_scaled = StandardScaler().fit_transform(x)
    unique_rows = len(np.unique(x_scaled, axis=0))
    max_k = min(k_max, len(vectors) - 1, unique_rows - 1)
    if max_k < k_min:
        raise ValueError(f"Cannot evaluate k range {k_min}-{k_max}; unique rows={unique_rows}")

    metric_rows: list[dict[str, object]] = []
    size_rows: list[dict[str, object]] = []

    for k in range(k_min, max_k + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=50)
        labels = model.fit_predict(x_scaled)
        counts = pd.Series(labels).value_counts().sort_index()
        largest = int(counts.max())
        smallest = int(counts.min())

        for cluster_label, n_issues in counts.items():
            size_rows.append(
                {
                    "k": k,
                    "cluster_label": int(cluster_label),
                    "n_issues": int(n_issues),
                    "issue_ratio": float(n_issues / len(vectors)),
                }
            )

        metric_rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(x_scaled, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(x_scaled, labels)),
                "davies_bouldin": float(davies_bouldin_score(x_scaled, labels)),
                "smallest_cluster_size": smallest,
                "largest_cluster_size": largest,
                "largest_cluster_ratio": float(largest / len(vectors)),
                "mean_cluster_size": float(counts.mean()),
                "median_cluster_size": float(counts.median()),
                "n_clusters_under_50": int((counts < 50).sum()),
                "n_clusters_under_100": int((counts < 100).sum()),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics["rank_silhouette"] = metrics["silhouette"].rank(ascending=False, method="min")
    metrics["rank_davies_bouldin"] = metrics["davies_bouldin"].rank(ascending=True, method="min")
    metrics["rank_calinski_harabasz"] = metrics["calinski_harabasz"].rank(ascending=False, method="min")
    metrics["rank_largest_balance"] = metrics["largest_cluster_ratio"].rank(ascending=True, method="min")
    metrics["rank_smallest_cluster"] = metrics["smallest_cluster_size"].rank(ascending=False, method="min")
    metrics["combined_rank_score"] = metrics[
        [
            "rank_silhouette",
            "rank_davies_bouldin",
            "rank_calinski_harabasz",
            "rank_largest_balance",
            "rank_smallest_cluster",
        ]
    ].sum(axis=1)
    return metrics, pd.DataFrame(size_rows)


def choose_recommended_k(
    metrics: pd.DataFrame,
    min_cluster_size_threshold: int,
    max_largest_cluster_ratio: float,
) -> tuple[int, str]:
    eligible = metrics.loc[
        metrics["smallest_cluster_size"].ge(min_cluster_size_threshold)
        & metrics["largest_cluster_ratio"].le(max_largest_cluster_ratio)
    ].copy()
    if eligible.empty:
        selected = int(metrics.sort_values(["combined_rank_score", "k"]).iloc[0]["k"])
        return (
            selected,
            "No k satisfied the size guardrails; selected the lowest combined rank score.",
        )

    selected = int(eligible.sort_values(["combined_rank_score", "k"]).iloc[0]["k"])
    return (
        selected,
        (
            f"Selected the best combined rank score among k values with smallest cluster "
            f">= {min_cluster_size_threshold} and largest cluster ratio <= "
            f"{max_largest_cluster_ratio:.2f}."
        ),
    )


def write_summary(
    metrics: pd.DataFrame,
    selected_k: int,
    selection_reason: str,
    output_path: Path,
    min_cluster_size_threshold: int,
    max_largest_cluster_ratio: float,
) -> None:
    best_sil = int(metrics.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    best_db = int(metrics.sort_values(["davies_bouldin", "k"], ascending=[True, True]).iloc[0]["k"])
    best_ch = int(metrics.sort_values(["calinski_harabasz", "k"], ascending=[False, True]).iloc[0]["k"])
    current_k6 = metrics.loc[metrics["k"].eq(6)].iloc[0] if metrics["k"].eq(6).any() else None
    selected_row = metrics.loc[metrics["k"].eq(selected_k)].iloc[0]

    lines = [
        "# Extended k Sensitivity for RQ3 Role-Composition Clustering",
        "",
        "## Purpose",
        "",
        (
            "This report extends the original RQ3 k-means model selection beyond "
            "`k=4--6`. It uses the same bug-level role-count vectors, the same "
            "`log1p` transformation, and standardization before k-means."
        ),
        "",
        "## Recommendation",
        "",
        f"- Recommended k under the extended diagnostic: **{selected_k}**.",
        f"- Selection rule: {selection_reason}",
        (
            f"- Guardrails: smallest cluster >= {min_cluster_size_threshold}; "
            f"largest cluster ratio <= {max_largest_cluster_ratio:.2f}."
        ),
        "",
        "## Best k by Individual Criteria",
        "",
        f"- Highest silhouette: k={best_sil}.",
        f"- Lowest Davies-Bouldin: k={best_db}.",
        f"- Highest Calinski-Harabasz: k={best_ch}.",
        "",
        "## Selected-k Metrics",
        "",
        (
            f"- k={selected_k}: silhouette={selected_row['silhouette']:.3f}, "
            f"Davies-Bouldin={selected_row['davies_bouldin']:.3f}, "
            f"Calinski-Harabasz={selected_row['calinski_harabasz']:.1f}, "
            f"smallest cluster={int(selected_row['smallest_cluster_size'])}, "
            f"largest cluster ratio={selected_row['largest_cluster_ratio']:.3f}."
        ),
        "",
    ]

    if current_k6 is not None:
        lines.extend(
            [
                "## Comparison with Original k=6",
                "",
                (
                    f"- Original k=6: silhouette={current_k6['silhouette']:.3f}, "
                    f"Davies-Bouldin={current_k6['davies_bouldin']:.3f}, "
                    f"Calinski-Harabasz={current_k6['calinski_harabasz']:.1f}, "
                    f"smallest cluster={int(current_k6['smallest_cluster_size'])}, "
                    f"largest cluster ratio={current_k6['largest_cluster_ratio']:.3f}."
                ),
                (
                    "- Interpretation: k=6 remains a concise, interpretable coarse taxonomy, "
                    "but the extended internal metrics favor a finer taxonomy."
                ),
                "",
            ]
        )

    table_cols = [
        "k",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "smallest_cluster_size",
        "largest_cluster_ratio",
        "n_clusters_under_100",
        "combined_rank_score",
    ]
    lines.extend(
        [
            "## Full k Table",
            "",
            "| k | silhouette | Davies-Bouldin | Calinski-Harabasz | smallest cluster | largest cluster ratio | clusters <100 | combined rank |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics[table_cols].itertuples(index=False):
        lines.append(
            f"| {int(row.k)} | {row.silhouette:.3f} | {row.davies_bouldin:.3f} | "
            f"{row.calinski_harabasz:.1f} | {int(row.smallest_cluster_size)} | "
            f"{row.largest_cluster_ratio:.3f} | {int(row.n_clusters_under_100)} | "
            f"{row.combined_rank_score:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Paper-use Note",
            "",
            (
                "Do not claim that any k reveals ground-truth bug types. The result is a "
                "descriptive clustering of role-composition vectors. If the paper keeps k=6, "
                "it should explicitly frame k=6 as a concise coarse taxonomy rather than the "
                "best-supported k under the extended sensitivity analysis."
            ),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(metrics: pd.DataFrame, selected_k: int, figure_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))
    axes = axes.ravel()
    plots = [
        ("silhouette", "Silhouette", "higher is better"),
        ("davies_bouldin", "Davies-Bouldin", "lower is better"),
        ("largest_cluster_ratio", "Largest cluster ratio", "lower is better"),
        ("smallest_cluster_size", "Smallest cluster size", "higher is better"),
    ]
    for ax, (col, title, subtitle) in zip(axes, plots):
        ax.plot(metrics["k"], metrics[col], marker="o")
        ax.axvline(selected_k, linestyle="--", linewidth=1)
        ax.set_xlabel("k")
        ax.set_title(f"{title}\n({subtitle})")
        ax.grid(alpha=0.25)
    fig.suptitle("Extended k sensitivity for RQ3 role-composition clustering")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def main() -> None:
    setup_logging()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    vectors = load_vectors(args.vectors)
    logging.info("Loaded %s issue role-composition vectors", len(vectors))
    metrics, sizes = evaluate_k_range(vectors, args.k_min, args.k_max, args.seed)
    selected_k, reason = choose_recommended_k(
        metrics,
        args.min_cluster_size_threshold,
        args.max_largest_cluster_ratio,
    )

    metrics["recommended"] = metrics["k"].eq(selected_k)
    metrics_path = args.output_dir / "kmeans_k_sensitivity_extended.csv"
    sizes_path = args.output_dir / "kmeans_k_sensitivity_cluster_sizes.csv"
    summary_path = args.output_dir / "kmeans_k_sensitivity_summary.md"
    figure_path = args.figure_dir / "figure_rq3_k_sensitivity_extended.png"

    metrics.to_csv(metrics_path, index=False)
    sizes.to_csv(sizes_path, index=False)
    write_summary(
        metrics,
        selected_k,
        reason,
        summary_path,
        args.min_cluster_size_threshold,
        args.max_largest_cluster_ratio,
    )
    write_figure(metrics, selected_k, figure_path)

    logging.info("Recommended k: %s", selected_k)
    logging.info("Wrote %s", metrics_path)
    logging.info("Wrote %s", summary_path)
    logging.info("Wrote %s", figure_path)


if __name__ == "__main__":
    main()
