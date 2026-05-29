from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from smartshark_roles.io import dataframe_to_markdown, write_dataframe, write_text


@dataclass(frozen=True)
class ClusteringPaths:
    labels_csv: Path
    labels_parquet: Path
    parameter_results_csv: Path
    cluster_size_table_csv: Path
    pca_explained_variance_csv: Path
    umap_png: Path
    summary_md: Path


@dataclass(frozen=True)
class ClusteringMetrics:
    input_actors: int
    input_features: int
    best_pca_components: int
    best_min_cluster_size: int
    best_min_samples: int
    n_clusters: int
    noise_count: int
    noise_ratio: float
    largest_cluster_size: int
    largest_cluster_ratio: float


def read_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature matrix file type: {path.suffix}")


def read_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Actor metadata does not exist: {path}")
    return pd.read_csv(path)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def validate_inputs(features: pd.DataFrame, metadata: pd.DataFrame) -> None:
    if len(features) != len(metadata):
        raise ValueError(f"Feature and metadata row counts differ: features={len(features)} metadata={len(metadata)}")
    if features.isna().sum().sum():
        raise ValueError("Feature matrix contains NaN values.")
    if "actor_id" not in metadata.columns:
        raise KeyError("Metadata must contain actor_id")
    if "is_bot" in metadata.columns and as_bool(metadata["is_bot"]).any():
        raise ValueError("Bot actors are present in clustering metadata.")
    if "is_low_activity" in metadata.columns and as_bool(metadata["is_low_activity"]).any():
        raise ValueError("Low activity actors are present in clustering metadata.")


def pca_explained_variance(features: pd.DataFrame, random_seed: int) -> tuple[PCA, pd.DataFrame]:
    max_components = min(len(features), len(features.columns))
    pca = PCA(n_components=max_components, random_state=random_seed)
    pca.fit(features)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    rows = [
        {
            "component": index + 1,
            "explained_variance_ratio": ratio,
            "cumulative_explained_variance": cumulative[index],
        }
        for index, ratio in enumerate(pca.explained_variance_ratio_)
    ]
    return pca, pd.DataFrame(rows)


def pca_candidates(explained: pd.DataFrame, n_features: int) -> list[int]:
    n90_values = explained.loc[explained["cumulative_explained_variance"] >= 0.90, "component"]
    n90 = int(n90_values.iloc[0]) if not n90_values.empty else n_features
    candidates = [10, 15, 20, n90]
    return sorted({min(max(1, value), n_features) for value in candidates})


def hdbscan_min_cluster_sizes(n_actors: int) -> list[int]:
    return sorted({20, 50, 100, max(20, int(0.01 * n_actors))})


def evaluate_labels(labels: np.ndarray) -> dict[str, Any]:
    n_actors = len(labels)
    noise_count = int((labels == -1).sum())
    cluster_labels = labels[labels != -1]
    if len(cluster_labels):
        counts = pd.Series(cluster_labels).value_counts().sort_index()
        n_clusters = int(len(counts))
        largest_cluster_size = int(counts.max())
        mean_cluster_size = float(counts.mean())
        median_cluster_size = float(counts.median())
    else:
        counts = pd.Series(dtype=int)
        n_clusters = 0
        largest_cluster_size = 0
        mean_cluster_size = 0.0
        median_cluster_size = 0.0

    return {
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_ratio": noise_count / n_actors if n_actors else 0.0,
        "largest_cluster_size": largest_cluster_size,
        "largest_cluster_ratio": largest_cluster_size / n_actors if n_actors else 0.0,
        "mean_cluster_size": mean_cluster_size,
        "median_cluster_size": median_cluster_size,
    }


def selection_score(metrics: dict[str, Any]) -> float:
    n_clusters = metrics["n_clusters"]
    noise_ratio = metrics["noise_ratio"]
    largest_ratio = metrics["largest_cluster_ratio"]
    if n_clusters == 0:
        return -10_000.0

    cluster_penalty = 0.0 if 3 <= n_clusters <= 12 else min(abs(n_clusters - 3), abs(n_clusters - 12))
    noise_penalty = max(0.0, noise_ratio - 0.45) * 8.0 + noise_ratio * 2.0
    largest_penalty = max(0.0, largest_ratio - 0.60) * 8.0 + largest_ratio
    target_cluster_penalty = abs(n_clusters - 6) * 0.2
    ideal_bonus = 5.0 if 3 <= n_clusters <= 12 and noise_ratio <= 0.45 and largest_ratio <= 0.60 else 0.0
    return ideal_bonus - cluster_penalty * 2.0 - noise_penalty - largest_penalty - target_cluster_penalty


def run_parameter_grid(features: pd.DataFrame, pca_component_values: list[int], random_seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    pca_cache: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    min_cluster_sizes = hdbscan_min_cluster_sizes(len(features))
    min_samples_values = [5, 10, 20]

    for n_components in pca_component_values:
        pca_model = PCA(n_components=n_components, random_state=random_seed)
        pca_cache[n_components] = pca_model.fit_transform(features)

        for min_cluster_size in min_cluster_sizes:
            for min_samples in min_samples_values:
                model = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric="euclidean",
                    cluster_selection_method="eom",
                )
                labels = model.fit_predict(pca_cache[n_components])
                metrics = evaluate_labels(labels)
                score = selection_score(metrics)
                rows.append(
                    {
                        "pca_components": n_components,
                        "min_cluster_size": min_cluster_size,
                        "min_samples": min_samples,
                        "metric": "euclidean",
                        **metrics,
                        "selection_score": score,
                    }
                )

    results = pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    if results.empty:
        raise RuntimeError("No HDBSCAN parameter results were produced.")
    best_row = results.iloc[0].to_dict()
    results["selected"] = False
    results.loc[0, "selected"] = True
    best_row["pca_scores"] = pca_cache[int(best_row["pca_components"])]
    return results, best_row


def final_hdbscan_labels(pca_scores: np.ndarray, best: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    model = hdbscan.HDBSCAN(
        min_cluster_size=int(best["min_cluster_size"]),
        min_samples=int(best["min_samples"]),
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = model.fit_predict(pca_scores)
    probabilities = getattr(model, "probabilities_", np.zeros(len(labels)))
    return labels, probabilities


def cluster_size_table(labels: np.ndarray) -> pd.DataFrame:
    total = len(labels)
    rows = []
    for label, count in pd.Series(labels).value_counts().sort_index().items():
        rows.append(
            {
                "cluster_label": int(label),
                "is_noise": int(label) == -1,
                "n_actors": int(count),
                "actor_ratio": int(count) / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["is_noise", "cluster_label"], ascending=[True, True])


def build_label_table(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    umap_embedding: np.ndarray,
    best: dict[str, Any],
) -> pd.DataFrame:
    table = metadata.copy()
    table["cluster_label"] = labels.astype(int)
    table["cluster_probability"] = probabilities
    table["is_noise"] = table["cluster_label"].eq(-1)
    table["umap_x"] = umap_embedding[:, 0]
    table["umap_y"] = umap_embedding[:, 1]
    table["pca_components"] = int(best["pca_components"])
    table["hdbscan_min_cluster_size"] = int(best["min_cluster_size"])
    table["hdbscan_min_samples"] = int(best["min_samples"])
    return table


def umap_embedding_and_plot(features: pd.DataFrame, labels: np.ndarray, output_path: Path, random_seed: int) -> np.ndarray:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mpl_cache = output_path.parent / "matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=random_seed)
    embedding = reducer.fit_transform(features)

    unique_labels = sorted(pd.Series(labels).unique())
    color_values = np.array([unique_labels.index(label) for label in labels])
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=color_values,
        s=16,
        cmap="tab20",
        alpha=0.82,
        linewidths=0,
    )
    ax.set_title("UMAP visualization of PCA + HDBSCAN actor clusters")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.18)

    handles = []
    handle_labels = []
    for label in unique_labels:
        color = scatter.cmap(scatter.norm(unique_labels.index(label)))
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", markersize=6, color=color))
        handle_labels.append("noise" if label == -1 else f"cluster {label}")
    ax.legend(handles, handle_labels, title="HDBSCAN label", loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return embedding


def summary_markdown(
    metrics: ClusteringMetrics,
    parameter_results: pd.DataFrame,
    cluster_sizes: pd.DataFrame,
    pca_variance: pd.DataFrame,
    features_path: Path,
    metadata_path: Path,
    umap_path: Path,
) -> str:
    best = parameter_results.loc[parameter_results["selected"]].iloc[0]
    top_results = parameter_results.head(10).drop(columns=["selected"], errors="ignore")
    pca_head = pca_variance.head(20)
    lines = [
        "# Role Clustering Summary",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Input features: `{features_path}`",
        f"- Input metadata: `{metadata_path}`",
        "- DB access: not used in this step.",
        "- Main method: PCA + HDBSCAN using the preprocessed standardized actor feature matrix.",
        "- UMAP is used only for 2D visualization. Clustering was not performed on the UMAP embedding.",
        "- Bot and low-activity actors were already excluded by preprocessing and are checked again before clustering.",
        "",
        "## Best Configuration",
        "",
        f"- Input actors: `{metrics.input_actors}`",
        f"- Input features: `{metrics.input_features}`",
        f"- PCA components: `{metrics.best_pca_components}`",
        f"- HDBSCAN min_cluster_size: `{metrics.best_min_cluster_size}`",
        f"- HDBSCAN min_samples: `{metrics.best_min_samples}`",
        f"- HDBSCAN metric: `euclidean`",
        f"- Cluster count: `{metrics.n_clusters}`",
        f"- Noise actors: `{metrics.noise_count}`",
        f"- Noise ratio: `{metrics.noise_ratio:.4f}`",
        f"- Largest cluster size/ratio: `{metrics.largest_cluster_size}` / `{metrics.largest_cluster_ratio:.4f}`",
        f"- UMAP figure: `{umap_path}`",
        "",
        "## Cluster Size Table",
        "",
        dataframe_to_markdown(cluster_sizes),
        "",
        "## Top Parameter Results",
        "",
        dataframe_to_markdown(top_results),
        "",
        "## PCA Explained Variance",
        "",
        dataframe_to_markdown(pca_head),
        "",
        "## Selection Note",
        "",
        "- The automatic score favors 3-12 clusters, lower noise ratio, and lower largest-cluster dominance.",
        f"- Selected score: `{best['selection_score']:.4f}`",
        "- Stability analysis, role naming, and profile interpretation are intentionally deferred to later steps.",
    ]
    return "\n".join(lines) + "\n"


def default_paths(project_root: Path) -> ClusteringPaths:
    cluster_dir = project_root / "data" / "processed" / "clusters"
    report_dir = project_root / "reports" / "clustering"
    return ClusteringPaths(
        labels_csv=cluster_dir / "actor_cluster_labels.csv",
        labels_parquet=cluster_dir / "actor_cluster_labels.parquet",
        parameter_results_csv=report_dir / "hdbscan_parameter_results.csv",
        cluster_size_table_csv=report_dir / "cluster_size_table.csv",
        pca_explained_variance_csv=report_dir / "pca_explained_variance.csv",
        umap_png=report_dir / "umap_roles.png",
        summary_md=report_dir / "clustering_summary.md",
    )


def run_role_clustering(
    features_path: Path,
    metadata_path: Path,
    project_root: Path,
    *,
    random_seed: int = 42,
    logger: Any | None = None,
) -> tuple[ClusteringMetrics, ClusteringPaths]:
    features = read_features(features_path)
    metadata = read_metadata(metadata_path)
    validate_inputs(features, metadata)

    _, pca_variance = pca_explained_variance(features, random_seed=random_seed)
    candidates = pca_candidates(pca_variance, len(features.columns))
    if logger is not None:
        logger.info("PCA component candidates: %s", candidates)
        logger.info("HDBSCAN min_cluster_size candidates: %s", hdbscan_min_cluster_sizes(len(features)))

    parameter_results, best = run_parameter_grid(features, candidates, random_seed=random_seed)
    labels, probabilities = final_hdbscan_labels(best["pca_scores"], best)
    final_metrics = evaluate_labels(labels)
    sizes = cluster_size_table(labels)
    paths = default_paths(project_root)
    embedding = umap_embedding_and_plot(features, labels, paths.umap_png, random_seed=random_seed)
    label_table = build_label_table(metadata, labels, probabilities, embedding, best)

    metrics = ClusteringMetrics(
        input_actors=len(features),
        input_features=len(features.columns),
        best_pca_components=int(best["pca_components"]),
        best_min_cluster_size=int(best["min_cluster_size"]),
        best_min_samples=int(best["min_samples"]),
        n_clusters=int(final_metrics["n_clusters"]),
        noise_count=int(final_metrics["noise_count"]),
        noise_ratio=float(final_metrics["noise_ratio"]),
        largest_cluster_size=int(final_metrics["largest_cluster_size"]),
        largest_cluster_ratio=float(final_metrics["largest_cluster_ratio"]),
    )

    paths.labels_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.summary_md.parent.mkdir(parents=True, exist_ok=True)
    write_dataframe(label_table, paths.labels_csv, paths.labels_parquet, logger=logger)
    write_dataframe(parameter_results.drop(columns=["pca_scores"], errors="ignore"), paths.parameter_results_csv, logger=logger)
    write_dataframe(sizes, paths.cluster_size_table_csv, logger=logger)
    write_dataframe(pca_variance, paths.pca_explained_variance_csv, logger=logger)
    write_text(summary_markdown(metrics, parameter_results, sizes, pca_variance, features_path, metadata_path, paths.umap_png), paths.summary_md, logger=logger)
    return metrics, paths
