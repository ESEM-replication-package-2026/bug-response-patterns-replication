from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

from smartshark_roles.io import dataframe_to_markdown, write_dataframe, write_text


@dataclass(frozen=True)
class StabilityConfig:
    n_bootstrap: int = 30
    sample_fraction: float = 0.80
    random_seed: int = 42
    pca_components: int = 12
    min_cluster_size: int = 20
    min_samples: int = 10


@dataclass(frozen=True)
class ValidationPaths:
    stability_bootstrap_csv: Path
    stability_summary_md: Path
    method_comparison_csv: Path
    method_comparison_summary_md: Path
    cluster_stability_by_label_csv: Path
    noise_stability_summary_csv: Path


def read_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature matrix file type: {path.suffix}")


def read_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cluster label file does not exist: {path}")
    labels = pd.read_csv(path)
    if "actor_id" not in labels.columns or "cluster_label" not in labels.columns:
        raise KeyError("Cluster labels must contain actor_id and cluster_label")
    if labels["actor_id"].duplicated().any():
        raise ValueError("Cluster labels contain duplicate actor_id values")
    return labels


def validate_inputs(features: pd.DataFrame, labels: pd.DataFrame) -> None:
    if len(features) != len(labels):
        raise ValueError(f"features and labels row counts differ: {len(features)} vs {len(labels)}")
    if features.isna().sum().sum():
        raise ValueError("Feature matrix contains NaN values")
    if "is_bot" in labels.columns and labels["is_bot"].astype(str).str.lower().isin({"true", "1"}).any():
        raise ValueError("Bot actors are present in clustering labels")
    if "is_low_activity" in labels.columns and labels["is_low_activity"].astype(str).str.lower().isin({"true", "1"}).any():
        raise ValueError("Low activity actors are present in clustering labels")


def pca_hdbscan(features: pd.DataFrame, config: StabilityConfig) -> np.ndarray:
    reduced = PCA(n_components=config.pca_components, random_state=config.random_seed).fit_transform(features)
    model = hdbscan.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return model.fit_predict(reduced).astype(int)


def cluster_metrics(labels: np.ndarray) -> dict[str, Any]:
    total = len(labels)
    noise_count = int((labels == -1).sum())
    nonnoise = labels[labels != -1]
    if len(nonnoise):
        sizes = pd.Series(nonnoise).value_counts()
        n_clusters = int(len(sizes))
        largest = int(sizes.max())
        mean_size = float(sizes.mean())
        median_size = float(sizes.median())
    else:
        n_clusters = 0
        largest = 0
        mean_size = 0.0
        median_size = 0.0
    return {
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_ratio": noise_count / total if total else 0.0,
        "largest_cluster_size": largest,
        "largest_cluster_ratio": largest / total if total else 0.0,
        "mean_cluster_size": mean_size,
        "median_cluster_size": median_size,
    }


def comparison_metrics(original: np.ndarray, boot: np.ndarray) -> dict[str, Any]:
    include = {
        "ari_include_noise": adjusted_rand_score(original, boot),
        "nmi_include_noise": normalized_mutual_info_score(original, boot),
        "v_measure_include_noise": v_measure_score(original, boot),
    }
    mask = (original != -1) & (boot != -1)
    if int(mask.sum()) >= 2 and len(np.unique(original[mask])) > 1 and len(np.unique(boot[mask])) > 1:
        exclude = {
            "ari_exclude_noise": adjusted_rand_score(original[mask], boot[mask]),
            "nmi_exclude_noise": normalized_mutual_info_score(original[mask], boot[mask]),
            "v_measure_exclude_noise": v_measure_score(original[mask], boot[mask]),
            "n_compared_exclude_noise": int(mask.sum()),
        }
    else:
        exclude = {
            "ari_exclude_noise": np.nan,
            "nmi_exclude_noise": np.nan,
            "v_measure_exclude_noise": np.nan,
            "n_compared_exclude_noise": int(mask.sum()),
        }
    return {**include, **exclude}


def dominant_fraction(values: np.ndarray) -> tuple[int, float]:
    if len(values) == 0:
        return -999, np.nan
    counts = pd.Series(values).value_counts()
    label = int(counts.index[0])
    return label, float(counts.iloc[0] / len(values))


def bootstrap_stability(
    features: pd.DataFrame,
    original_labels: np.ndarray,
    config: StabilityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.random_seed)
    n_actors = len(features)
    sample_size = int(round(config.sample_fraction * n_actors))
    bootstrap_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    original_cluster_labels = sorted(pd.Series(original_labels).unique())

    for iteration in range(1, config.n_bootstrap + 1):
        sample_indices = np.sort(rng.choice(n_actors, size=sample_size, replace=False))
        sample_features = features.iloc[sample_indices].reset_index(drop=True)
        sample_original = original_labels[sample_indices]
        boot_labels = pca_hdbscan(sample_features, config)
        metrics = cluster_metrics(boot_labels)
        comparison = comparison_metrics(sample_original, boot_labels)

        original_noise = sample_original == -1
        boot_noise = boot_labels == -1
        original_noise_count = int(original_noise.sum())
        original_nonnoise_count = int((~original_noise).sum())
        original_noise_to_noise = float((original_noise & boot_noise).sum() / original_noise_count) if original_noise_count else np.nan
        original_nonnoise_to_noise = float(((~original_noise) & boot_noise).sum() / original_nonnoise_count) if original_nonnoise_count else np.nan
        boot_noise_from_original_noise = float((original_noise & boot_noise).sum() / boot_noise.sum()) if int(boot_noise.sum()) else np.nan

        bootstrap_rows.append(
            {
                "iteration": iteration,
                "sample_size": sample_size,
                **comparison,
                **metrics,
                "original_noise_to_noise_rate": original_noise_to_noise,
                "original_nonnoise_to_noise_rate": original_nonnoise_to_noise,
                "boot_noise_from_original_noise_rate": boot_noise_from_original_noise,
            }
        )
        noise_rows.append(
            {
                "iteration": iteration,
                "original_noise_sampled": original_noise_count,
                "original_nonnoise_sampled": original_nonnoise_count,
                "bootstrap_noise_count": int(boot_noise.sum()),
                "original_noise_to_noise_rate": original_noise_to_noise,
                "original_nonnoise_to_noise_rate": original_nonnoise_to_noise,
                "boot_noise_from_original_noise_rate": boot_noise_from_original_noise,
            }
        )

        for cluster_label in original_cluster_labels:
            mask = sample_original == cluster_label
            n_sampled = int(mask.sum())
            if n_sampled:
                dominant_label, dom_fraction = dominant_fraction(boot_labels[mask])
                noise_fraction = float((boot_labels[mask] == -1).sum() / n_sampled)
            else:
                dominant_label, dom_fraction, noise_fraction = -999, np.nan, np.nan
            cluster_rows.append(
                {
                    "iteration": iteration,
                    "original_cluster_label": int(cluster_label),
                    "n_sampled": n_sampled,
                    "dominant_bootstrap_label": dominant_label,
                    "dominant_fraction": dom_fraction,
                    "bootstrap_noise_fraction": noise_fraction,
                }
            )

    bootstrap = pd.DataFrame(bootstrap_rows)
    cluster_detail = pd.DataFrame(cluster_rows)
    cluster_summary = (
        cluster_detail.groupby("original_cluster_label", dropna=False)
        .agg(
            bootstrap_runs=("iteration", "count"),
            mean_n_sampled=("n_sampled", "mean"),
            mean_dominant_fraction=("dominant_fraction", "mean"),
            median_dominant_fraction=("dominant_fraction", "median"),
            min_dominant_fraction=("dominant_fraction", "min"),
            max_dominant_fraction=("dominant_fraction", "max"),
            mean_bootstrap_noise_fraction=("bootstrap_noise_fraction", "mean"),
            median_bootstrap_noise_fraction=("bootstrap_noise_fraction", "median"),
        )
        .reset_index()
    )
    noise_detail = pd.DataFrame(noise_rows)
    noise_summary = pd.DataFrame(
        [
            {
                "metric": "original_noise_to_noise_rate",
                "mean": noise_detail["original_noise_to_noise_rate"].mean(),
                "median": noise_detail["original_noise_to_noise_rate"].median(),
                "std": noise_detail["original_noise_to_noise_rate"].std(),
                "min": noise_detail["original_noise_to_noise_rate"].min(),
                "max": noise_detail["original_noise_to_noise_rate"].max(),
            },
            {
                "metric": "original_nonnoise_to_noise_rate",
                "mean": noise_detail["original_nonnoise_to_noise_rate"].mean(),
                "median": noise_detail["original_nonnoise_to_noise_rate"].median(),
                "std": noise_detail["original_nonnoise_to_noise_rate"].std(),
                "min": noise_detail["original_nonnoise_to_noise_rate"].min(),
                "max": noise_detail["original_nonnoise_to_noise_rate"].max(),
            },
            {
                "metric": "boot_noise_from_original_noise_rate",
                "mean": noise_detail["boot_noise_from_original_noise_rate"].mean(),
                "median": noise_detail["boot_noise_from_original_noise_rate"].median(),
                "std": noise_detail["boot_noise_from_original_noise_rate"].std(),
                "min": noise_detail["boot_noise_from_original_noise_rate"].min(),
                "max": noise_detail["boot_noise_from_original_noise_rate"].max(),
            },
        ]
    )
    return bootstrap, cluster_summary, noise_summary


def metric_summary(bootstrap: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "ari_include_noise",
        "nmi_include_noise",
        "v_measure_include_noise",
        "ari_exclude_noise",
        "nmi_exclude_noise",
        "v_measure_exclude_noise",
        "noise_ratio",
        "n_clusters",
        "largest_cluster_ratio",
    ]
    rows = []
    for metric in metrics:
        series = pd.to_numeric(bootstrap[metric], errors="coerce")
        rows.append(
            {
                "metric": metric,
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
            }
        )
    return pd.DataFrame(rows)


def safe_cluster_scores(features: pd.DataFrame, labels: np.ndarray) -> tuple[float, float, float]:
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan, np.nan
    return (
        float(silhouette_score(features, labels)),
        float(calinski_harabasz_score(features, labels)),
        float(davies_bouldin_score(features, labels)),
    )


def method_comparison(features: pd.DataFrame, original_labels: np.ndarray, random_seed: int, n_clusters: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    nonnoise_mask = original_labels != -1

    hdb_nonnoise_features = features.loc[nonnoise_mask].reset_index(drop=True)
    hdb_nonnoise_labels = original_labels[nonnoise_mask]
    hdb_sil, hdb_ch, hdb_db = safe_cluster_scores(hdb_nonnoise_features, hdb_nonnoise_labels)
    rows.append(
        {
            "method": "hdbscan_nonnoise_reference",
            "n_clusters": n_clusters,
            "noise_count": int((original_labels == -1).sum()),
            "noise_ratio": float((original_labels == -1).mean()),
            "silhouette_score": hdb_sil,
            "calinski_harabasz_score": hdb_ch,
            "davies_bouldin_score": hdb_db,
            "ari_vs_hdbscan_nonnoise": 1.0,
            "nmi_vs_hdbscan_nonnoise": 1.0,
            "v_measure_vs_hdbscan_nonnoise": 1.0,
        }
    )

    kmeans_labels = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=20).fit_predict(features)
    agg_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(features)
    for method, labels in [("kmeans", kmeans_labels), ("agglomerative_ward", agg_labels)]:
        sil, ch, db = safe_cluster_scores(features, labels)
        rows.append(
            {
                "method": method,
                "n_clusters": int(len(np.unique(labels))),
                "noise_count": 0,
                "noise_ratio": 0.0,
                "silhouette_score": sil,
                "calinski_harabasz_score": ch,
                "davies_bouldin_score": db,
                "ari_vs_hdbscan_nonnoise": adjusted_rand_score(original_labels[nonnoise_mask], labels[nonnoise_mask]),
                "nmi_vs_hdbscan_nonnoise": normalized_mutual_info_score(original_labels[nonnoise_mask], labels[nonnoise_mask]),
                "v_measure_vs_hdbscan_nonnoise": v_measure_score(original_labels[nonnoise_mask], labels[nonnoise_mask]),
            }
        )
    return pd.DataFrame(rows)


def stable_cluster_labels(cluster_summary: pd.DataFrame) -> tuple[list[int], list[int]]:
    nonnoise = cluster_summary[cluster_summary["original_cluster_label"] != -1]
    stable = nonnoise[
        (nonnoise["mean_dominant_fraction"] >= 0.70)
        & (nonnoise["mean_bootstrap_noise_fraction"] <= 0.35)
    ]["original_cluster_label"].astype(int).tolist()
    unstable = nonnoise[
        (nonnoise["mean_dominant_fraction"] < 0.60)
        | (nonnoise["mean_bootstrap_noise_fraction"] > 0.50)
    ]["original_cluster_label"].astype(int).tolist()
    return stable, unstable


def stability_recommendation(summary: pd.DataFrame, cluster_summary: pd.DataFrame, noise_summary: pd.DataFrame) -> str:
    ari = float(summary.loc[summary["metric"] == "ari_include_noise", "mean"].iloc[0])
    ari_nonnoise = float(summary.loc[summary["metric"] == "ari_exclude_noise", "mean"].iloc[0])
    noise_mean = float(summary.loc[summary["metric"] == "noise_ratio", "mean"].iloc[0])
    _, unstable = stable_cluster_labels(cluster_summary)
    if ari >= 0.55 and ari_nonnoise >= 0.60 and len(unstable) <= 2 and noise_mean <= 0.55:
        return "maintain_current_hdbscan_and_proceed_to_stability_reporting"
    if ari >= 0.45 and ari_nonnoise >= 0.50:
        return "keep_current_hdbscan_for_now_but_report_noise_and_small_cluster_sensitivity"
    return "consider_parameter_retuning_before_final_role_claims"


def stability_summary_markdown(
    bootstrap: pd.DataFrame,
    summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    noise_summary: pd.DataFrame,
    config: StabilityConfig,
) -> str:
    stable, unstable = stable_cluster_labels(cluster_summary)
    recommendation = stability_recommendation(summary, cluster_summary, noise_summary)
    lines = [
        "# Stability Analysis Summary",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "- DB access: not used in this step.",
        "- Role naming, OpenAI API use, and RQ3 analysis were not performed.",
        "",
        "## Bootstrap Setup",
        "",
        f"- Bootstrap runs: `{config.n_bootstrap}`",
        f"- Actor sample fraction: `{config.sample_fraction}`",
        f"- PCA components: `{config.pca_components}`",
        f"- HDBSCAN min_cluster_size: `{config.min_cluster_size}`",
        f"- HDBSCAN min_samples: `{config.min_samples}`",
        "",
        "## Metric Summary",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Cluster Stability By Original Label",
        "",
        dataframe_to_markdown(cluster_summary),
        "",
        "## Noise Stability",
        "",
        dataframe_to_markdown(noise_summary),
        "",
        "## Interpretation",
        "",
        f"- Stable non-noise clusters by simple threshold: `{stable}`",
        f"- Potentially unstable non-noise clusters by simple threshold: `{unstable}`",
        "- Noise-included metrics evaluate whether the model reproduces the same dense regions and noise assignment.",
        "- Noise-excluded metrics evaluate agreement only among actors assigned to non-noise clusters in both runs.",
        f"- Recommendation: `{recommendation}`",
    ]
    return "\n".join(lines) + "\n"


def method_summary_markdown(methods: pd.DataFrame) -> str:
    hdb = methods[methods["method"] == "hdbscan_nonnoise_reference"].iloc[0]
    comparisons = methods[methods["method"] != "hdbscan_nonnoise_reference"].copy()
    best_ari = comparisons.sort_values("ari_vs_hdbscan_nonnoise", ascending=False).iloc[0]
    lines = [
        "# Method Comparison Summary",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "- DB access: not used in this step.",
        "- k-means and agglomerative clustering are comparison methods only; PCA + HDBSCAN remains the primary method.",
        "",
        "## Scores",
        "",
        dataframe_to_markdown(methods),
        "",
        "## Interpretation",
        "",
        f"- HDBSCAN reference uses its non-noise actors only for internal validity metrics; its noise ratio is `{hdb['noise_ratio']:.4f}`.",
        f"- The closest comparison method by ARI to HDBSCAN non-noise labels is `{best_ari['method']}` with ARI `{best_ari['ari_vs_hdbscan_nonnoise']:.4f}` and NMI `{best_ari['nmi_vs_hdbscan_nonnoise']:.4f}`.",
        "- If comparison ARI/NMI are modest, the exact partition is method-sensitive, but HDBSCAN remains useful because it explicitly models noise and irregular cluster density.",
        "- Final role claims should rely on HDBSCAN profiles plus stability evidence, not on k-means/agglomerative alone.",
    ]
    return "\n".join(lines) + "\n"


def default_paths(project_root: Path) -> ValidationPaths:
    result_dir = project_root / "reports" / "results"
    return ValidationPaths(
        stability_bootstrap_csv=result_dir / "stability_bootstrap.csv",
        stability_summary_md=result_dir / "stability_summary.md",
        method_comparison_csv=result_dir / "method_comparison.csv",
        method_comparison_summary_md=result_dir / "method_comparison_summary.md",
        cluster_stability_by_label_csv=result_dir / "cluster_stability_by_label.csv",
        noise_stability_summary_csv=result_dir / "noise_stability_summary.csv",
    )


def run_stability_analysis(
    features_path: Path,
    labels_path: Path,
    project_root: Path,
    *,
    config: StabilityConfig = StabilityConfig(),
    logger: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, ValidationPaths]:
    features = read_features(features_path)
    labels = read_labels(labels_path)
    validate_inputs(features, labels)
    original_labels = labels["cluster_label"].astype(int).to_numpy()
    if logger is not None:
        logger.info("Running %d bootstrap iterations", config.n_bootstrap)

    bootstrap, cluster_summary, noise_summary = bootstrap_stability(features, original_labels, config)
    summary = metric_summary(bootstrap)
    n_nonnoise_clusters = int(len(set(original_labels) - {-1}))
    methods = method_comparison(features, original_labels, config.random_seed, n_nonnoise_clusters)
    paths = default_paths(project_root)
    paths.stability_bootstrap_csv.parent.mkdir(parents=True, exist_ok=True)

    write_dataframe(bootstrap, paths.stability_bootstrap_csv, logger=logger)
    write_dataframe(cluster_summary, paths.cluster_stability_by_label_csv, logger=logger)
    write_dataframe(noise_summary, paths.noise_stability_summary_csv, logger=logger)
    write_dataframe(methods, paths.method_comparison_csv, logger=logger)
    write_text(stability_summary_markdown(bootstrap, summary, cluster_summary, noise_summary, config), paths.stability_summary_md, logger=logger)
    write_text(method_summary_markdown(methods), paths.method_comparison_summary_md, logger=logger)
    return bootstrap, cluster_summary, noise_summary, methods, paths
