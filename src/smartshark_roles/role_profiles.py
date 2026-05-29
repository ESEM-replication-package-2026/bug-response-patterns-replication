from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smartshark_roles.io import dataframe_to_markdown, write_dataframe, write_json, write_text


RATIO_FEATURES = [
    "reporting_ratio",
    "discussion_ratio",
    "triage_ratio",
    "fixing_ratio",
    "integration_ratio",
    "review_ratio",
    "closure_ratio",
    "unknown_ratio",
]

SOURCE_RATIO_FEATURES = [
    "issue_ratio",
    "issue_comment_ratio",
    "event_ratio",
    "commit_ratio",
    "pull_request_ratio",
    "pull_request_comment_ratio",
    "pull_request_review_ratio",
    "pull_request_review_comment_ratio",
]

ROLE_SIGNALS = {
    "Reporter": ["reporting_ratio", "issue_ratio", "n_issues_reported"],
    "Discussion Participant": ["discussion_ratio", "issue_comment_ratio", "n_issue_comments"],
    "Triager / Workflow Manager": ["triage_ratio", "event_ratio", "n_status_changes", "n_resolution_changes", "n_assignee_changes"],
    "Fixer": ["fixing_ratio", "commit_ratio", "n_bugfix_commits_authored", "n_commits_linked_to_bug"],
    "Integrator": ["integration_ratio", "pull_request_ratio", "n_prs_opened", "n_prs_closed", "n_prs_merged"],
    "Reviewer": ["review_ratio", "pull_request_comment_ratio", "pull_request_review_ratio", "pull_request_review_comment_ratio", "n_reviews_submitted", "n_review_comments"],
}


@dataclass(frozen=True)
class ProfilePaths:
    cluster_profiles_csv: Path
    cluster_profiles_json: Path
    cluster_profile_summary_md: Path
    role_cards_draft_md: Path
    cluster_top_features_csv: Path
    lifecycle_distribution_csv: Path
    source_distribution_csv: Path
    scope_distribution_csv: Path
    noise_profile_md: Path
    feature_heatmap_png: Path


def read_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature file does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Events file does not exist: {path}")
    columns = [
        "actor_id",
        "actor_unknown",
        "project_name",
        "timestamp",
        "source",
        "event_type",
        "lifecycle_stage",
        "event_scope",
        "issue_external_id",
    ]
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def load_inputs(features_path: Path, labels_path: Path, events_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = read_features(features_path)
    labels = pd.read_csv(labels_path)
    events = read_events(events_path)
    if "actor_id" not in features.columns or "actor_id" not in labels.columns:
        raise KeyError("features and labels must contain actor_id")
    if labels["actor_id"].duplicated().any():
        raise ValueError("actor_cluster_labels contains duplicate actor_id values")
    label_columns = [column for column in ["actor_id", "cluster_label", "cluster_probability", "is_noise"] if column in labels.columns]
    merged_features = labels[label_columns].merge(features, on="actor_id", how="left", validate="one_to_one")
    if merged_features["total_events"].isna().any():
        raise ValueError("Some cluster label actors were not found in actor_features_raw")
    events = events.loc[~events["actor_unknown"].fillna(True).astype(bool)].copy()
    clustered_events = events.merge(labels[["actor_id", "cluster_label"]], on="actor_id", how="inner")
    clustered_events["timestamp"] = pd.to_datetime(clustered_events["timestamp"], errors="coerce", utc=True)
    return merged_features, labels, clustered_events


def cluster_order(labels: pd.Series) -> list[int]:
    non_noise = sorted(label for label in labels.dropna().astype(int).unique() if label != -1)
    return [*non_noise, -1] if -1 in labels.astype(int).unique() else non_noise


def distribution(frame: pd.DataFrame, column: str, label_column: str = "cluster_label") -> pd.DataFrame:
    counts = frame.groupby([label_column, column], dropna=False).size().reset_index(name="n_events")
    totals = counts.groupby(label_column)["n_events"].transform("sum")
    counts["event_ratio"] = counts["n_events"] / totals
    return counts.sort_values([label_column, "n_events"], ascending=[True, False])


def project_distribution(events: pd.DataFrame) -> pd.DataFrame:
    counts = distribution(events, "project_name")
    return counts.rename(columns={"project_name": "project_name", "event_ratio": "project_event_ratio"})


def dominant_value(distribution_frame: pd.DataFrame, cluster_label: int, value_column: str, ratio_column: str) -> tuple[str, float]:
    subset = distribution_frame[distribution_frame["cluster_label"] == cluster_label]
    if subset.empty:
        return "", 0.0
    row = subset.sort_values(ratio_column, ascending=False).iloc[0]
    return str(row[value_column]), float(row[ratio_column])


def standardized_cluster_features(features: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    stats = features[numeric_columns].agg(["mean", "std"])
    std = stats.loc["std"].replace(0, np.nan)
    standardized = features[["actor_id", "cluster_label", *numeric_columns]].copy()
    standardized[numeric_columns] = (standardized[numeric_columns] - stats.loc["mean"]) / std
    standardized[numeric_columns] = standardized[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    return standardized


def cluster_top_features(features: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    excluded_columns = {
        "cluster_label",
        "row_index",
        "cluster_probability",
        "umap_x",
        "umap_y",
        "pca_components",
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
        "is_noise",
        "is_bot",
        "is_low_activity",
    }
    numeric_columns = [
        column
        for column in features.select_dtypes(include=["number", "bool"]).columns
        if column not in excluded_columns
    ]
    standardized = standardized_cluster_features(features, numeric_columns)
    means = standardized.groupby("cluster_label")[numeric_columns].mean()
    rows: list[dict[str, Any]] = []
    for label, row in means.iterrows():
        top = row.sort_values(ascending=False).head(top_n)
        bottom = row.sort_values(ascending=True).head(top_n)
        for rank, (feature, value) in enumerate(top.items(), start=1):
            rows.append({"cluster_label": int(label), "direction": "top", "rank": rank, "feature": feature, "standardized_mean": float(value)})
        for rank, (feature, value) in enumerate(bottom.items(), start=1):
            rows.append({"cluster_label": int(label), "direction": "bottom", "rank": rank, "feature": feature, "standardized_mean": float(value)})
    return pd.DataFrame(rows)


def candidate_role_name(cluster_means: pd.Series) -> str:
    role_scores: dict[str, float] = {}
    for role, features in ROLE_SIGNALS.items():
        values = [float(cluster_means.get(feature, 0.0)) for feature in features]
        role_scores[role] = float(np.mean(values)) if values else 0.0
    sorted_roles = sorted(role_scores.items(), key=lambda item: item[1], reverse=True)
    if not sorted_roles:
        return "Unclear"
    first, first_score = sorted_roles[0]
    second, second_score = sorted_roles[1]
    if first_score <= 0:
        return "Unclear"
    if second_score > 0 and second_score >= first_score * 0.75:
        return f"Hybrid: {first} + {second}"
    return first


def representative_actors(features: pd.DataFrame, cluster_label: int, max_actors: int = 5) -> list[str]:
    subset = features[features["cluster_label"] == cluster_label].copy()
    if subset.empty:
        return []
    if cluster_label == -1:
        subset["representative_score"] = subset["total_events"].rank(pct=True, method="first")
        subset["representative_score"] = (subset["representative_score"] - 0.5).abs()
        return subset.sort_values(["representative_score", "total_events"], ascending=[True, False])["actor_id"].head(max_actors).tolist()
    if "cluster_probability" in subset.columns:
        return subset.sort_values(["cluster_probability", "total_events"], ascending=[False, False])["actor_id"].head(max_actors).tolist()
    return subset.sort_values("total_events", ascending=False)["actor_id"].head(max_actors).tolist()


def actor_timeline(events: pd.DataFrame, actor_ids: list[str], max_events_per_actor: int = 8) -> pd.DataFrame:
    timeline = events.loc[events["actor_id"].isin(actor_ids)].copy()
    if timeline.empty:
        return timeline
    timeline = timeline.sort_values(["actor_id", "timestamp", "source", "event_type"], kind="mergesort")
    timeline = timeline.groupby("actor_id").head(max_events_per_actor)
    return timeline[
        [
            "actor_id",
            "project_name",
            "timestamp",
            "source",
            "event_type",
            "lifecycle_stage",
            "event_scope",
            "issue_external_id",
        ]
    ]


def build_cluster_profiles(features: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    lifecycle = distribution(events, "lifecycle_stage").rename(columns={"event_ratio": "lifecycle_ratio"})
    source = distribution(events, "source").rename(columns={"event_ratio": "source_ratio"})
    scope = distribution(events, "event_scope").rename(columns={"event_ratio": "scope_ratio"})
    projects = project_distribution(events)
    top_features = cluster_top_features(features)
    feature_means = features.groupby("cluster_label").mean(numeric_only=True)

    rows: list[dict[str, Any]] = []
    timelines: dict[int, pd.DataFrame] = {}
    for label in cluster_order(features["cluster_label"]):
        subset = features[features["cluster_label"] == label]
        cluster_events = events[events["cluster_label"] == label]
        dominant_project, dominant_project_ratio = dominant_value(projects, label, "project_name", "project_event_ratio")
        dominant_lifecycle, dominant_lifecycle_ratio = dominant_value(lifecycle, label, "lifecycle_stage", "lifecycle_ratio")
        dominant_source, dominant_source_ratio = dominant_value(source, label, "source", "source_ratio")
        dominant_scope, dominant_scope_ratio = dominant_value(scope, label, "event_scope", "scope_ratio")
        reps = representative_actors(features, label)
        timelines[label] = actor_timeline(cluster_events, reps)
        means = feature_means.loc[label] if label in feature_means.index else pd.Series(dtype=float)
        rows.append(
            {
                "cluster_label": int(label),
                "is_noise": label == -1,
                "candidate_role_name": "Noise / Outlier" if label == -1 else candidate_role_name(means),
                "n_actors": int(len(subset)),
                "n_events": int(len(cluster_events)),
                "total_events_mean": float(subset["total_events"].mean()),
                "total_events_median": float(subset["total_events"].median()),
                "n_unique_issues_mean": float(subset["n_unique_issues"].mean()),
                "n_unique_issues_median": float(subset["n_unique_issues"].median()),
                "dominant_project": dominant_project,
                "dominant_project_ratio": dominant_project_ratio,
                "dominant_lifecycle_stage": dominant_lifecycle,
                "dominant_lifecycle_ratio": dominant_lifecycle_ratio,
                "dominant_source": dominant_source,
                "dominant_source_ratio": dominant_source_ratio,
                "dominant_scope": dominant_scope,
                "dominant_scope_ratio": dominant_scope_ratio,
                "representative_actor_ids": ";".join(reps),
            }
        )
    return pd.DataFrame(rows), top_features, lifecycle, source, scope, timelines


def profiles_to_json(profiles: pd.DataFrame, top_features: pd.DataFrame) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for _, row in profiles.iterrows():
        label = int(row["cluster_label"])
        top = top_features[(top_features["cluster_label"] == label) & (top_features["direction"] == "top")].head(10)
        bottom = top_features[(top_features["cluster_label"] == label) & (top_features["direction"] == "bottom")].head(10)
        payload.append(
            {
                **{key: row[key] for key in profiles.columns if key != "representative_actor_ids"},
                "representative_actor_ids": str(row["representative_actor_ids"]).split(";") if row["representative_actor_ids"] else [],
                "top_features": top[["feature", "standardized_mean"]].to_dict(orient="records"),
                "bottom_features": bottom[["feature", "standardized_mean"]].to_dict(orient="records"),
            }
        )
    return payload


def top_feature_text(top_features: pd.DataFrame, cluster_label: int, direction: str = "top", n: int = 5) -> str:
    subset = top_features[(top_features["cluster_label"] == cluster_label) & (top_features["direction"] == direction)].head(n)
    return ", ".join(f"{row.feature} ({row.standardized_mean:.2f})" for row in subset.itertuples())


def role_cards_markdown(profiles: pd.DataFrame, top_features: pd.DataFrame, timelines: dict[int, pd.DataFrame]) -> str:
    lines = [
        "# Draft Role Cards",
        "",
        "- These candidate role names are rule-based and not final.",
        "- Raw actor identifiers, emails, usernames, and names are not shown.",
        "",
    ]
    for row in profiles.sort_values("cluster_label").itertuples():
        label = int(row.cluster_label)
        lines.extend(
            [
                f"## Cluster {label}: {row.candidate_role_name}",
                "",
                f"- Actors: `{row.n_actors}`",
                f"- Events: `{row.n_events}`",
                f"- Dominant lifecycle/source/scope: `{row.dominant_lifecycle_stage}` / `{row.dominant_source}` / `{row.dominant_scope}`",
                f"- Top features: {top_feature_text(top_features, label, 'top')}",
                f"- Bottom features: {top_feature_text(top_features, label, 'bottom')}",
                "",
                "### Representative Actor Timelines",
                "",
            ]
        )
        timeline = timelines.get(label, pd.DataFrame())
        if timeline.empty:
            lines.append("No timeline rows available.")
        else:
            lines.append(dataframe_to_markdown(timeline.head(40)))
        lines.append("")
    return "\n".join(lines) + "\n"


def profile_summary_markdown(profiles: pd.DataFrame, top_features: pd.DataFrame, projects: pd.DataFrame, events: pd.DataFrame) -> str:
    dominant_project_warning = profiles[profiles["dominant_project_ratio"] >= 0.60]
    noise = profiles[profiles["cluster_label"] == -1]
    noise_text = "Noise cluster not present."
    if not noise.empty:
        n = noise.iloc[0]
        noise_text = (
            f"Noise contains {int(n['n_actors'])} actors and {int(n['n_events'])} clustered-cohort events. "
            f"Dominant project ratio is {float(n['dominant_project_ratio']):.3f}; "
            f"median total_events is {float(n['total_events_median']):.1f}."
        )
    lines = [
        "# Cluster Profile Summary",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "- DB access: not used in this step.",
        "- Candidate role names are rule-based drafts for inspection, not final labels.",
        "- Reports use anonymized `actor_id` only; raw actor identifiers and personal metadata are not shown.",
        "",
        "## Cluster Overview",
        "",
        dataframe_to_markdown(profiles.drop(columns=["representative_actor_ids"], errors="ignore")),
        "",
        "## Top Features By Cluster",
        "",
        dataframe_to_markdown(top_features[top_features["direction"] == "top"].groupby("cluster_label").head(5)),
        "",
        "## Noise Summary",
        "",
        noise_text,
        "",
        "## Project Bias Check",
        "",
        dataframe_to_markdown(dominant_project_warning[["cluster_label", "dominant_project", "dominant_project_ratio"]]) if not dominant_project_warning.empty else "No cluster has a dominant project ratio >= 0.60.",
        "",
        "## Interpretation Check",
        "",
        "- Several clusters show distinct source/lifecycle signatures; role cards should be reviewed before final naming.",
        "- Noise is substantial and should be considered in stability analysis rather than discarded silently.",
        "- Cluster profiles should be used before deciding whether to tune clustering parameters further.",
    ]
    return "\n".join(lines) + "\n"


def noise_profile_markdown(profiles: pd.DataFrame, top_features: pd.DataFrame, projects: pd.DataFrame, lifecycle: pd.DataFrame, source: pd.DataFrame, scope: pd.DataFrame) -> str:
    noise_profile = profiles[profiles["cluster_label"] == -1]
    if noise_profile.empty:
        return "# Noise Profile\n\nNo noise cluster was produced.\n"
    top_projects = projects[projects["cluster_label"] == -1].head(10)
    top_lifecycle = lifecycle[lifecycle["cluster_label"] == -1].head(10)
    top_source = source[source["cluster_label"] == -1].head(10)
    top_scope = scope[scope["cluster_label"] == -1].head(10)
    top = top_features[(top_features["cluster_label"] == -1) & (top_features["direction"] == "top")].head(10)
    bottom = top_features[(top_features["cluster_label"] == -1) & (top_features["direction"] == "bottom")].head(10)
    row = noise_profile.iloc[0]
    lines = [
        "# Noise Profile",
        "",
        f"- Actors: `{int(row['n_actors'])}`",
        f"- Events: `{int(row['n_events'])}`",
        f"- Median total_events: `{float(row['total_events_median']):.1f}`",
        f"- Median n_unique_issues: `{float(row['n_unique_issues_median']):.1f}`",
        f"- Dominant project: `{row['dominant_project']}` / `{float(row['dominant_project_ratio']):.3f}`",
        "",
        "## Top Noise Features",
        "",
        dataframe_to_markdown(top),
        "",
        "## Bottom Noise Features",
        "",
        dataframe_to_markdown(bottom),
        "",
        "## Project Distribution",
        "",
        dataframe_to_markdown(top_projects),
        "",
        "## Lifecycle Distribution",
        "",
        dataframe_to_markdown(top_lifecycle),
        "",
        "## Source Distribution",
        "",
        dataframe_to_markdown(top_source),
        "",
        "## Scope Distribution",
        "",
        dataframe_to_markdown(top_scope),
        "",
        "## Interpretation",
        "",
        "- Noise is not automatically bad in HDBSCAN; it represents actors not assigned to dense behavioral regions.",
        "- Review whether noise contains high-activity hybrid actors before deciding on parameter retuning.",
    ]
    return "\n".join(lines) + "\n"


def heatmap(features: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = output_path.parent / "matplotlib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heatmap_features = [
        "reporting_ratio",
        "discussion_ratio",
        "triage_ratio",
        "fixing_ratio",
        "integration_ratio",
        "review_ratio",
        "closure_ratio",
        "unknown_ratio",
        "issue_ratio",
        "issue_comment_ratio",
        "event_ratio",
        "commit_ratio",
        "pull_request_ratio",
        "pull_request_comment_ratio",
        "pull_request_review_ratio",
        "pull_request_review_comment_ratio",
        "project_level_ratio",
        "total_events",
        "n_unique_issues",
    ]
    available = [column for column in heatmap_features if column in features.columns]
    matrix = features.groupby("cluster_label")[available].mean()
    standardized = (matrix - features[available].mean()) / features[available].std().replace(0, np.nan)
    standardized = standardized.fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    image = ax.imshow(standardized.values, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(available, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(standardized.index)))
    ax.set_yticklabels([str(int(label)) for label in standardized.index])
    ax.set_ylabel("Cluster label")
    ax.set_title("Cluster feature heatmap (standardized raw-feature means)")
    fig.colorbar(image, ax=ax, label="standardized mean")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def default_paths(project_root: Path) -> ProfilePaths:
    profile_dir = project_root / "reports" / "cluster_profiles"
    clustering_dir = project_root / "reports" / "clustering"
    return ProfilePaths(
        cluster_profiles_csv=profile_dir / "cluster_profiles.csv",
        cluster_profiles_json=profile_dir / "cluster_profiles.json",
        cluster_profile_summary_md=profile_dir / "cluster_profile_summary.md",
        role_cards_draft_md=profile_dir / "role_cards_draft.md",
        cluster_top_features_csv=profile_dir / "cluster_top_features.csv",
        lifecycle_distribution_csv=profile_dir / "cluster_lifecycle_distribution.csv",
        source_distribution_csv=profile_dir / "cluster_source_distribution.csv",
        scope_distribution_csv=profile_dir / "cluster_scope_distribution.csv",
        noise_profile_md=profile_dir / "noise_profile.md",
        feature_heatmap_png=clustering_dir / "feature_heatmap.png",
    )


def run_cluster_profiles(
    features_path: Path,
    labels_path: Path,
    events_path: Path,
    project_root: Path,
    logger: Any | None = None,
) -> tuple[pd.DataFrame, ProfilePaths]:
    features, labels, events = load_inputs(features_path, labels_path, events_path)
    profiles, top_features, lifecycle, source, scope, timelines = build_cluster_profiles(features, events)
    projects = project_distribution(events)
    paths = default_paths(project_root)
    paths.cluster_profiles_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.feature_heatmap_png.parent.mkdir(parents=True, exist_ok=True)

    write_dataframe(profiles, paths.cluster_profiles_csv, logger=logger)
    write_json(profiles_to_json(profiles, top_features), paths.cluster_profiles_json, logger=logger)
    write_dataframe(top_features, paths.cluster_top_features_csv, logger=logger)
    write_dataframe(lifecycle, paths.lifecycle_distribution_csv, logger=logger)
    write_dataframe(source, paths.source_distribution_csv, logger=logger)
    write_dataframe(scope, paths.scope_distribution_csv, logger=logger)
    write_text(profile_summary_markdown(profiles, top_features, projects, events), paths.cluster_profile_summary_md, logger=logger)
    write_text(role_cards_markdown(profiles, top_features, timelines), paths.role_cards_draft_md, logger=logger)
    write_text(noise_profile_markdown(profiles, top_features, projects, lifecycle, source, scope), paths.noise_profile_md, logger=logger)
    heatmap(features, paths.feature_heatmap_png)
    if logger is not None:
        logger.info("Wrote heatmap: %s", paths.feature_heatmap_png)
    return profiles, paths
