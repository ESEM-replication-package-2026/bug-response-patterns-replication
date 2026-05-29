from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartshark_roles.io import dataframe_to_markdown, write_dataframe, write_text


CLUSTER2_FEATURES = [
    "n_issues_reported",
    "n_issue_comments",
    "n_unique_issues_commented",
    "n_status_changes",
    "n_resolution_changes",
    "n_assignee_changes",
    "n_issue_field_changes",
    "reporting_ratio",
    "discussion_ratio",
    "triage_ratio",
    "unknown_ratio",
    "issue_ratio",
    "issue_comment_ratio",
    "event_ratio",
    "total_events",
    "n_unique_issues",
    "n_projects",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze project cross-cutting cluster distribution and cluster 2 diagnostics.")
    parser.add_argument("--features", default="data/interim/features/actor_features_raw.parquet")
    parser.add_argument("--labels", default="data/processed/clusters/actor_cluster_labels.csv")
    parser.add_argument("--events", default="data/interim/events/events_normalized.parquet")
    parser.add_argument("--cluster-profiles", default="reports/cluster_profiles/cluster_profiles.csv")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_features(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def read_events(path: Path) -> pd.DataFrame:
    columns = [
        "actor_id",
        "actor_unknown",
        "project_name",
        "timestamp",
        "source",
        "event_type",
        "lifecycle_stage",
        "issue_external_id",
    ]
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = read_features(project_path(args.features))
    labels = pd.read_csv(project_path(args.labels))
    events = read_events(project_path(args.events))
    profiles = pd.read_csv(project_path(args.cluster_profiles))
    labeled_features = labels[["actor_id", "cluster_label"]].merge(features, on="actor_id", how="left", validate="one_to_one")
    if labeled_features["total_events"].isna().any():
        raise ValueError("Some labeled actors are missing from actor_features_raw")
    labeled_events = events.loc[~events["actor_unknown"].fillna(True).astype(bool)].merge(
        labels[["actor_id", "cluster_label"]],
        on="actor_id",
        how="inner",
    )
    labeled_events["timestamp"] = pd.to_datetime(labeled_events["timestamp"], errors="coerce", utc=True)
    return labeled_features, labels, labeled_events, profiles, features


def project_cluster_distribution(events: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    counts = events.groupby(["project_name", "cluster_label"], dropna=False).size().reset_index(name="n_events")
    project_total = counts.groupby("project_name")["n_events"].transform("sum")
    cluster_total = counts.groupby("cluster_label")["n_events"].transform("sum")
    counts["project_cluster_ratio"] = counts["n_events"] / project_total
    counts["cluster_project_ratio"] = counts["n_events"] / cluster_total

    actor_counts = labels.merge(events[["actor_id", "project_name"]].drop_duplicates(), on="actor_id", how="inner")
    actors = actor_counts.groupby(["project_name", "cluster_label"])["actor_id"].nunique().reset_index(name="n_actors")
    return counts.merge(actors, on=["project_name", "cluster_label"], how="left").sort_values(["cluster_label", "n_events"], ascending=[True, False])


def dominant_project_table(distribution: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_label, subset in distribution.groupby("cluster_label"):
        row = subset.sort_values("cluster_project_ratio", ascending=False).iloc[0]
        rows.append(
            {
                "cluster_label": int(cluster_label),
                "dominant_project": row["project_name"],
                "dominant_project_event_ratio": float(row["cluster_project_ratio"]),
                "dominant_project_events": int(row["n_events"]),
                "n_projects_with_events": int(subset["project_name"].nunique()),
                "bias_flag_over_0_60": bool(row["cluster_project_ratio"] > 0.60),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_label")


def plot_project_cluster_heatmap(distribution: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = output_path.parent / "matplotlib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = distribution.pivot_table(index="project_name", columns="cluster_label", values="project_cluster_ratio", fill_value=0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.01, pivot.values.max()))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Cluster label")
    ax.set_ylabel("Project")
    ax.set_title("Project-level cluster event distribution")
    fig.colorbar(image, ax=ax, label="within-project event ratio")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def feature_distribution(cluster2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in CLUSTER2_FEATURES:
        series = pd.to_numeric(cluster2[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std()) if series.count() > 1 else np.nan,
                "min": float(series.min()),
                "p25": float(series.quantile(0.25)),
                "median": float(series.quantile(0.50)),
                "p75": float(series.quantile(0.75)),
                "p90": float(series.quantile(0.90)),
                "p95": float(series.quantile(0.95)),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


def cluster2_subgroups(cluster2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in cluster2.iterrows():
        candidates: list[tuple[str, float]] = []
        candidates.append(("Reporter-oriented", float(row["reporting_ratio"]) + 0.08 * float(row["n_issues_reported"])))
        candidates.append(("Commenter / Clarifier-oriented", float(row["discussion_ratio"]) + 0.03 * float(row["n_issue_comments"])))
        workflow_events = float(row["n_status_changes"] + row["n_resolution_changes"] + row["n_assignee_changes"] + row["n_issue_field_changes"])
        candidates.append(("Issue field-change / workflow-oriented", float(row["triage_ratio"] + row["unknown_ratio"]) + 0.05 * workflow_events))
        candidates.append(("High-activity issue participant", min(1.5, float(row["total_events"]) / 30.0) + min(1.0, float(row["n_unique_issues"]) / 10.0)))
        candidates.append(("General issue-side participant", 0.40 + float(row["issue_ratio"] + row["issue_comment_ratio"] + row["event_ratio"]) / 3.0))
        subgroup, score = sorted(candidates, key=lambda item: item[1], reverse=True)[0]
        rows.append(
            {
                "actor_id": row["actor_id"],
                "cluster_label": 2,
                "subgroup_candidate": subgroup,
                "subgroup_score": score,
                "total_events": row["total_events"],
                "n_unique_issues": row["n_unique_issues"],
                "n_issues_reported": row["n_issues_reported"],
                "n_issue_comments": row["n_issue_comments"],
                "n_issue_field_changes": row["n_issue_field_changes"],
                "reporting_ratio": row["reporting_ratio"],
                "discussion_ratio": row["discussion_ratio"],
                "triage_ratio": row["triage_ratio"],
                "unknown_ratio": row["unknown_ratio"],
            }
        )
    return pd.DataFrame(rows)


def representative_timelines(subgroups: pd.DataFrame, events: pd.DataFrame) -> str:
    lines = [
        "# Cluster 2 Representative Timelines",
        "",
        "- Actor IDs are anonymized.",
        "- Raw actor IDs, emails, usernames, and names are not shown.",
        "",
    ]
    cluster2_events = events[events["cluster_label"] == 2].copy()
    for subgroup, subset in subgroups.groupby("subgroup_candidate"):
        reps = subset.sort_values(["subgroup_score", "total_events"], ascending=[False, False]).head(3)["actor_id"].tolist()
        timeline = cluster2_events[cluster2_events["actor_id"].isin(reps)].sort_values(["actor_id", "timestamp", "source", "event_type"])
        timeline = timeline.groupby("actor_id").head(8)
        display = timeline[["actor_id", "timestamp", "project_name", "source", "event_type", "lifecycle_stage", "issue_external_id"]]
        lines.extend([f"## {subgroup}", "", dataframe_to_markdown(display) if not display.empty else "No events found.", ""])
    return "\n".join(lines) + "\n"


def project_distribution_markdown(distribution: pd.DataFrame, dominant: pd.DataFrame) -> str:
    top = distribution.groupby("cluster_label", group_keys=False).head(5)
    lines = [
        "# Project Cluster Distribution",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "- Values include clustered-cohort events only.",
        "",
        "## Dominant Project By Cluster",
        "",
        dataframe_to_markdown(dominant),
        "",
        "## Top Project Contributions By Cluster",
        "",
        dataframe_to_markdown(top),
    ]
    return "\n".join(lines) + "\n"


def bias_summary_markdown(dominant: pd.DataFrame, distribution: pd.DataFrame) -> str:
    flagged = dominant[dominant["bias_flag_over_0_60"]]
    noise = dominant[dominant["cluster_label"] == -1]
    noise_text = "Noise cluster not present."
    if not noise.empty:
        row = noise.iloc[0]
        noise_text = f"Noise dominant project is `{row['dominant_project']}` with event ratio `{row['dominant_project_event_ratio']:.3f}`."
    lines = [
        "# Cluster Project Bias Summary",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "- Dominant project ratio is computed within each cluster using event counts.",
        "",
        "## Bias Check",
        "",
        dataframe_to_markdown(flagged) if not flagged.empty else "No cluster has dominant project ratio > 0.60.",
        "",
        "## Noise Project Distribution",
        "",
        noise_text,
        "",
        "## Interpretation",
        "",
        "- A ratio above 0.60 would indicate strong project concentration requiring caution.",
        "- Current results should be interpreted alongside role profiles and stability analysis.",
    ]
    return "\n".join(lines) + "\n"


def cluster2_summary_markdown(feature_dist: pd.DataFrame, subgroups: pd.DataFrame, cluster2: pd.DataFrame, dominant: pd.DataFrame) -> str:
    subgroup_counts = subgroups["subgroup_candidate"].value_counts().reset_index()
    subgroup_counts.columns = ["subgroup_candidate", "n_actors"]
    cluster2_dom = dominant[dominant["cluster_label"] == 2]
    dom_text = ""
    if not cluster2_dom.empty:
        row = cluster2_dom.iloc[0]
        dom_text = f"- Dominant project: `{row['dominant_project']}` / `{row['dominant_project_event_ratio']:.3f}`"
    lines = [
        "# Cluster 2 Diagnostic Summary",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "- This is a diagnostic analysis only; main cluster labels are not changed.",
        "- DB access: not used.",
        "",
        "## Cluster 2 Size",
        "",
        f"- Actors: `{len(cluster2)}`",
        f"- Median total_events: `{cluster2['total_events'].median():.1f}`",
        f"- Median n_unique_issues: `{cluster2['n_unique_issues'].median():.1f}`",
        dom_text,
        "",
        "## Feature Distribution",
        "",
        dataframe_to_markdown(feature_dist),
        "",
        "## Rule-Based Subgroup Candidates",
        "",
        dataframe_to_markdown(subgroup_counts),
        "",
        "## Judgment",
        "",
        "- Cluster 2 combines reporting, issue comments, and issue event/unknown field-change activity.",
        "- `Discussion Participant` is probably too narrow because reporting and issue workflow signals are also visible.",
        "- A broader candidate name such as `Issue-side Participant` is more defensible at this stage.",
        "- The subgroup candidates show internal variation, but this diagnostic does not justify overwriting the main HDBSCAN result yet.",
        "- If later role cards need more granularity, split candidates should be based on reporting_ratio, discussion_ratio, issue_comment counts, and issue field-change / triage features.",
        "- Maintain the main clustering for now and use this diagnostic in role card refinement.",
    ]
    return "\n".join(lines) + "\n"


def noise_project_markdown(distribution: pd.DataFrame) -> str:
    noise = distribution[distribution["cluster_label"] == -1].sort_values("cluster_project_ratio", ascending=False)
    lines = [
        "# Noise Project Distribution",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "- Values are event-based within the original HDBSCAN noise label.",
        "",
        dataframe_to_markdown(noise),
        "",
        "## Interpretation",
        "",
        "- Noise should be considered project-skewed only if one project dominates strongly, e.g. > 0.60.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    labeled_features, labels, events, profiles, _ = load_inputs(args)
    results_dir = ROOT / "reports" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    distribution = project_cluster_distribution(events, labels)
    dominant = dominant_project_table(distribution)
    cluster2 = labeled_features[labeled_features["cluster_label"] == 2].copy()
    feature_dist = feature_distribution(cluster2)
    subgroups = cluster2_subgroups(cluster2)

    paths = {
        "project_cluster_distribution_csv": results_dir / "project_cluster_distribution.csv",
        "project_cluster_distribution_md": results_dir / "project_cluster_distribution.md",
        "project_cluster_heatmap_png": results_dir / "project_cluster_heatmap.png",
        "cluster_project_bias_summary_md": results_dir / "cluster_project_bias_summary.md",
        "cluster2_diagnostic_summary_md": results_dir / "cluster2_diagnostic_summary.md",
        "cluster2_feature_distribution_csv": results_dir / "cluster2_feature_distribution.csv",
        "cluster2_subgroup_candidates_csv": results_dir / "cluster2_subgroup_candidates.csv",
        "cluster2_representative_timelines_md": results_dir / "cluster2_representative_timelines.md",
        "noise_project_distribution_md": results_dir / "noise_project_distribution.md",
    }

    write_dataframe(distribution, paths["project_cluster_distribution_csv"])
    write_text(project_distribution_markdown(distribution, dominant), paths["project_cluster_distribution_md"])
    plot_project_cluster_heatmap(distribution, paths["project_cluster_heatmap_png"])
    write_text(bias_summary_markdown(dominant, distribution), paths["cluster_project_bias_summary_md"])
    write_dataframe(feature_dist, paths["cluster2_feature_distribution_csv"])
    write_dataframe(subgroups, paths["cluster2_subgroup_candidates_csv"])
    write_text(cluster2_summary_markdown(feature_dist, subgroups, cluster2, dominant), paths["cluster2_diagnostic_summary_md"])
    write_text(representative_timelines(subgroups, events), paths["cluster2_representative_timelines_md"])
    write_text(noise_project_markdown(distribution), paths["noise_project_distribution_md"])

    print("Project and cluster 2 diagnostics succeeded")
    print(f"- dominant_project_over_0_60_clusters: {dominant[dominant['bias_flag_over_0_60']]['cluster_label'].astype(int).tolist()}")
    print(f"- cluster2_actors: {len(cluster2)}")
    print("Cluster 2 subgroup candidates:")
    for row in subgroups["subgroup_candidate"].value_counts().items():
        print(f"- {row[0]}: {row[1]}")
    print("Generated:")
    for path in paths.values():
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
