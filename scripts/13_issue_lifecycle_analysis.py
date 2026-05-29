from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports" / "results" / "matplotlib_cache"))

from smartshark_roles.config import ensure_project_directories, load_config, project_path
from smartshark_roles.io import dataframe_to_markdown, utc_now_slug, write_dataframe, write_text
from smartshark_roles.logging_utils import setup_logging


ROLE_ORDER = [
    "Issue-side Boundary Participant",
    "Issue-side Participant",
    "Fixer / Code Contributor",
    "Review-focused Participant",
    "Review-Integration Hybrid",
    "PR Integrator",
    "noise_or_boundary",
]
ROLE_RANK = {role: index for index, role in enumerate(ROLE_ORDER)}
NO_CLUSTERED_ROLE = "NO_CLUSTERED_ROLE"
NO_NON_NOISE_ROLE = "NO_NON_NOISE_ROLE"


EVENT_COLUMNS = [
    "event_id",
    "project_name",
    "issue_id",
    "issue_external_id",
    "actor_id",
    "actor_unknown",
    "timestamp",
    "source",
    "event_type",
    "lifecycle_stage",
    "event_scope",
    "files_changed",
    "lines_added",
    "lines_deleted",
    "pr_id",
    "review_id",
    "commit_id",
]


@dataclass(frozen=True)
class AnalysisPaths:
    issue_role_sets_csv: Path
    role_set_frequency_csv: Path
    issue_lifecycle_metrics_csv: Path
    role_set_outcome_summary_csv: Path
    role_sequence_summary_csv: Path
    role_transition_counts_csv: Path
    time_to_close_by_role_set_csv: Path
    patch_size_by_role_set_csv: Path
    summary_md: Path
    time_to_close_figure: Path
    role_set_frequency_figure: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ3 issue-level lifecycle analysis with candidate role labels.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--events", default="data/interim/events/events_normalized.parquet")
    parser.add_argument("--labels", default="data/processed/clusters/actor_cluster_labels.csv")
    parser.add_argument("--taxonomy", default="reports/cluster_profiles/candidate_role_taxonomy.csv")
    parser.add_argument("--features", default="data/interim/features/actor_features_raw.parquet")
    parser.add_argument("--top-n-role-sets", type=int, default=20)
    parser.add_argument("--min-issues-for-trend", type=int, default=10)
    return parser.parse_args()


def read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Events file does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        events = pd.read_parquet(path, columns=EVENT_COLUMNS)
    else:
        events = pd.read_csv(path, usecols=EVENT_COLUMNS)
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)
    for column in ["files_changed", "lines_added", "lines_deleted"]:
        events[column] = pd.to_numeric(events[column], errors="coerce").fillna(0)
    events["actor_unknown"] = events["actor_unknown"].fillna(True).astype(bool)
    return events


def read_features(path: Path) -> pd.DataFrame:
    columns = ["actor_id", "is_bot", "is_low_activity"]
    if not path.exists():
        raise FileNotFoundError(f"Feature file does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def first_non_null(series: pd.Series) -> Any:
    values = series.dropna()
    return values.iloc[0] if not values.empty else pd.NA


def ordered_roles(values: pd.Series) -> list[str]:
    unique = {str(value) for value in values.dropna() if str(value)}
    return sorted(unique, key=lambda value: (ROLE_RANK.get(value, 999), value))


def join_role_set(values: pd.Series, empty_label: str) -> str:
    roles = ordered_roles(values)
    return " + ".join(roles) if roles else empty_label


def compressed_sequence(values: list[str]) -> list[str]:
    sequence: list[str] = []
    for value in values:
        if not value:
            continue
        if not sequence or sequence[-1] != value:
            sequence.append(value)
    return sequence


def role_lookup(labels: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    required_taxonomy = {"cluster_label", "candidate_role_label", "role_status", "is_primary_candidate"}
    missing = required_taxonomy - set(taxonomy.columns)
    if missing:
        raise KeyError(f"candidate_role_taxonomy.csv is missing required columns: {sorted(missing)}")

    lookup = labels[["actor_id", "cluster_label"]].merge(
        taxonomy[["cluster_label", "candidate_role_label", "role_status", "is_primary_candidate"]],
        on="cluster_label",
        how="left",
        validate="many_to_one",
    )
    if lookup["candidate_role_label"].isna().any():
        missing_labels = sorted(lookup.loc[lookup["candidate_role_label"].isna(), "cluster_label"].dropna().unique())
        raise ValueError(f"Missing candidate role labels for clusters: {missing_labels}")

    lookup["cluster_label"] = lookup["cluster_label"].astype(int)
    lookup["is_noise_cluster"] = lookup["cluster_label"] == -1
    lookup["analysis_role_label"] = lookup["candidate_role_label"]
    lookup.loc[lookup["is_noise_cluster"], "analysis_role_label"] = "noise_or_boundary"
    return lookup


def enrich_issue_events(events: pd.DataFrame, lookup: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    project_level_events = int((events["event_scope"] == "project_level").sum())
    issue_events = events.loc[(events["event_scope"] == "issue_linked") & events["issue_id"].notna()].copy()
    issue_events["issue_id"] = issue_events["issue_id"].astype(str)

    enriched = issue_events.merge(
        lookup,
        on="actor_id",
        how="left",
        validate="many_to_one",
    )
    enriched = enriched.merge(
        features,
        on="actor_id",
        how="left",
        validate="many_to_one",
    )
    enriched["actor_known"] = ~enriched["actor_unknown"]
    enriched["actor_clustered"] = enriched["cluster_label"].notna()
    enriched["actor_unclustered"] = enriched["actor_known"] & ~enriched["actor_clustered"]
    enriched["is_bot"] = enriched["is_bot"].fillna(False).astype(bool)
    enriched["is_low_activity"] = enriched["is_low_activity"].fillna(False).astype(bool)
    enriched["is_noise_cluster"] = enriched["is_noise_cluster"].fillna(False).astype(bool)
    enriched["role_set_eligible"] = enriched["actor_known"] & enriched["actor_clustered"]
    enriched["non_noise_role_eligible"] = enriched["role_set_eligible"] & ~enriched["is_noise_cluster"]

    diagnostics = {
        "input_events": int(len(events)),
        "issue_linked_events": int(len(issue_events)),
        "project_level_events_excluded": project_level_events,
        "issue_events_missing_timestamp": int(issue_events["timestamp"].isna().sum()),
        "actor_unknown_events_excluded_from_roles": int(issue_events["actor_unknown"].sum()),
        "known_unclustered_actor_events_excluded_from_roles": int(enriched["actor_unclustered"].sum()),
        "unique_known_issue_actors": int(enriched.loc[enriched["actor_known"], "actor_id"].nunique()),
        "unique_clustered_issue_actors": int(enriched.loc[enriched["actor_clustered"], "actor_id"].nunique()),
        "unique_unclustered_issue_actors": int(enriched.loc[enriched["actor_unclustered"], "actor_id"].nunique()),
        "unique_low_activity_issue_actors": int(enriched.loc[enriched["actor_unclustered"] & enriched["is_low_activity"], "actor_id"].nunique()),
        "unique_bot_issue_actors": int(enriched.loc[enriched["actor_unclustered"] & enriched["is_bot"], "actor_id"].nunique()),
    }
    return enriched, diagnostics


def build_issue_role_sets(enriched: pd.DataFrame) -> pd.DataFrame:
    base = (
        enriched.groupby("issue_id", dropna=False)
        .agg(
            project_name=("project_name", first_non_null),
            issue_external_id=("issue_external_id", first_non_null),
            n_actor_unknown_events=("actor_unknown", "sum"),
            n_unclustered_actor_events=("actor_unclustered", "sum"),
        )
        .reset_index()
    )

    actor_issue = enriched.loc[enriched["actor_known"], ["issue_id", "actor_id", "cluster_label", "analysis_role_label", "candidate_role_label", "is_noise_cluster", "actor_clustered", "actor_unclustered"]].drop_duplicates()
    clustered = actor_issue.loc[actor_issue["actor_clustered"]].copy()
    non_noise = clustered.loc[~clustered["is_noise_cluster"]].copy()

    with_noise = (
        clustered.groupby("issue_id")["analysis_role_label"]
        .apply(lambda values: join_role_set(values, NO_CLUSTERED_ROLE))
        .reset_index(name="role_set_with_noise")
    )
    without_noise = (
        non_noise.groupby("issue_id")["candidate_role_label"]
        .apply(lambda values: join_role_set(values, NO_NON_NOISE_ROLE))
        .reset_index(name="role_set_without_noise")
    )

    counts = (
        actor_issue.groupby("issue_id")
        .agg(
            n_actors=("actor_id", "nunique"),
            n_clustered_actors=("actor_clustered", "sum"),
            n_unclustered_actors=("actor_unclustered", "sum"),
            n_noise_actors=("is_noise_cluster", "sum"),
        )
        .reset_index()
    )
    non_noise_counts = non_noise.groupby("issue_id")["actor_id"].nunique().reset_index(name="n_non_noise_actors")

    issue_sets = base.merge(with_noise, on="issue_id", how="left")
    issue_sets = issue_sets.merge(without_noise, on="issue_id", how="left")
    issue_sets = issue_sets.merge(counts, on="issue_id", how="left")
    issue_sets = issue_sets.merge(non_noise_counts, on="issue_id", how="left")
    issue_sets["role_set_with_noise"] = issue_sets["role_set_with_noise"].fillna(NO_CLUSTERED_ROLE)
    issue_sets["role_set_without_noise"] = issue_sets["role_set_without_noise"].fillna(NO_NON_NOISE_ROLE)
    for column in ["n_actors", "n_clustered_actors", "n_unclustered_actors", "n_noise_actors", "n_non_noise_actors"]:
        issue_sets[column] = issue_sets[column].fillna(0).astype(int)
    issue_sets["has_role_set_with_noise"] = issue_sets["role_set_with_noise"] != NO_CLUSTERED_ROLE
    issue_sets["has_role_set_without_noise"] = issue_sets["role_set_without_noise"] != NO_NON_NOISE_ROLE
    issue_sets["n_roles_with_noise"] = issue_sets["role_set_with_noise"].apply(lambda value: 0 if value == NO_CLUSTERED_ROLE else len(str(value).split(" + ")))
    issue_sets["n_roles_without_noise"] = issue_sets["role_set_without_noise"].apply(lambda value: 0 if value == NO_NON_NOISE_ROLE else len(str(value).split(" + ")))
    return issue_sets


def count_by_issue(frame: pd.DataFrame, mask: pd.Series, column_name: str) -> pd.DataFrame:
    counts = frame.loc[mask].groupby("issue_id").size().reset_index(name=column_name)
    return counts


def unique_or_event_count(frame: pd.DataFrame, mask: pd.Series, id_column: str, column_name: str) -> pd.DataFrame:
    subset = frame.loc[mask, ["issue_id", "event_id", id_column]].copy()
    if subset.empty:
        return pd.DataFrame(columns=["issue_id", column_name])
    subset["_unit"] = subset[id_column].where(subset[id_column].notna() & (subset[id_column].astype(str) != ""), subset["event_id"])
    return subset.groupby("issue_id")["_unit"].nunique().reset_index(name=column_name)


def build_issue_lifecycle_metrics(enriched: pd.DataFrame, issue_sets: pd.DataFrame) -> pd.DataFrame:
    base = (
        enriched.groupby("issue_id", dropna=False)
        .agg(
            project_name=("project_name", first_non_null),
            issue_external_id=("issue_external_id", first_non_null),
            first_event_at=("timestamp", "min"),
            last_event_at=("timestamp", "max"),
            n_events=("event_id", "count"),
            files_changed=("files_changed", "sum"),
            lines_added=("lines_added", "sum"),
            lines_deleted=("lines_deleted", "sum"),
            n_timestamp_missing_events=("timestamp", lambda values: int(values.isna().sum())),
        )
        .reset_index()
    )

    created = (
        enriched.loc[enriched["event_type"] == "issue_created"]
        .groupby("issue_id")["timestamp"]
        .min()
        .reset_index(name="issue_created_at")
    )
    closed = (
        enriched.loc[(enriched["event_type"].isin(["issue_closed", "issue_resolved"])) | (enriched["lifecycle_stage"] == "closure")]
        .groupby("issue_id")["timestamp"]
        .max()
        .reset_index(name="issue_closed_or_resolved_at")
    )

    masks = {
        "n_comments": enriched["source"].isin(["issue_comment", "pull_request_comment", "pull_request_review_comment"]),
        "n_issue_comments": enriched["source"].eq("issue_comment") | enriched["event_type"].eq("issue_comment_added"),
        "n_review_comments": enriched["source"].isin(["pull_request_comment", "pull_request_review_comment"]) | enriched["event_type"].eq("review_comment_added"),
    }
    metrics = base.merge(created, on="issue_id", how="left").merge(closed, on="issue_id", how="left")
    for column_name, mask in masks.items():
        metrics = metrics.merge(count_by_issue(enriched, mask, column_name), on="issue_id", how="left")

    metrics = metrics.merge(unique_or_event_count(enriched, enriched["source"].eq("commit") | enriched["event_type"].eq("commit_authored"), "commit_id", "n_commits"), on="issue_id", how="left")
    metrics = metrics.merge(unique_or_event_count(enriched, enriched["source"].eq("pull_request") | enriched["event_type"].isin(["pr_opened", "pr_closed", "pr_merged"]), "pr_id", "n_prs"), on="issue_id", how="left")
    metrics = metrics.merge(unique_or_event_count(enriched, enriched["source"].eq("pull_request_review") | enriched["event_type"].eq("review_submitted"), "review_id", "n_reviews"), on="issue_id", how="left")

    count_columns = ["n_comments", "n_issue_comments", "n_review_comments", "n_commits", "n_prs", "n_reviews"]
    for column in count_columns:
        metrics[column] = metrics[column].fillna(0).astype(int)
    metrics["patch_size"] = metrics["lines_added"] + metrics["lines_deleted"]

    metrics = metrics.merge(
        issue_sets[
            [
                "issue_id",
                "role_set_with_noise",
                "role_set_without_noise",
                "n_actors",
                "n_clustered_actors",
                "n_non_noise_actors",
                "n_noise_actors",
                "n_unclustered_actors",
                "n_actor_unknown_events",
                "n_unclustered_actor_events",
                "has_role_set_with_noise",
                "has_role_set_without_noise",
            ]
        ],
        on="issue_id",
        how="left",
    )

    metrics["time_to_close_hours"] = (metrics["issue_closed_or_resolved_at"] - metrics["issue_created_at"]).dt.total_seconds() / 3600
    metrics["time_to_close_days"] = metrics["time_to_close_hours"] / 24
    metrics.loc[metrics["time_to_close_hours"] < 0, ["time_to_close_hours", "time_to_close_days"]] = pd.NA

    role_set = metrics["role_set_without_noise"].fillna("")
    metrics["has_fixer"] = role_set.str.contains("Fixer / Code Contributor", regex=False)
    metrics["has_reviewer"] = role_set.str.contains("Review-focused Participant", regex=False)
    metrics["has_integrator"] = role_set.str.contains("PR Integrator", regex=False)
    metrics["has_issue_side_participant"] = role_set.str.contains("Issue-side Participant", regex=False)
    metrics["has_review_integration_hybrid"] = role_set.str.contains("Review-Integration Hybrid", regex=False)

    ordered_columns = [
        "project_name",
        "issue_id",
        "issue_external_id",
        "first_event_at",
        "last_event_at",
        "issue_created_at",
        "issue_closed_or_resolved_at",
        "time_to_close_hours",
        "time_to_close_days",
        "n_events",
        "n_comments",
        "n_issue_comments",
        "n_commits",
        "n_prs",
        "n_reviews",
        "n_review_comments",
        "n_actors",
        "n_non_noise_actors",
        "n_noise_actors",
        "files_changed",
        "lines_added",
        "lines_deleted",
        "patch_size",
        "has_fixer",
        "has_reviewer",
        "has_integrator",
        "has_issue_side_participant",
        "has_review_integration_hybrid",
        "role_set_with_noise",
        "role_set_without_noise",
        "n_clustered_actors",
        "n_unclustered_actors",
        "n_actor_unknown_events",
        "n_unclustered_actor_events",
        "n_timestamp_missing_events",
        "has_role_set_with_noise",
        "has_role_set_without_noise",
    ]
    return metrics[ordered_columns].sort_values(["project_name", "issue_external_id", "issue_id"])


def build_role_set_frequency(issue_sets: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scope, column in [("including_noise", "role_set_with_noise"), ("excluding_noise", "role_set_without_noise")]:
        total = issue_sets.groupby(column).size().reset_index(name="n_issues").rename(columns={column: "role_set"})
        total["project_name"] = "ALL"
        total["role_set_scope"] = scope
        total["issue_ratio"] = total["n_issues"] / len(issue_sets)
        rows.append(total)

        by_project = (
            issue_sets.groupby(["project_name", column])
            .size()
            .reset_index(name="n_issues")
            .rename(columns={column: "role_set"})
        )
        project_totals = by_project.groupby("project_name")["n_issues"].transform("sum")
        by_project["role_set_scope"] = scope
        by_project["issue_ratio"] = by_project["n_issues"] / project_totals
        rows.append(by_project)
    return pd.concat(rows, ignore_index=True).sort_values(["role_set_scope", "project_name", "n_issues"], ascending=[True, True, False])


def summarize_group(group: pd.DataFrame, role_set: str) -> dict[str, Any]:
    return {
        "role_set": role_set,
        "issue_count": int(len(group)),
        "median_time_to_close_days": group["time_to_close_days"].median(),
        "mean_time_to_close_days": group["time_to_close_days"].mean(),
        "p25_time_to_close_days": group["time_to_close_days"].quantile(0.25),
        "p75_time_to_close_days": group["time_to_close_days"].quantile(0.75),
        "median_n_comments": group["n_comments"].median(),
        "median_n_commits": group["n_commits"].median(),
        "median_patch_size": group["patch_size"].median(),
        "median_n_actors": group["n_actors"].median(),
    }


def build_outcome_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    main = metrics.loc[metrics["role_set_without_noise"] != NO_NON_NOISE_ROLE].copy()
    rows = [summarize_group(group, role_set) for role_set, group in main.groupby("role_set_without_noise", dropna=False)]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["issue_count", "role_set"], ascending=[False, True])


def build_patch_size_table(metrics: pd.DataFrame) -> pd.DataFrame:
    main = metrics.loc[metrics["role_set_without_noise"] != NO_NON_NOISE_ROLE].copy()
    rows: list[dict[str, Any]] = []
    for role_set, group in main.groupby("role_set_without_noise", dropna=False):
        rows.append(
            {
                "role_set": role_set,
                "issue_count": int(len(group)),
                "median_patch_size": group["patch_size"].median(),
                "mean_patch_size": group["patch_size"].mean(),
                "p25_patch_size": group["patch_size"].quantile(0.25),
                "p75_patch_size": group["patch_size"].quantile(0.75),
            }
        )
    return pd.DataFrame(rows).sort_values(["issue_count", "role_set"], ascending=[False, True]) if rows else pd.DataFrame()


def sequence_frame(enriched: pd.DataFrame, include_noise: bool) -> pd.DataFrame:
    events = enriched.loc[enriched["role_set_eligible"] & enriched["timestamp"].notna()].copy()
    if not include_noise:
        events = events.loc[~events["is_noise_cluster"]].copy()
    if events.empty:
        return pd.DataFrame(columns=["issue_id", "role_sequence", "sequence_length"])

    role_column = "analysis_role_label" if include_noise else "candidate_role_label"
    events = events.sort_values(["issue_id", "timestamp", "event_id"], na_position="last")
    rows: list[dict[str, Any]] = []
    for issue_id, group in events.groupby("issue_id", sort=False):
        sequence = compressed_sequence(group[role_column].dropna().astype(str).tolist())
        if not sequence:
            continue
        rows.append(
            {
                "issue_id": issue_id,
                "role_sequence": " -> ".join(sequence),
                "sequence_length": len(sequence),
                "_sequence_list": sequence,
            }
        )
    return pd.DataFrame(rows)


def build_role_sequences(enriched: pd.DataFrame, issue_sets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[pd.DataFrame] = []
    transition_rows: list[dict[str, Any]] = []
    for scope, include_noise in [("including_noise", True), ("excluding_noise", False)]:
        sequences = sequence_frame(enriched, include_noise=include_noise)
        if sequences.empty:
            continue
        summary = sequences.groupby("role_sequence").agg(n_issues=("issue_id", "nunique"), median_sequence_length=("sequence_length", "median")).reset_index()
        summary["sequence_scope"] = scope
        summary_rows.append(summary)

        for _, row in sequences.iterrows():
            sequence = row["_sequence_list"]
            for from_role, to_role in zip(sequence, sequence[1:]):
                transition_rows.append(
                    {
                        "sequence_scope": scope,
                        "issue_id": row["issue_id"],
                        "from_role": from_role,
                        "to_role": to_role,
                    }
                )

    sequence_summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame(columns=["role_sequence", "n_issues", "median_sequence_length", "sequence_scope"])
    sequence_summary = sequence_summary.sort_values(["sequence_scope", "n_issues"], ascending=[True, False])

    if transition_rows:
        transitions = pd.DataFrame(transition_rows)
        transition_counts = (
            transitions.groupby(["sequence_scope", "from_role", "to_role"])
            .agg(n_transitions=("issue_id", "count"), n_issues=("issue_id", "nunique"))
            .reset_index()
            .sort_values(["sequence_scope", "n_transitions"], ascending=[True, False])
        )
    else:
        transition_counts = pd.DataFrame(columns=["sequence_scope", "from_role", "to_role", "n_transitions", "n_issues"])
    return sequence_summary, transition_counts


def plot_role_set_frequency(frequency: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = frequency.loc[(frequency["role_set_scope"] == "excluding_noise") & (frequency["project_name"] == "ALL")].copy()
    subset = subset.loc[subset["role_set"] != NO_NON_NOISE_ROLE].head(top_n)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if subset.empty:
        output_path.write_text("No role set frequency data available.", encoding="utf-8")
        return
    labels = [textwrap.shorten(role_set, width=70, placeholder="...") for role_set in subset["role_set"]]
    fig_height = max(4, 0.45 * len(subset) + 1)
    plt.figure(figsize=(11, fig_height))
    plt.barh(labels[::-1], subset["n_issues"].iloc[::-1])
    plt.xlabel("Issues")
    plt.ylabel("Role set")
    plt.title("Top role sets by issue frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_time_to_close(metrics: pd.DataFrame, frequency: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_role_sets = (
        frequency.loc[(frequency["role_set_scope"] == "excluding_noise") & (frequency["project_name"] == "ALL") & (frequency["role_set"] != NO_NON_NOISE_ROLE)]
        .head(top_n)["role_set"]
        .tolist()
    )
    data = [
        metrics.loc[(metrics["role_set_without_noise"] == role_set) & metrics["time_to_close_days"].notna(), "time_to_close_days"].to_numpy()
        for role_set in top_role_sets
    ]
    kept = [(role_set, values) for role_set, values in zip(top_role_sets, data) if len(values) > 0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not kept:
        output_path.write_text("No time-to-close data available.", encoding="utf-8")
        return

    labels = [textwrap.shorten(role_set, width=55, placeholder="...") for role_set, _ in kept]
    values = [values for _, values in kept]
    fig_height = max(4, 0.45 * len(values) + 1)
    plt.figure(figsize=(11, fig_height))
    plt.boxplot(values, vert=False, tick_labels=labels)
    plt.xlabel("Time to close (days)")
    plt.ylabel("Role set")
    plt.title("Time to close by top role set")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def trend_sentence(outcome: pd.DataFrame, column: str, label: str, min_issues: int) -> str:
    if outcome.empty or column not in outcome.columns:
        return "No outcome trend could be computed."
    eligible = outcome.loc[(outcome["issue_count"] >= min_issues) & outcome[column].notna()].copy()
    if eligible.empty:
        return f"No {label} trend met the minimum issue threshold of {min_issues}."
    low = eligible.sort_values(column, ascending=True).iloc[0]
    high = eligible.sort_values(column, ascending=False).iloc[0]
    return (
        f"Among role sets with at least {min_issues} issues, the lowest median {label} was "
        f"{low[column]:.3f} for `{low['role_set']}`, while the highest was {high[column]:.3f} "
        f"for `{high['role_set']}`."
    )


def build_summary(
    issue_sets: pd.DataFrame,
    metrics: pd.DataFrame,
    frequency: pd.DataFrame,
    outcome: pd.DataFrame,
    diagnostics: dict[str, int],
    args: argparse.Namespace,
) -> str:
    total_issues = len(issue_sets)
    with_noise_count = int(issue_sets["has_role_set_with_noise"].sum())
    without_noise_count = int(issue_sets["has_role_set_without_noise"].sum())
    no_non_noise_role_count = int((issue_sets["role_set_without_noise"] == NO_NON_NOISE_ROLE).sum())
    top_role_sets = frequency.loc[
        (frequency["role_set_scope"] == "excluding_noise")
        & (frequency["project_name"] == "ALL")
        & (frequency["role_set"] != NO_NON_NOISE_ROLE)
    ].head(10)

    role_rates = pd.DataFrame(
        [
            {"role_indicator": "has_fixer", "issue_count": int(metrics["has_fixer"].sum()), "issue_ratio": float(metrics["has_fixer"].mean())},
            {"role_indicator": "has_reviewer", "issue_count": int(metrics["has_reviewer"].sum()), "issue_ratio": float(metrics["has_reviewer"].mean())},
            {"role_indicator": "has_integrator", "issue_count": int(metrics["has_integrator"].sum()), "issue_ratio": float(metrics["has_integrator"].mean())},
            {"role_indicator": "has_issue_side_participant", "issue_count": int(metrics["has_issue_side_participant"].sum()), "issue_ratio": float(metrics["has_issue_side_participant"].mean())},
            {"role_indicator": "has_review_integration_hybrid", "issue_count": int(metrics["has_review_integration_hybrid"].sum()), "issue_ratio": float(metrics["has_review_integration_hybrid"].mean())},
        ]
    )

    time_trend = trend_sentence(outcome, "median_time_to_close_days", "time-to-close days", args.min_issues_for_trend)
    patch_trend = trend_sentence(outcome, "median_patch_size", "patch size", args.min_issues_for_trend)
    missing_time_to_close = int(metrics["time_to_close_days"].isna().sum())

    lines = [
        "# RQ3 Issue-Level Lifecycle Analysis Summary",
        "",
        "This analysis uses candidate_role_label values derived from observed cluster behavior. These labels are not treated as formal positions or ground-truth identities.",
        "",
        "## Scope",
        "",
        f"- Input events: {diagnostics['input_events']}",
        f"- Issue-linked events analyzed: {diagnostics['issue_linked_events']}",
        f"- Project-level events excluded from main issue-level analysis: {diagnostics['project_level_events_excluded']}",
        f"- Target issues: {total_issues}",
        f"- Issues with role set including noise_or_boundary: {with_noise_count}",
        f"- Issues with role set excluding noise: {without_noise_count}",
        f"- Issues with no non-noise candidate role set: {no_non_noise_role_count}",
        f"- Timestamp-missing issue-linked events: {diagnostics['issue_events_missing_timestamp']}",
        f"- Actor-unknown events excluded from role set construction: {diagnostics['actor_unknown_events_excluded_from_roles']}",
        f"- Known but unclustered actor events excluded from role set construction: {diagnostics['known_unclustered_actor_events_excluded_from_roles']}",
        "",
        "Project-level PR/review events were excluded because they are not directly linked to a specific issue and would contaminate issue-level lifecycle role sets.",
        "",
        "## Top Role Sets",
        "",
        dataframe_to_markdown(top_role_sets[["role_set", "n_issues", "issue_ratio"]]),
        "",
        "## Role Indicator Coverage",
        "",
        dataframe_to_markdown(role_rates),
        "",
        "## Outcome Trends",
        "",
        f"- {time_trend}",
        f"- {patch_trend}",
        f"- Issues missing time-to-close because issue creation or closure timestamps were absent: {missing_time_to_close}",
        "",
        "Count-based metrics include timestamp-missing rows. Temporal metrics and role sequence analysis exclude rows with missing timestamps because event order cannot be inferred.",
        "",
        "## Noise Handling",
        "",
        "Noise cluster -1 is mapped to `noise_or_boundary` only in the including-noise role set and is excluded from the main non-noise role set. Noise is not interpreted as a primary role.",
        "",
        "## Caveats",
        "",
        "- Results are descriptive associations, not causal estimates.",
        "- Low-activity and bot-filtered actors that were not present in actor_cluster_labels.csv cannot receive a candidate role label and are excluded from role set construction.",
        "- candidate_role_label should be described in the paper as an interpretation of observed behavior.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def run(args: argparse.Namespace, logger: logging.Logger) -> tuple[dict[str, pd.DataFrame], AnalysisPaths, dict[str, int]]:
    events = read_events(project_path(args.events))
    labels = pd.read_csv(project_path(args.labels))
    taxonomy = pd.read_csv(project_path(args.taxonomy))
    features = read_features(project_path(args.features))

    lookup = role_lookup(labels, taxonomy)
    enriched, diagnostics = enrich_issue_events(events, lookup, features)
    issue_sets = build_issue_role_sets(enriched)
    metrics = build_issue_lifecycle_metrics(enriched, issue_sets)
    frequency = build_role_set_frequency(issue_sets)
    outcome = build_outcome_summary(metrics)
    time_table = (
        outcome[
            [
                "role_set",
                "issue_count",
                "median_time_to_close_days",
                "mean_time_to_close_days",
                "p25_time_to_close_days",
                "p75_time_to_close_days",
            ]
        ].copy()
        if not outcome.empty
        else pd.DataFrame()
    )
    patch_table = build_patch_size_table(metrics)
    sequence_summary, transition_counts = build_role_sequences(enriched, issue_sets)

    paths = AnalysisPaths(
        issue_role_sets_csv=ROOT / "reports" / "results" / "issue_role_sets.csv",
        role_set_frequency_csv=ROOT / "reports" / "results" / "role_set_frequency.csv",
        issue_lifecycle_metrics_csv=ROOT / "reports" / "results" / "issue_lifecycle_metrics.csv",
        role_set_outcome_summary_csv=ROOT / "reports" / "results" / "role_set_outcome_summary.csv",
        role_sequence_summary_csv=ROOT / "reports" / "results" / "role_sequence_summary.csv",
        role_transition_counts_csv=ROOT / "reports" / "results" / "role_transition_counts.csv",
        time_to_close_by_role_set_csv=ROOT / "reports" / "results" / "time_to_close_by_role_set.csv",
        patch_size_by_role_set_csv=ROOT / "reports" / "results" / "patch_size_by_role_set.csv",
        summary_md=ROOT / "reports" / "results" / "rq3_lifecycle_analysis_summary.md",
        time_to_close_figure=ROOT / "reports" / "figures" / "time_to_close_by_role_set.png",
        role_set_frequency_figure=ROOT / "reports" / "figures" / "role_set_frequency.png",
    )

    write_dataframe(issue_sets, paths.issue_role_sets_csv, logger=logger)
    write_dataframe(frequency, paths.role_set_frequency_csv, logger=logger)
    write_dataframe(metrics, paths.issue_lifecycle_metrics_csv, logger=logger)
    write_dataframe(outcome, paths.role_set_outcome_summary_csv, logger=logger)
    write_dataframe(sequence_summary, paths.role_sequence_summary_csv, logger=logger)
    write_dataframe(transition_counts, paths.role_transition_counts_csv, logger=logger)
    write_dataframe(time_table, paths.time_to_close_by_role_set_csv, logger=logger)
    write_dataframe(patch_table, paths.patch_size_by_role_set_csv, logger=logger)
    plot_role_set_frequency(frequency, paths.role_set_frequency_figure, top_n=10)
    logger.info("Wrote figure: %s", paths.role_set_frequency_figure)
    plot_time_to_close(metrics, frequency, paths.time_to_close_figure, top_n=10)
    logger.info("Wrote figure: %s", paths.time_to_close_figure)
    write_text(build_summary(issue_sets, metrics, frequency, outcome, diagnostics, args), paths.summary_md, logger=logger)

    frames = {
        "issue_sets": issue_sets,
        "metrics": metrics,
        "frequency": frequency,
        "outcome": outcome,
        "sequence_summary": sequence_summary,
        "transition_counts": transition_counts,
        "time_table": time_table,
        "patch_table": patch_table,
    }
    return frames, paths, diagnostics


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    logger = setup_logging(ROOT / "reports" / "results" / "logs" / f"issue_lifecycle_analysis_{utc_now_slug()}.log", level=logging.INFO)

    try:
        frames, paths, diagnostics = run(args, logger)
    except Exception:
        logger.exception("RQ3 issue lifecycle analysis failed")
        return 1

    issue_sets = frames["issue_sets"]
    metrics = frames["metrics"]
    frequency = frames["frequency"]
    print("RQ3 issue lifecycle analysis succeeded")
    print(f"- input_events: {diagnostics['input_events']}")
    print(f"- issue_linked_events: {diagnostics['issue_linked_events']}")
    print(f"- project_level_events_excluded: {diagnostics['project_level_events_excluded']}")
    print(f"- target_issues: {len(issue_sets)}")
    print(f"- issues_with_role_set_excluding_noise: {int(issue_sets['has_role_set_without_noise'].sum())}")
    print(f"- has_fixer_ratio: {metrics['has_fixer'].mean():.6f}")
    print(f"- has_reviewer_ratio: {metrics['has_reviewer'].mean():.6f}")
    print(f"- has_integrator_ratio: {metrics['has_integrator'].mean():.6f}")
    print("Top role sets excluding noise:")
    top = frequency.loc[
        (frequency["role_set_scope"] == "excluding_noise")
        & (frequency["project_name"] == "ALL")
        & (frequency["role_set"] != NO_NON_NOISE_ROLE)
    ].head(10)
    for row in top.itertuples():
        print(f"- {row.role_set}: {row.n_issues} ({row.issue_ratio:.4f})")
    print("Generated:")
    for path in paths.__dict__.values():
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
