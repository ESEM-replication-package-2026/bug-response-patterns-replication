from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smartshark_roles.io import dataframe_to_markdown, write_dataframe, write_text


LIFECYCLE_STAGES = (
    "reporting",
    "discussion",
    "triage",
    "fixing",
    "integration",
    "review",
    "closure",
    "unknown",
)

SOURCES = (
    "issue",
    "issue_comment",
    "event",
    "commit",
    "pull_request",
    "pull_request_comment",
    "pull_request_review",
    "pull_request_review_comment",
)


@dataclass(frozen=True)
class FeatureBuildPaths:
    features_parquet: Path
    features_csv: Path
    summary_md: Path
    summary_csv: Path
    missingness_csv: Path
    build_counts_csv: Path


@dataclass(frozen=True)
class FeatureBuildMetrics:
    input_events: int
    actor_unknown_events: int
    events_used: int
    actor_feature_rows: int
    unique_actors: int
    bot_actors: int
    low_activity_actors: int
    feature_columns_including_actor_id: int
    feature_columns_excluding_actor_id: int


def read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Events file does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported events file type: {path.suffix}")


def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {"actor_id", "actor_unknown", "event_id", "timestamp", "event_type", "lifecycle_stage", "source", "event_scope"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise KeyError(f"events table is missing required columns: {missing}")

    actor_unknown = events["actor_unknown"].fillna(True).astype(bool)
    prepared = events.loc[~actor_unknown & events["actor_id"].notna()].copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce", utc=True)
    prepared["issue_id_clean"] = prepared.get("issue_id", pd.Series(index=prepared.index, dtype="object")).astype("string").replace("", pd.NA)
    prepared["project_name"] = prepared.get("project_name", pd.Series(index=prepared.index, dtype="object")).astype("string")
    prepared["is_bot"] = prepared.get("is_bot", False).fillna(False).astype(bool)

    for column in ("text_length", "files_changed", "lines_added", "lines_deleted"):
        if column not in prepared.columns:
            prepared[column] = 0
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0)
    return prepared


def count_by_actor(events: pd.DataFrame, condition: pd.Series, name: str) -> pd.Series:
    return events.loc[condition].groupby("actor_id").size().rename(name)


def unique_issues_by_actor(events: pd.DataFrame, condition: pd.Series, name: str) -> pd.Series:
    frame = events.loc[condition & events["issue_id_clean"].notna()]
    return frame.groupby("actor_id")["issue_id_clean"].nunique().rename(name)


def category_counts(events: pd.DataFrame, column: str, values: tuple[str, ...]) -> pd.DataFrame:
    counts = events.groupby(["actor_id", column], dropna=False).size().unstack(fill_value=0)
    return counts.reindex(columns=list(values), fill_value=0)


def temporal_features(events: pd.DataFrame, actors: pd.Index) -> pd.DataFrame:
    valid = events.loc[events["timestamp"].notna(), ["actor_id", "timestamp", "lifecycle_stage"]].copy()
    frame = pd.DataFrame(index=actors)
    if valid.empty:
        frame["first_event_at"] = pd.NaT
        frame["last_event_at"] = pd.NaT
        frame["active_days"] = 0
        frame["median_inter_event_hours"] = pd.NA
        frame["first_lifecycle_stage"] = pd.NA
        frame["last_lifecycle_stage"] = pd.NA
        return frame

    valid = valid.sort_values(["actor_id", "timestamp", "lifecycle_stage"], kind="mergesort")
    grouped = valid.groupby("actor_id", sort=False)
    frame["first_event_at"] = grouped["timestamp"].first().reindex(actors)
    frame["last_event_at"] = grouped["timestamp"].last().reindex(actors)
    active_days = valid.assign(event_date=valid["timestamp"].dt.date).groupby("actor_id")["event_date"].nunique()
    frame["active_days"] = active_days.reindex(actors, fill_value=0).astype(int)

    diffs = valid.assign(inter_event_hours=grouped["timestamp"].diff().dt.total_seconds() / 3600.0)
    medians = diffs.groupby("actor_id")["inter_event_hours"].median()
    frame["median_inter_event_hours"] = medians.reindex(actors)

    first_stage = grouped["lifecycle_stage"].first()
    last_stage = grouped["lifecycle_stage"].last()
    frame["first_lifecycle_stage"] = first_stage.reindex(actors)
    frame["last_lifecycle_stage"] = last_stage.reindex(actors)
    return frame


def build_actor_features(events: pd.DataFrame) -> tuple[pd.DataFrame, FeatureBuildMetrics]:
    prepared = prepare_events(events)
    actors = pd.Index(sorted(prepared["actor_id"].dropna().unique()), name="actor_id")
    features = pd.DataFrame(index=actors)
    features["total_events"] = prepared.groupby("actor_id").size().reindex(actors, fill_value=0).astype(int)
    features["n_projects"] = prepared.groupby("actor_id")["project_name"].nunique().reindex(actors, fill_value=0).astype(int)
    features["n_unique_issues"] = prepared.loc[prepared["issue_id_clean"].notna()].groupby("actor_id")["issue_id_clean"].nunique().reindex(actors, fill_value=0).astype(int)
    features["is_bot"] = prepared.groupby("actor_id")["is_bot"].max().reindex(actors, fill_value=False).astype(bool)

    scope_counts = category_counts(prepared, "event_scope", ("issue_linked", "project_level")).reindex(actors, fill_value=0)
    features["n_issue_linked_events"] = scope_counts["issue_linked"].astype(int)
    features["n_project_level_events"] = scope_counts["project_level"].astype(int)
    features["issue_linked_ratio"] = features["n_issue_linked_events"] / features["total_events"]
    features["project_level_ratio"] = features["n_project_level_events"] / features["total_events"]

    event_type = prepared["event_type"].fillna("")
    source = prepared["source"].fillna("")
    lifecycle = prepared["lifecycle_stage"].fillna("")

    features["n_issues_reported"] = count_by_actor(prepared, event_type.eq("issue_created"), "n_issues_reported").reindex(actors, fill_value=0).astype(int)
    issue_comment_condition = event_type.eq("issue_comment_added")
    features["n_issue_comments"] = count_by_actor(prepared, issue_comment_condition, "n_issue_comments").reindex(actors, fill_value=0).astype(int)
    features["n_unique_issues_commented"] = unique_issues_by_actor(prepared, issue_comment_condition, "n_unique_issues_commented").reindex(actors, fill_value=0).astype(int)
    median_comment_length = prepared.loc[issue_comment_condition].groupby("actor_id")["text_length"].median()
    features["median_comment_length"] = median_comment_length.reindex(actors)

    for event_name, column_name in (
        ("status_changed", "n_status_changes"),
        ("priority_changed", "n_priority_changes"),
        ("resolution_changed", "n_resolution_changes"),
        ("assignee_changed", "n_assignee_changes"),
        ("issue_field_changed", "n_issue_field_changes"),
    ):
        features[column_name] = count_by_actor(prepared, event_type.eq(event_name), column_name).reindex(actors, fill_value=0).astype(int)
    features["n_unknown_issue_field_changes"] = count_by_actor(
        prepared,
        event_type.eq("issue_field_changed") & lifecycle.eq("unknown"),
        "n_unknown_issue_field_changes",
    ).reindex(actors, fill_value=0).astype(int)

    commit_condition = source.eq("commit") | event_type.eq("commit_authored")
    commit_events = prepared.loc[commit_condition]
    features["n_bugfix_commits_authored"] = count_by_actor(prepared, event_type.eq("commit_authored"), "n_bugfix_commits_authored").reindex(actors, fill_value=0).astype(int)
    features["n_commits_linked_to_bug"] = commit_events.groupby("actor_id")["commit_id"].nunique().reindex(actors, fill_value=0).astype(int)
    for source_column, feature_column in (
        ("files_changed", "n_files_changed"),
        ("lines_added", "n_lines_added"),
        ("lines_deleted", "n_lines_deleted"),
    ):
        features[feature_column] = commit_events.groupby("actor_id")[source_column].sum().reindex(actors, fill_value=0)
    features["n_lines_added_deleted_log"] = np.log1p((features["n_lines_added"] + features["n_lines_deleted"]).clip(lower=0))

    for event_name, column_name in (
        ("pr_opened", "n_prs_opened"),
        ("pr_closed", "n_prs_closed"),
        ("pr_merged", "n_prs_merged"),
    ):
        features[column_name] = count_by_actor(prepared, event_type.eq(event_name), column_name).reindex(actors, fill_value=0).astype(int)
    features["n_pr_events_total"] = count_by_actor(prepared, source.eq("pull_request"), "n_pr_events_total").reindex(actors, fill_value=0).astype(int)

    features["n_reviews_submitted"] = count_by_actor(prepared, event_type.eq("review_submitted"), "n_reviews_submitted").reindex(actors, fill_value=0).astype(int)
    features["n_review_comments"] = count_by_actor(prepared, event_type.eq("review_comment_added"), "n_review_comments").reindex(actors, fill_value=0).astype(int)
    features["n_pr_review_comments"] = count_by_actor(prepared, source.eq("pull_request_review_comment"), "n_pr_review_comments").reindex(actors, fill_value=0).astype(int)

    features["n_issues_closed"] = count_by_actor(prepared, event_type.eq("issue_closed"), "n_issues_closed").reindex(actors, fill_value=0).astype(int)
    features["n_issues_resolved"] = count_by_actor(prepared, event_type.eq("issue_resolved"), "n_issues_resolved").reindex(actors, fill_value=0).astype(int)
    features["n_final_status_changes"] = count_by_actor(prepared, lifecycle.eq("closure"), "n_final_status_changes").reindex(actors, fill_value=0).astype(int)

    stage_counts = category_counts(prepared, "lifecycle_stage", LIFECYCLE_STAGES).reindex(actors, fill_value=0)
    for stage in LIFECYCLE_STAGES:
        features[f"{stage}_ratio"] = stage_counts[stage] / features["total_events"]

    source_counts = category_counts(prepared, "source", SOURCES).reindex(actors, fill_value=0)
    for event_source in SOURCES:
        features[f"{event_source}_ratio"] = source_counts[event_source] / features["total_events"]

    temporal = temporal_features(prepared, actors)
    features["active_days"] = temporal["active_days"].astype(int)
    features["first_event_at"] = temporal["first_event_at"]
    features["last_event_at"] = temporal["last_event_at"]
    features["median_inter_event_hours"] = temporal["median_inter_event_hours"]
    features["first_lifecycle_stage"] = temporal["first_lifecycle_stage"]
    features["last_lifecycle_stage"] = temporal["last_lifecycle_stage"]

    features["is_low_activity"] = (features["total_events"] < 3) | (features["n_unique_issues"] < 2)

    ordered_columns = [
        "total_events",
        "n_projects",
        "n_unique_issues",
        "active_days",
        "is_bot",
        "is_low_activity",
        "n_issue_linked_events",
        "n_project_level_events",
        "issue_linked_ratio",
        "project_level_ratio",
        "n_issues_reported",
        "n_issue_comments",
        "n_unique_issues_commented",
        "median_comment_length",
        "n_status_changes",
        "n_priority_changes",
        "n_resolution_changes",
        "n_assignee_changes",
        "n_issue_field_changes",
        "n_unknown_issue_field_changes",
        "n_bugfix_commits_authored",
        "n_commits_linked_to_bug",
        "n_files_changed",
        "n_lines_added",
        "n_lines_deleted",
        "n_lines_added_deleted_log",
        "n_prs_opened",
        "n_prs_closed",
        "n_prs_merged",
        "n_pr_events_total",
        "n_reviews_submitted",
        "n_review_comments",
        "n_pr_review_comments",
        "n_issues_closed",
        "n_issues_resolved",
        "n_final_status_changes",
        *[f"{stage}_ratio" for stage in LIFECYCLE_STAGES],
        *[f"{event_source}_ratio" for event_source in SOURCES],
        "first_event_at",
        "last_event_at",
        "median_inter_event_hours",
        "first_lifecycle_stage",
        "last_lifecycle_stage",
    ]
    features = features[ordered_columns].reset_index()

    metrics = FeatureBuildMetrics(
        input_events=len(events),
        actor_unknown_events=int(events["actor_unknown"].fillna(True).astype(bool).sum()) if "actor_unknown" in events else 0,
        events_used=len(prepared),
        actor_feature_rows=len(features),
        unique_actors=int(features["actor_id"].nunique()),
        bot_actors=int(features["is_bot"].sum()),
        low_activity_actors=int(features["is_low_activity"].sum()),
        feature_columns_including_actor_id=len(features.columns),
        feature_columns_excluding_actor_id=len(features.columns) - 1,
    )
    return features, metrics


def describe_feature(series: pd.Series, section: str) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    return {
        "section": section,
        "count": int(numeric.count()),
        "missing": int(numeric.isna().sum()),
        "mean": float(numeric.mean()) if numeric.count() else pd.NA,
        "std": float(numeric.std()) if numeric.count() > 1 else pd.NA,
        "min": float(numeric.min()) if numeric.count() else pd.NA,
        "p25": float(numeric.quantile(0.25)) if numeric.count() else pd.NA,
        "median": float(numeric.quantile(0.5)) if numeric.count() else pd.NA,
        "p75": float(numeric.quantile(0.75)) if numeric.count() else pd.NA,
        "p90": float(numeric.quantile(0.9)) if numeric.count() else pd.NA,
        "p95": float(numeric.quantile(0.95)) if numeric.count() else pd.NA,
        "max": float(numeric.max()) if numeric.count() else pd.NA,
    }


def feature_missingness(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(features)
    for column in features.columns:
        missing = int(features[column].isna().sum())
        rows.append({"feature": column, "missing_count": missing, "missing_rate": missing / total if total else 0.0})
    return pd.DataFrame(rows).sort_values(["missing_rate", "feature"], ascending=[False, True])


def build_summary_tables(features: pd.DataFrame, metrics: FeatureBuildMetrics) -> tuple[pd.DataFrame, pd.DataFrame]:
    build_counts = pd.DataFrame(
        [
            {"section": "overall", "metric": "input_events", "value": metrics.input_events, "rate": 1.0},
            {"section": "overall", "metric": "actor_unknown_events", "value": metrics.actor_unknown_events, "rate": metrics.actor_unknown_events / metrics.input_events if metrics.input_events else 0.0},
            {"section": "overall", "metric": "events_used", "value": metrics.events_used, "rate": metrics.events_used / metrics.input_events if metrics.input_events else 0.0},
            {"section": "overall", "metric": "actor_feature_rows", "value": metrics.actor_feature_rows, "rate": pd.NA},
            {"section": "overall", "metric": "unique_actors", "value": metrics.unique_actors, "rate": pd.NA},
            {"section": "overall", "metric": "bot_actors", "value": metrics.bot_actors, "rate": metrics.bot_actors / metrics.actor_feature_rows if metrics.actor_feature_rows else 0.0},
            {"section": "overall", "metric": "low_activity_actors", "value": metrics.low_activity_actors, "rate": metrics.low_activity_actors / metrics.actor_feature_rows if metrics.actor_feature_rows else 0.0},
            {"section": "overall", "metric": "feature_columns_including_actor_id", "value": metrics.feature_columns_including_actor_id, "rate": pd.NA},
            {"section": "overall", "metric": "feature_columns_excluding_actor_id", "value": metrics.feature_columns_excluding_actor_id, "rate": pd.NA},
        ]
    )
    distributions = pd.DataFrame(
        [
            describe_feature(features["total_events"], "total_events_distribution"),
            describe_feature(features["n_unique_issues"], "n_unique_issues_distribution"),
            describe_feature(features["project_level_ratio"], "project_level_ratio_distribution"),
        ]
    )
    summary = pd.concat([build_counts, distributions], ignore_index=True, sort=False)
    return summary, build_counts


def summary_markdown(
    features: pd.DataFrame,
    metrics: FeatureBuildMetrics,
    summary: pd.DataFrame,
    missingness: pd.DataFrame,
    events_path: Path,
) -> str:
    top_missing = missingness.loc[missingness["missing_count"] > 0].head(15)
    distributions = summary[summary["section"].str.endswith("_distribution")]
    lines = [
        "# Actor Feature Summary",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Input events: `{events_path}`",
        "- DB access: not used in this step.",
        "- Privacy note: raw actor identifiers and metadata strings are not included in feature outputs or reports.",
        "",
        "## Overall",
        "",
        f"- Input events: `{metrics.input_events}`",
        f"- actor_unknown excluded events: `{metrics.actor_unknown_events}`",
        f"- Events used for features: `{metrics.events_used}`",
        f"- Actor feature rows: `{metrics.actor_feature_rows}`",
        f"- Unique actors: `{metrics.unique_actors}`",
        f"- Bot actors retained: `{metrics.bot_actors}`",
        f"- Low activity actors: `{metrics.low_activity_actors}`",
        f"- Feature columns including actor_id: `{metrics.feature_columns_including_actor_id}`",
        f"- Feature columns excluding actor_id: `{metrics.feature_columns_excluding_actor_id}`",
        "",
        "## Missing Value Policy",
        "",
        "- `actor_unknown=true` rows are excluded before aggregation.",
        "- Bot actors are retained and marked by `is_bot`.",
        "- Count features include rows with missing timestamps.",
        "- Temporal features ignore rows with missing timestamps.",
        "- Project-level events with missing `issue_id` do not contribute to `n_unique_issues`.",
        "- Count and ratio features are zero-filled when an actor has no matching events; median and temporal features remain missing when undefined.",
        "",
        "## Distributions",
        "",
        dataframe_to_markdown(distributions),
        "",
        "## Top Missing Features",
        "",
        dataframe_to_markdown(top_missing) if not top_missing.empty else "No missing feature values.",
    ]
    return "\n".join(lines) + "\n"


def default_paths(project_root: Path) -> FeatureBuildPaths:
    features_dir = project_root / "data" / "interim" / "features"
    report_dir = project_root / "reports" / "features"
    return FeatureBuildPaths(
        features_parquet=features_dir / "actor_features_raw.parquet",
        features_csv=features_dir / "actor_features_raw.csv",
        summary_md=report_dir / "actor_feature_summary.md",
        summary_csv=report_dir / "actor_feature_summary.csv",
        missingness_csv=report_dir / "feature_missingness.csv",
        build_counts_csv=report_dir / "actor_feature_build_counts.csv",
    )


def run_actor_feature_build(events_path: Path, project_root: Path, logger: Any | None = None) -> tuple[pd.DataFrame, FeatureBuildMetrics, FeatureBuildPaths]:
    events = read_events(events_path)
    features, metrics = build_actor_features(events)
    summary, build_counts = build_summary_tables(features, metrics)
    missingness = feature_missingness(features)
    paths = default_paths(project_root)

    paths.features_parquet.parent.mkdir(parents=True, exist_ok=True)
    paths.summary_md.parent.mkdir(parents=True, exist_ok=True)
    write_dataframe(features, paths.features_csv, paths.features_parquet, logger=logger)
    write_dataframe(summary, paths.summary_csv, logger=logger)
    write_dataframe(missingness, paths.missingness_csv, logger=logger)
    write_dataframe(build_counts, paths.build_counts_csv, logger=logger)
    write_text(summary_markdown(features, metrics, summary, missingness, events_path), paths.summary_md, logger=logger)
    return features, metrics, paths
