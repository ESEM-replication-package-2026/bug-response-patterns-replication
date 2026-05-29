from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from smartshark_roles.io import dataframe_to_markdown, write_dataframe, write_text


BOT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("contains_bot", re.compile(r"bot", re.IGNORECASE)),
    ("jenkins", re.compile(r"jenkins", re.IGNORECASE)),
    ("travis", re.compile(r"travis", re.IGNORECASE)),
    ("github_actions", re.compile(r"github[-_ ]actions", re.IGNORECASE)),
    ("dependabot", re.compile(r"dependabot", re.IGNORECASE)),
    ("asfbot", re.compile(r"asfbot", re.IGNORECASE)),
    ("automation", re.compile(r"automation", re.IGNORECASE)),
    ("ci", re.compile(r"(^|[^a-z0-9])ci([^a-z0-9]|$)", re.IGNORECASE)),
    ("build", re.compile(r"build", re.IGNORECASE)),
    ("sonar", re.compile(r"sonar", re.IGNORECASE)),
    ("codecov", re.compile(r"codecov", re.IGNORECASE)),
    ("coveralls", re.compile(r"coveralls", re.IGNORECASE)),
)


@dataclass(frozen=True)
class ActorNormalizationPaths:
    events_parquet: Path
    events_csv: Path
    actor_mapping_csv: Path
    normalization_summary_md: Path
    normalization_counts_csv: Path
    bot_summary_md: Path
    bot_counts_csv: Path


@dataclass(frozen=True)
class ActorNormalizationMetrics:
    input_events: int
    output_events: int
    duplicate_event_ids: int
    missing_actor_id_raw: int
    unique_raw_actors: int
    unique_normalized_actors: int
    bot_actors: int
    bot_events: int
    bot_event_rate: float


def read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Events file does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported events file type: {path.suffix}")


def actor_raw_to_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def bot_reason_for_raw_actor(raw_actor: str | None) -> str:
    raw_actor_text = actor_raw_to_string(raw_actor)
    if raw_actor_text is None:
        return ""
    reasons = [rule_name for rule_name, pattern in BOT_RULES if pattern.search(raw_actor_text)]
    return ";".join(reasons)


def build_actor_mapping(raw_actor_values: pd.Series) -> pd.DataFrame:
    cleaned = raw_actor_values.map(actor_raw_to_string)
    unique_values = sorted(value for value in cleaned.dropna().unique())
    rows = [
        {
            "actor_id_raw": raw_actor,
            "actor_id": f"actor_{index:06d}",
        }
        for index, raw_actor in enumerate(unique_values, start=1)
    ]
    return pd.DataFrame(rows, columns=["actor_id_raw", "actor_id"])


def normalize_actor_columns(events: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if "actor_id_raw" not in events.columns:
        raise KeyError("events table must contain actor_id_raw")

    normalized = events.copy()
    raw_actor_key = normalized["actor_id_raw"].map(actor_raw_to_string)
    actor_lookup = dict(zip(mapping["actor_id_raw"], mapping["actor_id"], strict=True))

    normalized["actor_id"] = raw_actor_key.map(actor_lookup)
    normalized["actor_unknown"] = raw_actor_key.isna()
    bot_reason = raw_actor_key.map(bot_reason_for_raw_actor)
    normalized["bot_reason"] = bot_reason.fillna("")
    normalized["is_bot"] = (~normalized["actor_unknown"]) & normalized["bot_reason"].ne("")
    return normalized


def validate_normalization(input_events: pd.DataFrame, output_events: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if len(input_events) != len(output_events):
        issues.append(f"Event row count changed: input={len(input_events)} output={len(output_events)}")
    if "event_id" in output_events.columns:
        duplicate_ids = int(output_events["event_id"].duplicated().sum())
        if duplicate_ids:
            issues.append(f"Duplicate event_id values detected: {duplicate_ids}")
    else:
        issues.append("event_id column is missing")
    return issues


def scalar_metrics(input_events: pd.DataFrame, output_events: pd.DataFrame) -> ActorNormalizationMetrics:
    total = len(output_events)
    duplicate_event_ids = int(output_events["event_id"].duplicated().sum()) if "event_id" in output_events else 0
    missing_actor = int(output_events["actor_unknown"].sum())
    unique_raw = int(output_events.loc[~output_events["actor_unknown"], "actor_id_raw"].map(actor_raw_to_string).nunique())
    unique_normalized = int(output_events["actor_id"].dropna().nunique())
    bot_events = int(output_events["is_bot"].sum())
    bot_actors = int(output_events.loc[output_events["is_bot"], "actor_id"].dropna().nunique())
    return ActorNormalizationMetrics(
        input_events=len(input_events),
        output_events=total,
        duplicate_event_ids=duplicate_event_ids,
        missing_actor_id_raw=missing_actor,
        unique_raw_actors=unique_raw,
        unique_normalized_actors=unique_normalized,
        bot_actors=bot_actors,
        bot_events=bot_events,
        bot_event_rate=bot_events / total if total else 0.0,
    )


def grouped_actor_counts(events: pd.DataFrame, dimensions: list[str], section: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["section", *dimensions, "n_events", "actor_unknown_events", "bot_events", "actor_unknown_rate", "bot_event_rate"])
    grouped = (
        events.groupby(dimensions, dropna=False)
        .agg(
            n_events=("event_id", "size"),
            actor_unknown_events=("actor_unknown", "sum"),
            bot_events=("is_bot", "sum"),
        )
        .reset_index()
    )
    grouped["actor_unknown_rate"] = grouped["actor_unknown_events"] / grouped["n_events"]
    grouped["bot_event_rate"] = grouped["bot_events"] / grouped["n_events"]
    grouped.insert(0, "section", section)
    return grouped


def build_normalization_counts(events: pd.DataFrame, metrics: ActorNormalizationMetrics) -> pd.DataFrame:
    overall = pd.DataFrame(
        [
            {"section": "overall", "metric": "input_events", "value": metrics.input_events, "rate": 1.0},
            {"section": "overall", "metric": "output_events", "value": metrics.output_events, "rate": 1.0},
            {"section": "overall", "metric": "duplicate_event_ids", "value": metrics.duplicate_event_ids, "rate": metrics.duplicate_event_ids / metrics.output_events if metrics.output_events else 0.0},
            {"section": "overall", "metric": "missing_actor_id_raw", "value": metrics.missing_actor_id_raw, "rate": metrics.missing_actor_id_raw / metrics.output_events if metrics.output_events else 0.0},
            {"section": "overall", "metric": "unique_raw_actors", "value": metrics.unique_raw_actors, "rate": pd.NA},
            {"section": "overall", "metric": "unique_normalized_actors", "value": metrics.unique_normalized_actors, "rate": pd.NA},
        ]
    )
    return pd.concat(
        [
            overall,
            grouped_actor_counts(events, ["project_name"], "project"),
            grouped_actor_counts(events, ["source"], "source"),
            grouped_actor_counts(events, ["event_type"], "event_type"),
        ],
        ignore_index=True,
        sort=False,
    )


def build_bot_counts(events: pd.DataFrame, metrics: ActorNormalizationMetrics) -> pd.DataFrame:
    reason_counts = (
        events.loc[events["is_bot"]]
        .groupby("bot_reason", dropna=False)
        .agg(n_events=("event_id", "size"), n_actors=("actor_id", "nunique"))
        .reset_index()
        .sort_values(["n_events", "bot_reason"], ascending=[False, True])
    )
    reason_counts.insert(0, "section", "bot_reason")

    overall = pd.DataFrame(
        [
            {"section": "overall", "metric": "bot_actors", "value": metrics.bot_actors, "rate": metrics.bot_actors / metrics.unique_normalized_actors if metrics.unique_normalized_actors else 0.0},
            {"section": "overall", "metric": "bot_events", "value": metrics.bot_events, "rate": metrics.bot_event_rate},
        ]
    )
    return pd.concat(
        [
            overall,
            reason_counts,
            grouped_actor_counts(events, ["project_name"], "project_bot"),
            grouped_actor_counts(events, ["source"], "source_bot"),
            grouped_actor_counts(events, ["event_type"], "event_type_bot"),
        ],
        ignore_index=True,
        sort=False,
    )


def normalization_summary(
    metrics: ActorNormalizationMetrics,
    events: pd.DataFrame,
    validation_issues: list[str],
    events_path: Path,
    mapping_path: Path,
) -> str:
    project_counts = grouped_actor_counts(events, ["project_name"], "project").drop(columns=["section"], errors="ignore")
    source_counts = grouped_actor_counts(events, ["source"], "source").drop(columns=["section"], errors="ignore").sort_values("actor_unknown_events", ascending=False)
    event_type_counts = grouped_actor_counts(events, ["event_type"], "event_type").drop(columns=["section"], errors="ignore").sort_values("actor_unknown_events", ascending=False).head(25)

    validation_text = "No validation issues detected." if not validation_issues else "\n".join(f"- {issue}" for issue in validation_issues)
    lines = [
        "# Actor Normalization Summary",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Input events: `{events_path}`",
        f"- Internal actor mapping: `{mapping_path}`",
        "- Privacy note: reports intentionally contain only aggregate counts. Raw actor identifiers are stored only in the internal mapping CSV and should be excluded from replication packages.",
        "",
        "## Overall",
        "",
        f"- Input events: `{metrics.input_events}`",
        f"- Output events: `{metrics.output_events}`",
        f"- Duplicate event_id count: `{metrics.duplicate_event_ids}`",
        f"- actor_id_raw missing count/rate: `{metrics.missing_actor_id_raw}` / `{metrics.missing_actor_id_raw / metrics.output_events:.4f}`",
        f"- Unique raw actors: `{metrics.unique_raw_actors}`",
        f"- Unique normalized actors: `{metrics.unique_normalized_actors}`",
        "",
        "## Project Counts",
        "",
        dataframe_to_markdown(project_counts),
        "",
        "## Source Counts",
        "",
        dataframe_to_markdown(source_counts),
        "",
        "## Event Type Counts",
        "",
        dataframe_to_markdown(event_type_counts),
        "",
        "## Validation",
        "",
        validation_text,
    ]
    return "\n".join(lines) + "\n"


def bot_summary(metrics: ActorNormalizationMetrics, events: pd.DataFrame) -> str:
    project_counts = grouped_actor_counts(events, ["project_name"], "project").drop(columns=["section"], errors="ignore").sort_values("bot_events", ascending=False)
    source_counts = grouped_actor_counts(events, ["source"], "source").drop(columns=["section"], errors="ignore").sort_values("bot_events", ascending=False)
    event_type_counts = grouped_actor_counts(events, ["event_type"], "event_type").drop(columns=["section"], errors="ignore").sort_values("bot_events", ascending=False).head(25)
    reason_counts = (
        events.loc[events["is_bot"]]
        .groupby("bot_reason", dropna=False)
        .agg(n_events=("event_id", "size"), n_actors=("actor_id", "nunique"))
        .reset_index()
        .sort_values("n_events", ascending=False)
    )

    reason_text = dataframe_to_markdown(reason_counts) if not reason_counts.empty else "No bot actors matched the configured string rules."
    no_match_note = (
        "- No actors matched these rules. In the current events table, `actor_id_raw` is an opaque SmartSHARK actor identifier, "
        "so name-based automation detection may require a later read-only identity enrichment step if higher bot recall is needed."
        if metrics.bot_events == 0
        else ""
    )
    lines = [
        "# Bot Detection Summary",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "- Detection rules: case-insensitive raw actor string matches for `bot`, `jenkins`, `travis`, `github-actions`, `dependabot`, `asfbot`, `automation`, token `ci`, and `build`.",
        "- Missing actors are excluded from bot detection.",
        "- Privacy note: no raw actor identifiers are shown in this report.",
        "",
        "## Overall",
        "",
        f"- Bot actors: `{metrics.bot_actors}`",
        f"- Bot events: `{metrics.bot_events}`",
        f"- Bot event rate: `{metrics.bot_event_rate:.4f}`",
        "",
        "## Bot Reason Counts",
        "",
        reason_text,
        "",
        no_match_note,
        "",
        "## Project Bot Counts",
        "",
        dataframe_to_markdown(project_counts),
        "",
        "## Source Bot Counts",
        "",
        dataframe_to_markdown(source_counts),
        "",
        "## Event Type Bot Counts",
        "",
        dataframe_to_markdown(event_type_counts),
    ]
    return "\n".join(lines) + "\n"


def default_output_paths(project_root: Path) -> ActorNormalizationPaths:
    events_dir = project_root / "data" / "interim" / "events"
    report_dir = project_root / "reports" / "build_events"
    return ActorNormalizationPaths(
        events_parquet=events_dir / "events_normalized.parquet",
        events_csv=events_dir / "events_normalized.csv",
        actor_mapping_csv=events_dir / "actor_id_mapping.csv",
        normalization_summary_md=report_dir / "actor_normalization_summary.md",
        normalization_counts_csv=report_dir / "actor_normalization_counts.csv",
        bot_summary_md=report_dir / "bot_detection_summary.md",
        bot_counts_csv=report_dir / "bot_detection_counts.csv",
    )


def run_actor_normalization(events_path: Path, project_root: Path, logger: Any | None = None) -> tuple[ActorNormalizationMetrics, ActorNormalizationPaths]:
    events = read_events(events_path)
    mapping = build_actor_mapping(events["actor_id_raw"])
    normalized = normalize_actor_columns(events, mapping)
    validation_issues = validate_normalization(events, normalized)
    metrics = scalar_metrics(events, normalized)
    paths = default_output_paths(project_root)

    write_dataframe(normalized, paths.events_csv, paths.events_parquet, logger=logger)
    write_dataframe(mapping, paths.actor_mapping_csv, logger=logger)
    write_dataframe(build_normalization_counts(normalized, metrics), paths.normalization_counts_csv, logger=logger)
    write_dataframe(build_bot_counts(normalized, metrics), paths.bot_counts_csv, logger=logger)
    write_text(
        normalization_summary(metrics, normalized, validation_issues, events_path, paths.actor_mapping_csv),
        paths.normalization_summary_md,
        logger=logger,
    )
    write_text(bot_summary(metrics, normalized), paths.bot_summary_md, logger=logger)

    if validation_issues and logger is not None:
        for issue in validation_issues:
            logger.warning("Normalization validation issue: %s", issue)
    return metrics, paths
