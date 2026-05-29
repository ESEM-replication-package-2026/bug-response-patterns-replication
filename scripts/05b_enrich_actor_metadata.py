from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from bson import ObjectId
from pymongo.database import Database

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartshark_roles.anonymize import BOT_RULES, actor_raw_to_string, grouped_actor_counts
from smartshark_roles.config import ensure_project_directories, load_config, project_path
from smartshark_roles.io import dataframe_to_markdown, write_dataframe, write_text, utc_now_slug
from smartshark_roles.logging_utils import setup_logging
from smartshark_roles.mongo import get_database


METADATA_COLLECTION_CANDIDATES = (
    "people",
    "person",
    "persons",
    "identity",
    "identities",
    "user",
    "users",
    "account",
    "accounts",
)

METADATA_TEXT_FIELDS = (
    "name",
    "email",
    "username",
    "user_name",
    "login",
    "github_login",
    "display_name",
)

ACTOR_REFERENCE_FIELDS = (
    ("issue", "creator_id", False),
    ("issue", "reporter_id", False),
    ("issue", "assignee_id", False),
    ("issue_comment", "author_id", False),
    ("event", "author_id", False),
    ("commit", "author_id", False),
    ("commit", "committer_id", False),
    ("pull_request", "creator_id", False),
    ("pull_request", "assignee_id", False),
    ("pull_request", "requested_reviewer_ids", True),
    ("pull_request", "linked_user_ids", True),
    ("pull_request_comment", "author_id", False),
    ("pull_request_commit", "author_id", False),
    ("pull_request_commit", "committer_id", False),
    ("pull_request_event", "author_id", False),
    ("pull_request_review", "creator_id", False),
    ("pull_request_review_comment", "creator_id", False),
    ("message", "from_id", False),
    ("message", "to_ids", True),
    ("message", "cc_ids", True),
)

PR_EVENT_AUTHOR_LOGIN_PATHS = (
    ("additional_data", "assigner", "login"),
    ("additional_data", "review_requester", "login"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich normalized actors with read-only metadata and rerun bot detection.")
    parser.add_argument("--events", required=True, help="Input normalized events Parquet or CSV.")
    parser.add_argument("--mapping", required=True, help="Internal actor_id_raw to actor_id mapping CSV.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


def read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Events file does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def actor_object_ids(mapping: pd.DataFrame) -> dict[str, ObjectId]:
    result: dict[str, ObjectId] = {}
    for raw_actor in mapping["actor_id_raw"].map(actor_raw_to_string).dropna():
        if re.fullmatch(r"[0-9a-fA-F]{24}", raw_actor):
            result[raw_actor] = ObjectId(raw_actor)
    return result


def nested_get(document: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = document
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def safe_metadata_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def add_metadata(metadata: dict[str, set[str]], raw_actor: Any, value: Any) -> bool:
    raw = actor_raw_to_string(raw_actor)
    text = safe_metadata_text(value)
    if raw is None or text is None:
        return False
    metadata[raw].add(text)
    return True


def audit_candidate_collections(project_root: Path, collection_names: set[str]) -> pd.DataFrame:
    fields_path = project_root / "reports" / "data_audit" / "collection_fields.csv"
    rows: list[dict[str, Any]] = []
    fields = pd.read_csv(fields_path) if fields_path.exists() else pd.DataFrame(columns=["collection", "field_path"])
    field_column = "field_path" if "field_path" in fields.columns else "field"
    for collection in METADATA_COLLECTION_CANDIDATES:
        collection_fields = fields.loc[fields["collection"] == collection, field_column].astype(str).tolist() if not fields.empty else []
        metadata_fields = [field for field in collection_fields if field.lower() in METADATA_TEXT_FIELDS]
        rows.append(
            {
                "section": "metadata_collection_candidate",
                "collection": collection,
                "exists_in_db": collection in collection_names,
                "metadata_text_fields_detected": len(metadata_fields),
                "metadata_text_field_names": ";".join(metadata_fields),
            }
        )
    return pd.DataFrame(rows)


def reference_field_counts(db: Database, collection_names: set[str], actor_ids: list[ObjectId], logger: logging.Logger) -> pd.DataFrame:
    op_in = "$in"
    elem_match = "$elemMatch"
    rows: list[dict[str, Any]] = []
    for collection, field, is_array in ACTOR_REFERENCE_FIELDS:
        if collection not in collection_names:
            rows.append({"section": "actor_reference_field", "collection": collection, "field": field, "exists_in_db": False, "matching_documents": pd.NA})
            continue
        query = {field: {elem_match: {op_in: actor_ids}}} if is_array else {field: {op_in: actor_ids}}
        try:
            count = db[collection].count_documents(query, maxTimeMS=120000)
        except Exception as exc:
            logger.warning("Could not count actor reference field %s.%s: %s", collection, field, exc)
            count = pd.NA
        rows.append({"section": "actor_reference_field", "collection": collection, "field": field, "exists_in_db": True, "matching_documents": count})
    return pd.DataFrame(rows)


def metadata_from_direct_collections(
    db: Database,
    collection_names: set[str],
    actor_ids: dict[str, ObjectId],
    logger: logging.Logger,
) -> tuple[dict[str, set[str]], pd.DataFrame]:
    metadata: dict[str, set[str]] = defaultdict(set)
    rows: list[dict[str, Any]] = []
    reverse = {oid: raw for raw, oid in actor_ids.items()}
    object_ids = list(actor_ids.values())
    for collection in METADATA_COLLECTION_CANDIDATES:
        if collection not in collection_names:
            continue
        projection = {"_id": 1, **{field: 1 for field in METADATA_TEXT_FIELDS}}
        docs_seen = 0
        actors_resolved_before = len(metadata)
        text_values_added = 0
        try:
            cursor = db[collection].find({"_id": {"$in": object_ids}}, projection)
            for doc in cursor:
                docs_seen += 1
                raw_actor = reverse.get(doc.get("_id"))
                for field in METADATA_TEXT_FIELDS:
                    if add_metadata(metadata, raw_actor, doc.get(field)):
                        text_values_added += 1
        except Exception as exc:
            logger.warning("Could not read metadata collection %s: %s", collection, exc)
        rows.append(
            {
                "section": "metadata_source",
                "metadata_source": collection,
                "documents_seen": docs_seen,
                "actors_resolved": len(metadata) - actors_resolved_before,
                "text_values_added": text_values_added,
            }
        )
    return metadata, pd.DataFrame(rows)


def metadata_from_pull_request_events(
    db: Database,
    collection_names: set[str],
    actor_ids: dict[str, ObjectId],
    logger: logging.Logger,
) -> tuple[dict[str, set[str]], pd.DataFrame]:
    metadata: dict[str, set[str]] = defaultdict(set)
    if "pull_request_event" not in collection_names:
        return metadata, pd.DataFrame(
            [
                {
                    "section": "metadata_source",
                    "metadata_source": "pull_request_event_author_login",
                    "documents_seen": 0,
                    "actors_resolved": 0,
                    "text_values_added": 0,
                }
            ]
        )

    object_ids = list(actor_ids.values())
    projection = {
        "author_id": 1,
        "additional_data.assigner.login": 1,
        "additional_data.review_requester.login": 1,
    }
    docs_seen = 0
    text_values_added = 0
    try:
        cursor = db.pull_request_event.find({"author_id": {"$in": object_ids}}, projection)
        for doc in cursor:
            docs_seen += 1
            for path in PR_EVENT_AUTHOR_LOGIN_PATHS:
                value = nested_get(doc, path)
                if add_metadata(metadata, doc.get("author_id"), value):
                    text_values_added += 1
    except Exception as exc:
        logger.warning("Could not read pull_request_event author login metadata: %s", exc)

    return metadata, pd.DataFrame(
        [
            {
                "section": "metadata_source",
                "metadata_source": "pull_request_event_author_login",
                "documents_seen": docs_seen,
                "actors_resolved": len(metadata),
                "text_values_added": text_values_added,
            }
        ]
    )


def merge_metadata(*sources: dict[str, set[str]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for source in sources:
        for raw_actor, values in source.items():
            merged[raw_actor].update(values)
    return dict(merged)


def bot_reason_for_values(values: list[str | None]) -> str:
    reasons: list[str] = []
    for rule_name, pattern in BOT_RULES:
        if any((text := actor_raw_to_string(value)) is not None and pattern.search(text) for value in values):
            reasons.append(rule_name)
    return ";".join(reasons)


def update_bot_columns(events: pd.DataFrame, metadata: dict[str, set[str]]) -> pd.DataFrame:
    updated = events.copy()
    raw_actor_key = updated["actor_id_raw"].map(actor_raw_to_string)

    def reason(raw_actor: Any) -> str:
        raw_actor_text = actor_raw_to_string(raw_actor)
        if raw_actor_text is None:
            return ""
        return bot_reason_for_values([raw_actor_text, *sorted(metadata.get(raw_actor_text, set()))])

    updated["bot_reason"] = raw_actor_key.map(reason).fillna("")
    if "actor_unknown" not in updated.columns:
        updated["actor_unknown"] = raw_actor_key.isna()
    updated["is_bot"] = (~updated["actor_unknown"]) & updated["bot_reason"].ne("")
    return updated


def scalar_metrics(input_events: pd.DataFrame, output_events: pd.DataFrame, mapping: pd.DataFrame, metadata: dict[str, set[str]]) -> dict[str, Any]:
    total = len(output_events)
    bot_events = int(output_events["is_bot"].sum())
    bot_actors = int(output_events.loc[output_events["is_bot"], "actor_id"].dropna().nunique())
    metadata_actors = len(set(metadata).intersection(set(mapping["actor_id_raw"].map(actor_raw_to_string).dropna())))
    mapping_actors = int(mapping["actor_id_raw"].map(actor_raw_to_string).dropna().nunique())
    return {
        "input_events": len(input_events),
        "output_events": total,
        "duplicate_event_ids": int(output_events["event_id"].duplicated().sum()) if "event_id" in output_events else 0,
        "mapping_actors": mapping_actors,
        "metadata_resolved_actors": metadata_actors,
        "metadata_coverage": metadata_actors / mapping_actors if mapping_actors else 0.0,
        "bot_actors": bot_actors,
        "bot_events": bot_events,
        "bot_event_rate": bot_events / total if total else 0.0,
    }


def enrichment_counts(
    metrics: dict[str, Any],
    candidate_collections: pd.DataFrame,
    reference_counts: pd.DataFrame,
    metadata_sources: pd.DataFrame,
) -> pd.DataFrame:
    overall = pd.DataFrame(
        [
            {"section": "overall", "metric": "input_events", "value": metrics["input_events"], "rate": 1.0},
            {"section": "overall", "metric": "output_events", "value": metrics["output_events"], "rate": 1.0},
            {"section": "overall", "metric": "duplicate_event_ids", "value": metrics["duplicate_event_ids"], "rate": metrics["duplicate_event_ids"] / metrics["output_events"] if metrics["output_events"] else 0.0},
            {"section": "overall", "metric": "mapping_actors", "value": metrics["mapping_actors"], "rate": pd.NA},
            {"section": "overall", "metric": "metadata_resolved_actors", "value": metrics["metadata_resolved_actors"], "rate": metrics["metadata_coverage"]},
        ]
    )
    return pd.concat([overall, candidate_collections, reference_counts, metadata_sources], ignore_index=True, sort=False)


def bot_counts(events: pd.DataFrame, metrics: dict[str, Any]) -> pd.DataFrame:
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
            {"section": "overall", "metric": "bot_actors", "value": metrics["bot_actors"], "rate": metrics["bot_actors"] / metrics["mapping_actors"] if metrics["mapping_actors"] else 0.0},
            {"section": "overall", "metric": "bot_events", "value": metrics["bot_events"], "rate": metrics["bot_event_rate"]},
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


def enrichment_summary(
    metrics: dict[str, Any],
    candidate_collections: pd.DataFrame,
    reference_counts: pd.DataFrame,
    metadata_sources: pd.DataFrame,
    events_path: Path,
    mapping_path: Path,
) -> str:
    existing_metadata = candidate_collections[candidate_collections["exists_in_db"] == True]  # noqa: E712
    top_references = reference_counts.sort_values("matching_documents", ascending=False, na_position="last").head(20)
    lines = [
        "# Actor Metadata Enrichment Summary",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Input events: `{events_path}`",
        f"- Internal actor mapping: `{mapping_path}`",
        "- Privacy note: raw actor identifiers, email addresses, usernames, and names are not written to this report.",
        "- Actor metadata is used only transiently to update bot flags.",
        "",
        "## Overall",
        "",
        f"- Input events: `{metrics['input_events']}`",
        f"- Output events: `{metrics['output_events']}`",
        f"- Duplicate event_id count: `{metrics['duplicate_event_ids']}`",
        f"- Actors in mapping: `{metrics['mapping_actors']}`",
        f"- Actors with metadata resolved: `{metrics['metadata_resolved_actors']}`",
        f"- Metadata coverage: `{metrics['metadata_coverage']:.4f}`",
        "",
        "## Metadata Collections",
        "",
        dataframe_to_markdown(existing_metadata.drop(columns=["section"], errors="ignore")) if not existing_metadata.empty else "No standalone person/identity metadata collection was present in this SmartSHARK dump.",
        "",
        "## Metadata Sources Used",
        "",
        dataframe_to_markdown(metadata_sources.drop(columns=["section"], errors="ignore")),
        "",
        "## Actor Reference Fields",
        "",
        dataframe_to_markdown(top_references.drop(columns=["section"], errors="ignore")),
        "",
        "## Interpretation",
        "",
        "- The DB contains many actor reference fields such as issue/comment/commit/PR author fields.",
        "- The dump does not expose a standalone person or identity collection with names/emails for those ObjectIds.",
        "- A limited enrichment was possible from pull request event documents where `author_id` appears with GitHub login-like metadata in the same document.",
    ]
    return "\n".join(lines) + "\n"


def bot_summary(events: pd.DataFrame, metrics: dict[str, Any]) -> str:
    reason_counts = (
        events.loc[events["is_bot"]]
        .groupby("bot_reason", dropna=False)
        .agg(n_events=("event_id", "size"), n_actors=("actor_id", "nunique"))
        .reset_index()
        .sort_values("n_events", ascending=False)
    )
    project_counts = grouped_actor_counts(events, ["project_name"], "project").drop(columns=["section"], errors="ignore").sort_values("bot_events", ascending=False)
    source_counts = grouped_actor_counts(events, ["source"], "source").drop(columns=["section"], errors="ignore").sort_values("bot_events", ascending=False)
    reason_text = dataframe_to_markdown(reason_counts) if not reason_counts.empty else "No bot actors matched the enriched metadata rules."
    zero_note = (
        "- No bot actors were detected after enrichment. The available metadata coverage is limited because this dump lacks a standalone identity/person collection."
        if metrics["bot_events"] == 0
        else ""
    )
    lines = [
        "# Enriched Bot Detection Summary",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "- Detection rules: case-insensitive matches for `bot`, `jenkins`, `travis`, `github-actions`, `dependabot`, `asfbot`, `automation`, token `ci`, `build`, `sonar`, `codecov`, and `coveralls`.",
        "- Rules are evaluated against raw actor IDs and transiently resolved metadata strings; missing actors are excluded.",
        "- Privacy note: no raw actor identifiers, email addresses, usernames, or names are shown in this report.",
        "",
        "## Overall",
        "",
        f"- Bot actors: `{metrics['bot_actors']}`",
        f"- Bot events: `{metrics['bot_events']}`",
        f"- Bot event rate: `{metrics['bot_event_rate']:.4f}`",
        "",
        "## Bot Reason Counts",
        "",
        reason_text,
        "",
        zero_note,
        "",
        "## Project Bot Counts",
        "",
        dataframe_to_markdown(project_counts),
        "",
        "## Source Bot Counts",
        "",
        dataframe_to_markdown(source_counts),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    logger = setup_logging(ROOT / "reports" / "build_events" / "logs" / f"actor_metadata_enrichment_{utc_now_slug()}.log", level=logging.INFO)

    events_path = project_path(args.events)
    mapping_path = project_path(args.mapping)
    events_csv_path = events_path.with_suffix(".csv")
    enrichment_counts_path = ROOT / "reports" / "build_events" / "actor_metadata_enrichment_counts.csv"
    enrichment_summary_path = ROOT / "reports" / "build_events" / "actor_metadata_enrichment_summary.md"
    bot_counts_path = ROOT / "reports" / "build_events" / "bot_detection_enriched_counts.csv"
    bot_summary_path = ROOT / "reports" / "build_events" / "bot_detection_enriched_summary.md"

    try:
        input_events = read_events(events_path)
        mapping = pd.read_csv(mapping_path)
        db = get_database(config, logger=logger)
        collection_names = set(db.list_collection_names())
        actor_ids = actor_object_ids(mapping)

        candidate_collections = audit_candidate_collections(ROOT, collection_names)
        reference_counts = reference_field_counts(db, collection_names, list(actor_ids.values()), logger)
        direct_metadata, direct_sources = metadata_from_direct_collections(db, collection_names, actor_ids, logger)
        pr_event_metadata, pr_event_sources = metadata_from_pull_request_events(db, collection_names, actor_ids, logger)
        metadata = merge_metadata(direct_metadata, pr_event_metadata)
        metadata_sources = pd.concat([direct_sources, pr_event_sources], ignore_index=True, sort=False)

        output_events = update_bot_columns(input_events, metadata)
        metrics = scalar_metrics(input_events, output_events, mapping, metadata)
        counts = enrichment_counts(metrics, candidate_collections, reference_counts, metadata_sources)
        enriched_bot_counts = bot_counts(output_events, metrics)

        write_dataframe(output_events, events_csv_path, events_path, logger=logger)
        write_dataframe(counts, enrichment_counts_path, logger=logger)
        write_dataframe(enriched_bot_counts, bot_counts_path, logger=logger)
        write_text(enrichment_summary(metrics, candidate_collections, reference_counts, metadata_sources, events_path, mapping_path), enrichment_summary_path, logger=logger)
        write_text(bot_summary(output_events, metrics), bot_summary_path, logger=logger)
    except Exception:
        logger.exception("Actor metadata enrichment failed")
        return 1

    print("Actor metadata enrichment succeeded")
    print(f"- input_events: {metrics['input_events']}")
    print(f"- output_events: {metrics['output_events']}")
    print(f"- duplicate_event_ids: {metrics['duplicate_event_ids']}")
    print(f"- metadata_resolved_actors: {metrics['metadata_resolved_actors']}")
    print(f"- metadata_coverage: {metrics['metadata_coverage']:.6f}")
    print(f"- bot_actors: {metrics['bot_actors']}")
    print(f"- bot_events: {metrics['bot_events']}")
    print(f"- bot_event_rate: {metrics['bot_event_rate']:.6f}")
    print("Generated:")
    for path in [events_path, events_csv_path, enrichment_summary_path, enrichment_counts_path, bot_summary_path, bot_counts_path]:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
