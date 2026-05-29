from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from bson import ObjectId

from smartshark_roles.io import dataframe_to_markdown


@dataclass(frozen=True)
class CollectionSchema:
    role: str
    collection: str
    required_fields: list[str]
    optional_fields: list[str]
    extraction_use: str


COLLECTION_SCHEMAS = [
    CollectionSchema(
        role="project",
        collection="project",
        required_fields=["_id", "name"],
        optional_fields=[],
        extraction_use="Resolve selected project names and ObjectIds.",
    ),
    CollectionSchema(
        role="issue_system",
        collection="issue_system",
        required_fields=["_id", "project_id"],
        optional_fields=["url", "last_updated"],
        extraction_use="Map project._id to issue.issue_system_id.",
    ),
    CollectionSchema(
        role="issue",
        collection="issue",
        required_fields=["_id", "issue_system_id", "issue_type", "status", "resolution"],
        optional_fields=["external_id", "title", "created_at", "updated_at", "creator_id", "reporter_id", "assignee_id"],
        extraction_use="Select closed bug issues and provide core issue lifecycle records.",
    ),
    CollectionSchema(
        role="issue_comment",
        collection="issue_comment",
        required_fields=["_id", "issue_id", "author_id", "created_at", "comment"],
        optional_fields=["external_id"],
        extraction_use="Attach issue discussion events to issue._id.",
    ),
    CollectionSchema(
        role="issue_event",
        collection="event",
        required_fields=["_id", "issue_id", "status", "created_at"],
        optional_fields=["author_id", "external_id", "old_value", "new_value"],
        extraction_use="Attach issue status/history changes to issue._id.",
    ),
    CollectionSchema(
        role="commit",
        collection="commit",
        required_fields=["_id", "vcs_system_id", "revision_hash"],
        optional_fields=["author_id", "committer_id", "author_date", "committer_date", "linked_issue_ids", "fixed_issue_ids"],
        extraction_use="Attach commits to issues via linked_issue_ids/fixed_issue_ids and to projects via vcs_system_id.",
    ),
    CollectionSchema(
        role="file_action",
        collection="file_action",
        required_fields=["_id", "commit_id", "file_id"],
        optional_fields=["mode", "lines_added", "lines_deleted", "parent_revision_hash"],
        extraction_use="Later enrich commit activity with file-level change size; not required for linkage preflight.",
    ),
    CollectionSchema(
        role="pull_request",
        collection="pull_request",
        required_fields=["_id", "pull_request_system_id", "creator_id", "created_at", "state"],
        optional_fields=["external_id", "title", "updated_at", "merged_at", "merge_commit_id", "source_commit_id", "target_commit_id"],
        extraction_use="Attach pull request activity to projects and, where possible, to sampled issues through linked commits.",
    ),
    CollectionSchema(
        role="pull_request_event",
        collection="pull_request_event",
        required_fields=["_id", "pull_request_id", "event_type", "created_at"],
        optional_fields=["author_id", "commit_id", "commit_sha", "additional_data"],
        extraction_use="Attach pull request lifecycle events to pull_request._id.",
    ),
    CollectionSchema(
        role="pull_request_comment",
        collection="pull_request_comment",
        required_fields=["_id", "pull_request_id", "author_id", "created_at", "comment"],
        optional_fields=["external_id", "updated_at", "author_association"],
        extraction_use="Attach pull request conversation comments to pull_request._id.",
    ),
    CollectionSchema(
        role="pull_request_review",
        collection="pull_request_review",
        required_fields=["_id", "pull_request_id", "creator_id", "submitted_at", "state"],
        optional_fields=["external_id", "pull_request_commit_id", "commit_sha", "description"],
        extraction_use="Attach review events to pull_request._id.",
    ),
    CollectionSchema(
        role="pull_request_commit",
        collection="pull_request_commit",
        required_fields=["_id", "pull_request_id"],
        optional_fields=["commit_id", "commit_sha", "author_id", "committer_id"],
        extraction_use="Bridge linked issue commits to pull requests when commit_id is available.",
    ),
]


FIELD_ALIASES = {
    "issue_id_field": "_id",
    "issue_status_field": "status",
    "issue_type_field": "issue_type",
    "issue_resolution_field": "resolution",
    "comment_issue_ref_field": "issue_id",
    "event_issue_ref_field": "issue_id",
    "commit_link_collection": "commit",
    "commit_link_issue_ref_field": "linked_issue_ids;fixed_issue_ids",
    "commit_link_commit_ref_field": "_id",
    "pr_project_ref_field": "pull_request.pull_request_system_id -> pull_request_system.project_id",
    "review_pr_ref_field": "pull_request_id",
}


def bson_json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (ObjectId, Path)):
        return str(value)
    return repr(value)


def load_field_coverage(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Field coverage CSV does not exist: {path}")
    return pd.read_csv(path)


def build_schema_map(field_coverage: pd.DataFrame) -> dict[str, Any]:
    coverage_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in field_coverage.iterrows():
        coverage_lookup[(str(row["collection"]), str(row["field_path"]))] = {
            "present_rate": float(row.get("present_rate", 0) or 0),
            "missing_rate": float(row.get("missing_rate", 0) or 0),
            "value_types": str(row.get("value_types", "")),
            "field_audit_mode": str(row.get("field_audit_mode", "")),
        }

    collections: list[dict[str, Any]] = []
    warnings: list[str] = []
    for schema in COLLECTION_SCHEMAS:
        fields: list[dict[str, Any]] = []
        for field in schema.required_fields + schema.optional_fields:
            coverage = coverage_lookup.get((schema.collection, field))
            exists = coverage is not None
            if not exists and field in schema.required_fields:
                warnings.append(f"Missing required field in audit coverage: {schema.collection}.{field}")
            fields.append(
                {
                    "field": field,
                    "required": field in schema.required_fields,
                    "present_in_audit": exists,
                    **(coverage or {}),
                }
            )
        collections.append({**asdict(schema), "fields": fields})

    return {
        "generated_from": "field_coverage.csv",
        "collections": collections,
        "field_aliases": FIELD_ALIASES,
        "warnings": warnings,
    }


def schema_map_to_markdown(schema_map: dict[str, Any]) -> str:
    rows: list[dict[str, Any]] = []
    for collection in schema_map["collections"]:
        for field in collection["fields"]:
            rows.append(
                {
                    "role": collection["role"],
                    "collection": collection["collection"],
                    "field": field["field"],
                    "required": field["required"],
                    "present_in_audit": field.get("present_in_audit", False),
                    "present_rate": field.get("present_rate", ""),
                    "value_types": field.get("value_types", ""),
                }
            )
    table = pd.DataFrame(rows)
    lines = [
        "# Events Extraction Schema Map",
        "",
        "## Field Map",
        "",
        dataframe_to_markdown(table),
        "",
        "## Linkage Plan",
        "",
        "- Project to issues: `project._id -> issue_system.project_id -> issue.issue_system_id`.",
        "- Closed bug issue filter: `issue.issue_type == 'Bug'` and closed/resolved status or non-`Unresolved` resolution.",
        "- Issue comments: `issue_comment.issue_id -> issue._id`.",
        "- Issue events: `event.issue_id -> issue._id`.",
        "- Issue-linked commits: `commit.linked_issue_ids` or `commit.fixed_issue_ids` contains `issue._id`.",
        "- Pull requests by project: `pull_request.pull_request_system_id -> pull_request_system.project_id`.",
        "- Pull requests for sampled issues: linked commits can bridge through `pull_request_commit.commit_id -> commit._id` when available.",
        "- Reviews: `pull_request_review.pull_request_id -> pull_request._id`.",
        "",
        "## Warnings",
        "",
    ]
    warnings = schema_map.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No schema coverage warnings.")
    return "\n".join(lines) + "\n"
