from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from bson import ObjectId
from pymongo.database import Database

from smartshark_roles.extraction_schema import bson_json_default
from smartshark_roles.io import dataframe_to_markdown, write_dataframe, write_text
from smartshark_roles.mongo import get_database


EVENT_COLUMNS = [
    "event_id",
    "project_id",
    "project_name",
    "issue_id",
    "issue_external_id",
    "actor_id_raw",
    "actor_role_raw",
    "timestamp",
    "source",
    "event_type",
    "lifecycle_stage",
    "target_id",
    "event_scope",
    "linkage_type",
    "text_available",
    "text_length",
    "files_changed",
    "lines_added",
    "lines_deleted",
    "pr_id",
    "review_id",
    "commit_id",
    "metadata_json",
]

COUNT_COLUMNS = [
    "project_name",
    "project_id",
    "source",
    "event_type",
    "lifecycle_stage",
    "event_scope",
    "linkage_type",
    "n_events",
]

WARNING_COLUMNS = [
    "project_name",
    "project_id",
    "event_id",
    "warning_type",
    "source",
    "event_type",
    "message",
]

CLOSED_BUG_FILTER = {
    "$and": [
        {"issue_type": {"$regex": "^Bug$", "$options": "i"}},
        {
            "$or": [
                {"status": {"$in": ["Closed", "Resolved"]}},
                {"resolution": {"$exists": True, "$nin": [None, "", "Unresolved"]}},
            ]
        },
    ]
}

ISSUE_PROJECTION = {
    "issue_system_id": 1,
    "external_id": 1,
    "title": 1,
    "desc": 1,
    "issue_type": 1,
    "status": 1,
    "resolution": 1,
    "created_at": 1,
    "updated_at": 1,
    "creator_id": 1,
    "reporter_id": 1,
    "assignee_id": 1,
}


@dataclass(frozen=True)
class SelectedProject:
    project_name: str
    project_id: str

    @property
    def object_id(self) -> ObjectId:
        return ObjectId(self.project_id)


@dataclass
class ProjectContext:
    selected: SelectedProject
    issue_system_ids: list[ObjectId]
    vcs_system_ids: list[ObjectId]
    pull_request_system_ids: list[ObjectId]


def load_selected_projects(path: Path) -> list[SelectedProject]:
    if not path.exists():
        raise FileNotFoundError(f"Selected projects YAML does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    selected = payload.get("selected_projects", [])
    if not selected:
        raise ValueError(f"No selected_projects entries found in {path}")
    return [
        SelectedProject(project_name=str(row["project_name"]), project_id=str(row["project_id"]))
        for row in selected
    ]


def load_schema_map(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Extraction schema map does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_id(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def metadata_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=bson_json_default, ensure_ascii=False)


def text_length(*values: Any) -> int:
    return sum(len(value) for value in values if isinstance(value, str))


def lifecycle_stage(event_type: str) -> str:
    mapping = {
        "issue_created": "reporting",
        "issue_comment_added": "discussion",
        "status_changed": "triage",
        "priority_changed": "triage",
        "assignee_changed": "triage",
        "resolution_changed": "triage",
        "commit_authored": "fixing",
        "pr_opened": "integration",
        "pr_closed": "integration",
        "pr_merged": "integration",
        "review_submitted": "review",
        "review_comment_added": "review",
        "issue_closed": "closure",
        "issue_resolved": "closure",
    }
    return mapping.get(event_type, "unknown")


def normalize_issue_event_type(status: Any, new_value: Any) -> str:
    status_text = str(status or "").strip().lower()
    new_text = str(new_value or "").strip().lower()
    if status_text == "status":
        if new_text == "closed":
            return "issue_closed"
        if new_text == "resolved":
            return "issue_resolved"
        return "status_changed"
    if status_text in {"priority"}:
        return "priority_changed"
    if status_text in {"assignee", "assignee_id"}:
        return "assignee_changed"
    if status_text in {"resolution"}:
        return "resolution_changed"
    return "issue_field_changed"


def pr_event_types(pr: dict[str, Any]) -> list[tuple[str, Any]]:
    events = [("pr_opened", pr.get("created_at"))]
    if pr.get("merged_at") is not None:
        events.append(("pr_merged", pr.get("merged_at")))
    elif str(pr.get("state", "")).lower() == "closed":
        events.append(("pr_closed", pr.get("updated_at")))
    return events


def warning_row(
    *,
    project: SelectedProject,
    event_id: str,
    warning_type: str,
    source: str,
    event_type: str,
    message: str,
) -> dict[str, Any]:
    return {
        "project_name": project.project_name,
        "project_id": project.project_id,
        "event_id": event_id,
        "warning_type": warning_type,
        "source": source,
        "event_type": event_type,
        "message": message,
    }


def make_event(
    *,
    project: SelectedProject,
    event_id: str,
    issue_id: Any,
    issue_external_id: Any,
    actor_id_raw: Any,
    actor_role_raw: str | None,
    timestamp: Any,
    source: str,
    event_type: str,
    target_id: Any,
    event_scope: str,
    linkage_type: str,
    text_available: bool = False,
    text_length_value: int = 0,
    files_changed: int = 0,
    lines_added: int = 0,
    lines_deleted: int = 0,
    pr_id: Any = None,
    review_id: Any = None,
    commit_id: Any = None,
    metadata: dict[str, Any] | None = None,
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    row = {
        "event_id": event_id,
        "project_id": project.project_id,
        "project_name": project.project_name,
        "issue_id": to_id(issue_id),
        "issue_external_id": issue_external_id,
        "actor_id_raw": to_id(actor_id_raw),
        "actor_role_raw": actor_role_raw,
        "timestamp": timestamp,
        "source": source,
        "event_type": event_type,
        "lifecycle_stage": lifecycle_stage(event_type),
        "target_id": to_id(target_id),
        "event_scope": event_scope,
        "linkage_type": linkage_type,
        "text_available": bool(text_available),
        "text_length": int(text_length_value or 0),
        "files_changed": int(files_changed or 0),
        "lines_added": int(lines_added or 0),
        "lines_deleted": int(lines_deleted or 0),
        "pr_id": to_id(pr_id),
        "review_id": to_id(review_id),
        "commit_id": to_id(commit_id),
        "metadata_json": metadata_json(metadata or {}),
    }
    if warnings is not None:
        if timestamp is None:
            warnings.append(
                warning_row(
                    project=project,
                    event_id=event_id,
                    warning_type="missing_timestamp",
                    source=source,
                    event_type=event_type,
                    message="Event retained with missing timestamp.",
                )
            )
        if actor_id_raw is None:
            warnings.append(
                warning_row(
                    project=project,
                    event_id=event_id,
                    warning_type="missing_actor_id_raw",
                    source=source,
                    event_type=event_type,
                    message="Event retained with missing actor_id_raw.",
                )
            )
    return row


def fetch_project_context(db: Database, project: SelectedProject) -> ProjectContext:
    project_id = project.object_id
    issue_system_ids = [row["_id"] for row in db.issue_system.find({"project_id": project_id}, {"_id": 1})]
    vcs_system_ids = [row["_id"] for row in db.vcs_system.find({"project_id": project_id}, {"_id": 1})]
    pull_request_system_ids = [row["_id"] for row in db.pull_request_system.find({"project_id": project_id}, {"_id": 1})]
    return ProjectContext(
        selected=project,
        issue_system_ids=issue_system_ids,
        vcs_system_ids=vcs_system_ids,
        pull_request_system_ids=pull_request_system_ids,
    )


def closed_bug_issue_query(issue_system_ids: list[ObjectId]) -> dict[str, Any]:
    return {"issue_system_id": {"$in": issue_system_ids}, **CLOSED_BUG_FILTER}


def fetch_closed_bug_issues(
    db: Database,
    context: ProjectContext,
    *,
    max_issues: int | None,
) -> list[dict[str, Any]]:
    query = closed_bug_issue_query(context.issue_system_ids)
    cursor = db.issue.find(query, ISSUE_PROJECTION, batch_size=1000).sort("updated_at", -1)
    if max_issues is not None:
        cursor = cursor.limit(max_issues)
    return list(cursor)


def file_action_stats(db: Database, commit_ids: list[ObjectId]) -> dict[str, dict[str, int]]:
    if not commit_ids:
        return {}
    pipeline = [
        {"$match": {"commit_id": {"$in": commit_ids}}},
        {
            "$group": {
                "_id": "$commit_id",
                "files_changed": {"$sum": 1},
                "lines_added": {"$sum": {"$ifNull": ["$lines_added", 0]}},
                "lines_deleted": {"$sum": {"$ifNull": ["$lines_deleted", 0]}},
            }
        },
    ]
    return {
        str(row["_id"]): {
            "files_changed": int(row.get("files_changed", 0) or 0),
            "lines_added": int(row.get("lines_added", 0) or 0),
            "lines_deleted": int(row.get("lines_deleted", 0) or 0),
        }
        for row in db.file_action.aggregate(pipeline, allowDiskUse=True)
    }


def linked_commits_for_issues(
    db: Database,
    issue_ids: list[ObjectId],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]], dict[str, str]]:
    issue_id_set = {str(issue_id) for issue_id in issue_ids}
    issue_to_commits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    commit_to_issues: dict[str, list[str]] = defaultdict(list)
    revision_hash_to_commit: dict[str, str] = {}
    if not issue_ids:
        return issue_to_commits, commit_to_issues, revision_hash_to_commit

    cursor = db.commit.find(
        {
            "$or": [
                {"linked_issue_ids": {"$in": issue_ids}},
                {"fixed_issue_ids": {"$in": issue_ids}},
            ]
        },
        {
            "vcs_system_id": 1,
            "revision_hash": 1,
            "author_id": 1,
            "committer_id": 1,
            "author_date": 1,
            "committer_date": 1,
            "message": 1,
            "linked_issue_ids": 1,
            "fixed_issue_ids": 1,
        },
        batch_size=1000,
    )
    for commit in cursor:
        commit_id = str(commit["_id"])
        revision_hash = commit.get("revision_hash")
        if isinstance(revision_hash, str):
            revision_hash_to_commit[revision_hash] = commit_id
        linked_issue_values = []
        for field in ["linked_issue_ids", "fixed_issue_ids"]:
            if isinstance(commit.get(field), list):
                linked_issue_values.extend(commit[field])
        for issue_id in linked_issue_values:
            issue_key = str(issue_id)
            if issue_key in issue_id_set:
                issue_to_commits[issue_key].append(commit)
                commit_to_issues[commit_id].append(issue_key)
    return issue_to_commits, commit_to_issues, revision_hash_to_commit


def bridge_commits_to_prs(
    db: Database,
    commit_to_issues: dict[str, list[str]],
    revision_hash_to_commit: dict[str, str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    issue_to_prs: dict[str, set[str]] = defaultdict(set)
    pr_to_issues: dict[str, set[str]] = defaultdict(set)
    commit_ids = [ObjectId(commit_id) for commit_id in commit_to_issues]
    revision_hashes = list(revision_hash_to_commit)
    if not commit_ids and not revision_hashes:
        return issue_to_prs, pr_to_issues

    prc_query: dict[str, Any] = {"$or": []}
    if commit_ids:
        prc_query["$or"].append({"commit_id": {"$in": commit_ids}})
    if revision_hashes:
        prc_query["$or"].append({"commit_sha": {"$in": revision_hashes}})
    for row in db.pull_request_commit.find(prc_query, {"commit_id": 1, "commit_sha": 1, "pull_request_id": 1}, batch_size=1000):
        commit_key = to_id(row.get("commit_id"))
        if commit_key not in commit_to_issues and row.get("commit_sha") in revision_hash_to_commit:
            commit_key = revision_hash_to_commit[row.get("commit_sha")]
        pr_key = to_id(row.get("pull_request_id"))
        if commit_key is None or pr_key is None:
            continue
        for issue_key in commit_to_issues.get(commit_key, []):
            issue_to_prs[issue_key].add(pr_key)
            pr_to_issues[pr_key].add(issue_key)

    pr_query: dict[str, Any] = {"$or": []}
    if commit_ids:
        pr_query["$or"].extend(
            [
                {"merge_commit_id": {"$in": commit_ids}},
                {"source_commit_id": {"$in": commit_ids}},
                {"target_commit_id": {"$in": commit_ids}},
            ]
        )
    if revision_hashes:
        pr_query["$or"].extend(
            [
                {"source_commit_sha": {"$in": revision_hashes}},
                {"target_commit_sha": {"$in": revision_hashes}},
            ]
        )
    for row in db.pull_request.find(pr_query, {"merge_commit_id": 1, "source_commit_id": 1, "target_commit_id": 1, "source_commit_sha": 1, "target_commit_sha": 1}, batch_size=1000):
        pr_key = str(row["_id"])
        candidate_commit_keys = [
            to_id(row.get("merge_commit_id")),
            to_id(row.get("source_commit_id")),
            to_id(row.get("target_commit_id")),
        ]
        for sha_field in ["source_commit_sha", "target_commit_sha"]:
            if row.get(sha_field) in revision_hash_to_commit:
                candidate_commit_keys.append(revision_hash_to_commit[row.get(sha_field)])
        for commit_key in candidate_commit_keys:
            for issue_key in commit_to_issues.get(commit_key, []):
                issue_to_prs[issue_key].add(pr_key)
                pr_to_issues[pr_key].add(issue_key)
    return issue_to_prs, pr_to_issues


def load_pr_docs(db: Database, pr_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not pr_ids:
        return {}
    object_ids = [ObjectId(pr_id) for pr_id in pr_ids]
    projection = {
        "pull_request_system_id": 1,
        "external_id": 1,
        "title": 1,
        "description": 1,
        "creator_id": 1,
        "assignee_id": 1,
        "created_at": 1,
        "updated_at": 1,
        "merged_at": 1,
        "state": 1,
        "merge_commit_id": 1,
        "source_commit_id": 1,
        "target_commit_id": 1,
        "source_commit_sha": 1,
        "target_commit_sha": 1,
    }
    return {str(row["_id"]): row for row in db.pull_request.find({"_id": {"$in": object_ids}}, projection, batch_size=1000)}


def project_level_pr_docs(
    db: Database,
    context: ProjectContext,
    exclude_pr_ids: set[str],
    max_project_level_prs: int,
) -> list[dict[str, Any]]:
    query: dict[str, Any] = {"pull_request_system_id": {"$in": context.pull_request_system_ids}}
    if exclude_pr_ids:
        query["_id"] = {"$nin": [ObjectId(pr_id) for pr_id in exclude_pr_ids]}
    projection = {
        "pull_request_system_id": 1,
        "external_id": 1,
        "title": 1,
        "description": 1,
        "creator_id": 1,
        "assignee_id": 1,
        "created_at": 1,
        "updated_at": 1,
        "merged_at": 1,
        "state": 1,
    }
    return list(
        db.pull_request.find(query, projection, batch_size=1000)
        .sort("created_at", -1)
        .limit(max_project_level_prs)
    )


def issue_lookup(issues: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(issue["_id"]): issue for issue in issues}


def issue_event_rows(
    db: Database,
    project: SelectedProject,
    issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_issue = issue_lookup(issues)
    issue_ids = [issue["_id"] for issue in issues]
    for issue in issues:
        issue_id = issue["_id"]
        title = issue.get("title")
        desc = issue.get("desc")
        rows.append(
            make_event(
                project=project,
                event_id=f"issue:issue_created:{issue_id}",
                issue_id=issue_id,
                issue_external_id=issue.get("external_id"),
                actor_id_raw=issue.get("creator_id") or issue.get("reporter_id"),
                actor_role_raw="issue_creator",
                timestamp=issue.get("created_at"),
                source="issue",
                event_type="issue_created",
                target_id=issue_id,
                event_scope="issue_linked",
                linkage_type="direct_issue",
                text_available=bool(title or desc),
                text_length_value=text_length(title, desc),
                metadata={"status": issue.get("status"), "resolution": issue.get("resolution"), "issue_type": issue.get("issue_type")},
                warnings=warnings,
            )
        )

    if not issue_ids:
        return rows

    for comment in db.issue_comment.find({"issue_id": {"$in": issue_ids}}, {"issue_id": 1, "author_id": 1, "created_at": 1, "comment": 1, "external_id": 1}, batch_size=1000):
        issue = by_issue.get(str(comment.get("issue_id")), {})
        comment_text = comment.get("comment")
        rows.append(
            make_event(
                project=project,
                event_id=f"issue_comment:issue_comment_added:{comment['_id']}",
                issue_id=comment.get("issue_id"),
                issue_external_id=issue.get("external_id"),
                actor_id_raw=comment.get("author_id"),
                actor_role_raw="comment_author",
                timestamp=comment.get("created_at"),
                source="issue_comment",
                event_type="issue_comment_added",
                target_id=comment["_id"],
                event_scope="issue_linked",
                linkage_type="direct_issue",
                text_available=bool(comment_text),
                text_length_value=text_length(comment_text),
                metadata={"external_id": comment.get("external_id")},
                warnings=warnings,
            )
        )

    for event in db.event.find({"issue_id": {"$in": issue_ids}}, {"issue_id": 1, "author_id": 1, "created_at": 1, "status": 1, "old_value": 1, "new_value": 1, "external_id": 1}, batch_size=1000):
        issue = by_issue.get(str(event.get("issue_id")), {})
        event_type = normalize_issue_event_type(event.get("status"), event.get("new_value"))
        rows.append(
            make_event(
                project=project,
                event_id=f"event:{event_type}:{event['_id']}",
                issue_id=event.get("issue_id"),
                issue_external_id=issue.get("external_id"),
                actor_id_raw=event.get("author_id"),
                actor_role_raw="issue_event_author",
                timestamp=event.get("created_at"),
                source="event",
                event_type=event_type,
                target_id=event["_id"],
                event_scope="issue_linked",
                linkage_type="direct_issue",
                metadata={"status_field": event.get("status"), "old_value": event.get("old_value"), "new_value": event.get("new_value"), "external_id": event.get("external_id")},
                warnings=warnings,
            )
        )
    return rows


def commit_event_rows(
    db: Database,
    project: SelectedProject,
    issues: list[dict[str, Any]],
    issue_to_commits: dict[str, list[dict[str, Any]]],
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_issue = issue_lookup(issues)
    unique_commit_ids = sorted({commit["_id"] for commits in issue_to_commits.values() for commit in commits}, key=str)
    stats = file_action_stats(db, unique_commit_ids)
    for issue_key, commits in issue_to_commits.items():
        issue = by_issue.get(issue_key, {})
        for commit in commits:
            commit_id = commit["_id"]
            commit_stats = stats.get(str(commit_id), {})
            commit_message = commit.get("message")
            rows.append(
                make_event(
                    project=project,
                    event_id=f"commit:commit_authored:issue:{issue_key}:commit:{commit_id}",
                    issue_id=issue_key,
                    issue_external_id=issue.get("external_id"),
                    actor_id_raw=commit.get("author_id"),
                    actor_role_raw="commit_author",
                    timestamp=commit.get("author_date") or commit.get("committer_date"),
                    source="commit",
                    event_type="commit_authored",
                    target_id=commit_id,
                    event_scope="issue_linked",
                    linkage_type="commit_bridge",
                    text_available=bool(commit_message),
                    text_length_value=text_length(commit_message),
                    files_changed=commit_stats.get("files_changed", 0),
                    lines_added=commit_stats.get("lines_added", 0),
                    lines_deleted=commit_stats.get("lines_deleted", 0),
                    commit_id=commit_id,
                    metadata={"revision_hash": commit.get("revision_hash"), "vcs_system_id": commit.get("vcs_system_id")},
                    warnings=warnings,
                )
            )
    return rows


def pr_event_rows(
    project: SelectedProject,
    pr: dict[str, Any],
    *,
    issue: dict[str, Any] | None,
    event_scope: str,
    linkage_type: str,
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pr_id = pr["_id"]
    issue_id = issue.get("_id") if issue else None
    issue_suffix = f":issue:{issue_id}" if issue_id is not None else ""
    for event_type, timestamp in pr_event_types(pr):
        rows.append(
            make_event(
                project=project,
                event_id=f"pull_request:{event_type}:pr:{pr_id}{issue_suffix}:{event_scope}",
                issue_id=issue_id,
                issue_external_id=issue.get("external_id") if issue else None,
                actor_id_raw=pr.get("creator_id"),
                actor_role_raw="pr_creator",
                timestamp=timestamp,
                source="pull_request",
                event_type=event_type,
                target_id=pr_id,
                event_scope=event_scope,
                linkage_type=linkage_type,
                text_available=bool(pr.get("title") or pr.get("description")),
                text_length_value=text_length(pr.get("title"), pr.get("description")),
                pr_id=pr_id,
                metadata={"state": pr.get("state"), "external_id": pr.get("external_id")},
                warnings=warnings,
            )
        )
    return rows


def review_rows_for_prs(
    db: Database,
    project: SelectedProject,
    pr_to_issues: dict[str, set[str]],
    pr_docs: dict[str, dict[str, Any]],
    by_issue: dict[str, dict[str, Any]],
    *,
    event_scope: str,
    linkage_type: str,
    warnings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    review_ids: set[str] = set()
    pr_ids = [ObjectId(pr_id) for pr_id in pr_to_issues]
    if not pr_ids:
        return rows, review_ids
    projection = {"pull_request_id": 1, "creator_id": 1, "submitted_at": 1, "state": 1, "description": 1, "external_id": 1}
    for review in db.pull_request_review.find({"pull_request_id": {"$in": pr_ids}}, projection, batch_size=1000):
        review_ids.add(str(review["_id"]))
        pr_key = str(review.get("pull_request_id"))
        issue_keys = pr_to_issues.get(pr_key, {None})
        for issue_key in issue_keys:
            issue = by_issue.get(issue_key) if issue_key else None
            issue_suffix = f":issue:{issue_key}" if issue_key else ""
            rows.append(
                make_event(
                    project=project,
                    event_id=f"pull_request_review:review_submitted:review:{review['_id']}{issue_suffix}:{event_scope}",
                    issue_id=issue.get("_id") if issue else None,
                    issue_external_id=issue.get("external_id") if issue else None,
                    actor_id_raw=review.get("creator_id"),
                    actor_role_raw="reviewer",
                    timestamp=review.get("submitted_at"),
                    source="pull_request_review",
                    event_type="review_submitted",
                    target_id=review["_id"],
                    event_scope=event_scope,
                    linkage_type=linkage_type,
                    text_available=bool(review.get("description")),
                    text_length_value=text_length(review.get("description")),
                    pr_id=review.get("pull_request_id"),
                    review_id=review["_id"],
                    metadata={"state": review.get("state"), "external_id": review.get("external_id"), "pr_external_id": pr_docs.get(pr_key, {}).get("external_id")},
                    warnings=warnings,
                )
            )
    return rows, review_ids


def pr_comment_rows(
    db: Database,
    project: SelectedProject,
    pr_to_issues: dict[str, set[str]],
    by_issue: dict[str, dict[str, Any]],
    *,
    event_scope: str,
    linkage_type: str,
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pr_ids = [ObjectId(pr_id) for pr_id in pr_to_issues]
    if not pr_ids:
        return rows
    projection = {"pull_request_id": 1, "author_id": 1, "created_at": 1, "comment": 1, "external_id": 1}
    for comment in db.pull_request_comment.find({"pull_request_id": {"$in": pr_ids}}, projection, batch_size=1000):
        pr_key = str(comment.get("pull_request_id"))
        issue_keys = pr_to_issues.get(pr_key, {None})
        for issue_key in issue_keys:
            issue = by_issue.get(issue_key) if issue_key else None
            issue_suffix = f":issue:{issue_key}" if issue_key else ""
            comment_text = comment.get("comment")
            rows.append(
                make_event(
                    project=project,
                    event_id=f"pull_request_comment:review_comment_added:comment:{comment['_id']}{issue_suffix}:{event_scope}",
                    issue_id=issue.get("_id") if issue else None,
                    issue_external_id=issue.get("external_id") if issue else None,
                    actor_id_raw=comment.get("author_id"),
                    actor_role_raw="pr_comment_author",
                    timestamp=comment.get("created_at"),
                    source="pull_request_comment",
                    event_type="review_comment_added",
                    target_id=comment["_id"],
                    event_scope=event_scope,
                    linkage_type=linkage_type,
                    text_available=bool(comment_text),
                    text_length_value=text_length(comment_text),
                    pr_id=comment.get("pull_request_id"),
                    metadata={"external_id": comment.get("external_id")},
                    warnings=warnings,
                )
            )
    return rows


def review_comment_rows(
    db: Database,
    project: SelectedProject,
    review_ids: set[str],
    review_to_issue_keys: dict[str, set[str]],
    by_issue: dict[str, dict[str, Any]],
    *,
    event_scope: str,
    linkage_type: str,
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not review_ids:
        return rows
    object_ids = [ObjectId(review_id) for review_id in review_ids]
    projection = {"pull_request_review_id": 1, "creator_id": 1, "created_at": 1, "comment": 1, "external_id": 1}
    for comment in db.pull_request_review_comment.find({"pull_request_review_id": {"$in": object_ids}}, projection, batch_size=1000):
        review_key = str(comment.get("pull_request_review_id"))
        issue_keys = review_to_issue_keys.get(review_key, {None})
        for issue_key in issue_keys:
            issue = by_issue.get(issue_key) if issue_key else None
            issue_suffix = f":issue:{issue_key}" if issue_key else ""
            comment_text = comment.get("comment")
            rows.append(
                make_event(
                    project=project,
                    event_id=f"pull_request_review_comment:review_comment_added:comment:{comment['_id']}{issue_suffix}:{event_scope}",
                    issue_id=issue.get("_id") if issue else None,
                    issue_external_id=issue.get("external_id") if issue else None,
                    actor_id_raw=comment.get("creator_id"),
                    actor_role_raw="review_comment_author",
                    timestamp=comment.get("created_at"),
                    source="pull_request_review_comment",
                    event_type="review_comment_added",
                    target_id=comment["_id"],
                    event_scope=event_scope,
                    linkage_type=linkage_type,
                    text_available=bool(comment_text),
                    text_length_value=text_length(comment_text),
                    review_id=comment.get("pull_request_review_id"),
                    metadata={"external_id": comment.get("external_id")},
                    warnings=warnings,
                )
            )
    return rows


def project_events(
    db: Database,
    project: SelectedProject,
    *,
    smoke: bool,
    max_issues_per_project: int,
    max_project_level_prs: int,
    warnings: list[dict[str, Any]],
    logger: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = fetch_project_context(db, project)
    max_issues = max_issues_per_project if smoke else None
    closed_bug_count = db.issue.count_documents(closed_bug_issue_query(context.issue_system_ids))
    issues = fetch_closed_bug_issues(db, context, max_issues=max_issues)
    by_issue = issue_lookup(issues)
    issue_ids = [issue["_id"] for issue in issues]

    rows: list[dict[str, Any]] = []
    rows.extend(issue_event_rows(db, project, issues, warnings))

    issue_to_commits, commit_to_issues, revision_hash_to_commit = linked_commits_for_issues(db, issue_ids)
    rows.extend(commit_event_rows(db, project, issues, issue_to_commits, warnings))

    issue_to_prs, pr_to_issues = bridge_commits_to_prs(db, commit_to_issues, revision_hash_to_commit)
    issue_linked_pr_ids = set(pr_to_issues)
    issue_linked_pr_docs = load_pr_docs(db, issue_linked_pr_ids)
    for pr_key, issue_keys in pr_to_issues.items():
        pr = issue_linked_pr_docs.get(pr_key)
        if pr is None:
            continue
        for issue_key in issue_keys:
            rows.extend(
                pr_event_rows(
                    project,
                    pr,
                    issue=by_issue.get(issue_key),
                    event_scope="issue_linked",
                    linkage_type="commit_bridge",
                    warnings=warnings,
                )
            )

    rows.extend(
        pr_comment_rows(
            db,
            project,
            pr_to_issues,
            by_issue,
            event_scope="issue_linked",
            linkage_type="commit_bridge",
            warnings=warnings,
        )
    )
    review_rows, review_ids = review_rows_for_prs(
        db,
        project,
        pr_to_issues,
        issue_linked_pr_docs,
        by_issue,
        event_scope="issue_linked",
        linkage_type="commit_bridge",
        warnings=warnings,
    )
    rows.extend(review_rows)

    review_to_issue_keys: dict[str, set[str]] = defaultdict(set)
    for review in db.pull_request_review.find({"_id": {"$in": [ObjectId(review_id) for review_id in review_ids]}}, {"pull_request_id": 1}, batch_size=1000):
        review_key = str(review["_id"])
        pr_key = str(review.get("pull_request_id"))
        review_to_issue_keys[review_key].update(pr_to_issues.get(pr_key, set()))
    rows.extend(
        review_comment_rows(
            db,
            project,
            review_ids,
            review_to_issue_keys,
            by_issue,
            event_scope="issue_linked",
            linkage_type="commit_bridge",
            warnings=warnings,
        )
    )

    project_prs = project_level_pr_docs(db, context, issue_linked_pr_ids, max_project_level_prs)
    project_pr_ids = {str(pr["_id"]) for pr in project_prs}
    project_pr_docs = {str(pr["_id"]): pr for pr in project_prs}
    project_pr_to_issue_none = {pr_id: {None} for pr_id in project_pr_ids}
    for pr in project_prs:
        rows.extend(
            pr_event_rows(
                project,
                pr,
                issue=None,
                event_scope="project_level",
                linkage_type="project_level",
                warnings=warnings,
            )
        )
    rows.extend(
        pr_comment_rows(
            db,
            project,
            project_pr_to_issue_none,
            {},
            event_scope="project_level",
            linkage_type="project_level",
            warnings=warnings,
        )
    )
    project_review_rows, project_review_ids = review_rows_for_prs(
        db,
        project,
        project_pr_to_issue_none,
        project_pr_docs,
        {},
        event_scope="project_level",
        linkage_type="project_level",
        warnings=warnings,
    )
    rows.extend(project_review_rows)
    project_review_to_none = {review_id: {None} for review_id in project_review_ids}
    rows.extend(
        review_comment_rows(
            db,
            project,
            project_review_ids,
            project_review_to_none,
            {},
            event_scope="project_level",
            linkage_type="project_level",
            warnings=warnings,
        )
    )

    stats = {
        "project_name": project.project_name,
        "project_id": project.project_id,
        "n_closed_bug_issues_available": closed_bug_count,
        "n_closed_bug_issues_extracted": len(issues),
        "n_issue_linked_prs": len(issue_linked_pr_ids),
        "n_project_level_prs_extracted": len(project_prs),
        "n_events": len(rows),
    }
    logger.info("Extracted project=%s issues=%d events=%d issue_linked_prs=%d project_level_prs=%d", project.project_name, len(issues), len(rows), len(issue_linked_pr_ids), len(project_prs))
    return rows, stats


def counts_frame(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=COUNT_COLUMNS)
    return (
        events.groupby(["project_name", "project_id", "source", "event_type", "lifecycle_stage", "event_scope", "linkage_type"], dropna=False)
        .size()
        .reset_index(name="n_events")
        .sort_values(["project_name", "event_scope", "source", "event_type"])
    )


def duplicate_warnings(events: pd.DataFrame) -> list[dict[str, Any]]:
    if events.empty:
        return []
    duplicates = events[events["event_id"].duplicated(keep=False)]
    rows: list[dict[str, Any]] = []
    for _, row in duplicates.iterrows():
        rows.append(
            {
                "project_name": row.get("project_name"),
                "project_id": row.get("project_id"),
                "event_id": row.get("event_id"),
                "warning_type": "duplicate_event_id",
                "source": row.get("source"),
                "event_type": row.get("event_type"),
                "message": "Duplicate event_id detected in extracted events.",
            }
        )
    return rows


def summary_markdown(
    events: pd.DataFrame,
    counts: pd.DataFrame,
    warnings: pd.DataFrame,
    project_stats: pd.DataFrame,
    *,
    smoke: bool,
    max_issues_per_project: int,
    max_project_level_prs: int,
) -> str:
    project_counts = events.groupby("project_name").size().reset_index(name="n_events") if not events.empty else pd.DataFrame(columns=["project_name", "n_events"])
    source_counts = events.groupby("source").size().reset_index(name="n_events") if not events.empty else pd.DataFrame(columns=["source", "n_events"])
    stage_counts = events.groupby("lifecycle_stage").size().reset_index(name="n_events") if not events.empty else pd.DataFrame(columns=["lifecycle_stage", "n_events"])
    scope_counts = events.groupby("event_scope").size().reset_index(name="n_events") if not events.empty else pd.DataFrame(columns=["event_scope", "n_events"])
    timestamp_missing = int(events["timestamp"].isna().sum()) if "timestamp" in events else 0
    actor_missing = int(events["actor_id_raw"].isna().sum()) if "actor_id_raw" in events else 0
    issue_limit_label = str(max_issues_per_project) if smoke else "all"

    lines = [
        "# Event Extraction Summary",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Smoke mode: `{smoke}`",
        f"- Max issues per project: `{issue_limit_label}`",
        f"- Max project-level PRs per project: `{max_project_level_prs}`",
        f"- Total events: `{len(events)}`",
        f"- Missing timestamps: `{timestamp_missing}`",
        f"- Missing actor_id_raw: `{actor_missing}`",
        f"- Warning rows: `{len(warnings)}`",
        "",
        "## Project Stats",
        "",
        dataframe_to_markdown(project_stats),
        "",
        "## Project Event Counts",
        "",
        dataframe_to_markdown(project_counts),
        "",
        "## Source Counts",
        "",
        dataframe_to_markdown(source_counts),
        "",
        "## Lifecycle Stage Counts",
        "",
        dataframe_to_markdown(stage_counts),
        "",
        "## Scope Counts",
        "",
        dataframe_to_markdown(scope_counts),
        "",
        "## Detailed Counts",
        "",
        dataframe_to_markdown(counts.head(100)),
        "",
        "## Notes",
        "",
        "- Issue, issue comment, issue event, and commit rows are issue-linked.",
        "- Pull requests and reviews are split into `issue_linked` and `project_level` scopes.",
        "- Project-level PR/review rows are capped by `max_project_level_prs` and are not forced onto a specific issue.",
        "- No actor normalization, feature building, or clustering was performed.",
    ]
    return "\n".join(lines) + "\n"


def run_event_extraction(
    config: dict[str, Any],
    *,
    selected_path: Path,
    schema_map_path: Path,
    smoke: bool,
    max_issues_per_project: int,
    max_project_level_prs: int,
    logger: Any,
) -> dict[str, Path]:
    load_schema_map(schema_map_path)
    selected_projects = load_selected_projects(selected_path)
    db = get_database(config, logger=logger)

    warning_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    project_stat_rows: list[dict[str, Any]] = []
    for project in selected_projects:
        rows, stats = project_events(
            db,
            project,
            smoke=smoke,
            max_issues_per_project=max_issues_per_project,
            max_project_level_prs=max_project_level_prs,
            warnings=warning_rows,
            logger=logger,
        )
        event_rows.extend(rows)
        project_stat_rows.append(stats)

    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    duplicate_rows = duplicate_warnings(events)
    warning_rows.extend(duplicate_rows)
    warnings = pd.DataFrame(warning_rows, columns=WARNING_COLUMNS)
    counts = counts_frame(events)
    project_stats = pd.DataFrame(project_stat_rows)

    output_dir = Path(config["_project_root"]) / "data" / "interim" / "events"
    report_dir = Path(config["_project_root"]) / "reports" / "build_events"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    event_stem = "events_smoke" if smoke else "events"

    paths = {
        "events_csv": output_dir / f"{event_stem}.csv",
        "events_parquet": output_dir / f"{event_stem}.parquet",
        "summary_md": report_dir / "event_extraction_summary.md",
        "counts_csv": report_dir / "event_extraction_counts.csv",
        "warnings_csv": report_dir / "event_extraction_warnings.csv",
    }
    write_dataframe(events, paths["events_csv"], paths["events_parquet"], logger=logger)
    write_dataframe(counts, paths["counts_csv"], logger=logger)
    write_dataframe(warnings, paths["warnings_csv"], logger=logger)
    write_text(
        summary_markdown(
            events,
            counts,
            warnings,
            project_stats,
            smoke=smoke,
            max_issues_per_project=max_issues_per_project,
            max_project_level_prs=max_project_level_prs,
        ),
        paths["summary_md"],
        logger=logger,
    )
    return paths
