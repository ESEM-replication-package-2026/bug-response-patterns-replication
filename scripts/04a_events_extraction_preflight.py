from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from bson import ObjectId

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartshark_roles.config import ensure_project_directories, load_config, project_path
from smartshark_roles.extraction_schema import (
    FIELD_ALIASES,
    bson_json_default,
    build_schema_map,
    load_field_coverage,
    schema_map_to_markdown,
)
from smartshark_roles.io import dataframe_to_markdown, write_dataframe, write_json, write_text
from smartshark_roles.logging_utils import setup_logging
from smartshark_roles.mongo import get_database


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
    "external_id": 1,
    "title": 1,
    "issue_system_id": 1,
    "issue_type": 1,
    "status": 1,
    "resolution": 1,
    "created_at": 1,
    "updated_at": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checks for SmartSHARK events extraction.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--selected", default="configs/selected_projects.yaml", help="Selected projects YAML.")
    parser.add_argument(
        "--field-coverage",
        default="reports/data_audit/collection_fields.csv",
        help="Field coverage CSV from DB audit.",
    )
    parser.add_argument("--sample-size", type=int, default=5, help="Maximum closed bug issues sampled per project.")
    return parser.parse_args()


def load_selected_projects(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Selected projects YAML does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    projects = payload.get("selected_projects", [])
    if not isinstance(projects, list) or not projects:
        raise ValueError(f"No selected_projects entries found in {path}")
    return projects


def object_id(value: str, warnings: list[str], label: str) -> ObjectId | None:
    try:
        return ObjectId(value)
    except Exception:
        warnings.append(f"Invalid ObjectId for {label}: {value}")
        return None


def collection_names(db: Any) -> set[str]:
    return set(db.list_collection_names())


def require_collection(names: set[str], collection: str, warnings: list[str]) -> bool:
    if collection not in names:
        warnings.append(f"Missing collection: {collection}")
        return False
    return True


def group_counts_by_issue(db: Any, collection: str, issue_ids: list[ObjectId], logger: logging.Logger) -> dict[str, int]:
    if not issue_ids:
        return {}
    pipeline = [
        {"$match": {"issue_id": {"$in": issue_ids}}},
        {"$group": {"_id": "$issue_id", "count": {"$sum": 1}}},
    ]
    rows = db[collection].aggregate(pipeline, allowDiskUse=True)
    counts = {str(row["_id"]): int(row["count"]) for row in rows}
    logger.info("Grouped %s by issue_id for %d issues -> %d hits", collection, len(issue_ids), len(counts))
    return counts


def map_issue_commits(
    db: Any,
    issue_ids: list[ObjectId],
    logger: logging.Logger,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    issue_id_set = {str(issue_id) for issue_id in issue_ids}
    issue_to_commits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    commit_to_issues: dict[str, list[str]] = defaultdict(list)
    if not issue_ids:
        return issue_to_commits, commit_to_issues

    cursor = db.commit.find(
        {
            "$or": [
                {"linked_issue_ids": {"$in": issue_ids}},
                {"fixed_issue_ids": {"$in": issue_ids}},
            ]
        },
        projection={"revision_hash": 1, "linked_issue_ids": 1, "fixed_issue_ids": 1},
        batch_size=1000,
    )
    count = 0
    for commit in cursor:
        commit_id = str(commit["_id"])
        linked_issue_values = []
        for field in ["linked_issue_ids", "fixed_issue_ids"]:
            if isinstance(commit.get(field), list):
                linked_issue_values.extend(commit[field])
        for issue_id in linked_issue_values:
            issue_key = str(issue_id)
            if issue_key in issue_id_set:
                issue_to_commits[issue_key].append(
                    {
                        "commit_id": commit_id,
                        "revision_hash": commit.get("revision_hash"),
                    }
                )
                commit_to_issues[commit_id].append(issue_key)
        count += 1
    logger.info("Found %d issue-linked commits for %d issues", count, len(issue_ids))
    return issue_to_commits, commit_to_issues


def map_commits_to_pull_requests(
    db: Any,
    commit_to_issues: dict[str, list[str]],
    logger: logging.Logger,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    commit_ids = [ObjectId(commit_id) for commit_id in commit_to_issues]
    issue_to_prs: dict[str, set[str]] = defaultdict(set)
    pr_to_issues: dict[str, set[str]] = defaultdict(set)
    if not commit_ids:
        return issue_to_prs, pr_to_issues

    cursor = db.pull_request_commit.find(
        {"commit_id": {"$in": commit_ids}},
        projection={"commit_id": 1, "pull_request_id": 1},
        batch_size=1000,
    )
    bridge_rows = 0
    for row in cursor:
        commit_key = str(row.get("commit_id"))
        pr_key = str(row.get("pull_request_id"))
        if not pr_key:
            continue
        for issue_key in commit_to_issues.get(commit_key, []):
            issue_to_prs[issue_key].add(pr_key)
            pr_to_issues[pr_key].add(issue_key)
        bridge_rows += 1

    direct_cursor = db.pull_request.find(
        {
            "$or": [
                {"merge_commit_id": {"$in": commit_ids}},
                {"source_commit_id": {"$in": commit_ids}},
                {"target_commit_id": {"$in": commit_ids}},
            ]
        },
        projection={"merge_commit_id": 1, "source_commit_id": 1, "target_commit_id": 1},
        batch_size=1000,
    )
    direct_rows = 0
    for row in direct_cursor:
        pr_key = str(row["_id"])
        for field in ["merge_commit_id", "source_commit_id", "target_commit_id"]:
            commit_key = str(row.get(field))
            for issue_key in commit_to_issues.get(commit_key, []):
                issue_to_prs[issue_key].add(pr_key)
                pr_to_issues[pr_key].add(issue_key)
        direct_rows += 1

    logger.info("Mapped commits to PRs through %d pull_request_commit rows and %d direct PR commit refs", bridge_rows, direct_rows)
    return issue_to_prs, pr_to_issues


def map_reviews_to_issues(
    db: Any,
    pr_to_issues: dict[str, set[str]],
    logger: logging.Logger,
) -> dict[str, int]:
    pr_ids = [ObjectId(pr_id) for pr_id in pr_to_issues]
    issue_review_counts: dict[str, int] = defaultdict(int)
    if not pr_ids:
        return issue_review_counts
    cursor = db.pull_request_review.find(
        {"pull_request_id": {"$in": pr_ids}},
        projection={"pull_request_id": 1},
        batch_size=1000,
    )
    count = 0
    for review in cursor:
        pr_key = str(review.get("pull_request_id"))
        for issue_key in pr_to_issues.get(pr_key, []):
            issue_review_counts[issue_key] += 1
        count += 1
    logger.info("Mapped %d reviews to sampled issues through PRs", count)
    return issue_review_counts


def select_sample_issues(
    issues: list[dict[str, Any]],
    comments: dict[str, int],
    events: dict[str, int],
    commits: dict[str, list[dict[str, Any]]],
    prs: dict[str, set[str]],
    reviews: dict[str, int],
    sample_size: int,
) -> list[dict[str, Any]]:
    def score(issue: dict[str, Any]) -> tuple[int, str]:
        issue_key = str(issue["_id"])
        linkage_score = sum(
            [
                comments.get(issue_key, 0) > 0,
                events.get(issue_key, 0) > 0,
                len(commits.get(issue_key, [])) > 0,
                len(prs.get(issue_key, set())) > 0,
                reviews.get(issue_key, 0) > 0,
            ]
        )
        created_at = issue.get("created_at")
        created_key = created_at.isoformat() if hasattr(created_at, "isoformat") else ""
        return linkage_score, created_key

    return sorted(issues, key=score, reverse=True)[:sample_size]


def project_preflight(
    db: Any,
    names: set[str],
    selected_project: dict[str, Any],
    sample_size: int,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    warnings: list[str] = []
    project_name = selected_project["project_name"]
    project_id_text = selected_project["project_id"]
    project_id = object_id(project_id_text, warnings, f"{project_name}.project_id")

    for collection in [
        "project",
        "issue_system",
        "issue",
        "issue_comment",
        "event",
        "commit",
        "file_action",
        "pull_request",
        "pull_request_event",
        "pull_request_comment",
        "pull_request_review",
        "pull_request_commit",
    ]:
        require_collection(names, collection, warnings)

    issue_system_ids: list[ObjectId] = []
    pr_system_ids: list[ObjectId] = []
    vcs_system_ids: list[ObjectId] = []
    if project_id is not None:
        project = db.project.find_one({"_id": project_id}, {"name": 1})
        if project is None:
            warnings.append(f"project._id not found: {project_id_text}")
        issue_system_ids = [row["_id"] for row in db.issue_system.find({"project_id": project_id}, {"_id": 1})]
        pr_system_ids = [row["_id"] for row in db.pull_request_system.find({"project_id": project_id}, {"_id": 1})]
        vcs_system_ids = [row["_id"] for row in db.vcs_system.find({"project_id": project_id}, {"_id": 1})]

    if not issue_system_ids:
        warnings.append("No issue_system_id values found for project")
    if not pr_system_ids:
        warnings.append("No pull_request_system_id values found for project")
    if not vcs_system_ids:
        warnings.append("No vcs_system_id values found for project")

    project_pull_request_ids = (
        [
            row["_id"]
            for row in db.pull_request.find(
                {"pull_request_system_id": {"$in": pr_system_ids}},
                {"_id": 1},
                batch_size=1000,
            )
        ]
        if pr_system_ids
        else []
    )
    n_project_reviews = (
        db.pull_request_review.count_documents({"pull_request_id": {"$in": project_pull_request_ids}})
        if project_pull_request_ids
        else 0
    )

    issue_query = {"issue_system_id": {"$in": issue_system_ids}, **CLOSED_BUG_FILTER}
    n_closed_bug_issues = db.issue.count_documents(issue_query) if issue_system_ids else 0
    issues = list(db.issue.find(issue_query, projection=ISSUE_PROJECTION, batch_size=1000))
    issue_ids = [issue["_id"] for issue in issues]

    comments = group_counts_by_issue(db, "issue_comment", issue_ids, logger)
    events = group_counts_by_issue(db, "event", issue_ids, logger)
    issue_to_commits, commit_to_issues = map_issue_commits(db, issue_ids, logger)
    issue_to_prs, pr_to_issues = map_commits_to_pull_requests(db, commit_to_issues, logger)
    issue_review_counts = map_reviews_to_issues(db, pr_to_issues, logger)

    samples = select_sample_issues(
        issues,
        comments,
        events,
        issue_to_commits,
        issue_to_prs,
        issue_review_counts,
        sample_size,
    )

    sample_rows: list[dict[str, Any]] = []
    for issue in samples:
        issue_key = str(issue["_id"])
        sample_rows.append(
            {
                "project_name": project_name,
                "project_id": project_id_text,
                "issue_id": issue_key,
                "issue_external_id": issue.get("external_id"),
                "issue_type": issue.get("issue_type"),
                "issue_status": issue.get("status"),
                "issue_resolution": issue.get("resolution"),
                "issue_created_at": issue.get("created_at"),
                "issue_title": issue.get("title"),
                "n_comments": comments.get(issue_key, 0),
                "n_events": events.get(issue_key, 0),
                "n_linked_commits": len(issue_to_commits.get(issue_key, [])),
                "n_pull_requests": len(issue_to_prs.get(issue_key, set())),
                "n_reviews": issue_review_counts.get(issue_key, 0),
                "linked_commit_ids": ";".join(commit["commit_id"] for commit in issue_to_commits.get(issue_key, [])[:20]),
                "linked_pull_request_ids": ";".join(sorted(issue_to_prs.get(issue_key, set()))[:20]),
            }
        )

    if samples and not any(row["n_pull_requests"] > 0 for row in sample_rows):
        warnings.append("No sampled closed bug issue linked to pull requests through commit bridge")
    if samples and not any(row["n_reviews"] > 0 for row in sample_rows):
        warnings.append("No sampled closed bug issue linked to reviews through commit -> PR bridge")

    count_row = {
        "project_name": project_name,
        "project_id": project_id_text,
        "n_closed_bug_issues_detected": n_closed_bug_issues,
        "n_sampled_issues": len(samples),
        "n_sampled_issues_with_comments": sum(row["n_comments"] > 0 for row in sample_rows),
        "n_sampled_issues_with_events": sum(row["n_events"] > 0 for row in sample_rows),
        "n_sampled_issues_with_linked_commits": sum(row["n_linked_commits"] > 0 for row in sample_rows),
        "n_sampled_issues_with_pull_requests": sum(row["n_pull_requests"] > 0 for row in sample_rows),
        "n_sampled_issues_with_reviews": sum(row["n_reviews"] > 0 for row in sample_rows),
        "n_project_pull_requests": len(project_pull_request_ids),
        "n_project_reviews": n_project_reviews,
        "issue_id_field": FIELD_ALIASES["issue_id_field"],
        "issue_status_field": FIELD_ALIASES["issue_status_field"],
        "issue_type_field": FIELD_ALIASES["issue_type_field"],
        "issue_resolution_field": FIELD_ALIASES["issue_resolution_field"],
        "comment_issue_ref_field": FIELD_ALIASES["comment_issue_ref_field"],
        "event_issue_ref_field": FIELD_ALIASES["event_issue_ref_field"],
        "commit_link_collection": FIELD_ALIASES["commit_link_collection"],
        "commit_link_issue_ref_field": FIELD_ALIASES["commit_link_issue_ref_field"],
        "commit_link_commit_ref_field": FIELD_ALIASES["commit_link_commit_ref_field"],
        "pr_project_ref_field": FIELD_ALIASES["pr_project_ref_field"],
        "review_pr_ref_field": FIELD_ALIASES["review_pr_ref_field"],
        "warnings": "; ".join(warnings),
        "n_issue_system_ids": len(issue_system_ids),
        "n_vcs_system_ids": len(vcs_system_ids),
        "n_pull_request_system_ids": len(pr_system_ids),
    }
    logger.info("Preflight project=%s closed_bugs=%d samples=%d warnings=%d", project_name, n_closed_bug_issues, len(samples), len(warnings))
    return count_row, sample_rows


def build_summary(
    linkage_frame: pd.DataFrame,
    sample_frame: pd.DataFrame,
    schema_map: dict[str, Any],
    selected_path: Path,
) -> str:
    display_cols = [
        "project_name",
        "n_closed_bug_issues_detected",
        "n_sampled_issues",
        "n_sampled_issues_with_comments",
        "n_sampled_issues_with_events",
        "n_sampled_issues_with_linked_commits",
        "n_sampled_issues_with_pull_requests",
        "n_sampled_issues_with_reviews",
        "n_project_pull_requests",
        "n_project_reviews",
        "warnings",
    ]
    sample_cols = [
        "project_name",
        "issue_external_id",
        "issue_status",
        "issue_resolution",
        "n_comments",
        "n_events",
        "n_linked_commits",
        "n_pull_requests",
        "n_reviews",
    ]
    schema_rows = [
        {
            "role": item["role"],
            "collection": item["collection"],
            "required_fields": ", ".join(item["required_fields"]),
            "optional_fields": ", ".join(item["optional_fields"]),
        }
        for item in schema_map["collections"]
    ]
    lines = [
        "# Events Extraction Preflight Summary",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Selected projects file: `{selected_path}`",
        "- No full events CSV/Parquet was generated in this preflight.",
        "",
        "## Linkage Counts",
        "",
        dataframe_to_markdown(linkage_frame.loc[:, display_cols]),
        "",
        "## Sampled Closed Bug Issues",
        "",
        dataframe_to_markdown(sample_frame.loc[:, sample_cols]) if not sample_frame.empty else "No samples were produced.",
        "",
        "## Planned Collection/Field Sources",
        "",
        dataframe_to_markdown(pd.DataFrame(schema_rows)),
        "",
        "## Notes",
        "",
        "- Closed bug issues come from `issue` via `issue_system.project_id`.",
        "- Comments and issue events link directly through `issue_comment.issue_id` and `event.issue_id`.",
        "- Issue-linked commits use `commit.linked_issue_ids` and `commit.fixed_issue_ids`; there is no separate issue-commit link collection in the audited schema.",
        "- Pull request and review checks for sampled issues use linked commits bridged through `pull_request_commit.commit_id` where available.",
        "- Project-level pull request extraction will use `pull_request.pull_request_system_id -> pull_request_system.project_id`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)

    log_path = ROOT / "reports" / "build_events" / "logs" / f"events_preflight_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    logger = setup_logging(log_path, level=logging.INFO)

    selected_path = project_path(args.selected)
    field_coverage_path = project_path(args.field_coverage)
    output_dir = ROOT / "reports" / "build_events"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        selected_projects = load_selected_projects(selected_path)
        field_coverage = load_field_coverage(field_coverage_path)
        schema_map = build_schema_map(field_coverage)
        db = get_database(config, logger=logger)
        names = collection_names(db)

        linkage_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        for selected_project in selected_projects:
            linkage_row, rows = project_preflight(db, names, selected_project, args.sample_size, logger)
            linkage_rows.append(linkage_row)
            sample_rows.extend(rows)

        linkage_frame = pd.DataFrame(linkage_rows)
        sample_frame = pd.DataFrame(sample_rows)

        write_json(schema_map, output_dir / "extraction_schema_map.json", logger=logger)
        write_text(schema_map_to_markdown(schema_map), output_dir / "extraction_schema_map.md", logger=logger)
        write_dataframe(sample_frame, output_dir / "preflight_issue_samples.csv", logger=logger)
        write_dataframe(linkage_frame, output_dir / "preflight_linkage_counts.csv", logger=logger)
        write_text(
            build_summary(linkage_frame, sample_frame, schema_map, selected_path),
            output_dir / "preflight_summary.md",
            logger=logger,
        )
    except Exception:
        logger.exception("Events extraction preflight failed")
        return 1

    print("Generated:")
    print(r"- reports\build_events\extraction_schema_map.json")
    print(r"- reports\build_events\extraction_schema_map.md")
    print(r"- reports\build_events\preflight_issue_samples.csv")
    print(r"- reports\build_events\preflight_linkage_counts.csv")
    print(r"- reports\build_events\preflight_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
