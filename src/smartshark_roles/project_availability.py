from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pymongo.database import Database
from pymongo.errors import PyMongoError

from smartshark_roles.config import project_path
from smartshark_roles.io import dataframe_to_markdown, write_dataframe, write_json, write_text
from smartshark_roles.mongo import get_database


SOURCE_METRICS = [
    "n_issues",
    "n_closed_issues",
    "n_bug_issues",
    "n_closed_bug_issues",
    "n_issue_comments",
    "n_issue_events",
    "n_linked_commits",
    "n_commits",
    "n_pull_requests",
    "n_reviews",
    "n_actors_estimated",
]

SCORE_WEIGHTS = {
    "n_closed_bug_issues": 0.20,
    "n_issue_events": 0.15,
    "n_linked_commits": 0.15,
    "n_commits": 0.10,
    "n_issue_comments": 0.10,
    "n_pull_requests": 0.10,
    "n_reviews": 0.10,
    "n_actors_estimated": 0.10,
}


def as_key(value: Any) -> str | None:
    return None if value is None else str(value)


def has_collection(collection_names: set[str], collection_name: str) -> bool:
    return collection_name in collection_names


def is_non_empty_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def is_bug_issue(document: dict[str, Any]) -> bool:
    issue_type = document.get("issue_type")
    return isinstance(issue_type, str) and issue_type.strip().lower() == "bug"


def is_closed_issue(document: dict[str, Any]) -> bool:
    status = document.get("status")
    resolution = document.get("resolution")
    status_closed = isinstance(status, str) and status.strip().lower() in {"closed", "resolved"}
    resolution_closed = (
        isinstance(resolution, str)
        and resolution.strip() != ""
        and resolution.strip().lower() != "unresolved"
    )
    return status_closed or resolution_closed


def add_actor(actor_sets: dict[str, set[str]], project_id: str | None, actor_id: Any) -> None:
    actor_key = as_key(actor_id)
    if project_id is not None and actor_key is not None:
        actor_sets[project_id].add(actor_key)


def add_actor_values(actor_sets: dict[str, set[str]], project_id: str | None, values: Any) -> None:
    if isinstance(values, list):
        for value in values:
            add_actor(actor_sets, project_id, value)
    else:
        add_actor(actor_sets, project_id, values)


def fetch_projects(
    db: Database,
    collection_names: set[str],
    project_collection: str,
    logger: Any,
) -> pd.DataFrame:
    if not has_collection(collection_names, project_collection):
        logger.warning("Project collection is missing: %s", project_collection)
        return pd.DataFrame(columns=["project_id", "project_name"])

    rows: list[dict[str, Any]] = []
    projection = {"name": 1, "project_name": 1, "display_name": 1}
    for document in db[project_collection].find({}, projection=projection):
        project_id = as_key(document.get("_id"))
        rows.append(
            {
                "project_id": project_id,
                "project_name": document.get("name")
                or document.get("project_name")
                or document.get("display_name")
                or project_id,
            }
        )
    frame = pd.DataFrame(rows, columns=["project_id", "project_name"])
    if not frame.empty:
        frame = frame.sort_values(["project_name", "project_id"])
    logger.info("Loaded %d projects", len(frame))
    return frame


def load_id_to_field(
    db: Database,
    collection_names: set[str],
    collection_name: str,
    target_field: str,
    *,
    batch_size: int,
    logger: Any,
) -> tuple[dict[str, str], dict[str, Any]]:
    if not has_collection(collection_names, collection_name):
        return {}, {"collection": collection_name, "stage": f"load_{target_field}", "status": "missing_collection"}

    mapping: dict[str, str] = {}
    try:
        cursor = db[collection_name].find(
            {target_field: {"$exists": True, "$ne": None}},
            projection={target_field: 1},
            batch_size=batch_size,
        )
        for document in cursor:
            source = as_key(document.get("_id"))
            target = as_key(document.get(target_field))
            if source is not None and target is not None:
                mapping[source] = target
        logger.info("Loaded %d id mappings from %s.%s", len(mapping), collection_name, target_field)
        return mapping, {
            "collection": collection_name,
            "stage": f"load_{target_field}",
            "status": "ok",
            "n_keys": len(mapping),
        }
    except PyMongoError as exc:
        logger.exception("Failed to load mapping from %s", collection_name)
        return {}, {
            "collection": collection_name,
            "stage": f"load_{target_field}",
            "status": f"failed: {exc.__class__.__name__}: {exc}",
        }


def find_documents(
    db: Database,
    collection_names: set[str],
    collection_name: str,
    projection: dict[str, int],
    *,
    batch_size: int,
) -> Iterable[dict[str, Any]]:
    if not has_collection(collection_names, collection_name):
        return iter(())
    return db[collection_name].find({}, projection=projection, batch_size=batch_size)


def increment(metrics: dict[str, dict[str, int]], project_id: str | None, metric: str, amount: int = 1) -> None:
    if project_id is not None:
        metrics[project_id][metric] += amount


def process_issues(
    db: Database,
    collection_names: set[str],
    collections: dict[str, str],
    issue_system_to_project: dict[str, str],
    metrics: dict[str, dict[str, int]],
    actor_sets: dict[str, set[str]],
    *,
    batch_size: int,
    logger: Any,
) -> tuple[dict[str, str], dict[str, Any]]:
    collection_name = collections.get("issue", "issue")
    if not has_collection(collection_names, collection_name):
        return {}, {"collection": collection_name, "stage": "process_issues", "status": "missing_collection"}

    issue_to_project: dict[str, str] = {}
    count = 0
    projection = {
        "issue_system_id": 1,
        "issue_type": 1,
        "status": 1,
        "resolution": 1,
        "creator_id": 1,
        "reporter_id": 1,
        "assignee_id": 1,
    }
    try:
        for document in db[collection_name].find({}, projection=projection, batch_size=batch_size):
            issue_id = as_key(document.get("_id"))
            project_id = issue_system_to_project.get(as_key(document.get("issue_system_id")))
            if issue_id is not None and project_id is not None:
                issue_to_project[issue_id] = project_id

            closed = is_closed_issue(document)
            bug = is_bug_issue(document)
            increment(metrics, project_id, "n_issues")
            if closed:
                increment(metrics, project_id, "n_closed_issues")
            if bug:
                increment(metrics, project_id, "n_bug_issues")
            if closed and bug:
                increment(metrics, project_id, "n_closed_bug_issues")

            add_actor(actor_sets, project_id, document.get("creator_id"))
            add_actor(actor_sets, project_id, document.get("reporter_id"))
            add_actor(actor_sets, project_id, document.get("assignee_id"))
            count += 1
        logger.info("Processed %d issues", count)
        return issue_to_project, {
            "collection": collection_name,
            "stage": "process_issues",
            "status": "ok",
            "n_documents": count,
            "n_issue_project_links": len(issue_to_project),
        }
    except PyMongoError as exc:
        logger.exception("Failed to process issues")
        return issue_to_project, {
            "collection": collection_name,
            "stage": "process_issues",
            "status": f"failed: {exc.__class__.__name__}: {exc}",
        }


def process_issue_children(
    db: Database,
    collection_names: set[str],
    collection_name: str,
    metric: str,
    issue_to_project: dict[str, str],
    actor_field: str,
    metrics: dict[str, dict[str, int]],
    actor_sets: dict[str, set[str]],
    *,
    batch_size: int,
    logger: Any,
) -> dict[str, Any]:
    if not has_collection(collection_names, collection_name):
        return {"collection": collection_name, "stage": metric, "status": "missing_collection"}

    count = 0
    linked = 0
    try:
        for document in db[collection_name].find(
            {},
            projection={"issue_id": 1, actor_field: 1},
            batch_size=batch_size,
        ):
            project_id = issue_to_project.get(as_key(document.get("issue_id")))
            increment(metrics, project_id, metric)
            add_actor(actor_sets, project_id, document.get(actor_field))
            if project_id is not None:
                linked += 1
            count += 1
        logger.info("Processed %d documents from %s", count, collection_name)
        return {
            "collection": collection_name,
            "stage": metric,
            "status": "ok",
            "n_documents": count,
            "n_project_links": linked,
        }
    except PyMongoError as exc:
        logger.exception("Failed to process %s", collection_name)
        return {"collection": collection_name, "stage": metric, "status": f"failed: {exc.__class__.__name__}: {exc}"}


def process_commits(
    db: Database,
    collection_names: set[str],
    collections: dict[str, str],
    vcs_to_project: dict[str, str],
    metrics: dict[str, dict[str, int]],
    actor_sets: dict[str, set[str]],
    *,
    batch_size: int,
    logger: Any,
) -> dict[str, Any]:
    collection_name = collections.get("commit", "commit")
    if not has_collection(collection_names, collection_name):
        return {"collection": collection_name, "stage": "process_commits", "status": "missing_collection"}

    count = 0
    linked = 0
    projection = {
        "vcs_system_id": 1,
        "author_id": 1,
        "committer_id": 1,
        "linked_issue_ids": 1,
        "fixed_issue_ids": 1,
    }
    try:
        for document in db[collection_name].find({}, projection=projection, batch_size=batch_size):
            project_id = vcs_to_project.get(as_key(document.get("vcs_system_id")))
            increment(metrics, project_id, "n_commits")
            has_issue_link = is_non_empty_list(document.get("linked_issue_ids")) or is_non_empty_list(
                document.get("fixed_issue_ids")
            )
            if has_issue_link:
                increment(metrics, project_id, "n_linked_commits")
                linked += 1
            add_actor(actor_sets, project_id, document.get("author_id"))
            add_actor(actor_sets, project_id, document.get("committer_id"))
            count += 1
        logger.info("Processed %d commits", count)
        return {
            "collection": collection_name,
            "stage": "process_commits",
            "status": "ok",
            "n_documents": count,
            "n_linked_commits": linked,
        }
    except PyMongoError as exc:
        logger.exception("Failed to process commits")
        return {"collection": collection_name, "stage": "process_commits", "status": f"failed: {exc.__class__.__name__}: {exc}"}


def process_pull_requests(
    db: Database,
    collection_names: set[str],
    collections: dict[str, str],
    pr_system_to_project: dict[str, str],
    metrics: dict[str, dict[str, int]],
    actor_sets: dict[str, set[str]],
    *,
    batch_size: int,
    logger: Any,
) -> tuple[dict[str, str], dict[str, Any]]:
    collection_name = collections.get("pull_request", "pull_request")
    if not has_collection(collection_names, collection_name):
        return {}, {"collection": collection_name, "stage": "process_pull_requests", "status": "missing_collection"}

    pr_to_project: dict[str, str] = {}
    count = 0
    projection = {
        "pull_request_system_id": 1,
        "creator_id": 1,
        "assignee_id": 1,
        "requested_reviewer_ids": 1,
        "linked_user_ids": 1,
    }
    try:
        for document in db[collection_name].find({}, projection=projection, batch_size=batch_size):
            pr_id = as_key(document.get("_id"))
            project_id = pr_system_to_project.get(as_key(document.get("pull_request_system_id")))
            if pr_id is not None and project_id is not None:
                pr_to_project[pr_id] = project_id
            increment(metrics, project_id, "n_pull_requests")
            add_actor(actor_sets, project_id, document.get("creator_id"))
            add_actor(actor_sets, project_id, document.get("assignee_id"))
            add_actor_values(actor_sets, project_id, document.get("requested_reviewer_ids"))
            add_actor_values(actor_sets, project_id, document.get("linked_user_ids"))
            count += 1
        logger.info("Processed %d pull requests", count)
        return pr_to_project, {
            "collection": collection_name,
            "stage": "process_pull_requests",
            "status": "ok",
            "n_documents": count,
            "n_pr_project_links": len(pr_to_project),
        }
    except PyMongoError as exc:
        logger.exception("Failed to process pull requests")
        return pr_to_project, {
            "collection": collection_name,
            "stage": "process_pull_requests",
            "status": f"failed: {exc.__class__.__name__}: {exc}",
        }


def process_reviews(
    db: Database,
    collection_names: set[str],
    collections: dict[str, str],
    pr_to_project: dict[str, str],
    metrics: dict[str, dict[str, int]],
    actor_sets: dict[str, set[str]],
    *,
    batch_size: int,
    logger: Any,
) -> tuple[dict[str, str], dict[str, Any]]:
    collection_name = collections.get("pull_request_review", "pull_request_review")
    if not has_collection(collection_names, collection_name):
        return {}, {"collection": collection_name, "stage": "process_reviews", "status": "missing_collection"}

    review_to_project: dict[str, str] = {}
    count = 0
    try:
        for document in db[collection_name].find(
            {},
            projection={"pull_request_id": 1, "creator_id": 1},
            batch_size=batch_size,
        ):
            review_id = as_key(document.get("_id"))
            project_id = pr_to_project.get(as_key(document.get("pull_request_id")))
            if review_id is not None and project_id is not None:
                review_to_project[review_id] = project_id
            increment(metrics, project_id, "n_reviews")
            add_actor(actor_sets, project_id, document.get("creator_id"))
            count += 1
        logger.info("Processed %d pull request reviews", count)
        return review_to_project, {
            "collection": collection_name,
            "stage": "process_reviews",
            "status": "ok",
            "n_documents": count,
            "n_review_project_links": len(review_to_project),
        }
    except PyMongoError as exc:
        logger.exception("Failed to process reviews")
        return review_to_project, {
            "collection": collection_name,
            "stage": "process_reviews",
            "status": f"failed: {exc.__class__.__name__}: {exc}",
        }


def process_review_comments(
    db: Database,
    collection_names: set[str],
    collections: dict[str, str],
    review_to_project: dict[str, str],
    actor_sets: dict[str, set[str]],
    *,
    batch_size: int,
    logger: Any,
) -> dict[str, Any]:
    collection_name = collections.get("pull_request_review_comment", "pull_request_review_comment")
    if not has_collection(collection_names, collection_name):
        return {"collection": collection_name, "stage": "process_review_comments", "status": "missing_collection"}

    count = 0
    linked = 0
    try:
        for document in db[collection_name].find(
            {},
            projection={"pull_request_review_id": 1, "creator_id": 1},
            batch_size=batch_size,
        ):
            project_id = review_to_project.get(as_key(document.get("pull_request_review_id")))
            add_actor(actor_sets, project_id, document.get("creator_id"))
            if project_id is not None:
                linked += 1
            count += 1
        logger.info("Processed %d pull request review comments", count)
        return {
            "collection": collection_name,
            "stage": "process_review_comments",
            "status": "ok",
            "n_documents": count,
            "n_project_links": linked,
        }
    except PyMongoError as exc:
        logger.exception("Failed to process review comments")
        return {
            "collection": collection_name,
            "stage": "process_review_comments",
            "status": f"failed: {exc.__class__.__name__}: {exc}",
        }


def process_pull_request_events(
    db: Database,
    collection_names: set[str],
    collections: dict[str, str],
    pr_to_project: dict[str, str],
    actor_sets: dict[str, set[str]],
    *,
    batch_size: int,
    logger: Any,
) -> dict[str, Any]:
    collection_name = collections.get("pull_request_event", "pull_request_event")
    if not has_collection(collection_names, collection_name):
        return {"collection": collection_name, "stage": "process_pull_request_events", "status": "missing_collection"}

    count = 0
    linked = 0
    try:
        for document in db[collection_name].find(
            {},
            projection={"pull_request_id": 1, "author_id": 1},
            batch_size=batch_size,
        ):
            project_id = pr_to_project.get(as_key(document.get("pull_request_id")))
            add_actor(actor_sets, project_id, document.get("author_id"))
            if project_id is not None:
                linked += 1
            count += 1
        logger.info("Processed %d pull request events", count)
        return {
            "collection": collection_name,
            "stage": "process_pull_request_events",
            "status": "ok",
            "n_documents": count,
            "n_project_links": linked,
        }
    except PyMongoError as exc:
        logger.exception("Failed to process pull request events")
        return {
            "collection": collection_name,
            "stage": "process_pull_request_events",
            "status": f"failed: {exc.__class__.__name__}: {exc}",
        }


def compute_usable_score(projects: pd.DataFrame) -> pd.Series:
    if projects.empty:
        return pd.Series(dtype="float64")

    score = pd.Series(0.0, index=projects.index)
    for metric, weight in SCORE_WEIGHTS.items():
        maximum = float(projects[metric].max()) if metric in projects.columns else 0.0
        denominator = math.log1p(maximum) or 1.0
        score += weight * projects[metric].map(lambda value: math.log1p(float(value)) / denominator)
    return (score * 100).round(3)


def build_project_source_availability(
    db: Database,
    config: dict[str, Any],
    logger: Any,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    availability_config = config.get("project_availability", {})
    collections = availability_config.get("collections", {})
    batch_size = int(availability_config.get("batch_size", 1000))
    collection_names = set(db.list_collection_names())

    if collections.get("issue_event", "issue_event") not in collection_names and "event" in collection_names:
        collections = {**collections, "issue_event": "event"}

    projects = fetch_projects(db, collection_names, collections.get("project", "project"), logger)
    notes: list[dict[str, Any]] = []
    if projects.empty:
        return projects, [{"collection": "project", "stage": "fetch_projects", "status": "empty"}]

    project_ids = list(projects["project_id"])
    metrics = {project_id: {metric: 0 for metric in SOURCE_METRICS} for project_id in project_ids}
    actor_sets = {project_id: set() for project_id in project_ids}

    vcs_to_project, note = load_id_to_field(
        db,
        collection_names,
        collections.get("vcs_system", "vcs_system"),
        "project_id",
        batch_size=batch_size,
        logger=logger,
    )
    notes.append(note)
    issue_system_to_project, note = load_id_to_field(
        db,
        collection_names,
        collections.get("issue_system", "issue_system"),
        "project_id",
        batch_size=batch_size,
        logger=logger,
    )
    notes.append(note)
    pr_system_to_project, note = load_id_to_field(
        db,
        collection_names,
        collections.get("pull_request_system", "pull_request_system"),
        "project_id",
        batch_size=batch_size,
        logger=logger,
    )
    notes.append(note)

    issue_to_project, note = process_issues(
        db,
        collection_names,
        collections,
        issue_system_to_project,
        metrics,
        actor_sets,
        batch_size=batch_size,
        logger=logger,
    )
    notes.append(note)
    notes.append(
        process_issue_children(
            db,
            collection_names,
            collections.get("issue_comment", "issue_comment"),
            "n_issue_comments",
            issue_to_project,
            "author_id",
            metrics,
            actor_sets,
            batch_size=batch_size,
            logger=logger,
        )
    )
    notes.append(
        process_issue_children(
            db,
            collection_names,
            collections.get("issue_event", "event"),
            "n_issue_events",
            issue_to_project,
            "author_id",
            metrics,
            actor_sets,
            batch_size=batch_size,
            logger=logger,
        )
    )
    notes.append(
        process_commits(
            db,
            collection_names,
            collections,
            vcs_to_project,
            metrics,
            actor_sets,
            batch_size=batch_size,
            logger=logger,
        )
    )
    pr_to_project, note = process_pull_requests(
        db,
        collection_names,
        collections,
        pr_system_to_project,
        metrics,
        actor_sets,
        batch_size=batch_size,
        logger=logger,
    )
    notes.append(note)
    review_to_project, note = process_reviews(
        db,
        collection_names,
        collections,
        pr_to_project,
        metrics,
        actor_sets,
        batch_size=batch_size,
        logger=logger,
    )
    notes.append(note)
    notes.append(
        process_review_comments(
            db,
            collection_names,
            collections,
            review_to_project,
            actor_sets,
            batch_size=batch_size,
            logger=logger,
        )
    )
    notes.append(
        process_pull_request_events(
            db,
            collection_names,
            collections,
            pr_to_project,
            actor_sets,
            batch_size=batch_size,
            logger=logger,
        )
    )

    metric_frame = pd.DataFrame(
        [
            {
                "project_id": project_id,
                **metrics[project_id],
                "n_actors_estimated": len(actor_sets[project_id]),
            }
            for project_id in project_ids
        ]
    )
    projects = projects.merge(metric_frame, on="project_id", how="left")
    for metric in SOURCE_METRICS:
        projects[metric] = projects[metric].fillna(0).astype("int64")

    projects["has_issue_data"] = projects["n_issues"] > 0
    projects["has_vcs_data"] = projects["n_commits"] > 0
    projects["has_pr_data"] = projects["n_pull_requests"] > 0
    projects["has_bug_lifecycle_sources"] = (
        (projects["n_closed_bug_issues"] > 0)
        & (projects["n_issue_events"] > 0)
        & (projects["n_linked_commits"] > 0)
        & (projects["n_commits"] > 0)
    )
    projects["usable_score"] = compute_usable_score(projects)
    projects = projects.sort_values(["usable_score", "n_closed_bug_issues", "n_linked_commits"], ascending=False)
    return projects, notes


def build_project_report(projects: pd.DataFrame, notes: list[dict[str, Any]], config: dict[str, Any]) -> str:
    top_columns = [
        "project_name",
        "n_issues",
        "n_closed_issues",
        "n_bug_issues",
        "n_closed_bug_issues",
        "n_issue_comments",
        "n_issue_events",
        "n_linked_commits",
        "n_commits",
        "n_pull_requests",
        "n_reviews",
        "n_actors_estimated",
        "usable_score",
    ]
    lines = [
        "# SmartSHARK Project Source Availability",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- MongoDB database: `{config.get('mongodb', {}).get('database')}`",
        f"- Config: `{config.get('_config_path')}`",
        "",
        "## Top Projects",
        "",
    ]
    if projects.empty:
        lines.append("No projects were found.")
    else:
        lines.append(dataframe_to_markdown(projects.loc[:, top_columns].head(20)))

    lines.extend(["", "## Metric Status", ""])
    lines.append(dataframe_to_markdown(pd.DataFrame(notes)) if notes else "No metric notes were recorded.")
    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- This step performs read-only MongoDB operations.",
            "- `n_bug_issues` uses `issue.issue_type == 'Bug'` case-insensitively.",
            "- Closed issues are identified by `status in {Closed, Resolved}` or a non-empty `resolution` other than `Unresolved`.",
            "- Issue events use the SmartSHARK `event` collection linked through `event.issue_id -> issue._id -> issue_system.project_id`.",
            "- Linked commits are commits with non-empty `linked_issue_ids` or `fixed_issue_ids`.",
            "- `n_actors_estimated` is the union of observed actor ObjectIds across issue, commit, issue comment/event, pull request, review, review comment, and pull request event sources.",
            "- `usable_score` is a weighted log-normalized score over closed bug issues, issue events, linked commits, commits, comments, pull requests, reviews, and estimated actors.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_project_availability(config: dict[str, Any], logger: Any) -> dict[str, Path]:
    db = get_database(config, logger=logger)
    availability_config = config.get("project_availability", {})
    output_dir = project_path(availability_config.get("output_dir", "data/interim/db_audit"))
    report_dir = project_path(availability_config.get("report_dir", "reports/data_audit"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    projects, notes = build_project_source_availability(db, config, logger)

    paths = {
        "availability_csv": output_dir / "project_availability.csv",
        "availability_parquet": output_dir / "project_availability.parquet",
        "source_availability_csv": output_dir / "project_source_availability.csv",
        "source_availability_parquet": output_dir / "project_source_availability.parquet",
        "report_source_availability_csv": report_dir / "project_source_availability.csv",
        "report_source_availability_md": report_dir / "project_source_availability.md",
        "notes_json": output_dir / "project_source_availability_notes.json",
        "legacy_notes_json": output_dir / "project_availability_notes.json",
        "legacy_report_md": report_dir / "project_availability.md",
    }
    write_dataframe(projects, paths["availability_csv"], paths["availability_parquet"], logger=logger)
    write_dataframe(projects, paths["source_availability_csv"], paths["source_availability_parquet"], logger=logger)
    write_dataframe(projects, paths["report_source_availability_csv"], logger=logger)
    write_json(notes, paths["notes_json"], logger=logger)
    write_json(notes, paths["legacy_notes_json"], logger=logger)
    report_text = build_project_report(projects, notes, config)
    write_text(report_text, paths["report_source_availability_md"], logger=logger)
    write_text(report_text, paths["legacy_report_md"], logger=logger)
    logger.info("Project source availability completed")
    return paths
