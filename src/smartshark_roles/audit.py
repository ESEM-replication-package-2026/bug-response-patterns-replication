from __future__ import annotations

import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from bson import ObjectId
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from smartshark_roles.config import project_path
from smartshark_roles.io import dataframe_to_markdown, write_dataframe, write_json, write_text
from smartshark_roles.mongo import get_database


def type_label(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, ObjectId):
        return "ObjectId"
    return type(value).__name__


def collect_field_values(
    value: Any,
    prefix: str,
    fields: dict[str, list[Any]],
    *,
    flatten_nested_fields: bool,
) -> None:
    if prefix:
        fields[prefix].append(value)

    if not flatten_nested_fields:
        return

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            collect_field_values(
                child,
                child_prefix,
                fields,
                flatten_nested_fields=flatten_nested_fields,
            )
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping):
                for key, child in item.items():
                    child_prefix = f"{prefix}[].{key}" if prefix else f"[].{key}"
                    collect_field_values(
                        child,
                        child_prefix,
                        fields,
                        flatten_nested_fields=flatten_nested_fields,
                    )


def document_fields(document: Mapping[str, Any], *, flatten_nested_fields: bool) -> dict[str, list[Any]]:
    fields: dict[str, list[Any]] = defaultdict(list)
    for key, value in document.items():
        collect_field_values(
            value,
            str(key),
            fields,
            flatten_nested_fields=flatten_nested_fields,
        )
    return fields


def count_documents(collection: Collection, timeout_ms: int) -> tuple[int | None, str]:
    try:
        return collection.count_documents({}, maxTimeMS=timeout_ms), "ok"
    except PyMongoError as exc:
        return None, f"failed: {exc.__class__.__name__}: {exc}"


def collection_storage_stats(db: Database, collection_name: str) -> dict[str, Any]:
    try:
        stats = db.command("collStats", collection_name)
    except PyMongoError as exc:
        return {"collstats_status": f"failed: {exc.__class__.__name__}: {exc}"}
    return {
        "collstats_status": "ok",
        "size_bytes": stats.get("size"),
        "storage_size_bytes": stats.get("storageSize"),
        "total_index_size_bytes": stats.get("totalIndexSize"),
        "nindexes": stats.get("nindexes"),
    }


def iter_audit_documents(
    collection: Collection,
    *,
    exact: bool,
    max_documents: int,
    batch_size: int,
) -> Iterable[Mapping[str, Any]]:
    cursor = collection.find({}, batch_size=batch_size)
    if not exact and max_documents > 0:
        cursor = cursor.limit(max_documents)
    yield from cursor


def audit_collection(
    db: Database,
    collection_name: str,
    *,
    exact_field_coverage: bool,
    max_documents: int,
    batch_size: int,
    count_timeout_ms: int,
    flatten_nested_fields: bool,
    logger: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logger.info("Auditing collection=%s", collection_name)
    collection = db[collection_name]
    try:
        estimated_count = collection.estimated_document_count()
    except PyMongoError as exc:
        estimated_count = None
        logger.warning("Estimated count failed for %s: %s", collection_name, exc)
    exact_count, exact_count_status = count_documents(collection, count_timeout_ms)

    field_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"present_count": 0, "null_count": 0, "types": Counter()}
    )
    audited_documents = 0
    cursor_status = "ok"

    try:
        for document in iter_audit_documents(
            collection,
            exact=exact_field_coverage,
            max_documents=max_documents,
            batch_size=batch_size,
        ):
            audited_documents += 1
            fields = document_fields(document, flatten_nested_fields=flatten_nested_fields)
            for field_path, values in fields.items():
                stat = field_stats[field_path]
                stat["present_count"] += 1
                if values and all(value is None for value in values):
                    stat["null_count"] += 1
                for value in values:
                    stat["types"][type_label(value)] += 1
    except PyMongoError as exc:
        cursor_status = f"failed: {exc.__class__.__name__}: {exc}"
        logger.exception("Field audit failed for collection=%s", collection_name)

    collection_row = {
        "collection": collection_name,
        "estimated_document_count": estimated_count,
        "exact_document_count": exact_count,
        "exact_count_status": exact_count_status,
        "field_audit_document_count": audited_documents,
        "field_audit_mode": "exact" if exact_field_coverage else "sample",
        "field_audit_limit": None if exact_field_coverage else max_documents,
        "field_audit_status": cursor_status,
    }
    collection_row.update(collection_storage_stats(db, collection_name))

    field_rows: list[dict[str, Any]] = []
    for field_path in sorted(field_stats):
        stat = field_stats[field_path]
        present_count = int(stat["present_count"])
        null_count = int(stat["null_count"])
        missing_count = max(audited_documents - present_count, 0)
        denominator = audited_documents or 1
        type_counts = stat["types"]
        field_rows.append(
            {
                "collection": collection_name,
                "field_path": field_path,
                "audited_document_count": audited_documents,
                "present_count": present_count,
                "present_non_null_count": present_count - null_count,
                "null_count": null_count,
                "missing_count": missing_count,
                "present_rate": present_count / denominator,
                "missing_rate": missing_count / denominator,
                "missing_or_null_rate": (missing_count + null_count) / denominator,
                "value_types": ";".join(f"{key}:{type_counts[key]}" for key in sorted(type_counts)),
                "field_audit_mode": "exact" if exact_field_coverage else "sample",
            }
        )

    return collection_row, field_rows


def build_audit_report(
    *,
    config: dict[str, Any],
    collection_frame: pd.DataFrame,
    field_frame: pd.DataFrame,
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    audit_config = config.get("audit", {})
    lines = [
        "# SmartSHARK DB Audit",
        "",
        f"- Generated at UTC: `{generated_at}`",
        f"- MongoDB database: `{config.get('mongodb', {}).get('database')}`",
        f"- Config: `{config.get('_config_path')}`",
        f"- Field coverage mode: `{'exact' if audit_config.get('exact_field_coverage') else 'sample'}`",
        f"- Max documents per collection: `{audit_config.get('max_documents_per_collection')}`",
        "",
        "## Collections",
        "",
    ]
    if collection_frame.empty:
        lines.append("No collections were found.")
    else:
        display_cols = [
            "collection",
            "estimated_document_count",
            "exact_document_count",
            "field_audit_document_count",
            "field_audit_status",
        ]
        lines.append(dataframe_to_markdown(collection_frame[display_cols]))

    lines.extend(["", "## Missingness Hotspots", ""])
    if field_frame.empty:
        lines.append("No fields were audited.")
    else:
        hotspots = (
            field_frame.sort_values(["missing_or_null_rate", "collection", "field_path"], ascending=[False, True, True])
            .head(30)
            .loc[
                :,
                [
                    "collection",
                    "field_path",
                    "audited_document_count",
                    "missing_or_null_rate",
                    "missing_rate",
                    "null_count",
                    "value_types",
                ],
            ]
        )
        lines.append(dataframe_to_markdown(hotspots))

    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- This step performs read-only MongoDB operations.",
            "- Missing rates are computed over audited documents. In sample mode they are sample estimates, not full-population rates.",
            "- Nested dictionaries are flattened with dot paths. Fields inside arrays of dictionaries use `[]` in the path.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_audit_errors(collection_frame: pd.DataFrame) -> pd.DataFrame:
    if collection_frame.empty:
        return pd.DataFrame(columns=["collection", "stage", "status"])

    error_rows: list[dict[str, Any]] = []
    status_columns = [
        "exact_count_status",
        "field_audit_status",
        "collstats_status",
    ]
    for _, row in collection_frame.iterrows():
        for column in status_columns:
            status = row.get(column)
            if pd.notna(status) and str(status) not in {"ok", ""}:
                error_rows.append(
                    {
                        "collection": row.get("collection"),
                        "stage": column,
                        "status": status,
                    }
                )
    return pd.DataFrame(error_rows, columns=["collection", "stage", "status"])


def run_db_audit(config: dict[str, Any], logger: Any) -> dict[str, Path]:
    db = get_database(config, logger=logger)
    audit_config = config.get("audit", {})
    output_dir = project_path(audit_config.get("output_dir", "data/interim/db_audit"))
    report_dir = project_path(audit_config.get("report_dir", "reports/data_audit"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    exact_field_coverage = bool(audit_config.get("exact_field_coverage", False))
    max_documents = int(audit_config.get("max_documents_per_collection", 5000))
    batch_size = int(audit_config.get("batch_size", 1000))
    count_timeout_ms = int(audit_config.get("count_documents_timeout_ms", 120000))
    flatten_nested_fields = bool(audit_config.get("flatten_nested_fields", True))

    collection_names = sorted(db.list_collection_names())
    logger.info("Found %d collections", len(collection_names))

    collection_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    for collection_name in collection_names:
        collection_row, rows = audit_collection(
            db,
            collection_name,
            exact_field_coverage=exact_field_coverage,
            max_documents=max_documents,
            batch_size=batch_size,
            count_timeout_ms=count_timeout_ms,
            flatten_nested_fields=flatten_nested_fields,
            logger=logger,
        )
        collection_rows.append(collection_row)
        field_rows.extend(rows)

    collection_columns = [
        "collection",
        "estimated_document_count",
        "exact_document_count",
        "exact_count_status",
        "field_audit_document_count",
        "field_audit_mode",
        "field_audit_limit",
        "field_audit_status",
    ]
    field_columns = [
        "collection",
        "field_path",
        "audited_document_count",
        "present_count",
        "present_non_null_count",
        "null_count",
        "missing_count",
        "present_rate",
        "missing_rate",
        "missing_or_null_rate",
        "value_types",
        "field_audit_mode",
    ]
    collection_frame = pd.DataFrame(collection_rows, columns=collection_columns)
    field_frame = pd.DataFrame(field_rows, columns=field_columns)
    if not collection_frame.empty:
        collection_frame = collection_frame.sort_values("collection")
    if not field_frame.empty:
        field_frame = field_frame.sort_values(["collection", "field_path"])

    paths = {
        "collection_counts_csv": output_dir / "collection_counts.csv",
        "collection_counts_parquet": output_dir / "collection_counts.parquet",
        "field_coverage_csv": output_dir / "field_coverage.csv",
        "field_coverage_parquet": output_dir / "field_coverage.parquet",
        "metadata_json": output_dir / "db_audit_run_metadata.json",
        "report_md": report_dir / "db_audit_summary.md",
        "report_collection_counts_csv": report_dir / "collection_counts.csv",
        "report_collection_fields_csv": report_dir / "collection_fields.csv",
        "report_summary_md": report_dir / "data_audit_summary.md",
        "report_errors_csv": report_dir / "audit_errors.csv",
    }

    write_dataframe(
        collection_frame,
        paths["collection_counts_csv"],
        paths["collection_counts_parquet"],
        logger=logger,
    )
    write_dataframe(
        field_frame,
        paths["field_coverage_csv"],
        paths["field_coverage_parquet"],
        logger=logger,
    )
    write_dataframe(collection_frame, paths["report_collection_counts_csv"], logger=logger)
    write_dataframe(field_frame, paths["report_collection_fields_csv"], logger=logger)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config.get("_config_path"),
        "python": sys.version,
        "platform": platform.platform(),
        "collection_count": len(collection_names),
        "field_row_count": len(field_frame),
        "audit": audit_config,
    }
    write_json(metadata, paths["metadata_json"], logger=logger)
    report_text = build_audit_report(config=config, collection_frame=collection_frame, field_frame=field_frame)
    write_text(report_text, paths["report_md"], logger=logger)
    write_text(report_text, paths["report_summary_md"], logger=logger)

    error_frame = build_audit_errors(collection_frame)
    write_dataframe(error_frame, paths["report_errors_csv"], logger=logger)
    logger.info("DB audit completed")
    return paths
