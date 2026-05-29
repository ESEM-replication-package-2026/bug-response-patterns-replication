from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from bson import ObjectId


def utc_now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (ObjectId, Path)):
        return str(value)
    return repr(value)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def log_overwrite(path: Path, logger: Any | None) -> None:
    if path.exists() and logger is not None:
        logger.warning("Overwriting existing file: %s", path)


def write_dataframe(
    frame: pd.DataFrame,
    csv_path: Path,
    parquet_path: Path | None = None,
    *,
    logger: Any | None = None,
    index: bool = False,
) -> None:
    ensure_parent(csv_path)
    log_overwrite(csv_path, logger)
    frame.to_csv(csv_path, index=index)
    if logger is not None:
        logger.info("Wrote CSV: %s (%d rows)", csv_path, len(frame))

    if parquet_path is not None:
        ensure_parent(parquet_path)
        log_overwrite(parquet_path, logger)
        frame.to_parquet(parquet_path, index=index)
        if logger is not None:
            logger.info("Wrote Parquet: %s (%d rows)", parquet_path, len(frame))


def write_json(payload: Any, path: Path, *, logger: Any | None = None) -> None:
    ensure_parent(path)
    log_overwrite(path, logger)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")
    if logger is not None:
        logger.info("Wrote JSON: %s", path)


def write_text(text: str, path: Path, *, logger: Any | None = None) -> None:
    ensure_parent(path)
    log_overwrite(path, logger)
    path.write_text(text, encoding="utf-8")
    if logger is not None:
        logger.info("Wrote text: %s", path)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small dataframe as a GitHub-style Markdown table without extra dependencies."""
    if frame.empty:
        return ""

    display = frame.fillna("")
    columns = [str(column) for column in display.columns]

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(cell(row[column]) for column in display.columns) + " |")
    return "\n".join(lines)
