from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from smartshark_roles.io import dataframe_to_markdown, write_dataframe, write_text


METADATA_COLUMNS = (
    "actor_id",
    "is_bot",
    "is_low_activity",
    "first_event_at",
    "last_event_at",
    "first_lifecycle_stage",
    "last_lifecycle_stage",
)

EXPLICIT_EXCLUDED_FEATURES = {
    "median_inter_event_hours": "excluded_temporal_imputation_risk",
}

HIGH_MISSING_THRESHOLD = 0.5


@dataclass(frozen=True)
class PreprocessingPaths:
    model_input_parquet: Path
    model_input_csv: Path
    metadata_csv: Path
    summary_md: Path
    feature_list_csv: Path
    imputation_log_csv: Path
    excluded_features_csv: Path
    missingness_after_csv: Path


@dataclass(frozen=True)
class PreprocessingMetrics:
    raw_actors: int
    bot_excluded: int
    low_activity_excluded: int
    bot_and_low_activity: int
    main_clustering_actors: int
    input_feature_count: int
    excluded_feature_count: int
    final_feature_count: int
    nan_after_preprocessing: int


def read_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature file does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature file type: {path.suffix}")


def feature_kind(feature: str) -> str:
    if feature.endswith("_ratio"):
        return "ratio"
    if feature.startswith("n_") or feature in {"total_events", "active_days"}:
        return "count"
    if feature.startswith("median_"):
        return "median"
    return "numeric"


def metadata_frame(features: pd.DataFrame, main_mask: pd.Series) -> pd.DataFrame:
    available = [column for column in METADATA_COLUMNS if column in features.columns]
    metadata = features.loc[main_mask, available].copy()
    metadata.insert(0, "row_index", range(len(metadata)))
    return metadata


def candidate_feature_columns(features: pd.DataFrame) -> list[str]:
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [column for column in numeric_columns if column not in METADATA_COLUMNS]


def build_exclusion_log(features: pd.DataFrame, candidate_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in features.columns:
        if column in METADATA_COLUMNS:
            rows.append(
                {
                    "feature": column,
                    "reason": "metadata_separated",
                    "missing_rate": features[column].isna().mean(),
                    "feature_kind": "metadata",
                }
            )
        elif column in EXPLICIT_EXCLUDED_FEATURES:
            rows.append(
                {
                    "feature": column,
                    "reason": EXPLICIT_EXCLUDED_FEATURES[column],
                    "missing_rate": features[column].isna().mean(),
                    "feature_kind": feature_kind(column),
                }
            )
        elif column not in candidate_columns:
            rows.append(
                {
                    "feature": column,
                    "reason": "non_numeric_or_not_model_feature",
                    "missing_rate": features[column].isna().mean(),
                    "feature_kind": "non_numeric",
                }
            )
    return pd.DataFrame(rows).sort_values(["reason", "feature"])


def preprocessing_plan(model_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    working = model_features.copy()
    logs: list[dict[str, Any]] = []
    added_flags: list[str] = []

    for column in list(working.columns):
        missing_count = int(working[column].isna().sum())
        missing_rate = missing_count / len(working) if len(working) else 0.0
        kind = feature_kind(column)
        action = "none"
        fill_value: float | None = None

        if column == "median_comment_length":
            flag = "median_comment_length_missing"
            working[flag] = working[column].isna().astype(int)
            added_flags.append(flag)
            working[column] = working[column].fillna(0)
            action = "missing_flag_plus_zero_imputation"
            fill_value = 0.0
        elif missing_count and kind in {"count", "ratio", "median", "numeric"}:
            working[column] = working[column].fillna(0)
            action = "zero_imputation"
            fill_value = 0.0

        if missing_count or action != "none":
            logs.append(
                {
                    "feature": column,
                    "feature_kind": kind,
                    "missing_count_before": missing_count,
                    "missing_rate_before": missing_rate,
                    "action": action,
                    "fill_value": fill_value,
                    "missing_flag_added": "median_comment_length_missing" if column == "median_comment_length" else "",
                    "reason": "comment_absence_signal" if column == "median_comment_length" else "count_ratio_zero_policy",
                }
            )

        if missing_rate > HIGH_MISSING_THRESHOLD and column != "median_comment_length":
            logs.append(
                {
                    "feature": column,
                    "feature_kind": kind,
                    "missing_count_before": missing_count,
                    "missing_rate_before": missing_rate,
                    "action": "review_required",
                    "fill_value": pd.NA,
                    "missing_flag_added": "",
                    "reason": "missing_rate_above_50_percent",
                }
            )

    return working, pd.DataFrame(logs), added_flags


def transform_model_features(model_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    transformed = model_features.copy()
    rows: list[dict[str, Any]] = []
    for column in transformed.columns:
        kind = feature_kind(column)
        before_missing = int(transformed[column].isna().sum())
        if kind == "count":
            transformed[column] = np.log1p(pd.to_numeric(transformed[column], errors="coerce").fillna(0).clip(lower=0))
            transform = "log1p"
        else:
            transformed[column] = pd.to_numeric(transformed[column], errors="coerce")
            transform = "identity"

        p99 = transformed[column].quantile(0.99)
        if pd.notna(p99):
            transformed[column] = transformed[column].clip(upper=p99)
        rows.append(
            {
                "feature": column,
                "feature_kind": kind,
                "transform": transform,
                "winsorize_upper_quantile": 0.99,
                "winsorize_upper_value": p99,
                "missing_before_scaling": before_missing,
            }
        )

    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(transformed), columns=transformed.columns, index=transformed.index)
    transform_log = pd.DataFrame(rows)
    transform_log["scaler"] = "StandardScaler"
    return scaled, transform_log


def missingness_after(frame: pd.DataFrame) -> pd.DataFrame:
    total = len(frame)
    rows = [
        {
            "feature": column,
            "missing_count": int(frame[column].isna().sum()),
            "missing_rate": frame[column].isna().mean() if total else 0.0,
        }
        for column in frame.columns
    ]
    return pd.DataFrame(rows).sort_values(["missing_count", "feature"], ascending=[False, True])


def feature_list(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": column,
                "feature_kind": feature_kind(column),
                "used_for_main_clustering": True,
                "scaled": True,
            }
            for column in frame.columns
        ]
    )


def summary_markdown(
    metrics: PreprocessingMetrics,
    excluded: pd.DataFrame,
    imputation_log: pd.DataFrame,
    missing_after: pd.DataFrame,
    raw_path: Path,
) -> str:
    excluded_display = excluded[["feature", "reason", "missing_rate", "feature_kind"]].copy()
    imputed_display = imputation_log[["feature", "feature_kind", "missing_count_before", "missing_rate_before", "action", "missing_flag_added", "reason"]].copy()
    lines = [
        "# Feature Preprocessing Summary",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Input features: `{raw_path}`",
        "- DB access: not used in this step.",
        "- Privacy note: model input contains no raw actor IDs, emails, usernames, or names. Actor IDs and filter flags are stored separately in metadata.",
        "",
        "## Overall",
        "",
        f"- Raw actors: `{metrics.raw_actors}`",
        f"- Bot excluded: `{metrics.bot_excluded}`",
        f"- Low activity excluded: `{metrics.low_activity_excluded}`",
        f"- Bot and low activity overlap: `{metrics.bot_and_low_activity}`",
        f"- Main clustering actors: `{metrics.main_clustering_actors}`",
        f"- Input feature count: `{metrics.input_feature_count}`",
        f"- Excluded feature count: `{metrics.excluded_feature_count}`",
        f"- Final feature count: `{metrics.final_feature_count}`",
        f"- NaN after preprocessing: `{metrics.nan_after_preprocessing}`",
        "",
        "## Missing Value Policy",
        "",
        "- Count and ratio features are zero-imputed before transformation.",
        "- `median_comment_length` is zero-imputed and paired with `median_comment_length_missing`.",
        "- `median_inter_event_hours` is excluded from the main model input because temporal imputation can distort distances.",
        "- Count features are `log1p` transformed, then all model features are winsorized at p99 and standardized.",
        "",
        "## Excluded Features",
        "",
        dataframe_to_markdown(excluded_display),
        "",
        "## Imputation Log",
        "",
        dataframe_to_markdown(imputed_display) if not imputed_display.empty else "No imputation was required.",
        "",
        "## Missingness After Preprocessing",
        "",
        dataframe_to_markdown(missing_after.head(20)),
    ]
    return "\n".join(lines) + "\n"


def default_paths(project_root: Path) -> PreprocessingPaths:
    cluster_dir = project_root / "data" / "processed" / "clusters"
    report_dir = project_root / "reports" / "features"
    return PreprocessingPaths(
        model_input_parquet=cluster_dir / "actor_features_model_input.parquet",
        model_input_csv=cluster_dir / "actor_features_model_input.csv",
        metadata_csv=cluster_dir / "actor_metadata_for_clustering.csv",
        summary_md=report_dir / "preprocessing_summary.md",
        feature_list_csv=report_dir / "preprocessing_feature_list.csv",
        imputation_log_csv=report_dir / "preprocessing_imputation_log.csv",
        excluded_features_csv=report_dir / "excluded_features.csv",
        missingness_after_csv=report_dir / "missingness_after_preprocessing.csv",
    )


def run_preprocessing(features_path: Path, project_root: Path, logger: Any | None = None) -> tuple[pd.DataFrame, PreprocessingMetrics, PreprocessingPaths]:
    raw = read_features(features_path)
    if "actor_id" not in raw.columns or "is_bot" not in raw.columns or "is_low_activity" not in raw.columns:
        raise KeyError("Raw features must contain actor_id, is_bot, and is_low_activity")

    raw_actors = len(raw)
    bot_mask = raw["is_bot"].fillna(False).astype(bool)
    low_mask = raw["is_low_activity"].fillna(False).astype(bool)
    main_mask = ~bot_mask & ~low_mask
    main = raw.loc[main_mask].copy()

    candidates = candidate_feature_columns(raw)
    candidates = [column for column in candidates if column not in EXPLICIT_EXCLUDED_FEATURES]
    excluded = build_exclusion_log(raw, candidates)
    model_raw = main[candidates].copy()
    imputed, imputation_log, _ = preprocessing_plan(model_raw)
    scaled, transform_log = transform_model_features(imputed)
    nan_after = int(scaled.isna().sum().sum())
    if nan_after:
        raise ValueError(f"NaN remains after preprocessing: {nan_after}")

    combined_imputation = pd.merge(
        transform_log,
        imputation_log,
        on=["feature", "feature_kind"],
        how="left",
    )
    combined_imputation["missing_count_before"] = combined_imputation["missing_count_before"].fillna(0).astype(int)
    combined_imputation["missing_rate_before"] = combined_imputation["missing_rate_before"].fillna(0.0)
    combined_imputation["action"] = combined_imputation["action"].fillna("none")
    combined_imputation["missing_flag_added"] = combined_imputation["missing_flag_added"].fillna("")
    combined_imputation["reason"] = combined_imputation["reason"].fillna("")

    metadata = metadata_frame(raw, main_mask)
    missing_after = missingness_after(scaled)
    features_used = feature_list(scaled)

    metrics = PreprocessingMetrics(
        raw_actors=raw_actors,
        bot_excluded=int(bot_mask.sum()),
        low_activity_excluded=int(low_mask.sum()),
        bot_and_low_activity=int((bot_mask & low_mask).sum()),
        main_clustering_actors=len(main),
        input_feature_count=len(candidate_feature_columns(raw)),
        excluded_feature_count=len(excluded),
        final_feature_count=len(scaled.columns),
        nan_after_preprocessing=nan_after,
    )

    paths = default_paths(project_root)
    paths.model_input_parquet.parent.mkdir(parents=True, exist_ok=True)
    paths.summary_md.parent.mkdir(parents=True, exist_ok=True)
    write_dataframe(scaled, paths.model_input_csv, paths.model_input_parquet, logger=logger)
    write_dataframe(metadata, paths.metadata_csv, logger=logger)
    write_dataframe(features_used, paths.feature_list_csv, logger=logger)
    write_dataframe(combined_imputation, paths.imputation_log_csv, logger=logger)
    write_dataframe(excluded, paths.excluded_features_csv, logger=logger)
    write_dataframe(missing_after, paths.missingness_after_csv, logger=logger)
    write_text(summary_markdown(metrics, excluded, combined_imputation, missing_after, features_path), paths.summary_md, logger=logger)
    return scaled, metrics, paths
