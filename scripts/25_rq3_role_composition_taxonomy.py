"""RQ3 descriptive taxonomy of bug-level role composition patterns.

This script intentionally does not modify existing RQ3 outputs. It creates a
separate descriptive experiment that clusters closed bug issues by the number of
actors from each candidate behavioral role that participated in the issue.

Claim type: descriptive taxonomy, not causal effect estimation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(ROOT / "reports" / "figures" / "rq3_role_composition_taxonomy" / "matplotlib_cache"),
)
DEFAULT_EVENTS = ROOT / "data" / "interim" / "events" / "events_normalized.parquet"
DEFAULT_LABELS = ROOT / "data" / "processed" / "clusters" / "actor_cluster_labels.csv"
DEFAULT_TAXONOMY = ROOT / "reports" / "cluster_profiles" / "candidate_role_taxonomy.csv"
DEFAULT_METRICS = ROOT / "reports" / "results" / "issue_lifecycle_metrics.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "results" / "rq3_role_composition_taxonomy"
DEFAULT_FIGURES = ROOT / "reports" / "figures" / "rq3_role_composition_taxonomy"

ROLE_COLUMNS = {
    "Fixer / Code Contributor": "n_fixer",
    "Issue-side Participant": "n_issue_side_participant",
    "PR Integrator": "n_pr_integrator",
    "Review-Integration Hybrid": "n_review_integration_hybrid",
    "Review-focused Participant": "n_review_focused_participant",
    "Issue-side Boundary Participant": "n_issue_side_boundary_participant",
}

CLUSTER_FEATURES = list(ROLE_COLUMNS.values())


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logging.info("Reading events: %s", args.events)
    events = pd.read_parquet(
        args.events,
        columns=[
            "issue_id",
            "project_name",
            "issue_external_id",
            "event_scope",
            "source",
            "event_type",
            "lifecycle_stage",
            "actor_id",
            "actor_unknown",
            "timestamp",
        ],
    )
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)

    logging.info("Reading actor cluster labels: %s", args.labels)
    labels = pd.read_csv(args.labels)
    logging.info("Reading role taxonomy: %s", args.taxonomy)
    taxonomy = pd.read_csv(args.taxonomy)
    logging.info("Reading issue lifecycle metrics: %s", args.metrics)
    metrics = pd.read_csv(args.metrics)
    return events, labels, taxonomy, metrics


def role_lookup(labels: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    tax = taxonomy[["cluster_label", "candidate_role_label", "role_status", "is_primary_candidate"]].copy()
    tax["cluster_label"] = pd.to_numeric(tax["cluster_label"], errors="coerce").astype("Int64")
    lab = labels[["actor_id", "cluster_label", "is_noise"]].copy()
    lab["cluster_label"] = pd.to_numeric(lab["cluster_label"], errors="coerce").astype("Int64")
    out = lab.merge(tax, on="cluster_label", how="left")
    out["candidate_role_label"] = out["candidate_role_label"].fillna("No candidate role label")
    return out


def build_issue_role_vectors(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    taxonomy: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    issue_events = events.loc[events["event_scope"].eq("issue_linked")].copy()
    actor_roles = role_lookup(labels, taxonomy)
    known_actor_events = issue_events.loc[
        issue_events["actor_id"].notna() & ~issue_events["actor_unknown"].fillna(False).astype(bool)
    ].copy()
    known_actor_events = known_actor_events.merge(actor_roles, on="actor_id", how="left")
    known_actor_events["role_for_count"] = known_actor_events["candidate_role_label"].where(
        known_actor_events["cluster_label"].notna(),
        "Unclustered known actor",
    )
    known_actor_events.loc[known_actor_events["cluster_label"].eq(-1), "role_for_count"] = (
        "No primary role label"
    )

    distinct_issue_actor_role = known_actor_events[
        ["issue_id", "actor_id", "role_for_count"]
    ].drop_duplicates()
    role_counts = (
        distinct_issue_actor_role.groupby(["issue_id", "role_for_count"])["actor_id"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )

    base = metrics.copy()
    keep_metric_cols = [
        "project_name",
        "issue_id",
        "issue_external_id",
        "time_to_close_days",
        "n_events",
        "n_comments",
        "n_issue_comments",
        "n_commits",
        "n_prs",
        "n_reviews",
        "n_review_comments",
        "n_actors",
        "n_non_noise_actors",
        "n_noise_actors",
        "patch_size",
        "files_changed",
        "lines_added",
        "lines_deleted",
        "role_set_without_noise",
        "role_set_with_noise",
        "has_role_set_without_noise",
    ]
    base = base[[col for col in keep_metric_cols if col in base.columns]].copy()
    out = base.merge(role_counts, on="issue_id", how="left")

    for role_name, col in ROLE_COLUMNS.items():
        if role_name not in out.columns:
            out[role_name] = 0
        out[col] = out[role_name].fillna(0).astype(int)
        out = out.drop(columns=[role_name])

    for source_col, dest_col in [
        ("No primary role label", "n_noise_or_boundary_actor"),
        ("Unclustered known actor", "n_unclustered_known_actor"),
    ]:
        if source_col not in out.columns:
            out[source_col] = 0
        out[dest_col] = out[source_col].fillna(0).astype(int)
        out = out.drop(columns=[source_col])

    unknown_actor_events = (
        issue_events.loc[issue_events["actor_unknown"].fillna(False).astype(bool)]
        .groupby("issue_id")
        .size()
        .reset_index(name="n_actor_unknown_events_recomputed")
    )
    out = out.merge(unknown_actor_events, on="issue_id", how="left")
    out["n_actor_unknown_events_recomputed"] = (
        out["n_actor_unknown_events_recomputed"].fillna(0).astype(int)
    )

    out["n_stable_role_actors"] = out[CLUSTER_FEATURES].sum(axis=1)
    out["n_all_counted_known_actors"] = (
        out["n_stable_role_actors"]
        + out["n_noise_or_boundary_actor"]
        + out["n_unclustered_known_actor"]
    )
    out["issue_key"] = out["project_name"].astype(str) + ":" + out["issue_external_id"].astype(str)
    return out


def choose_k_and_cluster(vectors: pd.DataFrame, k_min: int, k_max: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    x = np.log1p(vectors[CLUSTER_FEATURES].astype(float).to_numpy())
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    max_possible_k = min(k_max, len(vectors) - 1, len(np.unique(x_scaled, axis=0)) - 1)
    if max_possible_k < k_min:
        max_possible_k = max(2, max_possible_k)
        k_min = 2

    rows = []
    fitted = {}
    for k in range(k_min, max_possible_k + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=50)
        labels = model.fit_predict(x_scaled)
        counts = pd.Series(labels).value_counts()
        if len(counts) < 2:
            continue
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(x_scaled, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(x_scaled, labels)),
                "davies_bouldin": float(davies_bouldin_score(x_scaled, labels)),
                "smallest_cluster_size": int(counts.min()),
                "largest_cluster_size": int(counts.max()),
                "largest_cluster_ratio": float(counts.max() / len(vectors)),
            }
        )
        fitted[k] = labels

    eval_df = pd.DataFrame(rows)
    if eval_df.empty:
        vectors = vectors.copy()
        vectors["role_composition_cluster"] = 0
        vectors["role_composition_k"] = 1
        return vectors, pd.DataFrame()

    eval_df["rank_silhouette"] = eval_df["silhouette"].rank(ascending=False, method="min")
    eval_df["rank_davies_bouldin"] = eval_df["davies_bouldin"].rank(ascending=True, method="min")
    eval_df["rank_largest_balance"] = eval_df["largest_cluster_ratio"].rank(ascending=True, method="min")
    eval_df["selection_score"] = (
        eval_df["rank_silhouette"] + eval_df["rank_davies_bouldin"] + eval_df["rank_largest_balance"]
    )
    selected_k = int(eval_df.sort_values(["selection_score", "k"]).iloc[0]["k"])
    vectors = vectors.copy()
    vectors["role_composition_cluster"] = fitted[selected_k]
    vectors["role_composition_k"] = selected_k
    eval_df["selected"] = eval_df["k"].eq(selected_k)
    return vectors, eval_df


def candidate_pattern_label(row: pd.Series) -> str:
    role_means = {col: row.get(f"mean_{col}", 0.0) for col in CLUSTER_FEATURES}
    total = sum(role_means.values())
    if row.get("zero_stable_role_ratio", 0.0) >= 0.50:
        return "No stable role / sparse PR-driven"
    if total < 0.25:
        return "No stable role observed"
    if role_means["n_issue_side_boundary_participant"] >= 0.50:
        return "Issue-side boundary / long-running"
    if role_means["n_review_focused_participant"] >= 0.50:
        return "Review-heavy coordination"
    if role_means["n_review_integration_hybrid"] >= 0.75 and role_means["n_pr_integrator"] >= 1.00:
        return "Review-integration intensive"
    if role_means["n_pr_integrator"] >= 0.50 and role_means["n_fixer"] < 0.20:
        return "PR-driven integration"
    if role_means["n_fixer"] >= 0.75 and role_means["n_pr_integrator"] >= 0.50:
        return "Code and PR integration"
    if role_means["n_issue_side_participant"] >= 0.50 and role_means["n_fixer"] >= 0.20:
        return "Issue discussion and fixing"
    top_col = max(role_means, key=role_means.get)
    active = {col for col, value in role_means.items() if value >= 0.20}
    has_fixer = "n_fixer" in active
    has_issue = "n_issue_side_participant" in active or "n_issue_side_boundary_participant" in active
    has_pr = "n_pr_integrator" in active or "n_review_integration_hybrid" in active
    has_review = "n_review_focused_participant" in active or "n_review_integration_hybrid" in active

    if has_fixer and not has_issue and not has_pr and not has_review:
        return "Fixer-centered"
    if has_issue and not has_fixer and not has_pr and not has_review:
        return "Issue-side dominated"
    if has_pr and has_fixer and not has_issue:
        return "Code and PR integration"
    if has_pr and not has_fixer:
        return "PR/review integration"
    if has_review and has_fixer:
        return "Fixing and review"
    if has_issue and has_fixer:
        return "Issue discussion and fixing"
    label_map = {
        "n_fixer": "Fixer-centered",
        "n_issue_side_participant": "Issue-side dominated",
        "n_pr_integrator": "PR-driven",
        "n_review_integration_hybrid": "Review-integration heavy",
        "n_review_focused_participant": "Review-focused",
        "n_issue_side_boundary_participant": "Issue-side boundary",
    }
    return label_map.get(top_col, "Mixed role composition")


def summarize_patterns(vectors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vectors = vectors.copy()
    vectors["has_zero_stable_role_actors"] = vectors["n_stable_role_actors"].eq(0)
    summary_aggs = {
        "issue_key": "count",
        "time_to_close_days": ["mean", "median"],
        "n_comments": ["mean", "median"],
        "n_issue_comments": ["mean", "median"],
        "n_commits": ["mean", "median"],
        "n_prs": ["mean", "median"],
        "n_reviews": ["mean", "median"],
        "patch_size": ["mean", "median"],
        "n_stable_role_actors": ["mean", "median"],
        "n_noise_or_boundary_actor": ["mean", "median"],
        "has_zero_stable_role_actors": "sum",
    }
    grouped = vectors.groupby("role_composition_cluster").agg(summary_aggs)
    grouped.columns = ["_".join([str(part) for part in col if part]) for col in grouped.columns]
    grouped = grouped.rename(columns={"issue_key_count": "n_issues"}).reset_index()

    role_means = (
        vectors.groupby("role_composition_cluster")[CLUSTER_FEATURES]
        .mean()
        .add_prefix("mean_")
        .reset_index()
    )
    role_medians = (
        vectors.groupby("role_composition_cluster")[CLUSTER_FEATURES]
        .median()
        .add_prefix("median_")
        .reset_index()
    )
    summary = grouped.merge(role_means, on="role_composition_cluster").merge(
        role_medians, on="role_composition_cluster"
    )
    summary["issue_ratio"] = summary["n_issues"] / len(vectors)
    summary["zero_stable_role_ratio"] = (
        summary["has_zero_stable_role_actors_sum"] / summary["n_issues"]
    )
    summary["candidate_pattern_label"] = summary.apply(candidate_pattern_label, axis=1)
    summary = summary.sort_values(["n_issues"], ascending=False).reset_index(drop=True)

    project_dist = (
        vectors.groupby(["role_composition_cluster", "project_name"])
        .size()
        .reset_index(name="n_issues")
    )
    project_total = project_dist.groupby("role_composition_cluster")["n_issues"].transform("sum")
    project_dist["project_ratio_within_pattern"] = project_dist["n_issues"] / project_total

    role_dist = summary[
        ["role_composition_cluster", "candidate_pattern_label", "n_issues"]
        + [f"mean_{col}" for col in CLUSTER_FEATURES]
        + [f"median_{col}" for col in CLUSTER_FEATURES]
    ].copy()
    return summary, project_dist, role_dist


def save_public_vectors(vectors: pd.DataFrame, output_dir: Path) -> None:
    drop_cols = ["issue_id"]
    public = vectors.drop(columns=[col for col in drop_cols if col in vectors.columns]).copy()
    public.to_csv(output_dir / "issue_role_composition_vectors.csv", index=False)
    public.to_parquet(output_dir / "issue_role_composition_vectors.parquet", index=False)


def write_summary(
    output_dir: Path,
    vectors: pd.DataFrame,
    k_eval: pd.DataFrame,
    summary: pd.DataFrame,
    project_dist: pd.DataFrame,
) -> None:
    selected_k = int(vectors["role_composition_k"].iloc[0])
    lines = [
        "# RQ3 Role Composition Pattern Taxonomy",
        "",
        "## Research Question",
        "",
        "How can closed bugs be categorized by the composition of behavioral roles involved, and what descriptive characteristics do the categories have?",
        "",
        "## Design",
        "",
        "- Unit of analysis: closed bug issue.",
        "- Vector: number of distinct actors per candidate behavioral role in issue-linked events.",
        "- Clustering: k-means on log1p-transformed role-count vectors.",
        "- Claim type: descriptive taxonomy, not causal effect estimation.",
        "- Noise/unclustered/unknown actors are retained for audit but are not primary role dimensions.",
        "",
        "## Dataset",
        "",
        f"- Issues analyzed: {len(vectors):,}",
        f"- Selected k: {selected_k}",
        f"- Issues with at least one stable non-noise role actor: {(vectors['n_stable_role_actors'] > 0).sum():,}",
        f"- Issues with no stable non-noise role actor: {(vectors['n_stable_role_actors'] == 0).sum():,}",
        "",
        "## k Selection",
        "",
    ]
    if not k_eval.empty:
        lines.extend(
            [
                "| k | silhouette | Davies-Bouldin | largest cluster ratio | selected |",
                "|---:|---:|---:|---:|:---:|",
            ]
        )
        for _, row in k_eval.iterrows():
            lines.append(
                f"| {int(row['k'])} | {row['silhouette']:.3f} | {row['davies_bouldin']:.3f} | {row['largest_cluster_ratio']:.3f} | {'yes' if row['selected'] else ''} |"
            )
    lines.extend(["", "## Pattern Summary", ""])
    lines.extend(
        [
            "| Pattern | Issues | Mean close days | Median close days | Mean comments | Mean commits | Zero stable-role issues |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['candidate_pattern_label']} (cluster {int(row['role_composition_cluster'])}) | "
            f"{int(row['n_issues']):,} | "
            f"{row['time_to_close_days_mean']:.1f} | "
            f"{row['time_to_close_days_median']:.1f} | "
            f"{row['n_comments_mean']:.1f} | "
            f"{row['n_commits_mean']:.2f} | "
            f"{row['zero_stable_role_ratio']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Pattern names are rule-based summaries assigned after clustering; they are not ground truth role labels.",
            "- The taxonomy describes observed role composition and issue characteristics. It does not estimate whether adding a role would improve an issue.",
            "- Large no-stable-role patterns should be interpreted as limitations of stable actor-role coverage, not as issues with no human activity.",
            "- Use the project distribution table to check whether a pattern is dominated by one project.",
            "",
            "## Outputs",
            "",
            "- `issue_role_composition_vectors.csv` / `.parquet`",
            "- `kmeans_k_selection.csv`",
            "- `bug_response_pattern_summary.csv`",
            "- `bug_response_pattern_role_composition.csv`",
            "- `bug_response_pattern_project_distribution.csv`",
            "- `bug_response_pattern_taxonomy_summary.md`",
        ]
    )
    (output_dir / "bug_response_pattern_taxonomy_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def make_figures(summary: pd.DataFrame, role_dist: pd.DataFrame, figure_dir: Path) -> None:
    cache_dir = figure_dir / "matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    plot_summary = summary.sort_values("n_issues", ascending=True).copy()
    labels = [
        f"{row.candidate_pattern_label}\n(n={int(row.n_issues)})"
        for row in plot_summary.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(labels, plot_summary["time_to_close_days_median"])
    ax.set_xlabel("Median time-to-close (days)")
    ax.set_ylabel("Bug response pattern")
    ax.set_title("Median close time by role-composition pattern")
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_rq3_pattern_close_time.png", dpi=200)
    plt.close(fig)

    heat = role_dist.set_index("candidate_pattern_label")[[f"mean_{col}" for col in CLUSTER_FEATURES]]
    heat = heat.rename(
        columns={
            "mean_n_fixer": "Commit-side\ncode contributor",
            "mean_n_issue_side_participant": "Issue-tracker\nparticipant",
            "mean_n_pr_integrator": "PR lifecycle\nparticipant",
            "mean_n_review_integration_hybrid": "Review-and-PR\nhybrid",
            "mean_n_review_focused_participant": "Review-side\nparticipant",
            "mean_n_issue_side_boundary_participant": "Boundary issue\ndiscussant",
        }
    )
    pattern_label_map = {
        "No stable role / sparse PR-driven": "Sparse-role quick\nPR-linked fixes",
        "Code and PR integration": "Commit-and-PR\nfixes",
        "Issue discussion and fixing": "Discussion-heavy\ndelayed fixes",
        "Review-integration intensive": "High-PR review-and-\nmerge fixes",
        "Review-heavy coordination": "Review-saturated\ncoordination",
        "Issue-side boundary / long-running": "Issue-boundary\nlong runners",
    }
    heat.index = [pattern_label_map.get(str(label), str(label)) for label in heat.index]

    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    image = ax.imshow(heat.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xlabel("Candidate behavioral label")
    ax.set_ylabel("Bug response pattern")
    ax.set_title("Mean actor counts by role-composition pattern")
    fig.colorbar(image, ax=ax, label="Mean actor count")
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_rq3_pattern_role_heatmap.png", dpi=200)
    plt.close(fig)


def integrity_check(output_dir: Path) -> pd.DataFrame:
    import re

    patterns = {
        "actor_id_raw": re.compile(r"actor_id_raw", re.IGNORECASE),
        "username": re.compile(r"username", re.IGNORECASE),
        "email_like": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "hex24": re.compile(r"\b[0-9a-fA-F]{24}\b"),
    }
    rows = []
    for path in sorted(output_dir.glob("*")):
        if path.suffix.lower() not in {".csv", ".md", ".json"}:
            continue
        if path.name == "artifact_integrity_checks.csv":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in patterns.items():
            count = len(pattern.findall(text))
            rows.append({"file": path.name, "check": name, "matches": count})
    return pd.DataFrame(rows)


def main() -> None:
    setup_logging()
    args = parse_args()
    ensure_dirs(args.output_dir, args.figure_dir)

    events, labels, taxonomy, metrics = read_inputs(args)
    vectors = build_issue_role_vectors(events, labels, taxonomy, metrics)
    vectors, k_eval = choose_k_and_cluster(vectors, args.k_min, args.k_max, args.seed)
    summary, project_dist, role_dist = summarize_patterns(vectors)

    save_public_vectors(vectors, args.output_dir)
    k_eval.to_csv(args.output_dir / "kmeans_k_selection.csv", index=False)
    summary.to_csv(args.output_dir / "bug_response_pattern_summary.csv", index=False)
    role_dist.to_csv(args.output_dir / "bug_response_pattern_role_composition.csv", index=False)
    project_dist.to_csv(args.output_dir / "bug_response_pattern_project_distribution.csv", index=False)
    write_summary(args.output_dir, vectors, k_eval, summary, project_dist)
    make_figures(summary, role_dist, args.figure_dir)

    metadata = {
        "n_issues": int(len(vectors)),
        "selected_k": int(vectors["role_composition_k"].iloc[0]),
        "cluster_features": CLUSTER_FEATURES,
        "claim_type": "descriptive taxonomy, not causal effect estimation",
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    integrity = integrity_check(args.output_dir)
    integrity.to_csv(args.output_dir / "artifact_integrity_checks.csv", index=False)
    logging.info("Completed RQ3 role composition taxonomy.")
    logging.info("Summary: %s", args.output_dir / "bug_response_pattern_taxonomy_summary.md")


if __name__ == "__main__":
    main()
