"""Prepare paper-ready assets for the k=9 RQ3 role-composition taxonomy.

This script reads the separate k=9 RQ3 outputs and creates display labels,
compact tables, and a heatmap for the paper package. It does not modify the
original k=6 RQ3 outputs.

The paper-facing labels in this script are the canonical labels for the ESEM
submission. The underlying clustering output may contain older internal labels;
those are mapped here to clearer descriptive names.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports" / "results" / "rq3_role_composition_taxonomy_k9"
DEFAULT_FIGURES = ROOT / "reports" / "figures" / "rq3_role_composition_taxonomy_k9"
DEFAULT_PAPER_TABLES = ROOT / "reports" / "paper_tables"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_FIGURES / "matplotlib_cache"))

ROLE_COLUMNS = [
    "mean_n_fixer",
    "mean_n_issue_side_participant",
    "mean_n_pr_integrator",
    "mean_n_review_integration_hybrid",
    "mean_n_review_focused_participant",
    "mean_n_issue_side_boundary_participant",
]

ROLE_DISPLAY = {
    "mean_n_fixer": "Commit-side\ncode contributor",
    "mean_n_issue_side_participant": "Issue-tracker\nparticipant",
    "mean_n_pr_integrator": "PR lifecycle\nparticipant",
    "mean_n_review_integration_hybrid": "Review-and-PR\nhybrid",
    "mean_n_review_focused_participant": "Review-side\nparticipant",
    "mean_n_issue_side_boundary_participant": "Boundary issue\ndiscussant",
}

PATTERN_DISPLAY_BY_CLUSTER = {
    1: "Sparse stable-role quick fixes",
    3: "Commit-side quick fixes",
    7: "Issue-tracker delayed fixes",
    8: "PR-lifecycle quick fixes",
    5: "PR-and-review merge fixes",
    0: "Discussion-and-code delayed fixes",
    6: "Code-and-heavy-PR fixes",
    2: "Review-saturated coordination",
    4: "Issue-boundary long runners",
}

PATTERN_HEATMAP_DISPLAY_BY_CLUSTER = {
    1: "Sparse stable-role\nquick fixes",
    3: "Commit-side\nquick fixes",
    7: "Issue-tracker\ndelayed fixes",
    8: "PR-lifecycle\nquick fixes",
    5: "PR-and-review\nmerge fixes",
    0: "Discussion-and-code\ndelayed fixes",
    6: "Code-and-heavy-PR\nfixes",
    2: "Review-saturated\ncoordination",
    4: "Issue-boundary\nlong runners",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--paper-tables-dir", type=Path, default=DEFAULT_PAPER_TABLES)
    return parser.parse_args()


def load_summary(input_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(input_dir / "bug_response_pattern_summary.csv")
    summary["paper_pattern_label"] = summary["role_composition_cluster"].map(
        PATTERN_DISPLAY_BY_CLUSTER
    )
    summary["heatmap_pattern_label"] = summary["role_composition_cluster"].map(
        PATTERN_HEATMAP_DISPLAY_BY_CLUSTER
    )
    missing = summary.loc[summary["paper_pattern_label"].isna(), "role_composition_cluster"].tolist()
    if missing:
        raise ValueError(f"Missing display labels for clusters: {missing}")
    return summary


def write_tables(summary: pd.DataFrame, paper_tables_dir: Path, input_dir: Path) -> None:
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    out = summary[
        [
            "role_composition_cluster",
            "paper_pattern_label",
            "n_issues",
            "time_to_close_days_median",
            "n_comments_mean",
            "n_issue_comments_mean",
            "n_commits_mean",
            "n_prs_mean",
            "n_reviews_mean",
            "patch_size_median",
            "zero_stable_role_ratio",
            "mean_n_fixer",
            "mean_n_issue_side_participant",
            "mean_n_pr_integrator",
            "mean_n_review_integration_hybrid",
            "mean_n_review_focused_participant",
            "mean_n_issue_side_boundary_participant",
        ]
    ].copy()
    # n_comments includes issue comments, pull-request comments, and
    # pull-request review comments that are linked to the closed bug issue.
    out["mean_pr_or_review_comments"] = out["n_comments_mean"] - out["n_issue_comments_mean"]
    out = out.rename(
        columns={
            "role_composition_cluster": "cluster",
            "paper_pattern_label": "pattern",
            "n_issues": "issues",
            "time_to_close_days_median": "median_close_days",
            "n_comments_mean": "mean_linked_comments",
            "n_issue_comments_mean": "mean_issue_comments",
            "n_commits_mean": "mean_commits",
            "n_prs_mean": "mean_prs",
            "n_reviews_mean": "mean_reviews",
            "patch_size_median": "median_patch_size",
            "zero_stable_role_ratio": "zero_stable_role_ratio",
            "mean_n_fixer": "mean_commit_side_code_contributor",
            "mean_n_issue_side_participant": "mean_issue_tracker_participant",
            "mean_n_pr_integrator": "mean_pr_lifecycle_participant",
            "mean_n_review_integration_hybrid": "mean_review_and_pr_hybrid",
            "mean_n_review_focused_participant": "mean_review_side_participant",
            "mean_n_issue_side_boundary_participant": "mean_boundary_issue_discussant",
        }
    )
    ordered_columns = [
        "cluster",
        "pattern",
        "issues",
        "median_close_days",
        "mean_linked_comments",
        "mean_issue_comments",
        "mean_pr_or_review_comments",
        "mean_commits",
        "mean_prs",
        "mean_reviews",
        "median_patch_size",
        "zero_stable_role_ratio",
        "mean_commit_side_code_contributor",
        "mean_issue_tracker_participant",
        "mean_pr_lifecycle_participant",
        "mean_review_and_pr_hybrid",
        "mean_review_side_participant",
        "mean_boundary_issue_discussant",
    ]
    out = out[ordered_columns]
    out.to_csv(paper_tables_dir / "table_rq3_k9_bug_response_patterns.csv", index=False)
    out.to_csv(input_dir / "bug_response_pattern_summary_k9_paper_labels.csv", index=False)
    write_pattern_summary_md(out, input_dir / "bug_response_pattern_summary_k9_paper_labels.md")


def write_pattern_summary_md(out: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Paper-facing k=9 Bug Response Pattern Summary",
        "",
        "These are the canonical pattern labels used for the ESEM submission. "
        "They are post-clustering descriptive labels, not ground-truth issue types "
        "or recommended assignments.",
        "",
        "`mean_linked_comments` includes issue comments, pull-request comments, "
        "and pull-request review comments linked to the closed bug issue. "
        "Review submissions are counted separately as `mean_reviews`.",
        "",
        "| Pattern | Issues | Median close days | Mean linked comments | Mean commits | Mean PRs | Mean reviews |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in out.itertuples(index=False):
        lines.append(
            f"| {row.pattern} | {int(row.issues):,} | {row.median_close_days:.1f} | "
            f"{row.mean_linked_comments:.1f} | {row.mean_commits:.2f} | "
            f"{row.mean_prs:.2f} | {row.mean_reviews:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Mean role counts per issue",
            "",
            "| Pattern | Commit-side code contributor | Issue-tracker participant | PR lifecycle participant | Review-and-PR hybrid | Review-side participant | Boundary issue discussant |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in out.itertuples(index=False):
        lines.append(
            f"| {row.pattern} | {row.mean_commit_side_code_contributor:.2f} | "
            f"{row.mean_issue_tracker_participant:.2f} | "
            f"{row.mean_pr_lifecycle_participant:.2f} | "
            f"{row.mean_review_and_pr_hybrid:.2f} | "
            f"{row.mean_review_side_participant:.2f} | "
            f"{row.mean_boundary_issue_discussant:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_heatmap(summary: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    heat = summary.set_index("heatmap_pattern_label")[ROLE_COLUMNS].rename(columns=ROLE_DISPLAY)

    fig, ax = plt.subplots(figsize=(9.8, 6.3))
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
    fig.savefig(figure_dir / "figure3_role_composition_heatmap_k9.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = load_summary(args.input_dir)
    write_tables(summary, args.paper_tables_dir, args.input_dir)
    write_heatmap(summary, args.figure_dir)


if __name__ == "__main__":
    main()
