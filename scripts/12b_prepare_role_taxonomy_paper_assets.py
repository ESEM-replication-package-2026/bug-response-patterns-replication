"""Prepare paper-facing role taxonomy labels for the ESEM submission.

The original role-naming protocol preserved earlier internal labels such as
``Fixer / Code Contributor`` and ``PR Integrator``. The submitted paper uses
clearer behavior-first labels. This script maps the cluster ids to the paper
labels and writes compact public tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports" / "cluster_profiles" / "candidate_role_taxonomy.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "cluster_profiles"
DEFAULT_PAPER_TABLES = ROOT / "reports" / "paper_tables"


PAPER_LABELS = {
    -1: {
        "candidate_behavioral_label": "No primary behavioral label",
        "dominant_signal": "HDBSCAN noise/boundary actors",
        "how_to_read_it": "Not interpreted as a role; these actors are not forced into a primary label.",
        "paper_role_category": "noise_or_boundary",
    },
    0: {
        "candidate_behavioral_label": "Boundary issue discussant",
        "dominant_signal": "Issue comments and discussion",
        "how_to_read_it": "Small boundary cluster; issue-side discussion is visible, but the label should be treated cautiously.",
        "paper_role_category": "boundary_candidate",
    },
    1: {
        "candidate_behavioral_label": "Commit-side code contributor",
        "dominant_signal": "Fixing lifecycle and commit source",
        "how_to_read_it": "Actors whose observable activity is dominated by bug-linked code changes.",
        "paper_role_category": "primary_candidate",
    },
    2: {
        "candidate_behavioral_label": "Issue-tracker participant",
        "dominant_signal": "Reporting, discussion, and issue-field activity",
        "how_to_read_it": "Broad issue-tracker activity, including reporting, commenting, and issue metadata changes.",
        "paper_role_category": "primary_candidate",
    },
    3: {
        "candidate_behavioral_label": "Review-side participant",
        "dominant_signal": "Review and review-comment sources",
        "how_to_read_it": "Review-related behavior is visible, but this candidate has explicit stability uncertainty.",
        "paper_role_category": "candidate_with_uncertainty",
    },
    4: {
        "candidate_behavioral_label": "Review-and-PR hybrid",
        "dominant_signal": "Review/comment activity plus PR integration",
        "how_to_read_it": "Actors whose activity combines review-side behavior with PR integration features.",
        "paper_role_category": "primary_candidate",
    },
    5: {
        "candidate_behavioral_label": "PR lifecycle participant",
        "dominant_signal": "PR opened, closed, and merged events",
        "how_to_read_it": "Actors whose observable activity is concentrated in pull-request lifecycle events.",
        "paper_role_category": "primary_candidate",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-tables-dir", type=Path, default=DEFAULT_PAPER_TABLES)
    return parser.parse_args()


def build_table(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_label, labels in PAPER_LABELS.items():
        row = source.loc[source["cluster_label"].astype(int).eq(cluster_label)]
        if row.empty:
            raise ValueError(f"Missing source taxonomy row for cluster {cluster_label}")
        src = row.iloc[0]
        rows.append(
            {
                "cluster_label": cluster_label,
                "candidate_behavioral_label": labels["candidate_behavioral_label"],
                "n_actors": int(src["n_actors"]),
                "paper_role_category": labels["paper_role_category"],
                "dominant_signal": labels["dominant_signal"],
                "how_to_read_it": labels["how_to_read_it"],
                "dominant_lifecycle_stage": src.get("dominant_lifecycle_stage", ""),
                "dominant_source": src.get("dominant_source", ""),
                "stability_note": (
                    f"dominant_fraction={float(src.get('stability_dominant_fraction_mean', 0)):.3f}; "
                    f"bootstrap_noise_fraction={float(src.get('stability_noise_fraction_mean', 0)):.3f}"
                ),
                "naming_caution": (
                    "Labels describe observable event patterns, not official project roles, "
                    "authority, expertise, seniority, or intent."
                ),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Paper-facing Candidate Behavioral Role Taxonomy",
        "",
        "These are the canonical role labels used in the ESEM submission. They are "
        "behavioral interpretations of actor clusters and are not official project roles.",
        "",
        "| Cluster | Candidate behavioral label | No. of actors | Dominant signal | How to read it |",
        "|---:|---|---:|---|---|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.cluster_label} | {row.candidate_behavioral_label} | "
            f"{row.n_actors:,} | {row.dominant_signal} | {row.how_to_read_it} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input)
    table = build_table(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "candidate_role_taxonomy_paper_labels.csv", index=False)
    table.to_csv(args.paper_tables_dir / "table_candidate_role_taxonomy_paper_labels.csv", index=False)
    write_markdown(table, args.output_dir / "candidate_role_taxonomy_paper_labels.md")


if __name__ == "__main__":
    main()
