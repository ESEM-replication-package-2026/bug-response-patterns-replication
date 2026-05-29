from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartshark_roles.config import ensure_project_directories, load_config, project_path
from smartshark_roles.io import dataframe_to_markdown, utc_now_slug, write_dataframe, write_text
from smartshark_roles.logging_utils import setup_logging


FORBIDDEN_LABELS = ["Maintainer", "Leader", "Expert", "Manager", "Decision Maker"]


@dataclass(frozen=True)
class RoleDecision:
    candidate_role_label: str
    role_status: str
    evidence_features: list[str]
    adopted_reason: str
    alternative_labels: list[str]
    uncertainty: str
    why_not_stronger_labels: str
    validation_priority: str


ROLE_DECISIONS: dict[int, RoleDecision] = {
    -1: RoleDecision(
        candidate_role_label="No candidate role label",
        role_status="noise_not_interpreted_as_role",
        evidence_features=[
            "review_ratio",
            "pull_request_review_ratio",
            "project_level_ratio",
            "active_days",
            "n_projects",
        ],
        adopted_reason=(
            "HDBSCAN marked these actors as noise rather than a dense behavioral cluster, "
            "so they are retained for audit but not interpreted as a role."
        ),
        alternative_labels=["Noise / Outlier", "Unassigned actors"],
        uncertainty="Noise is structurally large and mixed; it should be analyzed separately, not named as a role.",
        why_not_stronger_labels=(
            "No dense cluster evidence supports assigning a behavior role, and the data do not reveal authority, "
            "intent, or seniority."
        ),
        validation_priority="audit_only",
    ),
    0: RoleDecision(
        candidate_role_label="Issue-side Boundary Participant",
        role_status="boundary_minor_candidate",
        evidence_features=[
            "n_projects",
            "discussion_ratio",
            "issue_comment_ratio",
            "unknown_ratio",
            "event_ratio",
            "reporting_ratio",
        ],
        adopted_reason=(
            "The cluster is issue-side and comment-heavy, but it is small and noise-prone in bootstrap runs; "
            "therefore it is treated as a boundary/minor candidate rather than a primary role."
        ),
        alternative_labels=["Issue-side Participant", "Discussion-oriented Boundary Participant"],
        uncertainty=(
            "Small actor count and high bootstrap noise reassignment indicate that this cluster may be a boundary "
            "region around broader issue-side behavior."
        ),
        why_not_stronger_labels=(
            "The observed features only show issue-side activity. They do not establish project authority, "
            "decision power, leadership, or expertise."
        ),
        validation_priority="high",
    ),
    1: RoleDecision(
        candidate_role_label="Fixer / Code Contributor",
        role_status="primary_candidate",
        evidence_features=[
            "commit_ratio",
            "fixing_ratio",
            "n_bugfix_commits_authored",
            "n_commits_linked_to_bug",
            "n_files_changed",
            "n_lines_added_deleted_log",
        ],
        adopted_reason=(
            "The cluster is dominated by commit/fixing events linked to closed bug issues, with code-change "
            "features elevated relative to other clusters."
        ),
        alternative_labels=["Bug-fix Code Contributor", "Commit-focused Participant"],
        uncertainty="The label describes observed fixing events only; it does not imply ownership of the fix.",
        why_not_stronger_labels=(
            "Commit activity does not prove maintainer status, seniority, or decision authority."
        ),
        validation_priority="medium",
    ),
    2: RoleDecision(
        candidate_role_label="Issue-side Participant",
        role_status="primary_candidate_broad",
        evidence_features=[
            "issue_comment_ratio",
            "discussion_ratio",
            "reporting_ratio",
            "issue_ratio",
            "unknown_ratio",
            "event_ratio",
            "n_issue_field_changes",
        ],
        adopted_reason=(
            "The cluster combines issue comments, issue reporting, issue field-change events, and unknown "
            "issue-side events. A narrow Discussion Participant label would overstate comment activity."
        ),
        alternative_labels=["Reporter/Commenter Hybrid", "Issue Workflow Participant", "General Issue-side Participant"],
        uncertainty=(
            "Cluster 2 is large and contains diagnostic subgroups. It is interpretable as a broad role, but later "
            "manual validation should check whether subgroups merit separate discussion."
        ),
        why_not_stronger_labels=(
            "The observed issue-side actions do not reveal whether actors have formal permissions, leadership, "
            "decision-making authority, or expertise."
        ),
        validation_priority="high",
    ),
    3: RoleDecision(
        candidate_role_label="Review-focused Participant",
        role_status="candidate_with_stability_uncertainty",
        evidence_features=[
            "pull_request_review_ratio",
            "pull_request_review_comment_ratio",
            "review_ratio",
            "pull_request_comment_ratio",
        ],
        adopted_reason=(
            "The cluster is dominated by review-related lifecycle and pull request comment/review sources, "
            "but bootstrap stability is only moderate."
        ),
        alternative_labels=["Review-side Participant", "PR Review Commenter"],
        uncertainty=(
            "Bootstrap label stability is lower than for the main fixing and integration clusters, so this label "
            "should be validated carefully."
        ),
        why_not_stronger_labels=(
            "Review events do not prove reviewer authority, expertise, gatekeeping power, or decision status."
        ),
        validation_priority="high",
    ),
    4: RoleDecision(
        candidate_role_label="Review-Integration Hybrid",
        role_status="primary_candidate",
        evidence_features=[
            "pull_request_ratio",
            "integration_ratio",
            "pull_request_comment_ratio",
            "review_ratio",
        ],
        adopted_reason=(
            "The cluster combines pull request integration events with review/comment activity, so a hybrid "
            "behavior label is more faithful than a single-action label."
        ),
        alternative_labels=["PR Review-Integration Participant", "Integration and Review Participant"],
        uncertainty="The hybrid label reflects co-occurring event types, not a formal workflow position.",
        why_not_stronger_labels=(
            "Combined PR and review activity does not establish authority to merge, lead, or decide."
        ),
        validation_priority="medium",
    ),
    5: RoleDecision(
        candidate_role_label="PR Integrator",
        role_status="primary_candidate",
        evidence_features=[
            "pull_request_ratio",
            "integration_ratio",
            "n_prs_opened",
            "n_prs_closed",
            "n_pr_events_total",
        ],
        adopted_reason=(
            "The cluster is dominated by pull request lifecycle events, especially opened/closed PR activity."
        ),
        alternative_labels=["Pull Request Lifecycle Participant", "PR-focused Participant"],
        uncertainty="The label describes PR lifecycle activity only; it does not imply merge permission.",
        why_not_stronger_labels=(
            "PR events alone do not prove maintainer status, merge rights, project leadership, or decision authority."
        ),
        validation_priority="medium",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create evidence-based candidate role labels and manual validation material.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--cluster-profiles", default="reports/cluster_profiles/cluster_profiles.csv")
    parser.add_argument("--cluster-top-features", default="reports/cluster_profiles/cluster_top_features.csv")
    parser.add_argument("--stability", default="reports/results/cluster_stability_by_label.csv")
    parser.add_argument("--cluster2-subgroups", default="reports/results/cluster2_subgroup_candidates.csv")
    parser.add_argument("--events", default="data/interim/events/events_normalized.parquet")
    parser.add_argument("--events-per-actor", type=int, default=8)
    parser.add_argument("--actors-per-cluster", type=int, default=5)
    return parser.parse_args()


def read_events(path: Path) -> pd.DataFrame:
    columns = [
        "actor_id",
        "actor_unknown",
        "project_name",
        "timestamp",
        "source",
        "event_type",
        "lifecycle_stage",
        "event_scope",
        "issue_external_id",
    ]
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def semicolon(values: list[str]) -> str:
    return "; ".join(values)


def as_int_label(value: Any) -> int:
    return int(value)


def representative_actor_ids(value: Any, max_count: int) -> list[str]:
    if pd.isna(value):
        return []
    actors = [actor.strip() for actor in str(value).split(";") if actor.strip()]
    return actors[:max_count]


def build_taxonomy(profiles: pd.DataFrame, top_features: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    stability = stability.rename(columns={"original_cluster_label": "cluster_label"}).copy()
    stability["cluster_label"] = stability["cluster_label"].astype(int)
    top_feature_map = (
        top_features[top_features["direction"] == "top"]
        .sort_values(["cluster_label", "rank"])
        .groupby("cluster_label")["feature"]
        .apply(lambda values: "; ".join(values.head(6)))
        .to_dict()
    )

    rows: list[dict[str, Any]] = []
    for _, profile in profiles.sort_values("cluster_label").iterrows():
        label = as_int_label(profile["cluster_label"])
        decision = ROLE_DECISIONS[label]
        stability_row = stability[stability["cluster_label"] == label]
        stability_values = stability_row.iloc[0].to_dict() if not stability_row.empty else {}
        evidence_summary = (
            f"dominant_lifecycle={profile['dominant_lifecycle_stage']} ({profile['dominant_lifecycle_ratio']:.3f}); "
            f"dominant_source={profile['dominant_source']} ({profile['dominant_source_ratio']:.3f}); "
            f"top_standardized_features={top_feature_map.get(label, '')}"
        )
        rows.append(
            {
                "cluster_label": label,
                "original_candidate_name": profile.get("candidate_role_name", ""),
                "candidate_role_label": decision.candidate_role_label,
                "role_status": decision.role_status,
                "n_actors": int(profile["n_actors"]),
                "n_events": int(profile["n_events"]),
                "dominant_project": profile["dominant_project"],
                "dominant_project_ratio": float(profile["dominant_project_ratio"]),
                "dominant_lifecycle_stage": profile["dominant_lifecycle_stage"],
                "dominant_lifecycle_ratio": float(profile["dominant_lifecycle_ratio"]),
                "dominant_source": profile["dominant_source"],
                "dominant_source_ratio": float(profile["dominant_source_ratio"]),
                "dominant_scope": profile["dominant_scope"],
                "dominant_scope_ratio": float(profile["dominant_scope_ratio"]),
                "stability_dominant_fraction_mean": stability_values.get("mean_dominant_fraction", pd.NA),
                "stability_noise_fraction_mean": stability_values.get("mean_bootstrap_noise_fraction", pd.NA),
                "evidence_features": semicolon(decision.evidence_features),
                "evidence_summary": evidence_summary,
                "alternative_labels": semicolon(decision.alternative_labels),
                "adopted_reason": decision.adopted_reason,
                "uncertainty": decision.uncertainty,
                "why_not_stronger_labels": decision.why_not_stronger_labels,
                "validation_priority": decision.validation_priority,
                "is_primary_candidate": decision.role_status.startswith("primary"),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_cards(taxonomy: pd.DataFrame, cluster2_subgroups: pd.DataFrame) -> str:
    lines = [
        "# Candidate Role Cards",
        "",
        "These cards use candidate_role_label throughout. The labels are behavioral interpretations grounded in observed features, event distributions, stability results, and representative timelines.",
        "",
        "For the paper, use wording such as: \"we interpret clusters as roles\" and state that role labels are evidence-based interpretations rather than ground-truth titles.",
        "",
        "## Labeling Protocol",
        "",
        "1. Use only observable behavior: feature values, lifecycle/source/scope distributions, stability statistics, and anonymized timelines.",
        "2. Attach evidence_features to every candidate role label.",
        "3. Do not use labels that imply unobserved authority, intent, seniority, or expertise.",
        f"4. Forbidden labels: {', '.join(FORBIDDEN_LABELS)}.",
        "5. Treat HDBSCAN noise (-1) as unassigned/no-role, not as a candidate role.",
        "6. Treat cluster 0 as a boundary/minor candidate and cluster 3 as a candidate with explicit stability uncertainty.",
        "",
    ]

    for _, row in taxonomy.sort_values("cluster_label").iterrows():
        label = int(row["cluster_label"])
        title = f"Cluster {label}: {row['candidate_role_label']}"
        lines.extend(
            [
                f"## {title}",
                "",
                f"- Role status: {row['role_status']}",
                f"- Actors: {row['n_actors']}",
                f"- Events: {row['n_events']}",
                f"- Evidence features: {row['evidence_features']}",
                f"- Dominant lifecycle/source/scope: {row['dominant_lifecycle_stage']} / {row['dominant_source']} / {row['dominant_scope']}",
                f"- Stability: dominant_fraction_mean={row['stability_dominant_fraction_mean']:.3f}, bootstrap_noise_fraction_mean={row['stability_noise_fraction_mean']:.3f}",
                f"- Interpretation: {row['adopted_reason']}",
                f"- Alternative labels: {row['alternative_labels']}",
                f"- Uncertainty: {row['uncertainty']}",
                f"- Why not stronger labels: {row['why_not_stronger_labels']}",
                "",
            ]
        )
        if label == 2 and not cluster2_subgroups.empty:
            subgroup_counts = cluster2_subgroups["subgroup_candidate"].value_counts().reset_index()
            subgroup_counts.columns = ["subgroup_candidate", "n_actors"]
            lines.extend(
                [
                    "Cluster 2 diagnostic subgroups are retained as diagnostic evidence only and do not overwrite the main HDBSCAN labels:",
                    "",
                    dataframe_to_markdown(subgroup_counts),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def build_decision_log(taxonomy: pd.DataFrame) -> str:
    decision_columns = [
        "cluster_label",
        "original_candidate_name",
        "candidate_role_label",
        "adopted_reason",
        "evidence_features",
        "alternative_labels",
        "uncertainty",
        "why_not_stronger_labels",
    ]
    lines = [
        "# Role Naming Decisions",
        "",
        "This document records the candidate label protocol. It avoids labels that imply authority, intent, seniority, or expertise not observable in SmartSHARK events.",
        "",
        f"Forbidden stronger labels: {', '.join(FORBIDDEN_LABELS)}.",
        "",
        dataframe_to_markdown(taxonomy[decision_columns].sort_values("cluster_label")),
        "",
        "## Notes",
        "",
        "- Cluster 2 is labeled Issue-side Participant because issue comments, reporting, issue field-change events, and unknown issue-side events are mixed.",
        "- Cluster 0 is a boundary/minor candidate rather than a primary role because it is small and bootstrap noise-prone.",
        "- Cluster 3 is review-focused but carries explicit stability uncertainty.",
        "- Noise cluster -1 is not assigned a role label.",
        "- In the paper, describe this as interpreting clusters as roles based on observed behavioral evidence.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def actor_subgroup_lookup(cluster2_subgroups: pd.DataFrame) -> dict[str, str]:
    if cluster2_subgroups.empty or "actor_id" not in cluster2_subgroups.columns:
        return {}
    return dict(zip(cluster2_subgroups["actor_id"], cluster2_subgroups["subgroup_candidate"]))


def selected_actors_for_validation(
    profiles: pd.DataFrame,
    cluster2_subgroups: pd.DataFrame,
    actors_per_cluster: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for _, profile in profiles.sort_values("cluster_label").iterrows():
        label = int(profile["cluster_label"])
        if label == 2 and not cluster2_subgroups.empty:
            rank = 0
            for subgroup, subset in cluster2_subgroups.sort_values("subgroup_score", ascending=False).groupby("subgroup_candidate"):
                for actor_id in subset["actor_id"].head(3):
                    rank += 1
                    selected.append(
                        {
                            "cluster_label": label,
                            "actor_id": actor_id,
                            "actor_sample_rank": rank,
                            "subgroup_candidate": subgroup,
                        }
                    )
            continue

        for rank, actor_id in enumerate(representative_actor_ids(profile.get("representative_actor_ids"), actors_per_cluster), start=1):
            selected.append(
                {
                    "cluster_label": label,
                    "actor_id": actor_id,
                    "actor_sample_rank": rank,
                    "subgroup_candidate": "",
                }
            )
    return selected


def build_manual_validation_sample(
    profiles: pd.DataFrame,
    taxonomy: pd.DataFrame,
    cluster2_subgroups: pd.DataFrame,
    events: pd.DataFrame,
    events_per_actor: int,
    actors_per_cluster: int,
) -> pd.DataFrame:
    selected = selected_actors_for_validation(profiles, cluster2_subgroups, actors_per_cluster)
    selected_frame = pd.DataFrame(selected)
    if selected_frame.empty:
        return selected_frame

    taxonomy_for_join = taxonomy[
        [
            "cluster_label",
            "candidate_role_label",
            "role_status",
            "evidence_features",
            "validation_priority",
        ]
    ]
    selected_frame = selected_frame.merge(taxonomy_for_join, on="cluster_label", how="left")
    events = events.loc[~events["actor_unknown"].fillna(True).astype(bool)].copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)
    events = events.merge(selected_frame, on="actor_id", how="inner")
    events = events.sort_values(["cluster_label", "actor_sample_rank", "timestamp"], na_position="last")
    events["timeline_event_rank"] = events.groupby("actor_id").cumcount() + 1
    events = events[events["timeline_event_rank"] <= events_per_actor].copy()
    events["validation_question"] = (
        "Does this anonymized timeline support the candidate_role_label using only the observed events?"
    )
    events["human_label_fit"] = ""
    events["human_notes"] = ""

    columns = [
        "cluster_label",
        "candidate_role_label",
        "role_status",
        "validation_priority",
        "evidence_features",
        "subgroup_candidate",
        "actor_id",
        "actor_sample_rank",
        "timeline_event_rank",
        "project_name",
        "timestamp",
        "source",
        "event_type",
        "lifecycle_stage",
        "event_scope",
        "issue_external_id",
        "validation_question",
        "human_label_fit",
        "human_notes",
    ]
    return events[columns]


def build_manual_validation_guide(taxonomy: pd.DataFrame, sample: pd.DataFrame) -> str:
    summary = (
        sample.groupby(["cluster_label", "candidate_role_label"])["actor_id"].nunique().reset_index(name="sampled_actors")
        if not sample.empty
        else pd.DataFrame(columns=["cluster_label", "candidate_role_label", "sampled_actors"])
    )
    lines = [
        "# Manual Validation Guide",
        "",
        "Purpose: check whether each candidate_role_label is consistent with anonymized actor timelines and evidence features.",
        "",
        "## Validation Rules",
        "",
        "1. Judge only observable behavior in the sample rows: source, event_type, lifecycle_stage, event_scope, project_name, and issue_external_id.",
        "2. Do not infer authority, intent, seniority, or expertise.",
        f"3. Do not use forbidden labels: {', '.join(FORBIDDEN_LABELS)}.",
        "4. Mark human_label_fit as one of: yes, partial, no, unclear.",
        "5. Use human_notes to record concise evidence or contradictions.",
        "6. Treat noise cluster -1 as no-role audit material.",
        "7. Treat cluster 0 as boundary/minor and cluster 3 as uncertain unless manual evidence strongly supports the candidate label.",
        "",
        "## Sample Coverage",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Candidate Labels Under Review",
        "",
        dataframe_to_markdown(
            taxonomy[
                [
                    "cluster_label",
                    "candidate_role_label",
                    "role_status",
                    "evidence_features",
                    "uncertainty",
                ]
            ].sort_values("cluster_label")
        ),
        "",
        "The replication package should include this guide and the validation sample, but not actor_id_mapping.csv or raw actor identifiers.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def assert_no_forbidden_labels(frame: pd.DataFrame) -> None:
    labels = " ".join(str(value) for value in frame.get("candidate_role_label", pd.Series(dtype=str)).dropna())
    for forbidden in FORBIDDEN_LABELS:
        if forbidden.lower() in labels.lower():
            raise ValueError(f"Forbidden label appears in candidate_role_label: {forbidden}")


def run(args: argparse.Namespace, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    profiles = pd.read_csv(project_path(args.cluster_profiles))
    top_features = pd.read_csv(project_path(args.cluster_top_features))
    stability = pd.read_csv(project_path(args.stability))
    cluster2_path = project_path(args.cluster2_subgroups)
    cluster2_subgroups = pd.read_csv(cluster2_path) if cluster2_path.exists() else pd.DataFrame()
    events = read_events(project_path(args.events))

    taxonomy = build_taxonomy(profiles, top_features, stability)
    assert_no_forbidden_labels(taxonomy)
    sample = build_manual_validation_sample(
        profiles=profiles,
        taxonomy=taxonomy,
        cluster2_subgroups=cluster2_subgroups,
        events=events,
        events_per_actor=args.events_per_actor,
        actors_per_cluster=args.actors_per_cluster,
    )

    cluster_dir = ROOT / "reports" / "cluster_profiles"
    validation_dir = ROOT / "reports" / "manual_validation"
    paths = [
        cluster_dir / "candidate_role_taxonomy.csv",
        cluster_dir / "candidate_role_cards.md",
        cluster_dir / "role_naming_decisions.md",
        validation_dir / "manual_validation_sample.csv",
        validation_dir / "manual_validation_guide.md",
    ]

    write_dataframe(taxonomy.sort_values("cluster_label"), paths[0], logger=logger)
    write_text(build_candidate_cards(taxonomy, cluster2_subgroups), paths[1], logger=logger)
    write_text(build_decision_log(taxonomy), paths[2], logger=logger)
    write_dataframe(sample.sort_values(["cluster_label", "actor_sample_rank", "timeline_event_rank"]), paths[3], logger=logger)
    write_text(build_manual_validation_guide(taxonomy, sample), paths[4], logger=logger)
    return taxonomy, sample, paths


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    logger = setup_logging(ROOT / "reports" / "cluster_profiles" / "logs" / f"candidate_role_protocol_{utc_now_slug()}.log", level=logging.INFO)

    try:
        taxonomy, sample, paths = run(args, logger)
    except Exception:
        logger.exception("Candidate role protocol failed")
        return 1

    print("Candidate role protocol succeeded")
    print(f"- candidate_rows: {len(taxonomy)}")
    print(f"- manual_validation_rows: {len(sample)}")
    print("Candidate labels:")
    for row in taxonomy.sort_values("cluster_label").itertuples():
        print(f"- cluster {row.cluster_label}: {row.candidate_role_label} ({row.role_status})")
    print("Generated:")
    for path in paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
