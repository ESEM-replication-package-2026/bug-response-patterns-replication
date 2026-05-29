# Missing Value Policy

This project keeps raw SmartSHARK-derived events whenever a record is linkable to the selected project and lifecycle source, even when selected analytical fields are missing. Missing values are recorded explicitly and summarized in audit reports before actor normalization or feature construction.

## Event Extraction

`timestamp` and `raw_actor_identifier_field` may be missing for some events. These rows are retained in the events table and recorded in `reports/build_events/event_extraction_warnings.csv`.

In the smoke extraction, missing `timestamp` and missing `raw_actor_identifier_field` are concentrated in `pull_request_review_comment` rows. This matches the DB audit: `pull_request_review_comment.created_at` and `pull_request_review_comment.creator_id` are only partially covered. The rows are retained because review comments still indicate review activity and can contribute to source-level activity counts, but downstream temporal analyses and actor-level features must either exclude these rows from time-dependent calculations or use explicit missingness indicators.

## Lifecycle Stage Unknown

`lifecycle_stage=unknown` is allowed. In the smoke extraction, these rows are `event.issue_field_changed` records from heterogeneous JIRA fields such as `RemoteIssueLink`, `issue_links`, `fix_versions`, labels, sprint, summary, and story points.

These field changes are not mapped wholesale to `triage` or `workflow` because they mix unrelated semantics. For example, remote links may indicate external references or pull request links, while fix version changes are release-planning metadata. They remain `unknown` until a narrower, validated mapping is introduced.

## Downstream Handling

- Actor normalization must keep an explicit `actor_id_missing` indicator.
- Feature construction must report whether missing-actor events are excluded, counted only as aggregate activity, or represented with a sentinel actor category.
- Temporal features must report whether missing-timestamp events are excluded from time-window and ordering features.
- Clustering inputs must include a machine-readable preprocessing report documenting missing-value decisions.
