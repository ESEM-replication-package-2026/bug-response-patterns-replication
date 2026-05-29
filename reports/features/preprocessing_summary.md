# Feature Preprocessing Summary

- Generated at UTC: `2026-05-06T10:38:30.845997+00:00`
- Input features: `<workspace>\data\interim\features\actor_features_raw.parquet`
- DB access: not used in this step.
- Privacy note: model input contains no raw actor IDs, emails, account name fields, or names. Actor IDs and filter flags are stored separately in metadata.

## Overall

- Raw actors: `5973`
- Bot excluded: `1`
- Low activity excluded: `4072`
- Bot and low activity overlap: `1`
- Main clustering actors: `1901`
- Input feature count: `51`
- Excluded feature count: `8`
- Final feature count: `51`
- NaN after preprocessing: `0`

## Missing Value Policy

- Count and ratio features are zero-imputed before transformation.
- `median_comment_length` is zero-imputed and paired with `median_comment_length_missing`.
- `median_inter_event_hours` is excluded from the main model input because temporal imputation can distort distances.
- Count features are `log1p` transformed, then all model features are winsorized at p99 and standardized.

## Excluded Features

| feature | reason | missing_rate | feature_kind |
| --- | --- | --- | --- |
| median_inter_event_hours | excluded_temporal_imputation_risk | 0.22517997656119204 | median |
| actor_id | metadata_separated | 0.0 | metadata |
| first_event_at | metadata_separated | 0.0 | metadata |
| first_lifecycle_stage | metadata_separated | 0.0 | metadata |
| is_bot | metadata_separated | 0.0 | metadata |
| is_low_activity | metadata_separated | 0.0 | metadata |
| last_event_at | metadata_separated | 0.0 | metadata |
| last_lifecycle_stage | metadata_separated | 0.0 | metadata |

## Imputation Log

| feature | feature_kind | missing_count_before | missing_rate_before | action | missing_flag_added | reason |
| --- | --- | --- | --- | --- | --- | --- |
| total_events | count | 0 | 0.0 | none |  |  |
| n_projects | count | 0 | 0.0 | none |  |  |
| n_unique_issues | count | 0 | 0.0 | none |  |  |
| active_days | count | 0 | 0.0 | none |  |  |
| n_issue_linked_events | count | 0 | 0.0 | none |  |  |
| n_project_level_events | count | 0 | 0.0 | none |  |  |
| issue_linked_ratio | ratio | 0 | 0.0 | none |  |  |
| project_level_ratio | ratio | 0 | 0.0 | none |  |  |
| n_issues_reported | count | 0 | 0.0 | none |  |  |
| n_issue_comments | count | 0 | 0.0 | none |  |  |
| n_unique_issues_commented | count | 0 | 0.0 | none |  |  |
| median_comment_length | median | 909 | 0.47816938453445557 | missing_flag_plus_zero_imputation | median_comment_length_missing | comment_absence_signal |
| n_status_changes | count | 0 | 0.0 | none |  |  |
| n_priority_changes | count | 0 | 0.0 | none |  |  |
| n_resolution_changes | count | 0 | 0.0 | none |  |  |
| n_assignee_changes | count | 0 | 0.0 | none |  |  |
| n_issue_field_changes | count | 0 | 0.0 | none |  |  |
| n_unknown_issue_field_changes | count | 0 | 0.0 | none |  |  |
| n_bugfix_commits_authored | count | 0 | 0.0 | none |  |  |
| n_commits_linked_to_bug | count | 0 | 0.0 | none |  |  |
| n_files_changed | count | 0 | 0.0 | none |  |  |
| n_lines_added | count | 0 | 0.0 | none |  |  |
| n_lines_deleted | count | 0 | 0.0 | none |  |  |
| n_lines_added_deleted_log | count | 0 | 0.0 | none |  |  |
| n_prs_opened | count | 0 | 0.0 | none |  |  |
| n_prs_closed | count | 0 | 0.0 | none |  |  |
| n_prs_merged | count | 0 | 0.0 | none |  |  |
| n_pr_events_total | count | 0 | 0.0 | none |  |  |
| n_reviews_submitted | count | 0 | 0.0 | none |  |  |
| n_review_comments | count | 0 | 0.0 | none |  |  |
| n_pr_review_comments | count | 0 | 0.0 | none |  |  |
| n_issues_closed | count | 0 | 0.0 | none |  |  |
| n_issues_resolved | count | 0 | 0.0 | none |  |  |
| n_final_status_changes | count | 0 | 0.0 | none |  |  |
| reporting_ratio | ratio | 0 | 0.0 | none |  |  |
| discussion_ratio | ratio | 0 | 0.0 | none |  |  |
| triage_ratio | ratio | 0 | 0.0 | none |  |  |
| fixing_ratio | ratio | 0 | 0.0 | none |  |  |
| integration_ratio | ratio | 0 | 0.0 | none |  |  |
| review_ratio | ratio | 0 | 0.0 | none |  |  |
| closure_ratio | ratio | 0 | 0.0 | none |  |  |
| unknown_ratio | ratio | 0 | 0.0 | none |  |  |
| issue_ratio | ratio | 0 | 0.0 | none |  |  |
| issue_comment_ratio | ratio | 0 | 0.0 | none |  |  |
| event_ratio | ratio | 0 | 0.0 | none |  |  |
| commit_ratio | ratio | 0 | 0.0 | none |  |  |
| pull_request_ratio | ratio | 0 | 0.0 | none |  |  |
| pull_request_comment_ratio | ratio | 0 | 0.0 | none |  |  |
| pull_request_review_ratio | ratio | 0 | 0.0 | none |  |  |
| pull_request_review_comment_ratio | ratio | 0 | 0.0 | none |  |  |
| median_comment_length_missing | median | 0 | 0.0 | none |  |  |

## Missingness After Preprocessing

| feature | missing_count | missing_rate |
| --- | --- | --- |
| active_days | 0 | 0.0 |
| closure_ratio | 0 | 0.0 |
| commit_ratio | 0 | 0.0 |
| discussion_ratio | 0 | 0.0 |
| event_ratio | 0 | 0.0 |
| fixing_ratio | 0 | 0.0 |
| integration_ratio | 0 | 0.0 |
| issue_comment_ratio | 0 | 0.0 |
| issue_linked_ratio | 0 | 0.0 |
| issue_ratio | 0 | 0.0 |
| median_comment_length | 0 | 0.0 |
| median_comment_length_missing | 0 | 0.0 |
| n_assignee_changes | 0 | 0.0 |
| n_bugfix_commits_authored | 0 | 0.0 |
| n_commits_linked_to_bug | 0 | 0.0 |
| n_files_changed | 0 | 0.0 |
| n_final_status_changes | 0 | 0.0 |
| n_issue_comments | 0 | 0.0 |
| n_issue_field_changes | 0 | 0.0 |
| n_issue_linked_events | 0 | 0.0 |
