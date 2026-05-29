# Replication Package: Mining Role-Based Bug Response Patterns in OSS

This replication package supports the ESEM submission "Mining Role-Based Bug
Response Patterns in Open-Source Software".

## What This Package Reproduces

The package reproduces the analysis pipeline used in the paper:

1. Audit SmartSHARK MongoDB Release 2.2 small.
2. Select five Apache pilot projects: Kafka, NiFi, Storm, Zeppelin, ZooKeeper.
3. Extract normalized bug-fixing lifecycle events.
4. Build actor-level behavioral features.
5. Cluster actors with PCA + HDBSCAN and interpret non-noise clusters as
   candidate behavioral role labels.
6. Build closed-bug role-composition vectors.
7. Evaluate k for the bug-response taxonomy and prepare the paper-facing k=9
   bug-response pattern outputs.
8. Run supporting subsampling and project-distribution diagnostics.

## Data

The package uses the released SmartSHARK MongoDB Release 2.2 small dataset. The
raw SmartSHARK archive, MongoDB dump/restored database, full event tables, full
feature matrices, and internal actor mappings are not included.

## Canonical Paper Labels

Some internal scripts preserve earlier exploratory labels. The submitted paper's
canonical labels are provided in:

- `reports/cluster_profiles/candidate_role_taxonomy_paper_labels.csv`
- `reports/results/rq3_role_composition_taxonomy_k9/bug_response_pattern_summary_k9_paper_labels.csv`

Use those files when comparing package outputs to the paper.

## Privacy and Identifier Policy

The public package excludes raw actor mappings and large intermediate event
tables. Project database object identifiers in configuration/report files are
redacted. Source code may contain schema-field names needed for reproduction,
but packaged reports and sample outputs should not contain direct identifiers.
