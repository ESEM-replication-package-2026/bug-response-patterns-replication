# SmartSHARK Roles Pipeline

This project is organized as a reproducible experiment pipeline for extracting participant roles in bug fixing lifecycles from SmartSHARK MongoDB Release 2.2 small.

## Stage Order

1. DB audit: enumerate collections, document counts, field coverage, missingness, and storage metadata.
2. Project availability: summarize project-level availability before choosing target projects.
3. Events table: build immutable CSV/Parquet event tables only for selected projects.
4. Actor features: aggregate event tables into participant-level features.
5. Preprocessing: record bot filtering, low-activity filtering, missing-value handling, scaling, and feature selection.
6. Clustering: use HDBSCAN as the primary method, UMAP for visualization, and k-means/agglomerative clustering as comparison methods.
7. Role profiles: produce role summaries, cluster descriptions, validation tables, and replication package metadata.

## Read-only Data Policy

Pipeline code must not write to MongoDB or modify raw SmartSHARK data. Generated outputs are written under `data/interim`, `data/processed`, `reports`, and `artifact/replication_package`.

MongoDB must be started outside the pipeline. This repository intentionally does not launch `mongod` against `data/mongodb/db`, because starting a database process can update lock, journal, or storage-engine metadata even when analysis queries are read-only.

## Current Executable Steps

```powershell
.\.venv\Scripts\python.exe scripts\01_db_audit.py --config configs\default.yaml
.\.venv\Scripts\python.exe scripts\02_project_availability.py --config configs\default.yaml
```

For final paper runs, set exact field coverage explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\01_db_audit.py --config configs\default.yaml --exact
```

Two-hop project counts for issue events and pull request events can be expensive. Enable them only when needed:

```powershell
.\.venv\Scripts\python.exe scripts\02_project_availability.py --config configs\default.yaml --include-two-hop
```

After project availability, create the pilot project selection file before extracting events:

```powershell
.\.venv\Scripts\python.exe scripts\03_select_projects.py --config configs\default.yaml
```

The next pipeline stage is events extraction for `configs/selected_projects.yaml`. Events extraction should run only for the selected pilot projects first, not all projects.

Run the events smoke extraction before full extraction:

```powershell
.\.venv\Scripts\python.exe scripts\04_extract_events.py --config configs\default.yaml --selected configs\selected_projects.yaml --smoke --max-issues-per-project 50 --max-project-level-prs 200
.\.venv\Scripts\python.exe scripts\04b_audit_events.py --events data\interim\events\events_smoke.parquet
```

Missing `timestamp` and `raw_actor_identifier_field` values are retained and audited, not silently dropped. In the smoke run these missing values come from partially covered `pull_request_review_comment` fields. `lifecycle_stage=unknown` is also retained for heterogeneous issue field changes where a defensible lifecycle mapping is not yet available. See `docs/missing_value_policy.md`.
