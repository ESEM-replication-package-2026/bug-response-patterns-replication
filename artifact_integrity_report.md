# Artifact Integrity Report

- Generated at UTC: `2026-05-29T18:43:22.144954+00:00`
- File count: `75`
- Package size: `3754851` bytes
- Forbidden raw/intermediate paths found: `0`
- Sensitive-pattern findings outside source code: `0`

## Interpretation

Source code may contain schema field names such as actor identifier fields because they are required for reproduction. Non-source reports and sample outputs are expected not to contain direct identifiers or raw data.

## Forbidden Path Findings

- None.

## Sensitive Findings Outside Source Code

- None.

## All Sensitive Findings

- `scripts/04b_audit_events.py` (source): raw_actor_identifier_literal x 5
- `scripts/05_normalize_actors.py` (source): raw_actor_identifier_literal x 2
- `scripts/05b_enrich_actor_metadata.py` (source): raw_actor_identifier_literal x 5
- `scripts/05b_enrich_actor_metadata.py` (source): account_name_literal x 3
- `scripts/11_project_and_cluster2_diagnostics.py` (source): account_name_literal x 1
- `scripts/25_rq3_role_composition_taxonomy.py` (source): raw_actor_identifier_literal x 2
- `scripts/25_rq3_role_composition_taxonomy.py` (source): account_name_literal x 2
- `scripts/26_rq3_role_composition_validation.py` (source): raw_actor_identifier_literal x 2
- `scripts/26_rq3_role_composition_validation.py` (source): account_name_literal x 2
- `scripts/28_build_esem_replication_package.py` (source): raw_actor_identifier_literal x 2
- `scripts/28_build_esem_replication_package.py` (source): account_name_literal x 2
- `src/smartshark_roles/anonymize.py` (source): raw_actor_identifier_literal x 17
- `src/smartshark_roles/extraction.py` (source): raw_actor_identifier_literal x 18
- `src/smartshark_roles/preprocessing.py` (source): account_name_literal x 1
- `src/smartshark_roles/role_profiles.py` (source): account_name_literal x 1
