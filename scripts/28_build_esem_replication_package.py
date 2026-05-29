"""Build the ESEM submission replication package.

The package is aligned with the current paper:

- RQ1 actor-level candidate behavioral roles.
- RQ2 k=9 bug-response patterns from role-composition vectors.
- Supporting k-sensitivity and subsampling diagnostics.

Large data, restored MongoDB files, actor mappings, full event tables, and full
feature matrices are excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_DIR = ROOT / "artifact" / "esem_replication_package"
DEFAULT_ZIP = ROOT / "artifact" / "esem_replication_package.zip"
DEFAULT_PAPER_PDF = ROOT / "paper" / "submitted_paper_reference.pdf"

TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml", ".tex", ".bib"}
IMAGE_SUFFIXES = {".png"}
DOC_SUFFIXES = {".pdf"}
EXCLUDED_DIR_NAMES = {".git", ".venv", "__pycache__", "matplotlib_cache", "logs"}
SENSITIVE_PATTERNS = {
    "email_like": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "hex24": re.compile(r"\b[0-9a-fA-F]{24}\b"),
    "raw_actor_identifier_literal": re.compile(r"actor_id_raw", re.IGNORECASE),
    "account_name_literal": re.compile(r"username", re.IGNORECASE),
}

SCRIPT_ALLOWLIST = [
    "01_db_audit.py",
    "02_project_availability.py",
    "03_select_projects.py",
    "04a_events_extraction_preflight.py",
    "04_extract_events.py",
    "04b_audit_events.py",
    "05_normalize_actors.py",
    "05b_enrich_actor_metadata.py",
    "06_build_actor_features.py",
    "07_preprocess_features.py",
    "08_cluster_roles.py",
    "09_build_cluster_profiles.py",
    "10_stability_analysis.py",
    "11_project_and_cluster2_diagnostics.py",
    "12_candidate_role_protocol.py",
    "12b_prepare_role_taxonomy_paper_assets.py",
    "13_issue_lifecycle_analysis.py",
    "25_rq3_role_composition_taxonomy.py",
    "25b_rq3_k_sensitivity_extended.py",
    "25c_prepare_rq3_k9_paper_assets.py",
    "26_rq3_role_composition_validation.py",
    "28_build_esem_replication_package.py",
]


@dataclass
class ManifestRow:
    path: str
    size_bytes: int
    sha256: str
    role: str
    contains_raw_data: bool
    public_artifact_allowed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--paper-pdf", type=Path, default=DEFAULT_PAPER_PDF)
    parser.add_argument("--skip-pdf", action="store_true")
    return parser.parse_args()


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_clean(package_dir: Path) -> None:
    resolved = package_dir.resolve()
    artifact = (ROOT / "artifact").resolve()
    if artifact not in resolved.parents:
        raise ValueError(f"Refusing to clean unexpected path: {resolved}")
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def redact_identifiers(text: str) -> str:
    text = re.sub(r"[A-Za-z]:\\Users\\[^\\\r\n]+\\", r"<local_user_dir>\\", text)
    text = re.sub(r"/home/[^/\r\n]+/", r"<local_user_dir>/", text)
    text = text.replace(r"<local_user_dir>\hackathon\\", r"<workspace>\\")
    text = text.replace(r"<local_user_dir>\hackathon\\", r"<workspace>\\")
    text = text.replace(r"<local_user_dir>\hackathon", r"<workspace>")
    text = text.replace("<local_user_dir>/hackathon/", "<workspace>/")
    text = re.sub(r"\b[0-9a-fA-F]{24}\b", "<mongodb_object_id_redacted>", text)
    text = re.sub(r"actor_id_raw", "raw_actor_identifier_field", text, flags=re.IGNORECASE)
    text = re.sub(r"usernames?", "account name fields", text, flags=re.IGNORECASE)
    text = text.replace("missing_raw_actor_identifier_field", "missing_raw_actor_identifier")
    return text


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_file(src: Path, dst: Path, *, sanitize_hex: bool = False) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sanitize_hex and src.suffix.lower() in TEXT_SUFFIXES:
        write_text(dst, redact_identifiers(read_text(src)))
    else:
        shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, *, suffixes: set[str] | None = None, sanitize_hex: bool = False) -> None:
    if not src.exists():
        return
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in path.relative_to(src).parts):
            continue
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        copy_file(path, dst / path.relative_to(src), sanitize_hex=sanitize_hex)


def copy_source(package_dir: Path) -> None:
    copy_tree(ROOT / "src", package_dir / "src", suffixes={".py"})
    for script_name in SCRIPT_ALLOWLIST:
        copy_file(ROOT / "scripts" / script_name, package_dir / "scripts" / script_name)


def copy_configs(package_dir: Path) -> None:
    copy_file(ROOT / "configs" / "default.yaml", package_dir / "configs" / "default.yaml")
    copy_file(
        ROOT / "configs" / "selected_projects.yaml",
        package_dir / "configs" / "selected_projects.yaml",
        sanitize_hex=True,
    )


def copy_docs(package_dir: Path) -> None:
    copy_tree(ROOT / "docs", package_dir / "docs", suffixes={".md"}, sanitize_hex=True)


def copy_root_files(package_dir: Path) -> None:
    for name in ["requirements.txt", "pyproject.toml", "README.md"]:
        source = ROOT / name
        if source.exists():
            copy_file(source, package_dir / name, sanitize_hex=True)


def copy_report_outputs(package_dir: Path) -> None:
    targets = [
        ("reports/data_audit/selected_projects.csv", "reports/data_audit/selected_projects.csv", True),
        ("reports/build_events/events_quality_counts.csv", "reports/build_events/events_quality_counts.csv", True),
        ("reports/features/actor_feature_summary.csv", "reports/features/actor_feature_summary.csv", True),
        ("reports/features/preprocessing_summary.md", "reports/features/preprocessing_summary.md", True),
        ("reports/clustering/cluster_size_table.csv", "reports/clustering/cluster_size_table.csv", True),
        ("reports/clustering/hdbscan_parameter_results.csv", "reports/clustering/hdbscan_parameter_results.csv", True),
        ("reports/results/stability_summary.md", "reports/results/stability_summary.md", True),
        ("reports/results/cluster_stability_by_label.csv", "reports/results/cluster_stability_by_label.csv", True),
        ("reports/results/method_comparison_summary.md", "reports/results/method_comparison_summary.md", True),
        ("reports/cluster_profiles/candidate_role_taxonomy_paper_labels.csv", "reports/cluster_profiles/candidate_role_taxonomy_paper_labels.csv", True),
        ("reports/cluster_profiles/candidate_role_taxonomy_paper_labels.md", "reports/cluster_profiles/candidate_role_taxonomy_paper_labels.md", True),
        ("reports/results/rq3_role_composition_taxonomy/kmeans_k_sensitivity_extended.csv", "reports/results/rq3_role_composition_taxonomy/kmeans_k_sensitivity_extended.csv", True),
        ("reports/results/rq3_role_composition_taxonomy/kmeans_k_sensitivity_cluster_sizes.csv", "reports/results/rq3_role_composition_taxonomy/kmeans_k_sensitivity_cluster_sizes.csv", True),
        ("reports/results/rq3_role_composition_taxonomy/kmeans_k_sensitivity_summary.md", "reports/results/rq3_role_composition_taxonomy/kmeans_k_sensitivity_summary.md", True),
        ("reports/results/rq3_role_composition_taxonomy_k9/bug_response_pattern_summary_k9_paper_labels.csv", "reports/results/rq3_role_composition_taxonomy_k9/bug_response_pattern_summary_k9_paper_labels.csv", True),
        ("reports/results/rq3_role_composition_taxonomy_k9/bug_response_pattern_summary_k9_paper_labels.md", "reports/results/rq3_role_composition_taxonomy_k9/bug_response_pattern_summary_k9_paper_labels.md", True),
        ("reports/results/rq3_role_composition_taxonomy_k9/role_composition_validation_summary.md", "reports/results/rq3_role_composition_taxonomy_k9/role_composition_validation_summary.md", True),
        ("reports/results/rq3_role_composition_taxonomy_k9/role_composition_bootstrap_stability.csv", "reports/results/rq3_role_composition_taxonomy_k9/role_composition_bootstrap_stability.csv", True),
        ("reports/results/rq3_role_composition_taxonomy_k9/role_composition_project_independence_test.csv", "reports/results/rq3_role_composition_taxonomy_k9/role_composition_project_independence_test.csv", True),
        ("reports/results/rq3_role_composition_taxonomy_k9/role_composition_project_dominance.csv", "reports/results/rq3_role_composition_taxonomy_k9/role_composition_project_dominance.csv", True),
        ("reports/paper_tables/table_candidate_role_taxonomy_paper_labels.csv", "reports/paper_tables/table_candidate_role_taxonomy_paper_labels.csv", True),
        ("reports/paper_tables/table_rq3_k9_bug_response_patterns.csv", "reports/paper_tables/table_rq3_k9_bug_response_patterns.csv", True),
        ("reports/figures/rq3_role_composition_taxonomy_k9/figure3_role_composition_heatmap_k9.png", "reports/figures/rq3_role_composition_taxonomy_k9/figure3_role_composition_heatmap_k9.png", False),
        ("reports/figures/rq3_role_composition_taxonomy/figure_rq3_k_sensitivity_extended.png", "reports/figures/rq3_role_composition_taxonomy/figure_rq3_k_sensitivity_extended.png", False),
    ]
    for src_rel, dst_rel, sanitize in targets:
        source = ROOT / src_rel
        if source.exists():
            copy_file(source, package_dir / dst_rel, sanitize_hex=sanitize)


def copy_paper_reference(package_dir: Path, paper_pdf: Path, skip_pdf: bool) -> None:
    paper_dir = package_dir / "paper"
    if not skip_pdf and paper_pdf.exists():
        copy_file(paper_pdf, paper_dir / "submitted_paper_reference.pdf")
    write_text(
        paper_dir / "README.md",
        "# Paper Reference\n\n"
        "This folder contains the submitted-paper PDF as a reference point for the "
        "replication package. The reproducible experimental assets are under "
        "`scripts/`, `src/`, `configs/`, and `reports/`.\n\n"
        "The package intentionally does not rely on an editable Overleaf source "
        "snapshot because the experimental package should track the executable "
        "pipeline and canonical paper-facing outputs.\n",
    )


def package_readme() -> str:
    return """# Replication Package: Mining Role-Based Bug Response Patterns in OSS

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
"""


def reproduce_md() -> str:
    commands = [
        ".\\.venv\\Scripts\\python.exe scripts\\01_db_audit.py --config configs\\default.yaml",
        ".\\.venv\\Scripts\\python.exe scripts\\02_project_availability.py --config configs\\default.yaml",
        ".\\.venv\\Scripts\\python.exe scripts\\03_select_projects.py --config configs\\default.yaml",
        ".\\.venv\\Scripts\\python.exe scripts\\04a_events_extraction_preflight.py --config configs\\default.yaml --selected configs\\selected_projects.yaml",
        ".\\.venv\\Scripts\\python.exe scripts\\04_extract_events.py --config configs\\default.yaml --selected configs\\selected_projects.yaml --max-project-level-prs 1000",
        ".\\.venv\\Scripts\\python.exe scripts\\04b_audit_events.py --events data\\interim\\events\\events.parquet",
        ".\\.venv\\Scripts\\python.exe scripts\\05_normalize_actors.py --events data\\interim\\events\\events.parquet --config configs\\default.yaml",
        ".\\.venv\\Scripts\\python.exe scripts\\05b_enrich_actor_metadata.py --events data\\interim\\events\\events_normalized.parquet --mapping data\\interim\\events\\actor_id_mapping.csv --config configs\\default.yaml",
        ".\\.venv\\Scripts\\python.exe scripts\\06_build_actor_features.py --events data\\interim\\events\\events_normalized.parquet",
        ".\\.venv\\Scripts\\python.exe scripts\\07_preprocess_features.py --features data\\interim\\features\\actor_features_raw.parquet",
        ".\\.venv\\Scripts\\python.exe scripts\\08_cluster_roles.py --features data\\processed\\clusters\\actor_features_model_input.parquet --metadata data\\processed\\clusters\\actor_metadata_for_clustering.csv",
        ".\\.venv\\Scripts\\python.exe scripts\\09_build_cluster_profiles.py --features data\\interim\\features\\actor_features_raw.parquet --labels data\\processed\\clusters\\actor_cluster_labels.csv --events data\\interim\\events\\events_normalized.parquet",
        ".\\.venv\\Scripts\\python.exe scripts\\10_stability_analysis.py --features data\\processed\\clusters\\actor_features_model_input.parquet --labels data\\processed\\clusters\\actor_cluster_labels.csv",
        ".\\.venv\\Scripts\\python.exe scripts\\12_candidate_role_protocol.py",
        ".\\.venv\\Scripts\\python.exe scripts\\12b_prepare_role_taxonomy_paper_assets.py",
        ".\\.venv\\Scripts\\python.exe scripts\\13_issue_lifecycle_analysis.py",
        ".\\.venv\\Scripts\\python.exe scripts\\25_rq3_role_composition_taxonomy.py --k-min 9 --k-max 9 --output-dir reports\\results\\rq3_role_composition_taxonomy_k9 --figure-dir reports\\figures\\rq3_role_composition_taxonomy_k9",
        ".\\.venv\\Scripts\\python.exe scripts\\25b_rq3_k_sensitivity_extended.py",
        ".\\.venv\\Scripts\\python.exe scripts\\25c_prepare_rq3_k9_paper_assets.py",
        ".\\.venv\\Scripts\\python.exe scripts\\26_rq3_role_composition_validation.py --input-dir reports\\results\\rq3_role_composition_taxonomy_k9 --figure-dir reports\\figures\\rq3_role_composition_taxonomy_k9",
    ]
    lines = [
        "# Reproduction Instructions",
        "",
        "## Prerequisites",
        "",
        "- Windows environment.",
        "- Python virtual environment with dependencies from `requirements.txt`.",
        "- MongoDB running on `mongodb://localhost:27018`.",
        "- SmartSHARK MongoDB Release 2.2 small restored as database `smartshark_2_2`.",
        "",
        "## Important Notes",
        "",
        "- The packaged `configs/selected_projects.yaml` redacts MongoDB object ids. Run `scripts/03_select_projects.py` after restoring the database to regenerate local project ids.",
        "- Database operations are read-only.",
        "- Full event tables and actor mappings are regenerated locally and are intentionally not included in the public package.",
        "- The paper-facing role and pattern names are generated by scripts `12b_prepare_role_taxonomy_paper_assets.py` and `25c_prepare_rq3_k9_paper_assets.py`.",
        "",
        "## Command Sequence",
        "",
    ]
    for command in commands:
        lines.extend(["```powershell", command, "```", ""])
    return "\n".join(lines)


def write_manifest(package_dir: Path) -> list[ManifestRow]:
    rows = []
    for path in sorted(package_dir.rglob("*")):
        if path.is_dir():
            continue
        rel_path = rel(path, package_dir)
        raw_data = rel_path.startswith("data/") or "events.parquet" in rel_path or "actor_id_mapping" in rel_path
        allowed = not raw_data
        role = "source_or_script"
        if rel_path.startswith("reports/"):
            role = "aggregate_report_or_figure"
        elif rel_path.startswith("paper/"):
            role = "paper_reference"
        elif rel_path in {"README.md", "REPRODUCE.md", "MANIFEST.md", "artifact_integrity_report.md", "file_manifest.csv"}:
            role = "package_metadata"
        elif rel_path.startswith("configs/"):
            role = "configuration"
        rows.append(ManifestRow(rel_path, path.stat().st_size, sha256(path), role, raw_data, allowed))

    manifest_path = package_dir / "file_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["path", "size_bytes", "sha256", "role", "contains_raw_data", "public_artifact_allowed"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    return rows


def integrity_scan(package_dir: Path, rows: list[ManifestRow]) -> dict[str, object]:
    findings = []
    for row in rows:
        path = package_dir / row.path
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = read_text(path)
        for name, pattern in SENSITIVE_PATTERNS.items():
            count = len(pattern.findall(text))
            if count:
                findings.append(
                    {
                        "path": row.path,
                        "pattern": name,
                        "count": count,
                        "is_source_code": row.path.startswith("scripts/") or row.path.startswith("src/"),
                    }
                )
    forbidden_paths = [
        row.path
        for row in rows
        if row.path.startswith("data/")
        or "actor_id_mapping" in row.path
        or row.path.endswith("events.parquet")
        or row.path.endswith("events.csv")
        or row.path.endswith("events_normalized.parquet")
        or row.path.endswith("actor_features_raw.parquet")
        or row.path.endswith("actor_features_model_input.parquet")
    ]
    return {
        "package_size_bytes": sum(row.size_bytes for row in rows),
        "file_count": len(rows),
        "forbidden_paths": forbidden_paths,
        "sensitive_findings": findings,
        "sensitive_findings_non_source": [f for f in findings if not f["is_source_code"]],
    }


def write_integrity_report(package_dir: Path, report: dict[str, object]) -> None:
    non_source = report["sensitive_findings_non_source"]
    lines = [
        "# Artifact Integrity Report",
        "",
        f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- File count: `{report['file_count']}`",
        f"- Package size: `{report['package_size_bytes']}` bytes",
        f"- Forbidden raw/intermediate paths found: `{len(report['forbidden_paths'])}`",
        f"- Sensitive-pattern findings outside source code: `{len(non_source)}`",
        "",
        "## Interpretation",
        "",
        "Source code may contain schema field names such as actor identifier fields because they are required for reproduction. Non-source reports and sample outputs are expected not to contain direct identifiers or raw data.",
        "",
        "## Forbidden Path Findings",
        "",
    ]
    if report["forbidden_paths"]:
        lines.extend(f"- `{path}`" for path in report["forbidden_paths"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Sensitive Findings Outside Source Code", ""])
    if non_source:
        for item in non_source:
            lines.append(f"- `{item['path']}`: {item['pattern']} x {item['count']}")
    else:
        lines.append("- None.")
    lines.extend(["", "## All Sensitive Findings", ""])
    findings = report["sensitive_findings"]
    if findings:
        for item in findings:
            scope = "source" if item["is_source_code"] else "non-source"
            lines.append(f"- `{item['path']}` ({scope}): {item['pattern']} x {item['count']}")
    else:
        lines.append("- None.")
    write_text(package_dir / "artifact_integrity_report.md", "\n".join(lines) + "\n")


def write_manifest_md(package_dir: Path, rows: list[ManifestRow]) -> None:
    lines = [
        "# Manifest",
        "",
        "The full machine-readable manifest is in `file_manifest.csv`.",
        "",
        "| Path | Size bytes | Role | Public artifact allowed |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(f"| `{row.path}` | {row.size_bytes} | {row.role} | {row.public_artifact_allowed} |")
    write_text(package_dir / "MANIFEST.md", "\n".join(lines) + "\n")


def py_compile_package(package_dir: Path) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", str(package_dir / "scripts"), str(package_dir / "src")],
        cwd=package_dir,
        text=True,
        capture_output=True,
    )
    write_text(
        package_dir / "py_compile_report.txt",
        f"returncode={result.returncode}\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}\n",
    )
    for cache_dir in package_dir.rglob("__pycache__"):
        shutil.rmtree(cache_dir)
    return result.returncode == 0


def make_zip(package_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    base = zip_path.with_suffix("")
    shutil.make_archive(str(base), "zip", package_dir)
    generated = base.with_suffix(".zip")
    if generated != zip_path:
        generated.replace(zip_path)


def main() -> None:
    args = parse_args()
    ensure_clean(args.package_dir)
    copy_source(args.package_dir)
    copy_configs(args.package_dir)
    copy_docs(args.package_dir)
    copy_root_files(args.package_dir)
    copy_report_outputs(args.package_dir)
    copy_paper_reference(args.package_dir, args.paper_pdf, args.skip_pdf)
    write_text(args.package_dir / "README.md", package_readme())
    write_text(args.package_dir / "REPRODUCE.md", reproduce_md())
    py_ok = py_compile_package(args.package_dir)
    rows = write_manifest(args.package_dir)
    write_manifest_md(args.package_dir, rows)
    report = integrity_scan(args.package_dir, rows)
    write_integrity_report(args.package_dir, report)
    # Refresh manifest after writing manifest and integrity files.
    rows = write_manifest(args.package_dir)
    write_manifest_md(args.package_dir, rows)
    report = integrity_scan(args.package_dir, rows)
    write_integrity_report(args.package_dir, report)
    make_zip(args.package_dir, args.zip_path)
    print(f"package_dir={args.package_dir}")
    print(f"zip_path={args.zip_path}")
    print(f"py_compile_ok={py_ok}")
    print(f"file_count={len(rows)}")
    print(f"package_size_bytes={sum(row.size_bytes for row in rows)}")
    print(f"non_source_sensitive_findings={len(report['sensitive_findings_non_source'])}")
    print(f"forbidden_paths={len(report['forbidden_paths'])}")


if __name__ == "__main__":
    main()
