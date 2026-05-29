"""Validation checks for the RQ3 role-composition taxonomy.

Adds two checks without modifying the original taxonomy outputs:

1. Bootstrap/subsampling stability for bug response patterns.
2. Pattern x project independence test with expected-count diagnostics,
   Monte Carlo fallback, and Cramer's V effect size.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "reports" / "results" / "rq3_role_composition_taxonomy"
DEFAULT_FIGURE_DIR = ROOT / "reports" / "figures" / "rq3_role_composition_taxonomy"

ROLE_FEATURES = [
    "n_fixer",
    "n_issue_side_participant",
    "n_pr_integrator",
    "n_review_integration_hybrid",
    "n_review_focused_participant",
    "n_issue_side_boundary_participant",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--bootstrap-reps", type=int, default=30)
    parser.add_argument("--sample-frac", type=float, default=0.80)
    parser.add_argument("--monte-carlo-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vectors_path = input_dir / "issue_role_composition_vectors.csv"
    summary_path = input_dir / "bug_response_pattern_summary.csv"
    k_path = input_dir / "kmeans_k_selection.csv"
    logging.info("Reading vectors: %s", vectors_path)
    vectors = pd.read_csv(vectors_path)
    summary = pd.read_csv(summary_path)
    k_eval = pd.read_csv(k_path)
    return vectors, summary, k_eval


def selected_k(k_eval: pd.DataFrame, vectors: pd.DataFrame) -> int:
    if "selected" in k_eval.columns and k_eval["selected"].astype(str).str.lower().eq("true").any():
        return int(k_eval.loc[k_eval["selected"].astype(str).str.lower().eq("true"), "k"].iloc[0])
    return int(vectors["role_composition_k"].dropna().iloc[0])


def bootstrap_stability(
    vectors: pd.DataFrame,
    summary: pd.DataFrame,
    k: int,
    reps: int,
    sample_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
        v_measure_score,
    )
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    x_all = np.log1p(vectors[ROLE_FEATURES].astype(float).to_numpy())
    original = vectors["role_composition_cluster"].astype(int).to_numpy()
    n = len(vectors)
    sample_n = max(k + 1, int(round(n * sample_frac)))
    cluster_to_label = dict(
        zip(
            summary["role_composition_cluster"].astype(int),
            summary["candidate_pattern_label"].astype(str),
        )
    )

    rows = []
    label_rows = []
    for rep in range(1, reps + 1):
        idx = np.sort(rng.choice(n, size=sample_n, replace=False))
        scaler = StandardScaler()
        x = scaler.fit_transform(x_all[idx])
        model = KMeans(n_clusters=k, random_state=seed + rep, n_init=50)
        boot_labels = model.fit_predict(x)
        original_sample = original[idx]
        ari = adjusted_rand_score(original_sample, boot_labels)
        nmi = normalized_mutual_info_score(original_sample, boot_labels)
        v_measure = v_measure_score(original_sample, boot_labels)
        counts = pd.Series(boot_labels).value_counts()
        rows.append(
            {
                "rep": rep,
                "sample_n": sample_n,
                "k": k,
                "ari": ari,
                "nmi": nmi,
                "v_measure": v_measure,
                "silhouette": silhouette_score(x, boot_labels),
                "largest_cluster_size": int(counts.max()),
                "largest_cluster_ratio": float(counts.max() / sample_n),
                "smallest_cluster_size": int(counts.min()),
            }
        )

        for cluster in sorted(np.unique(original)):
            mask = original_sample == cluster
            n_in_sample = int(mask.sum())
            if n_in_sample == 0:
                continue
            mapped_counts = pd.Series(boot_labels[mask]).value_counts()
            dominant_fraction = float(mapped_counts.max() / n_in_sample)
            label_rows.append(
                {
                    "rep": rep,
                    "role_composition_cluster": int(cluster),
                    "candidate_pattern_label": cluster_to_label.get(int(cluster), str(cluster)),
                    "n_in_sample": n_in_sample,
                    "dominant_bootstrap_cluster": int(mapped_counts.idxmax()),
                    "dominant_fraction": dominant_fraction,
                    "n_bootstrap_clusters_touched": int(mapped_counts.size),
                }
            )

    stability = pd.DataFrame(rows)
    label_detail = pd.DataFrame(label_rows)
    by_label = (
        label_detail.groupby(["role_composition_cluster", "candidate_pattern_label"])
        .agg(
            reps=("rep", "nunique"),
            mean_n_in_sample=("n_in_sample", "mean"),
            mean_dominant_fraction=("dominant_fraction", "mean"),
            median_dominant_fraction=("dominant_fraction", "median"),
            min_dominant_fraction=("dominant_fraction", "min"),
            max_dominant_fraction=("dominant_fraction", "max"),
            mean_bootstrap_clusters_touched=("n_bootstrap_clusters_touched", "mean"),
        )
        .reset_index()
        .sort_values("role_composition_cluster")
    )
    return stability, by_label


def chi_square_stat(observed: np.ndarray, expected: np.ndarray) -> float:
    return float(((observed - expected) ** 2 / expected).sum())


def cramers_v(chi2: float, observed: np.ndarray) -> tuple[float, float]:
    n = observed.sum()
    r, c = observed.shape
    raw = math.sqrt(chi2 / (n * min(r - 1, c - 1)))
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((c - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    c_corr = c - ((c - 1) ** 2) / (n - 1)
    corrected = math.sqrt(phi2_corr / min(c_corr - 1, r_corr - 1)) if min(c_corr - 1, r_corr - 1) > 0 else 0.0
    return raw, corrected


def monte_carlo_chi_square(
    pattern_codes: np.ndarray,
    project_codes: np.ndarray,
    shape: tuple[int, int],
    expected: np.ndarray,
    observed_chi2: float,
    reps: int,
    seed: int,
) -> tuple[float, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    simulated = np.empty(reps, dtype=float)
    fixed_pattern = pattern_codes.copy()
    project = project_codes.copy()
    for i in range(reps):
        shuffled_project = rng.permutation(project)
        table = np.zeros(shape, dtype=int)
        np.add.at(table, (fixed_pattern, shuffled_project), 1)
        simulated[i] = chi_square_stat(table, expected)
    p_value = float((np.sum(simulated >= observed_chi2) + 1) / (reps + 1))
    sim_summary = pd.DataFrame(
        [
            {
                "monte_carlo_reps": reps,
                "simulated_chi2_mean": float(simulated.mean()),
                "simulated_chi2_p95": float(np.quantile(simulated, 0.95)),
                "simulated_chi2_max": float(simulated.max()),
            }
        ]
    )
    return p_value, sim_summary


def project_independence(
    vectors: pd.DataFrame,
    summary: pd.DataFrame,
    monte_carlo_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from scipy.stats import chi2

    label_map = dict(
        zip(
            summary["role_composition_cluster"].astype(int),
            summary["candidate_pattern_label"].astype(str),
        )
    )
    work = vectors[["project_name", "role_composition_cluster"]].copy()
    work["role_composition_cluster"] = work["role_composition_cluster"].astype(int)
    work["candidate_pattern_label"] = work["role_composition_cluster"].map(label_map)

    contingency = pd.crosstab(work["candidate_pattern_label"], work["project_name"])
    observed = contingency.to_numpy(dtype=float)
    n = observed.sum()
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / n
    chi2_stat = chi_square_stat(observed, expected)
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    asymptotic_p = float(chi2.sf(chi2_stat, dof))
    min_expected = float(expected.min())
    cells_expected_lt5 = int((expected < 5).sum())
    cells_expected_lt1 = int((expected < 1).sum())
    raw_v, corrected_v = cramers_v(chi2_stat, observed)

    use_monte_carlo = cells_expected_lt5 > 0
    monte_carlo_p = np.nan
    sim_summary = pd.DataFrame()
    if use_monte_carlo:
        pattern_codes, pattern_names = pd.factorize(work["candidate_pattern_label"])
        project_codes, project_names = pd.factorize(work["project_name"])
        monte_carlo_p, sim_summary = monte_carlo_chi_square(
            pattern_codes=pattern_codes,
            project_codes=project_codes,
            shape=(len(pattern_names), len(project_names)),
            expected=expected,
            observed_chi2=chi2_stat,
            reps=monte_carlo_reps,
            seed=seed,
        )
        p_used = monte_carlo_p
        p_method = "monte_carlo_chi_square_due_to_expected_counts"
    else:
        p_used = asymptotic_p
        p_method = "asymptotic_chi_square"

    result = pd.DataFrame(
        [
            {
                "test": "pattern_project_independence",
                "n": int(n),
                "n_patterns": int(observed.shape[0]),
                "n_projects": int(observed.shape[1]),
                "chi_square": chi2_stat,
                "dof": int(dof),
                "p_value_asymptotic": asymptotic_p,
                "p_value_monte_carlo": monte_carlo_p,
                "p_value_used": p_used,
                "p_value_method": p_method,
                "min_expected_count": min_expected,
                "cells_expected_lt5": cells_expected_lt5,
                "cells_expected_lt1": cells_expected_lt1,
                "cramers_v": raw_v,
                "cramers_v_bias_corrected": corrected_v,
                "effect_size_note": "Cramer's V near 0 indicates weak association; p-values can be significant with large n.",
            }
        ]
    )
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
    residuals = (observed - expected) / np.sqrt(expected)
    residuals_df = pd.DataFrame(residuals, index=contingency.index, columns=contingency.columns)
    return result, contingency, expected_df, residuals_df, sim_summary


def project_dominance(contingency: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pattern, row in contingency.iterrows():
        total = int(row.sum())
        dominant_project = str(row.idxmax())
        dominant_n = int(row.max())
        rows.append(
            {
                "candidate_pattern_label": pattern,
                "n_issues": total,
                "dominant_project": dominant_project,
                "dominant_project_n": dominant_n,
                "dominant_project_ratio": dominant_n / total if total else 0.0,
                "n_projects_with_issues": int((row > 0).sum()),
                "project_specific_gt_0_60": dominant_n / total > 0.60 if total else False,
            }
        )
    return pd.DataFrame(rows).sort_values("dominant_project_ratio", ascending=False)


def make_figures(
    by_label: pd.DataFrame,
    residuals: pd.DataFrame,
    figure_dir: Path,
) -> None:
    cache_dir = figure_dir / "matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    ordered = by_label.sort_values("mean_dominant_fraction", ascending=True)
    labels = [
        f"{row.candidate_pattern_label}\n(cluster {int(row.role_composition_cluster)})"
        for row in ordered.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(labels, ordered["mean_dominant_fraction"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean dominant bootstrap fraction")
    ax.set_ylabel("Pattern")
    ax.set_title("Bootstrap stability by role-composition pattern")
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_rq3_pattern_bootstrap_stability.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    image = ax.imshow(residuals.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(residuals.columns)))
    ax.set_xticklabels(residuals.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(residuals.index)))
    ax.set_yticklabels(residuals.index)
    ax.set_title("Standardized residuals: pattern x project")
    fig.colorbar(image, ax=ax, label="Std. residual")
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_rq3_project_residual_heatmap.png", dpi=200)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    stability: pd.DataFrame,
    by_label: pd.DataFrame,
    chi_result: pd.DataFrame,
    contingency: pd.DataFrame,
    dominance: pd.DataFrame,
) -> None:
    ari_mean = stability["ari"].mean()
    ari_median = stability["ari"].median()
    nmi_mean = stability["nmi"].mean()
    v_mean = stability["v_measure"].mean()
    chi = chi_result.iloc[0]
    lines = [
        "# RQ3 Role Composition Validation",
        "",
        "## Purpose",
        "",
        "This validation extends the descriptive RQ3 role-composition taxonomy without changing the original taxonomy outputs.",
        "",
        "## Bootstrap Stability",
        "",
        f"- Bootstrap/subsampling repetitions: {stability['rep'].nunique()}",
        f"- Sample fraction per repetition: {stability['sample_n'].iloc[0] / contingency.values.sum():.2f}",
        f"- Mean ARI: {ari_mean:.3f}",
        f"- Median ARI: {ari_median:.3f}",
        f"- Mean NMI: {nmi_mean:.3f}",
        f"- Mean V-measure: {v_mean:.3f}",
        "",
        "| Pattern | Mean dominant fraction | Min | Max |",
        "|---|---:|---:|---:|",
    ]
    for _, row in by_label.iterrows():
        lines.append(
            f"| {row['candidate_pattern_label']} | "
            f"{row['mean_dominant_fraction']:.3f} | "
            f"{row['min_dominant_fraction']:.3f} | "
            f"{row['max_dominant_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Pattern x Project Independence",
            "",
            f"- Chi-square: {chi['chi_square']:.3f}",
            f"- Degrees of freedom: {int(chi['dof'])}",
            f"- p-value used: {chi['p_value_used']:.6g} ({chi['p_value_method']})",
            f"- Minimum expected count: {chi['min_expected_count']:.3f}",
            f"- Cells with expected count < 5: {int(chi['cells_expected_lt5'])}",
            f"- Cramer's V: {chi['cramers_v']:.3f}",
            f"- Bias-corrected Cramer's V: {chi['cramers_v_bias_corrected']:.3f}",
            f"- Patterns with dominant project ratio > 0.60: {int(dominance['project_specific_gt_0_60'].sum())}",
            f"- Maximum dominant project ratio: {dominance['dominant_project_ratio'].max():.3f}",
            "",
            "| Pattern | Dominant project | Dominant ratio | Projects represented |",
            "|---|---|---:|---:|",
        ]
    )
    for _, row in dominance.iterrows():
        lines.append(
            f"| {row['candidate_pattern_label']} | {row['dominant_project']} | "
            f"{row['dominant_project_ratio']:.3f} | {int(row['n_projects_with_issues'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Bootstrap stability should be interpreted as stability of the descriptive bug-pattern clustering, not as validation of causal claims.",
            "- The project test checks whether pattern frequencies are independent of project. A significant p-value indicates some dependence, but Cramer's V is the main indicator of practical association strength.",
            "- If expected-count assumptions are violated, the script reports a Monte Carlo chi-square p-value and uses it as the primary p-value.",
            "- Project dependence does not invalidate the taxonomy, but strong dependence would mean that pattern interpretation should be project-aware.",
            "",
            "## Outputs",
            "",
            "- `role_composition_bootstrap_stability.csv`",
            "- `role_composition_pattern_stability_by_label.csv`",
            "- `role_composition_project_independence_test.csv`",
            "- `role_composition_project_contingency.csv`",
            "- `role_composition_project_expected_counts.csv`",
            "- `role_composition_project_standardized_residuals.csv`",
            "- `role_composition_project_dominance.csv`",
            "- `role_composition_validation_summary.md`",
        ]
    )
    (output_dir / "role_composition_validation_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def integrity_check(output_dir: Path) -> pd.DataFrame:
    import re

    patterns = {
        "actor_id_raw": re.compile(r"actor_id_raw", re.IGNORECASE),
        "username": re.compile(r"username", re.IGNORECASE),
        "email_like": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "hex24": re.compile(r"\b[0-9a-fA-F]{24}\b"),
    }
    rows = []
    for path in sorted(output_dir.glob("role_composition_*")):
        if path.suffix.lower() not in {".csv", ".md", ".json"}:
            continue
        if path.name.endswith("integrity_checks.csv"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in patterns.items():
            rows.append({"file": path.name, "check": name, "matches": len(pattern.findall(text))})
    return pd.DataFrame(rows)


def main() -> None:
    setup_logging()
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    vectors, summary, k_eval = read_inputs(args.input_dir)
    k = selected_k(k_eval, vectors)
    logging.info("Selected k=%s", k)

    stability, by_label = bootstrap_stability(
        vectors=vectors,
        summary=summary,
        k=k,
        reps=args.bootstrap_reps,
        sample_frac=args.sample_frac,
        seed=args.seed,
    )
    chi_result, contingency, expected, residuals, sim_summary = project_independence(
        vectors=vectors,
        summary=summary,
        monte_carlo_reps=args.monte_carlo_reps,
        seed=args.seed,
    )
    dominance = project_dominance(contingency)

    stability.to_csv(args.input_dir / "role_composition_bootstrap_stability.csv", index=False)
    by_label.to_csv(args.input_dir / "role_composition_pattern_stability_by_label.csv", index=False)
    chi_result.to_csv(args.input_dir / "role_composition_project_independence_test.csv", index=False)
    contingency.to_csv(args.input_dir / "role_composition_project_contingency.csv")
    expected.to_csv(args.input_dir / "role_composition_project_expected_counts.csv")
    residuals.to_csv(args.input_dir / "role_composition_project_standardized_residuals.csv")
    dominance.to_csv(args.input_dir / "role_composition_project_dominance.csv", index=False)
    if not sim_summary.empty:
        sim_summary.to_csv(args.input_dir / "role_composition_project_monte_carlo_summary.csv", index=False)

    make_figures(by_label, residuals, args.figure_dir)
    write_summary(args.input_dir, stability, by_label, chi_result, contingency, dominance)
    integrity = integrity_check(args.input_dir)
    integrity.to_csv(args.input_dir / "role_composition_validation_integrity_checks.csv", index=False)
    metadata = {
        "bootstrap_reps": args.bootstrap_reps,
        "sample_frac": args.sample_frac,
        "monte_carlo_reps": args.monte_carlo_reps,
        "selected_k": k,
        "claim_type": "validation of descriptive taxonomy, not causal inference",
    }
    (args.input_dir / "role_composition_validation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    logging.info("Validation completed: %s", args.input_dir / "role_composition_validation_summary.md")


if __name__ == "__main__":
    main()
