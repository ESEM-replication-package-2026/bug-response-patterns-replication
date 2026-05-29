# RQ3 Role Composition Validation

## Purpose

This validation extends the descriptive RQ3 role-composition taxonomy without changing the original taxonomy outputs.

## Bootstrap Stability

- Bootstrap/subsampling repetitions: 30
- Sample fraction per repetition: 0.80
- Mean ARI: 0.996
- Median ARI: 1.000
- Mean NMI: 0.997
- Mean V-measure: 0.997

| Pattern | Mean dominant fraction | Min | Max |
|---|---:|---:|---:|
| Issue discussion and fixing | 0.997 | 0.982 | 1.000 |
| No stable role / sparse PR-driven | 0.996 | 0.891 | 1.000 |
| Review-heavy coordination | 1.000 | 1.000 | 1.000 |
| Fixer-centered | 1.000 | 1.000 | 1.000 |
| Issue-side boundary / long-running | 1.000 | 1.000 | 1.000 |
| Review-integration intensive | 1.000 | 1.000 | 1.000 |
| Code and PR integration | 0.997 | 0.922 | 1.000 |
| PR/review integration | 1.000 | 0.999 | 1.000 |
| PR-driven integration | 1.000 | 1.000 | 1.000 |

## Pattern x Project Independence

- Chi-square: 2478.642
- Degrees of freedom: 32
- p-value used: 0 (asymptotic_chi_square)
- Minimum expected count: 14.251
- Cells with expected count < 5: 0
- Cramer's V: 0.268
- Bias-corrected Cramer's V: 0.267
- Patterns with dominant project ratio > 0.60: 0
- Maximum dominant project ratio: 0.559

| Pattern | Dominant project | Dominant ratio | Projects represented |
|---|---|---:|---:|
| Issue-side boundary / long-running | kafka | 0.559 | 5 |
| Fixer-centered | nifi | 0.527 | 5 |
| PR-driven integration | kafka | 0.497 | 4 |
| PR/review integration | kafka | 0.493 | 5 |
| Review-heavy coordination | kafka | 0.463 | 5 |
| Code and PR integration | zookeeper | 0.447 | 4 |
| Review-integration intensive | kafka | 0.432 | 5 |
| No stable role / sparse PR-driven | kafka | 0.419 | 5 |
| Issue discussion and fixing | zookeeper | 0.388 | 5 |

## Interpretation

- Bootstrap stability should be interpreted as stability of the descriptive bug-pattern clustering, not as validation of causal claims.
- The project test checks whether pattern frequencies are independent of project. A significant p-value indicates some dependence, but Cramer's V is the main indicator of practical association strength.
- If expected-count assumptions are violated, the script reports a Monte Carlo chi-square p-value and uses it as the primary p-value.
- Project dependence does not invalidate the taxonomy, but strong dependence would mean that pattern interpretation should be project-aware.

## Outputs

- `role_composition_bootstrap_stability.csv`
- `role_composition_pattern_stability_by_label.csv`
- `role_composition_project_independence_test.csv`
- `role_composition_project_contingency.csv`
- `role_composition_project_expected_counts.csv`
- `role_composition_project_standardized_residuals.csv`
- `role_composition_project_dominance.csv`
- `role_composition_validation_summary.md`
