# Extended k Sensitivity for RQ3 Role-Composition Clustering

## Purpose

This report extends the original RQ3 k-means model selection beyond `k=4--6`. It uses the same bug-level role-count vectors, the same `log1p` transformation, and standardization before k-means.

## Recommendation

- Recommended k under the extended diagnostic: **9**.
- Selection rule: Selected the best combined rank score among k values with smallest cluster >= 100 and largest cluster ratio <= 0.50.
- Guardrails: smallest cluster >= 100; largest cluster ratio <= 0.50.

## Best k by Individual Criteria

- Highest silhouette: k=12.
- Lowest Davies-Bouldin: k=2.
- Highest Calinski-Harabasz: k=12.

## Selected-k Metrics

- k=9: silhouette=0.740, Davies-Bouldin=0.653, Calinski-Harabasz=6111.3, smallest cluster=111, largest cluster ratio=0.378.

## Comparison with Original k=6

- Original k=6: silhouette=0.601, Davies-Bouldin=0.705, Calinski-Harabasz=4563.7, smallest cluster=111, largest cluster ratio=0.455.
- Interpretation: k=6 remains a concise, interpretable coarse taxonomy, but the extended internal metrics favor a finer taxonomy.

## Full k Table

| k | silhouette | Davies-Bouldin | Calinski-Harabasz | smallest cluster | largest cluster ratio | clusters <100 | combined rank |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.732 | 0.433 | 1701.3 | 111 | 0.987 | 0 | 29.0 |
| 3 | 0.409 | 1.099 | 2080.3 | 111 | 0.624 | 0 | 43.0 |
| 4 | 0.438 | 0.921 | 2677.7 | 111 | 0.612 | 0 | 39.0 |
| 5 | 0.489 | 0.823 | 3405.6 | 111 | 0.578 | 0 | 34.0 |
| 6 | 0.601 | 0.705 | 4563.7 | 111 | 0.455 | 0 | 28.0 |
| 7 | 0.684 | 0.651 | 5891.3 | 111 | 0.378 | 0 | 17.0 |
| 8 | 0.722 | 0.654 | 6040.2 | 111 | 0.378 | 0 | 17.0 |
| 9 | 0.740 | 0.653 | 6111.3 | 111 | 0.378 | 0 | 13.0 |
| 10 | 0.741 | 0.746 | 6127.9 | 36 | 0.378 | 1 | 22.0 |
| 11 | 0.743 | 0.843 | 6166.6 | 36 | 0.378 | 1 | 23.0 |
| 12 | 0.753 | 0.795 | 6364.8 | 36 | 0.378 | 1 | 19.0 |

## Paper-use Note

Do not claim that any k reveals ground-truth bug types. The result is a descriptive clustering of role-composition vectors. If the paper keeps k=6, it should explicitly frame k=6 as a concise coarse taxonomy rather than the best-supported k under the extended sensitivity analysis.
