# Method Comparison Summary

- Generated at UTC: `2026-05-06T11:14:26.652330+00:00`
- DB access: not used in this step.
- k-means and agglomerative clustering are comparison methods only; PCA + HDBSCAN remains the primary method.

## Scores

| method | n_clusters | noise_count | noise_ratio | silhouette_score | calinski_harabasz_score | davies_bouldin_score | ari_vs_hdbscan_nonnoise | nmi_vs_hdbscan_nonnoise | v_measure_vs_hdbscan_nonnoise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hdbscan_nonnoise_reference | 6 | 818 | 0.43029984218832196 | 0.34262294277517774 | 474.1580930392049 | 1.0689203582326807 | 1.0 | 1.0 | 1.0 |
| kmeans | 6 | 0 | 0.0 | 0.3060230576928446 | 495.4555254479502 | 1.1640345367708977 | 0.9130847642733496 | 0.9163267362734968 | 0.9163267362734969 |
| agglomerative_ward | 6 | 0 | 0.0 | 0.25717182087375134 | 434.7480208760885 | 1.3387554491604943 | 0.9135404791017485 | 0.9096205315644299 | 0.9096205315644299 |

## Interpretation

- HDBSCAN reference uses its non-noise actors only for internal validity metrics; its noise ratio is `0.4303`.
- The closest comparison method by ARI to HDBSCAN non-noise labels is `agglomerative_ward` with ARI `0.9135` and NMI `0.9096`.
- If comparison ARI/NMI are modest, the exact partition is method-sensitive, but HDBSCAN remains useful because it explicitly models noise and irregular cluster density.
- Final role claims should rely on HDBSCAN profiles plus stability evidence, not on k-means/agglomerative alone.
