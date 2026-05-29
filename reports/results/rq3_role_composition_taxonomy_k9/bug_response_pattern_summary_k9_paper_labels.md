# Paper-facing k=9 Bug Response Pattern Summary

These are the canonical pattern labels used for the ESEM submission. They are post-clustering descriptive labels, not ground-truth issue types or recommended assignments.

`mean_linked_comments` includes issue comments, pull-request comments, and pull-request review comments linked to the closed bug issue. Review submissions are counted separately as `mean_reviews`.

| Pattern | Issues | Median close days | Mean linked comments | Mean commits | Mean PRs | Mean reviews |
|---|---:|---:|---:|---:|---:|---:|
| Sparse stable-role quick fixes | 3,250 | 11.4 | 10.5 | 0.76 | 0.66 | 0.94 |
| Commit-side quick fixes | 2,100 | 7.0 | 17.7 | 1.65 | 1.33 | 2.33 |
| Issue-tracker delayed fixes | 1,006 | 37.1 | 18.5 | 0.67 | 0.78 | 0.66 |
| PR-lifecycle quick fixes | 722 | 5.5 | 68.2 | 2.69 | 7.60 | 3.93 |
| PR-and-review merge fixes | 424 | 8.0 | 67.6 | 2.24 | 6.31 | 6.75 |
| Discussion-and-code delayed fixes | 423 | 43.7 | 26.6 | 1.93 | 1.56 | 2.31 |
| Code-and-heavy-PR fixes | 409 | 21.5 | 58.8 | 2.85 | 9.80 | 2.83 |
| Review-saturated coordination | 162 | 11.8 | 121.2 | 2.12 | 8.99 | 23.83 |
| Issue-boundary long runners | 111 | 76.4 | 16.8 | 0.79 | 1.14 | 0.19 |

## Mean role counts per issue

| Pattern | Commit-side code contributor | Issue-tracker participant | PR lifecycle participant | Review-and-PR hybrid | Review-side participant | Boundary issue discussant |
|---|---:|---:|---:|---:|---:|---:|
| Sparse stable-role quick fixes | 0.00 | 0.00 | 0.11 | 0.00 | 0.00 | 0.00 |
| Commit-side quick fixes | 1.05 | 0.00 | 0.15 | 0.00 | 0.00 | 0.00 |
| Issue-tracker delayed fixes | 0.00 | 1.14 | 0.32 | 0.00 | 0.00 | 0.00 |
| PR-lifecycle quick fixes | 0.00 | 0.07 | 4.28 | 0.00 | 0.00 | 0.00 |
| PR-and-review merge fixes | 0.34 | 0.22 | 1.97 | 1.29 | 0.00 | 0.00 |
| Discussion-and-code delayed fixes | 1.07 | 1.20 | 0.48 | 0.00 | 0.00 | 0.00 |
| Code-and-heavy-PR fixes | 1.04 | 0.24 | 5.36 | 0.00 | 0.00 | 0.00 |
| Review-saturated coordination | 0.40 | 0.26 | 2.00 | 0.44 | 1.17 | 0.00 |
| Issue-boundary long runners | 0.34 | 0.59 | 0.57 | 0.03 | 0.00 | 1.04 |
