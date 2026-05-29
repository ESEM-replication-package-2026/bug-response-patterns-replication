# Paper-facing Candidate Behavioral Role Taxonomy

These are the canonical role labels used in the ESEM submission. They are behavioral interpretations of actor clusters and are not official project roles.

| Cluster | Candidate behavioral label | No. of actors | Dominant signal | How to read it |
|---:|---|---:|---|---|
| -1 | No primary behavioral label | 818 | HDBSCAN noise/boundary actors | Not interpreted as a role; these actors are not forced into a primary label. |
| 0 | Boundary issue discussant | 27 | Issue comments and discussion | Small boundary cluster; issue-side discussion is visible, but the label should be treated cautiously. |
| 1 | Commit-side code contributor | 195 | Fixing lifecycle and commit source | Actors whose observable activity is dominated by bug-linked code changes. |
| 2 | Issue-tracker participant | 571 | Reporting, discussion, and issue-field activity | Broad issue-tracker activity, including reporting, commenting, and issue metadata changes. |
| 3 | Review-side participant | 58 | Review and review-comment sources | Review-related behavior is visible, but this candidate has explicit stability uncertainty. |
| 4 | Review-and-PR hybrid | 138 | Review/comment activity plus PR integration | Actors whose activity combines review-side behavior with PR integration features. |
| 5 | PR lifecycle participant | 94 | PR opened, closed, and merged events | Actors whose observable activity is concentrated in pull-request lifecycle events. |
