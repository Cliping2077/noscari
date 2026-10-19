# Core Lab

Strict, controlled study of persistent learning, retrieval, inquiry, recovery and
retention. Existing runs are kept at their original locations to preserve links:

- `../validated/`: Round 1, five seeds and A/B/C/D plus strong comparators/ablations.
- `../one-demo/`: selected one-demonstration engineering check; not a benchmark.
- `../retention/`: old/new distribution and replay probe.
- `../round2/`: shared-prefix recovery evaluation and training audits.
- `../round2-demo/`: selected recovery demonstration; not a benchmark.
- `first-live-teacher/`: one live assistant-teacher pilot with a sealed 300-task set;
  task-seed groups are not independent teacher/learner-training replicates.
- `first-teacher-controls/`: explicitly post-hoc and synthetic failure controls.
- `cognitive-v1/`: frozen five-seed conditional-selection study; persistent GRU
  state versus parameter-matched MLPs and explicit history statistics. See
  `../../docs/cognitive-research.md` for the protocol and information boundary.

New strict replications belong under this directory with a unique run ID. Every
run must reference a pre-run protocol, hypothesis, primary metric, comparisons,
data splits, budget, random seeds, software versions and source manifest. Preserve
raw chosen-action logs, parameter checks, errors and deviations. Keep final test
data out of training/tuning. Report seed-level uncertainty, not independent-task
intervals that ignore shared training. Record null and adverse outcomes.

A run is not overwritten after inspection. Corrected implementations require a
new run ID if they change numerical behavior; retain invalidated results outside
the validated collection and explain why. A demonstration cannot be promoted into
evidence by renaming its directory. Frontier findings require a new controlled
replication before promotion; their original exploratory label remains intact.
