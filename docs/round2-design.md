# Round 2: learned recovery after feedback

Date: 2026-10-18. Authorized continuation of the V0.1 research brief.

## Hypothesis and scope

After an identical observed failed semantic attempt, a small trainable controller
can learn when to continue traversal, search, or ask, using remaining budget and
observed failures. This targets the fixed-fallback limitation identified in round 1.
It does not invent skills, learn language, or introduce a general multi-agent system.

Alternatives: keep fixed fallback (strong transparent baseline); contextual linear
reward prediction (chosen for inspectability); an MLP comparator. No parameter
sweep on the final test set. Recovery is a separate experimental module, preserving
round-1 artifacts and the existing Axiom API.

## Implementation

- `core/recovery_models.py`: typed visible episode state and contiguous transitions.
- `environments/interaction.py`: one menu operation or one bounded help-and-open
  operation per action. Session snapshots contain only acquired evidence.
- `learning/recovery.py`: linear per-action Monte Carlo return regression, dynamic
  features, hard action filters; LinUCB and MLP reuse standard algorithms.
- `core/recovery.py`: local journal-backed recovery learner with explicit record,
  learn, checkpoint, rollback, retract and privacy deletion, using strict event
  validation. Public module is experimental; initial semantic teaching remains the
  round-1 capability. Candidate alternatives are reevaluated after each observation.
- `benchmarks/recovery.py`: common failed-prefix evaluation, raw traces, comparisons
  and ablations. Same logged chosen-action trajectories for trainable comparators.

Each test episode starts by attempting the same semantic candidate if available.
Only post-prefix actions are chosen by policies. Report all tasks AND the common
failed-prefix cohort, whose membership is independent of the evaluated policy.
Every operation, including the prefix and help, consumes the shared episode budget.

Training follows the same prefix then uniform random available actions. Targets are
realized return-to-go under that behavior policy: terminal success minus future
operation cost. This is one-pass policy improvement from logged Monte Carlo returns,
not an optimal RL algorithm. Its behavior-policy distribution mismatch is a risk.

Features: fixed visible menu context plus remaining resources, observed missing
menus, semantic failure and help history. No target position, regime or seed enters
the feature vector. Hard action masks remain in every ablation. Initial-context
ablation removes dynamic scoring features but preserves safety/availability masks.

## Fixed experiment plan

5 independent seeds; 2 teacher-rule conditions (clean/partial error); 480 training
tasks per learner; 4 layouts × 3 help-cost settings × 30 heldout tasks per seed and
teacher. Test prefixes and noisy hints are paired between policies. Evaluation
parameters are frozen. Primary descriptive metric: utility on the common semantic
failure cohort, with seed-cluster paired intervals; success, costs, help usage,
failure recovery and calibration are also reported. Hyperparameters are unchanged
from round 1 (ridge=1; LinUCB alpha=.25; MLP 150 epochs).

Compare fixed traversal, fixed cheap-help heuristic, same-memory nearest-neighbor
retrieval, trained linear controller, LinUCB, MLP, frozen, initial-context and
no-inquiry. Fixed traversal equals round-1 semantic-then-traverse fallback after
the prefix. Round-1 full benchmark stays available and is not relabeled as round 2.

## Work checklist

- [x] Contract tests fail for missing step interaction and feedback-sensitive ranking.
- [x] Implement visible step state, budgets, no duplicate menu/help actions.
- [x] Explicit validated episode learning; save/load/rollback/forget regressions.
- [x] Equal-data baselines, common failure cohort, deterministic benchmark tests.
- [x] Run fixed 5-seed experiment; preserve negative outcomes and uncertainties.
- [x] Independent review and fixes; all tests, package install, source scan.
- [x] Round-2 report and updated source deliverable, with precise capability limits.
