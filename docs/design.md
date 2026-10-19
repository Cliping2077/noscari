# AXIOM Round 1 design — 2026-10-18

## Scope and hypothesis

The approved V0.1 task brief authorizes implementation. This round tests whether
verified experience changes a small internal method-ranking model and changes
behavior on held-out randomized menu layouts. It does not test AGI, consciousness,
language acquisition, autonomous skill invention, or a full cognitive architecture.

Chosen approach: a regularized linear reward regressor over observable context and
method identity, with explicit replay and versioned checkpoints. Alternatives are
a fixed memory retrieval policy (no trained weights) and a small MLP. LinUCB supplies
a contextual-bandit comparator. The linear model is inexpensive and inspectable;
the synthetic feature representation limits any claim of semantic generalization.

## Boundaries

- `core/models.py`: validated observations, demonstrations, feedback, candidates.
- `environments/menu.py`: private randomized layouts and budgeted public interaction.
- `learning/policies.py`: linear replay regression, LinUCB, optional PyTorch MLP.
- `memory/store.py`: transactional SQLite event store and JSONL export/replay.
- `core/agent.py`: observe/suggest/teach/feedback/learn/save/load; hard authorization
  filters, versioned skill records, pending candidates, explicit learning receipts.
- `benchmarks/run.py`: equal-information logged experience, A/B/C/D comparisons,
  frozen/no-association/no-inquiry ablations, paired seeded evaluation and costs.
- `examples/teach_menu.py`: one validated teaching event, exact parameter and action
  comparison, transfer attempts, persistence and rollback.

The menu methods are bounded macro-actions. Semantic search first tries a guessed
menu, then falls back to traversal; inquiry obtains a noisy simulated hint and
then tries it. Competing proposals persist in a workspace. Intra-method concurrent
planning and compositional candidate execution are explicitly deferred.

## Data and learning

The public observation includes goal text, root menu labels, action budget, allowed
help channels, cost, reliability and risk. It excludes seed, regime, item placement,
correct method and target path. Context features use a fixed toy vocabulary,
explicitly a preexisting semantic capability, not language learned from one demo.
Teaching registers a declarative allowed method and a trace, never executable code.
Only externally verified outcome feedback is eligible for training. Record IDs and
weights are separately auditable. Teacher confidence alone is insufficient.

Each experiment has one demo plus, in a separate phase, shared logged randomized
experience. Report the one-demo result separately from feedback-trained results.
Train and evaluation use disjoint seed namespaces and disjoint function names.
Evaluation freezes parameters and does not add evaluation outcomes to replay.
Policies see the same available methods, observations and logged training feedback.
The serial baseline intentionally discards cross-task memory. No commercial API.

## Governance and failure behavior

SQLite is authoritative. JSONL is an explicit export, avoiding a two-writer journal.
Corrections/retractions preserve provenance; physical privacy deletion rewrites the
owned store, invalidates learned state and requires retraining. Previously exported
files and backups must be removed by their owners; this is not machine unlearning.
Checkpoints use validated JSON arrays, never pickle or arbitrary code loading.
Default providers are offline. Help authorization is filtered before any learned
ranking. External publishing and real account/tool actions are out of scope.

## Acceptance evidence

Offline unit and integration tests; package wheel install smoke test; seeded A/B/C/D
and strong comparator results; parameter digests and norms; raw episode metrics;
paired differences and seed-cluster intervals; transfer failures; retention under
distribution shift; exact rollback; source secret scan. Results may be negative.
