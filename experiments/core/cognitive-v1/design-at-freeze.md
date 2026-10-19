# Cognitive Model Research — audit and bounded design

2026-10-19. This is a new experimental route within the existing package, not a
replacement of the earlier core. The user authorizes implementation without another
routine design approval. All alpha 1–3 artifacts remain immutable.

## What currently learns, and what does not

| Component | Actual mechanism | Boundary |
|---|---|---|
| Axiom selector | NumPy ridge return regression, 40 coefficients | Hand-engineered features and fixed method executors; no learned language |
| Recovery selector | Linear/LinUCB/PyTorch MLP on 20 supplied features | No persistent learned state across tasks; old observation exposes configured teacher reliability |
| Semantic skill | Stored category-to-label mapping and fixed alias matcher | Teaching supplies a rule; parameters mainly change whether to use it |
| Experience/skill records | SQLite events, versions, checkpoints | Persistence is not neural learning; latest active semantic rule is used, not arbitrary skill discovery |
| Candidate workspace | Recomputed lists and scores | No independent parallel cognition or learned scheduler |
| Teacher bridge | Trained ask score; template question; external assistant reply | Single live lesson did not train the ask selector; easy task did not establish calibrated need |
| Motivation | Verified progress EMA and persistent goal journal | Fixed progress definition and template-generated opportunities; not learned intrinsic objectives |
| External LLM | Optional protocol; manual assistant teacher in alpha 3 | Does not supply the learning core's weights; no autonomous API integration |

The 59%→76% result is preserved; direct rule memory also achieved 76%, while
scrambled layouts declined 63%→58%. Parameter movement is insufficient evidence
of improvement. Dynamic recovery features previously failed to improve results.

## Research map and limits of the analogy

- **Complementary Learning Systems:** separate fast experience capture from slower
  integration. This motivates distinct logs, fast activation state and replay-trained
  weights; it does not establish biological equivalence. [McClelland et al., 1995](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf).
- **Continual learning / stability–plasticity:** replay can preserve old experience,
  while consolidation penalties restrict parameter changes. Neither guarantees
  retention under arbitrary distribution shifts or bounded memory. We use replay,
  with a no-replay control; EWC is deferred. [CLEAR](https://papers.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html),
  [EWC](https://doi.org/10.1073/pnas.1611835114).
- **Episodic memory / associative retrieval:** key-value episodic value memories
  can reuse particular experiences quickly; metric quality and memory growth matter.
  We preserve raw episodes but do not add learned episodic retrieval this round.
  [Neural Episodic Control](https://proceedings.mlr.press/v70/pritzel17a.html).
- **Recurrent persistent state:** recurrent controllers can combine incomplete
  observations over time. This is the selected mechanism; recurrence may be harder
  to optimize than explicit statistics and need not outperform them.
  [DRQN](https://arxiv.org/abs/1507.06527).
- **Meta-learning:** RL² provides the relevant distinction between fast recurrent
  activations and slow learned parameters. Our masked Monte Carlo regression is
  not a replication of RL² policy-gradient training. MAML is an alternative that
  adapts parameters directly, but adds inner-loop optimization we do not need yet.
  [RL²](https://arxiv.org/abs/1611.02779), [MAML](https://proceedings.mlr.press/v70/finn17a.html).
- **Hierarchical RL:** reusable options separate skill execution from selection;
  this round retains engineered skills, not learned new options or termination.
  [Sutton et al., 1999](https://www.sciencedirect.com/science/article/pii/S0004370299000521).
- **Active information seeking:** delegation must account for other decision-makers'
  errors and cost. Hidden teacher quality is inferred only from observed results;
  asking here is a simulated information action, not a real conversational teacher.
  [Learning to defer](https://arxiv.org/abs/1711.06664).
- **Intrinsic motivation:** prediction error or information gain can encourage
  exploration but can reward noise and inherits a chosen objective. No new curiosity
  mechanism is added. [VIME](https://proceedings.neurips.cc/paper_files/paper/2016/hash/abd815286ba1007abfbb8415b83ae2cf-Abstract.html),
  [ICM](https://arxiv.org/abs/1705.05363).
- **Dynamic architectures:** expansion can add capacity, with capacity/control
  costs; it does not address missing evidence by itself. We defer growth until a
  fixed lightweight model has an identified capacity failure. [DEN](https://arxiv.org/abs/1708.01547).

These are computational motivations, not claims of new learning principles or
subjective experience. A trained outcome head is an action-success estimate,
not a complete self-model. Identity and autonomous goal creation are not expanded.

## Design choice

Alternatives: (1) explicit EMA statistics + context policy, interpretable and cheap;
(2) GRU feedback state + shared action-value/success heads, learned history summary;
(3) attention over all episodes, flexible but more compute and more confounds.
Implement (2) as the hypothesis and (1) as the strong comparator; defer (3).

Only actual feedback changes the recurrent state. Only explicit replay training
changes weights. JSONL episodes remain separate from both. GRU state persists
within a software session across tasks; new sessions reset it. State is not treated
as a semantic explanation of what the agent believes.

The environment owns private layouts, teacher accuracy/willingness and change
points. The learner-facing schema omits all those fields and environment labels.
It receives visible menu labels, task category, tried menus, current goal, budget,
permission, cost and feedback from its chosen action. No private task IDs are used
as features. Exact menu matching, traversal and asking are supplied capabilities.

Every step ranks all currently executable candidates, including semantic selection
before any failure. Permission masks only enforce availability/budget, never hidden
regime. This is interleaved selection, not simultaneous tool execution.

## Fixed experiment protocol, before evaluation

- Five independent training seeds 0–4. Separate train/test/development namespaces.
- Per seed: phase A 16 sessions of reliable categories; phase B 48 sessions across
  unreliable, partial and changing layouts. 16 tasks per training session.
- Final test: 16 new sessions, four per reliable/unreliable/partial/changing regime,
  24 tasks each. Category mapping and layout are randomized; reliable probability
  .9, unreliable .1. Partial regimes reverse by visible category. Changing regimes
  swap halfway without a visible change flag. This remains a finite toy family.
- Teacher accuracy .9/.4, answer probability .9/.5, cost 1/3, permission enabled/
  disabled; these factors vary independently of layout type. Budget 4/6; 6–8 menus.
- Random behavior collects only chosen-action transitions. All trainable methods
  receive the same recorded episodes, with no counterfactual/oracle labels. Reward
  is observed completion minus 0.6 × consumed/budget. Targets are within-task
  Monte Carlo returns under this behavior, not optimal Q labels.
- Model: GRU 24 state units over feedback tokens, 32-unit tanh heads predicting
  return and immediate success. Adam .003, 60 epochs A, 100 epochs B with A replay;
  MSE return + .25 BCE success. Fixed architecture/config; no test tuning.
  Before any full training/evaluation, baseline widths are set to match parameter
  budgets: GRU 4,078; stateless MLP (width 204) 4,086; statistics MLP (width 128)
  4,102. Same data/epochs; recurrent computation still has different wall time.
- Baselines: traversal; rule memory; EMA-greedy; linear context; stateless MLP;
  explicit-statistics MLP. Controls: untrained GRU; reset GRU at each task;
  no-help GRU; GRU trained on B without A replay.
- Evaluation freezes weights but permits causal state/statistics updates from
  each policy's own feedback. Same test tasks and budgets, 5% fixed exploration
  for learned/statistical policies; it is not learned intrinsic motivation.
- Primary: paired per-seed mean utility versus explicit-statistics MLP. Secondary:
  success, operations, help, wrong semantic attempts, exploration cost, phase-A
  retention, early/late task adaptation, outcome calibration and choice changes.
- Seal test cases and source/protocol hashes before training/evaluation. Development
  smoke tests may test execution only; no performance-driven retuning. Publish all
  seed results, failures, training loss, checkpoints and causal decision traces.
- An advantage claim requires positive mean utility difference with a paired
  seed-level 95% t interval above zero; five seeds remain limited evidence. No
  architecture advantage if it merely matches the simple baseline. A failure does
  not trigger additional hard-coded regime rules or further architectures this round.

## Implementation sequence

1. Add failing tests for public/private separation, causal feedback, budget masks
   and deterministic replay. Implement `environments/contextual.py` using the same
   bounded menu semantics, with a public schema that excludes hidden truth.
2. Add tests for feedback-vs-parameter updates, save/load, rollback, causality and
   unverified-data rejection. Implement `learning/cognitive.py` and its statistics
   comparators. Torch stays an optional extra and is not imported by the base API.
3. Add `benchmarks/cognitive.py`: freeze, collect, train, evaluate, replay commands;
   persist the initial/A/B parameters, raw episodes and every evaluation decision.
4. Read-only code review before final evaluation; run the full suite. Conduct the
   fixed five-seed run, analyze without retuning, preserve adverse results.
5. Update README/API/release documentation; build and test an isolated wheel,
   scan release files, verify prior archive hashes, package alpha 4. Local GitHub
   readiness is distinct from actual remote publication or remote CI execution.
