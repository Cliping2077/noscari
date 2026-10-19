# Experiment protocol

固定配置在首次完整运行前确定；最终结果不据此调参。

## Stages and information budget

One-demo: one witnessed successful file-category example; teach registers a fixed
semantic method, learn fits its reward. Compare identical skill availability with
untrained versus trained weights. 4 regimes × 3 help conditions × 20 heldout tasks.

Main: 5 independent training seeds × 2 teacher conditions. Each learner gets the
same demonstration and 240 uniformly randomized **chosen-method** outcomes in its
seed/teacher condition. These are logged offline exploration, not autonomous
AXIOM training. No unchosen method's counterfactual reward is provided. Teacher
partial_error maps image to file while the demonstrated file-category case works.
The demonstration cannot reveal that error; outcomes on later image tasks can.

Each policy is evaluated on 4 regimes × 3 help conditions × 20 tasks per seed and
teacher: 2,400 episodes/policy, 21,600 total with all nine policies. Identical layouts,
budgets and pre-sampled noisy help outcomes are shared between policies. A serial
baseline discards cross-task information by definition; B/C/D receive equal logs,
though C never updates its fixed scoring rule. No neural language model in any arm.

Training/test/demo use separate random namespaces and disjoint function names.
Only the simulator sees actual placement. Predictors receive 10 fixed features
of visible context; not task IDs, seeds, regimes, item placement or target paths.
Counterfactual evaluation across policies happens inside the evaluator only.

## Methods and limitations

Methods are macro-actions: traverse; semantic guess then traverse; search hint then
traverse; teacher hint then traverse. Only the initial macro-action is learned.
Fallback traversal is engineered, not a learned recovery strategy. All policies
can use the same executable methods (A retains only traversal). One root opening
reveals the target's presence in that submenu at unit cost; leaf clicks are omitted.

AXIOM: disjoint linear ridge regression with ridge=1 and a fixed traversal prior.
LinUCB: same regression plus uncertainty bonus alpha=0.25, also used at evaluation.
MLP: 10→16 tanh→4, CPU PyTorch, masked MSE on chosen actions, Adam lr=0.02,
weight_decay=0.01, 150 epochs, no hyperparameter sweep. B: fixed k=8 inverse-distance
nearest-outcome retrieval per method. C: published fixed cost-aware semantic heuristic.

No-association removes the match feature and semantic candidate. It therefore mixes
representation and candidate-availability ablation; it is not a pure neural recall
ablation. No-inquiry removes search/ask candidates. Frozen retains the same skills,
observation and stored experiences, but keeps initial weights. Evaluation never fits.

## Metrics

Success is finding the requested item before the total operation budget. Operations
include menu opens and help costs. Utility = success − (0.6 + 0.2 × visible risk) ×
operations / budget. This arbitrary predeclared utility is not a universal preference.
Teacher/search counts are simulated interactions, not commercial API requests.

New-method selection = choosing semantic macro-action. Semantic attempt = making
a semantic menu guess (renamed/unknown labels can prevent it). Wrong transfer =
semantic first guess misses, regardless of eventual fallback success. Report both
per-episode frequency and conditional frequency among semantic attempts.
Fallback usage is behavioral evidence; old-method availability is separately tested.

Aggregate success, operations, utility, help counts and per-condition breakdowns are
reported. API calls/dollars are zero. Wall time and per-selection CPU latency are
measured; these are machine-specific and exclude engineering/research time, electricity
and this development chat. No FLOP or GPU-energy estimate is asserted.

Paired differences average within seed across all conditions. 95% intervals use
Student-t across the 5 seed means (4 degrees of freedom), not 2,400 independent
episodes. Small seed count and shared vocabulary limit inference. No adjusted
multiple-comparison significance or broad superiority claim is made.

Retention probe: 120 semantic-layout training episodes, then 120 scrambled-layout
episodes; compare old model, latest-only refit and complete replay on fixed heldout
semantic and scrambled tasks. This is a narrow distribution-shift check.
