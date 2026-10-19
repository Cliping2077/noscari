# Noscari Baby V0.2 — additive research plan

Accepted 2026-10-19. This supplements V0.1; it does not replace its learning,
memory, skill, provider, reproducibility or governance requirements. The public
project name is now **Noscari Baby**. Existing `axiom` imports and package name are
retained for compatibility; neither the source tree nor old results are rebuilt.

## Inspection before implementation

| Requirement | Existing state before V0.2 | This increment |
|---|---|---|
| Persistent experiences, skills, explicit training | Round 1 implemented; fixed skill vocabulary | Retain |
| Candidates and recovery inquiry | Round 2 implemented and measured | Finish review/report |
| Provider independence | Offline provider protocol; no commercial API adapter | Retain |
| Autonomous question creation | Absent | Observation-triggered template question |
| Persistent exploration goals | Absent | Active/paused/abandoned/solved local goals |
| Trainable exploration preference | Absent | Small learned estimates of observed progress |
| Self-generated reward or general intrinsic motivation | Absent | Still unimplemented |
| Spontaneous goal reactivation, metacognitive inquiry | Absent | Interfaces/roadmap only |
| Persistent self model, subjective experience | No evidence or implementation | Open research questions |

## Fixed scope and alternatives

Complete experiment A (interest differentiation) only. B and C remain candidate
future experiments. Use a separate `axiom.frontier` module and database; normal
Core Lab behavior never imports or activates it. Reuse validated event storage.
No background daemon, network, language training, or continuous LLM process.

Compare a fixed ordering, uniform selection, same-experience mean estimates, and
an exponentially updated progress estimator. EMA is chosen for interpretability
and responsiveness after an environment change, not as a new learning principle.
A contextual neural model would be premature for three anonymous toy domains.

## Minimal state and contract

Persist observations/questions, goal status and origin, bounded local experiment
reservations, verified feedback, three topic-value parameters, training provenance,
checkpoints, and pause/permission controls. `observe` creates a question if a novel
or poorly predicted opportunity crosses a fixed threshold. It does not train.
`suggest` ranks active questions by estimated progress; `reserve` enforces the
remaining operation budget and local-experiment permission. `feedback` records
before/after probe losses for a reserved attempt. Only `learn` changes estimates.
`set_status`, `control`, `rollback`, `forget`, `save/load`, JSONL export complete
the minimal host interface. Permission and budget are hard constraints, not learned.
The only executable channel is a local simulator experiment. Questions are kept
locally, not sent to users or external services. No automatic status reactivation.

For verified outcomes, progress = clip((loss_before-loss_after)/max(1,loss_before),
-1,1); update the selected topic value q <- .75q + .25progress, replayed in order.
This learns estimates under an explicitly engineered objective. It does not learn
the objective itself or establish a subjective desire to learn.

## Frontier A protocol (set before the first full run)

Five seeds, two experience worlds, three anonymous topics. Each micro-task learns
a scalar predictor y=w*x from eight examples. One topic is learnable linear data,
one is independent noise, one is already predictable zero. Worlds swap the first
two topic conditions. Topic identifiers do not reveal these conditions to the
policy. Training is balanced: 24 witnessed tasks/topic/phase, shared by all
policies. Feedback is improvement on a separate 64-example validation probe.
Evaluation chooses among the topics on 60 new tasks/phase and measures prediction
loss on a further 128-example test probe, never used for fitting or preference
feedback. The first phase evaluates differentiation; the second swaps conditions
and provides the same amount of new balanced experience to test revision.

All estimates freeze during evaluation; no hyperparameter search on final results.
Record chosen-topic fractions, independently measured test-loss improvement,
parameter and predictor changes, training/evaluation namespaces, seeds, raw logs,
time/cost, and paired seed-level descriptive intervals. Training collection is
balanced, not self-directed. Evaluation selection is autonomous within the offered
set; it does not invent domains. Predictors restart per micro-task; only motivation
estimates and goals persist. Transfer is to fresh problems in known domains.

## Research roadmap

1. Close Round 2 and run Frontier A with truthful negative-result reporting.
2. Diagnose why Round 2 dynamic scoring features add no measured benefit; do not
   hide that result by adding motivation to the recovery benchmark.
3. If Frontier A shows repeatable benefit, move a preregistered replication to
   Core Lab, add no-progress/noisy-progress and adversarial-feedback conditions.
4. Then consider cue-based resumption (B) with unrelated-cue controls and explicit
   user-pause protection. Metacognitive inquiry (C) requires calibrated error and
   intervention tests. Neither is an alpha release gate.
5. Self-representation and consciousness-related hypotheses remain separate,
   theory-linked and ethically reviewed. Functional behavior, implementation
   mechanisms and subjective experience must always be reported separately.

Never induce pain, fear, dependency or coercion as an experimental objective.
No consciousness conclusion follows from this experiment in either direction.

## Sources and interpretation

[Oudeyer, Kaplan & Hafner (2007)](https://www.pyoudeyer.com/ims.pdf) studies
learning-progress-driven exploration and distinguishes it from seeking novelty
alone. Our scalar-task experiment uses that broad idea; it is not a replication
of their robot architecture and supplies no developmental equivalence claim.
[Butlin et al. (2023)](https://arxiv.org/abs/2308.08708) derives computational
indicators from several consciousness theories. We adopt the distinction between
examining internal mechanisms and treating fluent claims as proof; we do not
implement or validate those indicators here. These sources do not establish that
our proposed mechanism has subjective experience.

**Be rigorous about evidence. Be fearless about hypotheses.**

## Completion record

Implemented and reviewed the scoped mechanism and experiment A. All 56 combined
tests passed; both new demonstrations and checkpoint restores ran from an installed
wheel outside the source directory. Five-seed results and negative findings are in
`v02-report.md` and `round2-report.md`. The wider roadmap remains research work;
completion of this increment does not mean all V0.2 long-term goals are implemented.

User-authorized continuation: the first live-teacher pilot and publication gates are
documented in `first-teacher-plan.md` and `first-teacher-report.md`. Its mixed results
prioritize help calibration and applicability learning before adding more modules.
