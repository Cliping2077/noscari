# Noscari — cognitive learning research · Alpha 0.1.0a5

English | [简体中文](README.zh-CN.md)

**Every agent can learn. Every user can teach.**

A continual-learning research prototype that separates teaching records, verified feedback,
and internal parameter updates to test whether experience changes choices on unseen tasks.
Runs on a CPU, with no API key or model download; experiments do not access the network.
V0.2 has been integrated into the existing project. The architecture is named Noscari and
the current model is named Baby. The `axiom-learning-core` package name and `axiom` imports
are retained for compatibility. V0.2 is the project brief version; software versions continue
to advance through the alpha sequence.

**Noscari is a cognitive architecture research project; Baby is its current limited model
and experimental implementation.** The architecture defines the boundaries between experience,
skills, current state, candidates, and external capabilities. Baby implements small trainable
selectors, statistical state, and optional recurrent state; it does not yet implement all the
long-term goals in the brief. Contributions from databases and templates, engineering
primitives, internal learned parameters, and external teachers are recorded separately.
An external model completing a task does not mean Baby has learned it; saving a teaching
record does not count as parameter learning.

This a5 release strengthens release checks, reproduction tools, and research design only.
**It does not change a4's cognitive algorithms or experimental data.**
The research alpha is publicly available as a
[GitHub Release](https://github.com/Cliping2077/noscari/releases/tag/v0.1.0a5); it has not been published to PyPI.
Noscari's software release version is 0.1.0a5. Baby checkpoints are identified by experimental
protocol, training seed, and parameter hash; no separate model-generation scheme is defined yet.
All five jobs in [remote CI](https://github.com/Cliping2077/noscari/actions/runs/39251025287)
passed: Windows/Ubuntu × Python 3.11/3.14 each passed 89 tests; the no-PyTorch job passed
75 tests with 3 skipped. CI does not rerun the full research study.
The [release checklist](docs/github-release.md) retains the local preparation record from
the freeze; see the Release for subsequent remote verification.
The [next-round information-value study design](docs/help-value-design.md) has not entered experimentation.

This is not a complete human-like learning system or AGI. Semantic matching methods and
vocabulary are supplied by the implementation; the model learns when to use existing
methods. It cannot yet create new methods from free-form language teaching.

The detailed historical research reports and supporting documentation linked below are
currently in Chinese. This English README preserves their reported results and limitations.

## Installation and demos

Python 3.11+. A separate virtual environment is recommended. The initial dependency
installation requires network access; demos and tests can run offline afterward.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,torch]"
.venv/Scripts/python -m axiom.examples.teach_menu --output work/demo-1
.venv/Scripts/python -m axiom.examples.recovery --output work/recovery-demo
.venv/Scripts/python -m axiom.examples.motivation --output work/motivation-demo
.venv/Scripts/python -m pytest -q --basetemp work/pytest-1
```

On Linux/macOS, replace `.venv/Scripts/python` with `.venv/bin/python`. Install `.[dev]`
if you do not need the MLP; the base package depends only on NumPy and Pydantic.
The local test environment for this round was Windows, Python 3.14.6, and CPU PyTorch;
the remote CI above has also verified Windows/Ubuntu with Python 3.11/3.14.

Experiment commands refuse to overwrite an existing nonempty output directory. Use a new
output directory for another run.

## Public interface

```python
from axiom import Axiom
from axiom.environments.menu import make_demo

agent = Axiom("work/personal.db")
observation, demonstration = make_demo()
agent.observe(observation)
print(agent.suggest())
event_id = agent.teach(demonstration)  # Store a demonstration without updating weights.
receipt = agent.learn()               # Explicitly train; return before/after summaries and parameter deltas.
agent.save("work/checkpoint.json")
agent.rollback()
agent.close()
```

`feedback(Feedback(...))` submits execution results; `learn()` replays valid, verified
samples; `Axiom.load(checkpoint, new_db)` restores a checkpoint; `correct` / `retract`
correct or retract records; `forget(subject_id)` deletes local records associated with
a teacher and invalidates the old training state.

`verified=True` is **an assertion by a trusted host verifier**. Do not forward this field
directly from a teacher, webpage, or LLM. JSON checkpoints are a trusted local recovery
format, not signed files; load only files from trusted sources.
The replaceable asynchronous `LLMProvider` protocol is in `tools/providers.py`; its default
implementation is fully offline. The current program does not call external LLM APIs.
In the first teacher–learner experiment, the Codex assistant in the development session
acted as an external teacher through a response file. Adapters for real search, browsers,
and commercial APIs have not been implemented.

## Experiments

The latest Cognitive Model Research completed a fixed-protocol study with 5 training seeds.
A 4,078-parameter PyTorch GRU updates persistent state from actual feedback and changes
method selection on new tasks, but its success rate of 82.92% did not exceed the 83.33%
of explicit statistics + MLP. The primary utility difference was −0.00542, and its
95% paired-seed interval crosses zero. **There is no evidence that the new structure is
superior, and negative transfer under scrambled categories remains unresolved.** This
component remains an optional research implementation; the existing stable interface is
unchanged. This is a new task protocol, so its results cannot be compared across protocols
with the earlier 59%→76% result.

All final policies on the same fixed test set are listed below (five training seeds;
1,920 tasks per policy):

| Policy | Success rate | Mean net utility | Help requests / task |
|---|---:|---:|---:|
| Exhaustive traversal | 71.51% | 0.2897 | 0.000 |
| Rule memory | 83.80% | 0.5332 | 0.000 |
| EMA statistics | 83.07% | 0.5221 | 0.245 |
| Contextual linear | 82.92% | 0.5107 | 0.176 |
| MLP without persistent history | 83.33% | 0.5116 | 0.244 |
| Explicit statistics + MLP | 83.33% | 0.5224 | 0.224 |
| GRU | 82.92% | 0.5170 | 0.202 |
| Untrained GRU | 74.06% | 0.3395 | 0.408 |
| GRU state reset for every task | 83.18% | 0.5109 | 0.261 |
| GRU without help requests | 83.85% | 0.5294 | 0.000 |
| GRU without replay of earlier phases | 82.50% | 0.5048 | 0.219 |

The utility interval for the primary comparison, GRU minus explicit statistics + MLP,
is **[−0.01265, 0.00182]**. There is no evidence that Noscari's new structure outperforms
strong, simple methods; the rule-memory result must also be retained. The improvement
when help requests are disabled is a policy ablation and cannot be interpreted directly
as showing that every individual question is harmful. See the full report and raw data
for per-environment results, early/late phases, retention on earlier distributions,
the phase A model, and all parameter changes.

- [Model audit and research rationale](docs/cognitive-research.md)
- [Full results, ablations, and failure analysis](docs/cognitive-report.md)
- [Replayable fixed first-session growth trace](docs/cognitive-growth-trace.md)
- [Cognitive component integration and reproduction commands](docs/cognitive-api.md)
- [GitHub research alpha release notes](docs/github-release.md)

```powershell
python -m axiom.benchmarks.cognitive freeze --output work/cognitive-smoke --smoke
python -m axiom.benchmarks.cognitive train --output work/cognitive-smoke
python -m axiom.benchmarks.cognitive evaluate --output work/cognitive-smoke
python -m axiom.benchmarks.cognitive replay --output work/cognitive-smoke
```

Requires `.[dev,torch]`. Remove `--smoke` to run the full fixed study; the small demo
does not support performance conclusions. Current weights/state and raw replay records
can all be saved and restored independently of an external LLM.

The first real teacher–learner interaction has been recorded: a pretrained recovery
selector issued a request, the Codex assistant in the development session supplied one
suggestion, and the original Axiom selection module was trained only after the simulator
executed and verified it. Across 300 related tasks sealed in advance, success rose from
59% → 76%. The rule-memory baseline also achieved 76%, and scrambled categories produced
negative transfer. The learner asked before attempting the task, and the teaching example
could already be solved by the first traversal step. This therefore does not yet show
that it can accurately determine when help is necessary. The
[full teacher–learner experiment report](docs/first-teacher-report.md) retains these limitations.

See the [teacher interface](docs/teacher-bridge.md) for instructions on human teachers and
recorded-response replay. There is no persistent connection to Codex: the first response
was supplied after the development session read the request. Standalone operation requires
a human response file or an optional API adapter to be implemented in the future.
Templates provide the wording; parameters govern the current help-seeking choice.

The second Core Lab round added stepwise recovery. On the shared set of semantic-failure
cases, learned recovery succeeded in 91.39% of cases versus 85.74% for fixed traversal
fallback. However, removing dynamic scoring features still achieved 91.46%, so there is
no evidence of additional benefit from those dynamic features. The
[round-two report](docs/round2-report.md) retains this negative result and the strong-baseline comparison.

Frontier Lab's minimal motivation module generates template questions from observations,
retains goals, and explicitly trains exploration preferences. In the interest experiment,
learners with identical initial states diverged with experience and revised their
preferences after conditions were reversed; independent testing measured an increase in
predicted progress. A historical-mean baseline performed equally well in the first phase.
This is learning to estimate under fixed goal definitions. Self-created rewards, free-form
questioning, subjective curiosity, and validation of consciousness have not been implemented.
See the [V0.2 report](docs/v02-report.md).

```powershell
python -m axiom.benchmarks.run --output work/benchmark-1 --seeds 5 --train-tasks 240 --tasks-per-cell 20
python -m axiom.benchmarks.retention --output work/retention-1
python -m axiom.benchmarks.recovery --output work/recovery-1 --seeds 5 --train-tasks 480 --tasks-per-cell 30
python -m axiom.frontier.interest --output work/interest-1 --seeds 5 --train-per-topic 24 --eval-tasks 60
```

Without PyTorch, add `--without-mlp` to the first command. The recovery experiment also
supports `--without-mlp`; the motivation experiment does not depend on PyTorch.

See the [formal addendum](docs/brief-v02.md) and [V0.2 plan](docs/v02-plan.md) for the
research direction and explicit lists of unimplemented work. The two experiment
specifications are in [Core Lab](experiments/core/README.md) and
[Frontier Lab](experiments/frontier/README.md).

Experiments include A: serial execution; B: memory retrieval with the same experience;
C: a fixed set of multiple candidates; D: AXIOM; LinUCB/MLP controls; and three ablations:
frozen parameters, disabled association, and disabled help requests. The six categories
belong to a fixed toy semantic vocabulary. Tests use new feature names and random layouts;
they do not demonstrate transfer to unseen concepts.

The early first- and second-round environments exposed simulated teacher/search
reliability features, a simplified information setting. This does not demonstrate that
Baby inferred reliability on its own. The new cognitive-v1 protocol strictly hides these
ground truths: the learner can use only observations and feedback from the selected
action. Results from different protocols cannot be joined into a single "growth curve."

See the [reproduction guide](docs/reproducibility.md) for full retraining reproduction,
replay of historical teacher responses, parameter comparisons, and table generation.
a5 provides a one-command check, and all reference data are retained in the source archive.

See the [round-one experiment report](docs/round1-report.md),
[research distinctions](docs/research.md), [protocol and metrics](docs/experiment-protocol.md),
and [security and data](SECURITY.md). Raw records are in `experiments/`. Do not cite only
the README summary while omitting failure results.

## Directory layout

```text
src/axiom/
  core/          Public models, persistent state, candidate lists, explicit learning API
  memory/        Authoritative SQLite event records, JSONL import/export
  learning/      Linear ranker, LinUCB, small PyTorch MLP
  environments/  Offline simulator with hidden true menu locations
  benchmarks/    Controls, ablations, distribution shifts, and retention tests
  examples/      Single-teaching demonstration
  frontier/      Separately enabled minimal motivation module and interest experiments
  tools/         Optional capability-provider protocols
tests/ docs/ experiments/ scripts/
```

See the [design](docs/design.md) for detailed boundaries. Current candidates are finite
macro-actions, not parallel thinking modules. Skill versions are linked to outcomes,
but the default selector uses only the latest semantic version. Older versions are
retained for audit; automatic composition, reactivation of older semantic versions,
and general long-term subgoal planning have not been implemented. Frontier already has
limited persistent question state, but it is not yet connected to the core menu decision
module. MIT licensed. The GitHub research alpha is public; it has not been published to PyPI.
