# Testing and reproducibility

```powershell
python -m pytest -q --basetemp work/pytest-run
python scripts/scan_secrets.py .
python -m build --no-isolation
```

Use a dedicated temporary directory. In the development sandbox Windows' default
temporary folder was not writable, so runs used `../../work/pytest-*`.
The Windows Python event loop also blocked during its loopback self-pipe setup in
the network-denied sandbox. The no-I/O provider stub test drives its coroutine
directly; a real asynchronous network provider integration has not been tested.

Tests cover observation visibility, all four layout regimes, enforced budgets,
one-demo parameter/action change, recording/learning separation, unverified data,
parameter freezing, authority filtering, save/load/reopen/JSONL replay, exact rollback,
corrections, deletion, faulty teaching, deterministic comparators and equal evaluation
tasks. A socket guard checks the full demo stays offline. MLP test skips explicitly
when optional torch is absent. CI installs torch, so it should run there.

Source checks only detect common credential patterns; a clean scan is not a proof
that arbitrary data is non-sensitive. Never commit private experiment logs.

The checked-in experiment configuration is fixed. To reproduce use README commands
with a fresh output directory; timing, UUID event identifiers and provenance hashes
containing those IDs can differ. Core parameter values, task identities and outcome
aggregates are deterministic on the tested stack. Cross-version floating point
rounding can differ. See `requirements-executed.txt` for this run's exact versions.

## Alpha 2 / V0.2 additions

`experiments/tests-a2.txt` records 56 passing tests on the local stack. New checks
cover stepwise recovery, immutable persisted feature configuration, impossible
transition rejection, common-prefix comparisons, offline motivation demonstration,
strict verified-feedback separation, persistent goals, budget/permission controls,
privacy deletion without budget refund, replay validation and isolated evaluation.
The prior alpha's records remain unchanged. See `experiments/development-failures.md`.

Run the new examples and fixed protocols with the README commands. Source-only runs
need `PYTHONPATH=src` or an editable install. Installed-wheel smoke verification runs
outside this source tree and exercises both new examples plus checkpoint restore.
The exact commands, results and artifact/source hashes are in `verification-a2.json`
and `source-manifest-a2.json` under `experiments/`.

## Alpha 3 first-teacher pilot

The combined suite has 66 tests. The new checks include learned versus untrained
question selection, denied search/teacher and exhausted budget, forbidden teacher
fields, wrong request IDs, unsuccessful-advice rejection, frozen fixture and
checkpoint tampering, and an interruption after committed learning. The interrupted
session is locked and cannot replace its original reply. That last defect and the
checkpoint-swap gap were reproduced before the live experiment and then fixed.
Logs are `experiments/tests-a3.txt` and `verification-a3.json`. No remote CI run or
independent human-teacher replication is claimed.

## Alpha 4 cognitive experiment

The combined suite now has 83 tests. New checks cover strict public/private schema
separation, permissions and teacher refusal, causal batch/online agreement,
experience versus activation versus weight changes, parameter-matched baselines,
rollback/load, invalid and duplicate evidence, modified fixtures/configuration,
trace coverage and exploration/action replay. Each review defect was reproduced
by a failing regression before correction and before freezing the study.

Run the four cognitive CLI commands in README with `--smoke` on freeze for a small
execution check. The full protocol in `docs/cognitive-research.md` uses five seeds
and keeps test weights frozen. A source hash mismatch intentionally stops reuse of
an old study directory; retain it and start a separate version. A completed or
interrupted train/evaluate run refuses silent overwrite. The installed wheel's
experiment can generate its own fixtures and does not require bundled research data.

The shipped formal GRU trajectory has 4,595 decisions checked against its sealed
environment, all candidate values, action selection, feedback states and final
saved state. `replay` also requires complete session/task/step coverage and checks
all evaluation artifact hashes. The existing verification output is immutable;
to independently replay shipped data, copy the run to a new folder and move the
existing `replay-verification.json` aside in that copy before running `replay`.
Floating-point exact replay is verified on the recorded CPU/runtime; cross-platform
numerical equivalence has not been established. CI configuration is supplied, but
no remote CI result is claimed without a real GitHub run.

## Alpha 5 publication preparation

The release adds five tests for nested archive credentials/privacy, corrupt archives,
environment files, numerical/parameter regression comparison and nonfinite values.
See `docs/reproducibility.md` for a fresh-venv install, complete experiment reproduction,
installed-wheel smoke, exact parameter checks and all historical scan inputs.
No cognitive model algorithm is changed. `docs/help-value-design.md` is design only.
The remote CI matrix remains unexecuted until a repository is authorized.
