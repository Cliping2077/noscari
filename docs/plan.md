# Round 1 implementation plan

**Goal:** an installable, offline learning core with falsifiable menu experiments.
**Architecture:** event-backed state and explicit replay learning; bounded method
ranking, isolated simulator, reproducible paired benchmarks.
**Stack:** Python >=3.11, NumPy, Pydantic, SQLite, pytest; optional PyTorch MLP.

## Global constraints

No commercial API calls, no private uploads, no external publishing. Correct paths
stay inside the simulator. Records are not parameter updates. One-demo results and
multi-experience results must be labeled separately. Test seeds and names are held
out. User authorization overrides policy scores. CPU-first, bounded experiments.

## Steps

- [x] 1. Read primary papers and code; write research comparison with source links.
- [x] 2. Write failing contract tests for public observations, deterministic budgets,
  method retention, feedback gating, learned behavior, replay, save/load and deletion.
- [x] 3. Implement `models.py`, `menu.py`, `policies.py`, `store.py`, `agent.py`.
  Public signatures: `observe(Observation)`, `suggest()->list[Candidate]`,
  `teach(Demonstration)->str`, `feedback(Feedback)->str`, `learn()->LearningReceipt`,
  `save(path)`, `load(path, store_path)`. Run `python -m pytest -q`.
- [x] 4. Add tested baselines/ablations and paired benchmark runner. Store per-episode
  rows, configuration, training data digests, parameters, aggregate and seed results.
- [x] 5. Run one-demo transfer and distribution-shift/retention probes. Record failures
  and bounded claims. Review source and scientific validity independently.
- [x] 6. Add README, MIT license, contribution/testing/security docs and CI. Build
  sdist/wheel, install the wheel in an isolated target and run demo outside source.
- [x] 7. Run final tests and secret scan, freeze exact executed dependencies, package
  source and deliver experiment report with capabilities, omissions and next steps.

Implementation is tightly coupled and proceeds inline. Independent code review uses
the requesting-code-review skill. No repository or upstream remote exists initially.
