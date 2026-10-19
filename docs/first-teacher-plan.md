# First live teacher interaction — bounded pilot

Client date: 2026-10-18. User asks Codex to act as Baby's first live teacher.
This is an incremental pilot on the existing core, not a new architecture.

The user's subsequent teacher-experiment instruction also applies: do not take over
execution, do not volunteer all solutions, preserve failures, freeze independent
tests and distinguish actual autonomous choice from forced questioning. Future
retrieval and an auditable self-model remain roadmap items. This pilot exposes only
identity, current task, acquired skill availability, attempts and resource limits;
it does not claim learned self-awareness.

## Design and publication decision

Recommended path: a local request/reply bridge. Baby's existing trained recovery
selector can choose `ask` while search is unavailable. It emits the visible task,
attempts and scores to a pending request. The Codex assistant reads that request
in this active conversation and supplies a declarative suggestion. No commercial
API or independent background assistant is needed. A human can use the same files.
Alternatives: an API adapter enables unattended teachers but adds cost; simulated
teachers are reproducible but cannot establish that this live bridge works.

Use a new menu family with ordinary English labels outside the built-in lexicon.
The fixed teaching task asks about audio processing. The first request is generated
by the first clean seed-0 recovery checkpoint from Round 2, selected in advance.
No hand-written condition forces `ask`; if it does not ask, report that outcome.
The request must never include hidden target placement or a verified teacher answer.
The question renderer is a template; choosing to ask is the trained part.

The live lesson uses seed 0 and four operation units; development tests use seed
101 and a distinct heldout namespace. Before emitting the live question, serialize
300 private test cases and seal their SHA-256 into the pending-session record.
The teacher reads only request.json. Resume and evaluation reject modified test
fixtures. No live test metric is inspected until after the reply and learning are
recorded. This is a procedural separation within one assistant, not independent
human/developer/teacher blinding. Do not claim otherwise.

The teacher returns only request ID, explanation and category-to-visible-label rule.
No executable code, permissions or `verified` flag is accepted. The host tries the
proposed semantic action in the simulator. Only success produces a verified teaching
record; only an explicit `learn` updates AXIOM weights. A failed suggestion is logged,
not promoted into a stable skill. Existing traversal remains available.

For this pilot, the recovery chooser is already trained on synthetic interaction,
and is not retrained by the live lesson. The lesson updates the original Axiom
skill/selection subsystem. The modules are not yet a single end-to-end controller.
This is the first live assistant teacher, not the first source of simulated teaching.

## Deliverables and fixed checks

1. Typed request/reply bridge, new bounded menu family, prepare/resume CLI.
2. Tests for unavailable search, a learned `ask` decision, permission/budget masks,
   reply binding, forbidden fields, failed-advice rejection, recording vs learning,
   retained traversal and checkpoint restoration. First observe missing-module or
   failing regressions; then implement and run the whole suite.
3. Actually read the emitted request and author a teacher reply in this conversation.
   Preserve request, reply, provenance, execution trace and before/after parameters.
4. Frozen evaluation: 5 task-seed groups × 3 layouts × 20 tasks, using new task
   namespaces/orders. Compare traversal, same-record frozen selection, a direct
   rule-memory baseline, and the trained selector. Disable teacher and search in
   evaluation. The groups are task variation, not five independent live teachers;
   no five-seed confidence claim about teacher effectiveness is permitted.
5. Report success, operations, wrong transfer, help requests, and exact rollback.
   A simple rule-memory baseline may match the learner; report that honestly.

Implementation files: `tools/teacher_bridge.py`, `environments/classroom.py`,
`examples/first_teacher.py`, `tests/test_first_teacher.py`. Reuse core models,
learning, memory and existing comparisons. No new dependencies, UI or daemon.

## GitHub research-alpha gate

The existing package, license, tests and raw baselines already make a research
repository useful. For a first public teaching demo, additionally require:

- A fresh clone installs and runs without a key; documented live/manual teacher
  flow and recorded replay are clearly distinguished.
- The emitted question precedes the recorded reply; verified execution precedes
  parameter learning; new-task evaluation has no answer-path leakage.
- Wrong teacher/teacher absent/authorization denied cases are tested and logged.
- Same-information baselines, frozen parameters, rollback and old-method retention
  are visible. Publish negative results and limitations alongside the demonstration.
- Inspect the proposed release files for credentials and private data, run remote
  CI after a repository exists, and publish only with explicit user authorization.

These are our engineering/research gates, not GitHub platform requirements. A useful
alpha need not outperform every baseline or have human-level learning. It must make
its narrow behavior reproducible and its claims falsifiable. This pilot alone does
not establish general knowledge acquisition, learned question language, a unified
autonomous agent, or subjective experience.
