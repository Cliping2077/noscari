# Running the first-teacher bridge

Run from an installed source checkout. The example uses a supplied trusted Round 2
checkpoint. The source archive includes it; the wheel alone does not bundle research
data. The checkpoint determines whether to ask; an untrained selector may never ask.

```powershell
python -m axiom.examples.first_teacher prepare --output work/my-lesson --selector-checkpoint experiments/round2/seed-0-clean/checkpoint.json
```

If the result is `pending_teacher`, read only `work/my-lesson/request.json` while
acting as teacher. It contains visible context, current goal/identity, tried menus,
available candidates, learned scores, and a template-rendered question. The private
test fixture is sealed before the request. Do not inspect it before answering.

Write a UTF-8 JSON reply with these **three fields only**:

```json
{
  "request_id": "copy-the-actual-request-id",
  "rule": {"audio": "a visible menu label in lowercase"},
  "explanation": "A limited explanation or suggestion for the learner to test."
}
```

The rule is a category-to-label suggestion, not a menu ID/path or executable code.
The example above is a schema illustration, not a runnable answer. Use the actual
request and a label it displays. The explanation is preserved as text; the prototype
does not parse free prose into new algorithms. It executes the single declared rule
through an existing semantic method. Do not include verification or authorization.

```powershell
python -m axiom.examples.first_teacher resume --output work/my-lesson --reply work/my-reply.json --teacher-source manual-human-teacher
python -m axiom.examples.first_teacher evaluate --output work/my-lesson
```

The host executes the suggestion, verifies the observed outcome, records it and
explicitly learns. Invalid advice is retained as a failed lesson without skill or
parameter promotion. Search is absent; new-task evaluation disables teacher access.
Evaluation is frozen and refuses changed fixture/checkpoint hashes. A lesson with
no accepted teaching is recorded as failure and cannot generate a fabricated
transfer result.

For recorded replay, copy the previously published `teacher-reply.json`, replace
only its request ID with the new request ID, and use
`--teacher-source recorded-teacher-replay`. This reproduces a known response; it
is not another live or blinded teacher experiment. Public test fixtures are public
after this initial run; a future confirmatory study requires a fresh preregistered
namespace and an independent teacher/evaluator.

If a teacher never responds, the request remains pending with one reserved budget
unit. It sends no messages and performs no further work. This is an active-session
or manual bridge, not a free background connection to an assistant service.

If interrupted after `resume-started.json`, the session is locked and refuses
replacement replies. Keep all files for diagnosis. Automatic crash recovery is not
implemented; do not delete the marker or silently rerun a failed live session.
Create a separately labelled recovery attempt from preserved evidence if needed.
The pending and completed files are trusted local records, not signed attestations.

The real first request/reply, parameter receipts, failed controls and transfer logs
are in `experiments/core/first-live-teacher/` and
`experiments/core/first-teacher-controls/`. Controls are explicitly synthetic or
post-hoc, not extra live teaching successes.
