# Experimental cognitive selector integration

Install `axiom-learning-core[torch]`. The original `Axiom` API remains available;
`CognitiveSelector` is a separate experimental selector, not a replacement for all
memory, motivation, tools or goal management. It currently supports the two-category
menu observation schema. Arbitrary Agent tasks need their own tested adapter.

```python
from axiom.learning.cognitive import CognitiveSelector
from axiom.environments.contextual import ContextMenu, make_session

baby = CognitiveSelector(kind="gru", seed=0)
session_evidence = []
# The host owns the simulator and its private data. Only public observations and
# verified, actually executed feedback are submitted to the learner.
for private_case in make_session("integration-example", 0, "partial", tasks=4):
    env = ContextMenu(private_case)
    while not env.observe().done:
        baby.observe(env.observe())
        candidates = baby.suggest()
        event = env.step(candidates[0]["method"])
        baby.feedback(event)
        session_evidence.append(event)
receipt = baby.learn([session_evidence], epochs=4)
baby.save("baby-state.json")
restored = CognitiveSelector.load("baby-state.json")
assert restored.parameter_digest == baby.parameter_digest
```

This snippet demonstrates interfaces, not trained performance. A few epochs on
four tasks do not establish conditional competence. The full frozen experiment
and every baseline are reproducible without API keys:

```powershell
python -m axiom.benchmarks.cognitive freeze --output work/cognitive-new
python -m axiom.benchmarks.cognitive train --output work/cognitive-new
python -m axiom.benchmarks.cognitive evaluate --output work/cognitive-new
python -m axiom.benchmarks.cognitive replay --output work/cognitive-new
```

For an installation smoke test, add `--smoke` to `freeze` only. Smoke data and
training counts are separate and unsuitable for performance claims.

## Distinct changes and host responsibilities

| Operation | Episode records | Explicit statistics | GRU activation state | Network weights |
|---|---|---|---|---|
| `observe(public_state)` | No feedback invented | unchanged | unchanged | unchanged |
| `suggest()` | unchanged | unchanged | unchanged | unchanged |
| `feedback(verified_event)` | appends event | updates from observed outcome | learned transition updates state | unchanged |
| `learn(sequences)` | uses supplied replay | preserved | re-encodes current events | explicit gradient update |
| `rollback()` | retained | retained | re-encodes under old parameters | restores previous weights |
| `reset_context()` | clears current-session buffer | clears | resets | retained |
| `save/load` | preserved | reconstructed | reconstructed | preserved |

`verified=True` is an assertion by the trusted execution host, not an entitlement
granted by JSON validation. Never accept it directly from a teacher, webpage or
external LLM. `PublicState` has no teacher-quality or environment-regime fields;
strict validation rejects extra fields. Costs and permissions are user/host
constraints and cannot be overridden by learned scores.

State persists across tasks within one host-defined software session. The host
must call `reset_context` when switching independent sessions; autonomous context
identification is not implemented. Saves use JSON and refuse to overwrite files.
Checkpoint loading validates schemas, weights and feedback, but does not certify
file provenance. Load trusted local checkpoints only. Optimizer state is not
persisted: each explicit `learn` invocation starts a fresh Adam optimizer.

The in-memory episode buffer describes the current session. Full experiment
history is in external JSONL logs. `reset_context` is not a privacy-delete API:
previous exports/checkpoints must also be removed when deleting personal records.
The established SQLite core retains its separate governance API; arbitrary
concurrent writers and automatic crash recovery are outside this prototype.

## Replaying growth and interpreting candidates

Each `trace-gru.jsonl` row records only public feedback, all eligible candidates,
their expected-return/success estimates, the chosen action, exploration flag,
explicit statistics and GRU state before/after feedback. `replay` reconstructs
each environment from the sealed fixtures, verifies the executed observation,
recomputes candidates/states, and checks the final saved state.

Candidate scores estimate return under the logged training behavior and are not
optimal Q values. The success head estimates immediate observed completion, not
general mastery or confidence in a natural-language belief. Hidden-state values
are not human-readable beliefs. Evaluation updates activations from each policy's
own experience while holding weights fixed; between policies the experienced
actions differ even though initial tasks and budgets are identical.

Exploration is an explicitly configured 5% randomized action choice in the study,
not emergent curiosity. All methods can remain available after earlier failures
subject to current budget, task state and permissions; a teacher refusal in one
task does not revoke future permission. Teacher responses are simulated bounded
hints; no external teacher/API/search integration is introduced this round.
