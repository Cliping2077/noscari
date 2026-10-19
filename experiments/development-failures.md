# Development failures and review corrections — a2

- Round 2 review reproduced two real defects: writable runtime configuration could
  diverge from the persisted feature/rule definition; a help transition could claim
  success without opening a new menu. Added regressions first: 2 failed / 10 passed.
  Read-only configuration and evidence consistency checks fixed both; reviewer
  confirmed the fixes. The simulator never generated either invalid case, so the
  already-run benchmark's numerical outputs are unchanged.
- Frontier API/benchmark/demo tests initially failed because the modules did not
  exist. After implementation, one invalid-journal test exposed an exception-type
  mismatch: unknown goals were treated as permission failures. Unknown references
  now raise validation errors before import; ordinary denied execution still raises
  PermissionError. No failed measurement was discarded or tuned away.
- Direct module commands first failed because this interpreter did not yet have
  the package installed. Source runs then explicitly set PYTHONPATH=src. Final
  delivery also verifies an installed wheel from outside the source directory.
- Round 2 negative scientific result: initial-context scoring matched or slightly
  exceeded the dynamic scorer. The reported aggregate improvement over traversal
  does not demonstrate extra benefit from dynamic scoring features.
- Frontier A limitation: the simple memory-mean comparator exactly matched initial
  learned selections. Its reversal weakness is consistent with recency weighting;
  no uniquely human mechanism or new learning principle has been established.

Final test counts and package checks are in `verification-a2.json` and
`tests-a2.txt`. Historical round-1 verification and archives remain separate.
# Alpha 4 cognitive experiment — pre-freeze audit regressions

All fixes below preceded the formal `core/cognitive-v1` freeze. No formal test
results were used to choose them or retune models.

- Missing contextual/cognitive/benchmark APIs first failed their behavior tests;
  implementations then passed the intended tests.
- Parameter-budget regression first found 4,078 vs 646/1,030 neural parameters.
  Baseline widths were set to 4,086/4,102 before formal training, without performance tuning.
- Flipping an exploration flag initially passed trajectory replay; a failed
  regression led to deterministic action/flag verification.
- Removing earlier sessions initially still passed replay; full expected session,
  task, step and completion coverage is now required, plus output artifact hashes.
- Duplicated checkpoint feedback initially changed restored hidden state silently;
  complete episode chronology is now validated on append and load.
- Altering manifest epochs initially bypassed sealed protocol configuration;
  strict configuration agreement is now checked before training/evaluation.
- Independent re-review ran 17 targeted tests successfully; full suite passed 83.
  Formal outcome is a negative architecture comparison, preserved in the research
  report and all original logs. No post-test policy correction was applied.

