import json

from axiom.frontier.interest import micro_task, run_interest_experiment


def test_microtask_reproducible_and_splits_independent():
    first = micro_task(0, 0, 0, "alpha", "train", 0)
    assert micro_task(0, 0, 0, "alpha", "train", 0) == first
    other = micro_task(0, 0, 0, "alpha", "test", 0)
    assert first["task_id"] != other["task_id"]
    assert len(first["train_x"]) == len(first["train_y"]) == 8
    assert first["predictor_before"] == 0
    assert first["predictor_after"] != 0


def test_interest_protocol_frozen_evaluation_and_saved_audit(tmp_path):
    result = run_interest_experiment(tmp_path / "run", seeds=(0,), train_per_topic=2, eval_tasks=3)
    assert result["integrity"]["train_test_overlap"] == 0
    assert result["integrity"]["evaluation_parameters_unchanged"]
    assert result["integrity"]["identical_initial_parameters"]
    audit = json.loads((tmp_path / "run" / "parameter_audit.json").read_text())
    assert all(row["learning"]["updated"] for row in audit)
    for phase in ("initial", "reversed"):
        assert result["metrics"][phase]["learned"]["n"] == 6
        assert result["metrics"][phase]["frozen"]["n"] == 6
