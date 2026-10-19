import json

import pytest

from axiom.benchmarks.run import collect_training, run_benchmark
from axiom.benchmarks.baselines import MemoryPolicy, FixedPolicy, available_methods
from axiom.environments.menu import MenuEnvironment, make_demo
from axiom.learning.policies import LinUCBPolicy, MLPPolicy, features


def test_training_log_only_contains_executed_action_feedback():
    records = collect_training(11, 24, {})
    assert len(records) == 24
    for row in records:
        assert row.verified and row.success is not None
        assert row.operations <= row.observation.budget
        assert "target_menu" not in row.model_dump_json()


def test_fixed_and_memory_policies_respect_hard_permission_filters():
    obs = MenuEnvironment(3, help_condition="restricted").observe()
    assert "ask" not in available_methods(obs)
    for policy in [FixedPolicy(), MemoryPolicy([])]:
        choice = max(available_methods(obs), key=policy.scores(features(obs, {})).get)
        assert choice != "ask"


@pytest.mark.parametrize("kind", ["linucb", "mlp"])
def test_strong_comparators_fit_actual_feedback(kind):
    if kind == "mlp":
        pytest.importorskip("torch")
    policy = LinUCBPolicy() if kind == "linucb" else MLPPolicy(0)
    obs, demo = make_demo()
    x = features(obs, {})
    old = policy.digest
    policy.fit([(x.tolist(), "semantic", demo.reward)])
    assert policy.digest != old
    assert set(policy.scores(x)) == {"traverse", "semantic", "ask", "search"}


def test_benchmark_has_paired_tasks_frozen_evaluation_and_reproducible_metrics(tmp_path):
    a = run_benchmark(tmp_path / "a", seeds=[0], train_tasks=16, tasks_per_cell=2, include_mlp=False)
    b = run_benchmark(tmp_path / "b", seeds=[0], train_tasks=16, tasks_per_cell=2, include_mlp=False)
    assert a["aggregate"] == b["aggregate"]
    assert a["integrity"]["evaluation_parameters_unchanged"]
    rows = [json.loads(line) for line in (tmp_path / "a" / "episodes.jsonl").read_text().splitlines()]
    methods = {r["agent"] for r in rows}
    assert {"A_serial", "B_memory", "C_candidates", "D_axiom", "frozen", "no_association", "no_inquiry", "LinUCB"} <= methods
    tasks = [{r["task_id"] for r in rows if r["agent"] == name} for name in methods]
    assert all(t == tasks[0] for t in tasks)
    assert a["integrity"]["train_test_task_overlap"] == 0
