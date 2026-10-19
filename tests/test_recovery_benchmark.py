import json

from axiom.benchmarks.recovery import collect_recovery_training, run_recovery_benchmark
from axiom.learning.recovery import episode_rows


def test_training_returns_charge_future_actions_but_not_sunk_prefix():
    episodes = collect_recovery_training(0, 24, {})
    assert len(episodes) == 24
    assert any(e.initial.semantic_failed for e in episodes)
    for episode in episodes:
        rows = episode_rows(episode)
        if not rows:
            continue
        penalty = (0.6 + 0.2 * episode.initial.base.risk) / episode.initial.base.budget
        expected = float(episode.final.success) - penalty * (episode.final.spent - episode.initial.spent)
        assert abs(rows[0][2] - expected) < 1e-12


def test_recovery_benchmark_pairs_the_same_failure_cohort_and_freezes_parameters(tmp_path):
    result = run_recovery_benchmark(tmp_path / "experiment", seeds=[0], train_tasks=24,
                                    tasks_per_cell=2, include_mlp=False)
    assert result["integrity"]["evaluation_parameters_unchanged"]
    assert result["integrity"]["train_test_overlap"] == 0
    rows = [json.loads(s) for s in (tmp_path / "experiment/episodes.jsonl").read_text().splitlines()]
    names = {r["agent"] for r in rows}
    cohorts = [{(r["seed"], r["teacher"], r["task_id"]) for r in rows
                if r["agent"] == name and r["failed_prefix"]} for name in names]
    assert all(c == cohorts[0] for c in cohorts) and cohorts[0]
    assert {"fixed_fallback", "cost_heuristic", "memory", "D_recovery", "LinUCB",
            "frozen", "initial_context", "no_inquiry"} <= names
    assert all(r["operations"] <= r["budget"] for r in rows)
    assert result["all_tasks"]["fixed_fallback"] == result["all_tasks"]["frozen"]
