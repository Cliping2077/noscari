import argparse
from pathlib import Path

from axiom import Axiom
from axiom.benchmarks.run import HELP, REGIMES, aggregate, choose, write_json
from axiom.environments.menu import MenuEnvironment, execute_method, make_demo
from axiom.learning.policies import LinearPolicy


def run_demo(output, tasks_per_cell=20):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    agent = Axiom(output / "state.db")
    obs, demo = make_demo()
    agent.observe(obs)
    before = agent.parameter_digest
    before_choice = agent.suggest()[0].method
    agent.teach(demo)
    after_recording = agent.parameter_digest
    receipt = agent.learn()
    after_choice = agent.suggest()[0].method
    rows = []
    policies = {"before": LinearPolicy(), "after": agent.policy}
    for regime in REGIMES:
        for condition in HELP:
            for seed in range(tasks_per_cell):
                for name, policy in policies.items():
                    env = MenuEnvironment(90000000+seed, regime=regime, help_condition=condition)
                    method = choose(policy, env.observe(), {})
                    result = execute_method(env, method, {})
                    rows.append(dict(phase=name, regime=regime, help_condition=condition,
                        task_id=env.observe().task_id, method=method, **result.model_dump(mode="json")))
    agent.save(output / "checkpoint.json")
    agent.store.export_jsonl(output / "events.jsonl")
    agent.rollback()
    rolled_back = agent.parameter_digest
    report = dict(demo=demo.model_dump(mode="json"), before_digest=before,
        after_recording_digest=after_recording, learning=receipt.model_dump(),
        before_choice=before_choice, after_choice=after_choice,
        rollback_digest=rolled_back, skills=list(agent.skills),
        aggregate={name: aggregate([r for r in rows if r["phase"] == name]) for name in policies},
        cells={f"{name}/{regime}": aggregate([r for r in rows if r["phase"] == name and r["regime"] == regime]) for name in policies for regime in REGIMES},
        episodes=rows)
    write_json(output / "demo.json", report)
    agent.close()
    return report


def main():
    parser = argparse.ArgumentParser(description="One teaching event; inspect recording, parameter change and transfer.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_demo(args.output)
    print(f"Recorded without parameter change: {result['before_digest'] == result['after_recording_digest']}")
    print(f"Choice: {result['before_choice']} -> {result['after_choice']}")
    print(f"Parameter delta: {result['learning']['parameter_delta']:.6f}")
    print(f"Exact rollback: {result['before_digest'] == result['rollback_digest']}")
    print(f"Transfer success: {result['aggregate']['before']['success']:.3f} -> {result['aggregate']['after']['success']:.3f}")


if __name__ == "__main__":
    main()
