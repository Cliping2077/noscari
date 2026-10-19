"""Distribution-shift probe, comparing latest-only training with full replay."""
import argparse
from pathlib import Path

from axiom.benchmarks.run import aggregate, choose, collect_training, write_json
from axiom.environments.menu import MenuEnvironment, execute_method, make_demo
from axiom.learning.policies import LinearPolicy, features


def run_retention(output, seeds=range(5), tasks=60):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    obs, demo = make_demo()
    demo_row = (features(obs, {}).tolist(), "semantic", demo.reward)
    rows, states = [], []
    for seed in seeds:
        old = collect_training(seed, 120, {}, regimes=("semantic",), offset=50000)
        new = collect_training(seed, 120, {}, regimes=("scrambled",), offset=60000)
        convert = lambda data: [(features(r.observation, {}).tolist(), r.method, r.reward) for r in data]
        old_rows, new_rows = convert(old), convert(new)
        policies = {}
        for phase, training in {"old_trained": [demo_row]+old_rows,
                                "latest_only": [demo_row]+new_rows,
                                "full_replay": [demo_row]+old_rows+new_rows}.items():
            policy = LinearPolicy()
            policy.fit(training)
            policies[phase] = policy
            states.append(dict(seed=seed, phase=phase, state=policy.state(), digest=policy.digest))
        for regime in ("semantic", "scrambled"):
            for i in range(tasks):
                for phase, policy in policies.items():
                    env = MenuEnvironment(80000000 + seed*10000+i, regime=regime)
                    method = choose(policy, env.observe(), {})
                    outcome = execute_method(env, method, {})
                    rows.append(dict(seed=seed, phase=phase, regime=regime, method=method,
                                     **outcome.model_dump(mode="json")))
    report = dict(config=dict(seeds=list(seeds), phase_training_tasks=120, test_tasks_per_regime=tasks),
        aggregate={f"{phase}/{regime}": aggregate([r for r in rows if r["phase"]==phase and r["regime"]==regime])
                   for phase in policies for regime in ("semantic", "scrambled")},
        states=states, episodes=rows,
        limitation="Latest-only vs replay is a controlled regression probe, not an empirical claim about all continual-learning algorithms.")
    write_json(output / "retention.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_retention(args.output)
    print(f"Wrote {args.output / 'retention.json'}")


if __name__ == "__main__":
    main()
