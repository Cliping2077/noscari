"""Predeclared paired benchmark. No tuning or updating on evaluation tasks."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

from axiom import Axiom, Feedback
from axiom.benchmarks.baselines import (FixedPolicy, MemoryPolicy, SerialPolicy,
                                        available_methods)
from axiom.environments.menu import MenuEnvironment, execute_method, make_demo
from axiom.learning.policies import (FEATURE_NAMES, LinearPolicy, LinUCBPolicy,
                                     MLPPolicy, digest_state, features)

REGIMES = ("semantic", "scrambled", "legacy", "renamed")
HELP = ("standard", "cheap", "restricted")
METRICS = ("success", "operations", "menu_opens", "teacher_calls", "search_calls",
           "reward", "semantic_attempt", "wrong_transfer", "fallback_used", "api_calls", "api_cost_usd")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def collect_training(seed, count, rule, regimes=REGIMES, offset=0):
    """Only the chosen method's observed terminal outcome is logged."""
    rng = random.Random(f"behavior:{seed}:{offset}")
    rows = []
    for i in range(count):
        env = MenuEnvironment(seed * 100000 + offset + i, split="train",
            regime=rng.choice(regimes), help_condition=rng.choice(HELP))
        obs = env.observe()
        method = rng.choice(available_methods(obs))
        outcome = execute_method(env, method, rule)
        rows.append(Feedback(observation=obs, method=method, reward=outcome.reward,
            verified=True, source="simulator", success=outcome.success, operations=outcome.operations))
    return rows


def aggregate(rows):
    if not rows:
        return {}
    result = {metric: float(np.mean([r[metric] for r in rows])) for metric in METRICS}
    attempts = sum(r["semantic_attempt"] for r in rows)
    result["wrong_transfer_given_semantic"] = sum(r["wrong_transfer"] for r in rows) / attempts if attempts else None
    result["new_method_selection"] = sum(r["method"] == "semantic" for r in rows) / len(rows)
    result["traversal_selection"] = sum(r["method"] == "traverse" for r in rows) / len(rows)
    result["n"] = len(rows)
    return result


def interval(values):
    """Student t interval across independent seed means, NOT individual episodes."""
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return {"mean": mean, "ci95": None, "n_seeds": n}
    # Exact common small-n quantiles, conservative 1.96 only for n > 31.
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 2.228 if n <= 31 else 1.96)
    half = critical * float(np.std(values, ddof=1)) / np.sqrt(n)
    return {"mean": mean, "ci95": [mean-half, mean+half], "n_seeds": n}


def choose(policy, obs, rule, *, association=True, inquiry=True):
    scores = policy.scores(features(obs, rule, association))
    return sorted(available_methods(obs, association=association, inquiry=inquiry),
                  key=lambda method: (-scores[method], method))[0]


def run_benchmark(output, seeds=(0, 1, 2, 3, 4), train_tasks=240,
                  tasks_per_cell=20, include_mlp=True):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    if train_tasks < 1 or tasks_per_cell < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("positive sample counts and distinct seeds required")
    output.mkdir(parents=True, exist_ok=True)
    if include_mlp:
        import torch
        torch.set_num_threads(1)
    started = time.perf_counter()
    rows = []
    audit = []
    training_ids, testing_ids = set(), set()
    for seed in seeds:
        for teacher in ("clean", "partial_error"):
            folder = output / f"seed-{seed}-{teacher}"
            folder.mkdir()
            obs, demo = make_demo(partial_error=teacher == "partial_error")
            rule = demo.rule
            training = collect_training(seed, train_tasks, rule)
            training_ids.update(r.observation.task_id for r in training)
            agent = Axiom(folder / "state.db")
            prior = agent.parameter_digest
            agent.teach(demo)
            for experience in training:
                agent.feedback(experience)
            fit_started = time.perf_counter()
            receipt = agent.learn()
            fit_seconds = time.perf_counter() - fit_started
            replay, _ = agent.replay_rows()
            agent.save(folder / "checkpoint.json")
            agent.store.export_jsonl(folder / "events.jsonl")
            policies = {"A_serial": SerialPolicy(), "B_memory": MemoryPolicy(replay),
                "C_candidates": FixedPolicy(), "D_axiom": agent.policy,
                "frozen": LinearPolicy(), "no_association": LinearPolicy(),
                "no_inquiry": LinearPolicy(), "LinUCB": LinUCBPolicy()}
            no_assoc_rows = [([v if j != 1 else 0 for j, v in enumerate(x)], a, r) for x, a, r in replay]
            policies["no_association"].fit(no_assoc_rows)
            policies["no_inquiry"].fit(replay)
            policies["LinUCB"].fit(replay)
            training_times = {"D_axiom": fit_seconds}
            if include_mlp:
                fit_started = time.perf_counter()
                policies["MLP"] = MLPPolicy(seed)
                policies["MLP"].fit(replay)
                training_times["MLP"] = time.perf_counter() - fit_started
            before = {name: p.digest for name, p in policies.items()}
            for regime in REGIMES:
                for condition in HELP:
                    for task in range(tasks_per_cell):
                        env_seed = 10000000 + seed * 100000 + task
                        for name, policy in policies.items():
                            env = MenuEnvironment(env_seed, regime=regime, split="test", help_condition=condition)
                            observation = env.observe()
                            testing_ids.add(observation.task_id)
                            tick = time.perf_counter()
                            method = choose(policy, observation, rule,
                                association=name != "no_association", inquiry=name != "no_inquiry")
                            selection_ms = 1000 * (time.perf_counter() - tick)
                            result = execute_method(env, method, rule)
                            rows.append(dict(seed=seed, teacher=teacher, regime=regime,
                                help_condition=condition, task_id=observation.task_id, agent=name,
                                method=method, selection_ms=selection_ms, **result.model_dump(mode="json")))
            unchanged = all(before[name] == p.digest for name, p in policies.items())
            audit.append(dict(seed=seed, teacher=teacher, prior_digest=prior,
                learning=receipt.model_dump(), training_rows=len(replay),
                training_digest=digest_state({"rows": replay}), training_seconds=training_times,
                evaluation_unchanged=unchanged, states={name: p.state() for name, p in policies.items()}))
            agent.close()
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    names = sorted({r["agent"] for r in rows})
    aggregates = {name: aggregate([r for r in rows if r["agent"] == name]) for name in names}
    per_seed = {str(seed): {name: aggregate([r for r in rows if r["agent"] == name and r["seed"] == seed]) for name in names} for seed in seeds}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["agent"], row["teacher"], row["regime"], row["help_condition"])].append(row)
    comparisons = {}
    for name in names:
        if name == "D_axiom":
            continue
        comparisons[name] = {metric: interval([per_seed[str(seed)]["D_axiom"][metric] - per_seed[str(seed)][name][metric] for seed in seeds]) for metric in ("reward", "success", "operations")}
    summary = dict(config=dict(seeds=list(seeds), train_tasks=train_tasks,
        tasks_per_cell=tasks_per_cell, include_mlp=include_mlp, features=list(FEATURE_NAMES),
        ridge=1.0, mlp_epochs=150, linucb_alpha=0.25),
        runtime=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                     seconds=time.perf_counter()-started, device="cpu", api_cost_usd=0),
        aggregate=aggregates, per_seed=per_seed, paired_axiom_minus_baseline=comparisons,
        cells=[dict(agent=k[0], teacher=k[1], regime=k[2], help_condition=k[3], **aggregate(v)) for k, v in groups.items()],
        integrity=dict(evaluation_parameters_unchanged=all(a["evaluation_unchanged"] for a in audit),
                       train_test_task_overlap=len(training_ids & testing_ids),
                       training_task_ids=sorted(training_ids), test_task_ids=sorted(testing_ids)))
    write_json(output / "summary.json", summary)
    write_json(output / "parameter_audit.json", audit)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train-tasks", type=int, default=240)
    parser.add_argument("--tasks-per-cell", type=int, default=20)
    parser.add_argument("--without-mlp", action="store_true")
    args = parser.parse_args()
    summary = run_benchmark(args.output, list(range(args.seeds)), args.train_tasks,
                            args.tasks_per_cell, not args.without_mlp)
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
