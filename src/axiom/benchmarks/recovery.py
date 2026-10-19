"""Paired recovery evaluation after a common semantic prefix; no test-time fit."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

from axiom.benchmarks.baselines import MemoryPolicy, SerialPolicy
from axiom.benchmarks.run import HELP, REGIMES, interval, write_json
from axiom.core.recovery import RecoveryLearner
from axiom.environments.interaction import MenuSession, allowed_actions
from axiom.environments.menu import MenuEnvironment, make_demo
from axiom.learning.policies import digest_state
from axiom.learning.recovery import (RECOVERY_FEATURES, RecoveryLinear, RecoveryLinUCB,
    RecoveryMLP, episode_rows, recovery_features, select_action)


class RecoveryMemory(MemoryPolicy):
    def __init__(self, rows):
        super().__init__(rows)
        self.prior = RecoveryLinear()


class CostHeuristic(SerialPolicy):
    """Fixed conservative cheap-help rule. No learned state or hidden placement."""
    def scores(self, x):
        # x[10] is remaining fraction. x[4:6] are total-budget cost fractions.
        remaining = x[10]
        search_cost, ask_cost = x[4], x[5]
        return dict(traverse=0.15, semantic=-0.1,
            search=0.45 if x[6] >= 0.8 and search_cost <= remaining/3 else -0.1,
            ask=0.5 if x[7] >= 0.9 and ask_cost <= remaining/3 else -0.1)

    def state(self):
        return {"kind": "fixed-recovery-cost-heuristic", "version": 1}


def collect_recovery_training(seed, count, rule):
    rng = random.Random(f"recovery-behavior-v1:{seed}")
    episodes = []
    for i in range(count):
        env = MenuEnvironment(20000000 + seed*100000+i, split="train",
            regime=rng.choice(REGIMES), help_condition=rng.choice(HELP))
        session = MenuSession(env)
        session.semantic_prefix(rule)
        while not session.observe().done:
            action = rng.choice(allowed_actions(session.observe(), rule))
            session.step(action, rule)
        episodes.append(session.episode(verified=True))
    return episodes


def summarize(rows):
    if not rows:
        return {"n": 0}
    metrics = ("success", "reward", "operations", "recovery_operations", "recovery_steps",
               "teacher_calls", "search_calls", "method_switches", "help_after_semantic_failure")
    result = {key: float(np.mean([r[key] for r in rows])) for key in metrics}
    result["n"] = len(rows)
    return result


def evaluate_policy(policy, env, rule, *, initial_context=False, inquiry=True):
    session = MenuSession(env)
    initial = session.semantic_prefix(rule)
    predictions, timings = [], []
    while not session.observe().done:
        state = session.observe()
        tick = time.perf_counter()
        action = select_action(policy, state, rule, initial_context=initial_context, inquiry=inquiry)
        timings.append(1000*(time.perf_counter()-tick))
        predictions.append(policy.scores(recovery_features(state, rule, initial_context))[action])
        session.step(action, rule)
    episode = session.episode(verified=True)
    final = episode.final
    penalty = (0.6 + 0.2 * final.base.risk) / final.base.budget
    returns = episode_rows(episode, initial_context)
    actions = [t.action for t in episode.transitions]
    compact = [dict(action=t.action, spent_before=t.before.spent,
        spent_after=t.after.spent, opened_before=list(t.before.opened),
        opened_after=list(t.after.opened), success=t.after.success,
        help_used=list(t.after.help_used), hint_failures=t.after.hint_failures,
        prediction=float(pred), realized_future_return=float(row[2]))
        for t, pred, row in zip(episode.transitions, predictions, returns)]
    return dict(task_id=final.base.task_id, budget=final.base.budget,
        failed_prefix=initial.semantic_failed, prefix_spent=initial.spent,
        success=final.success, reward=float(final.success)-penalty*final.spent,
        operations=final.spent, recovery_operations=final.spent-initial.spent,
        recovery_steps=len(actions), teacher_calls=int("ask" in final.help_used),
        search_calls=int("search" in final.help_used),
        method_switches=sum(a != b for a, b in zip(actions, actions[1:])),
        help_after_semantic_failure=bool(initial.semantic_failed and final.help_used),
        first_recovery_action=actions[0] if actions else None,
        selection_ms=sum(timings), trace=compact, api_calls=0, api_cost_usd=0)


def run_recovery_benchmark(output, seeds=(0,1,2,3,4), train_tasks=480,
                           tasks_per_cell=30, include_mlp=True):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    if not seeds or len(set(seeds)) != len(seeds) or min(train_tasks, tasks_per_cell) < 1:
        raise ValueError("positive counts and distinct seeds required")
    output.mkdir(parents=True, exist_ok=True)
    if include_mlp:
        import torch
        torch.set_num_threads(1)
    started = time.perf_counter()
    records, audits = [], []
    train_ids, test_ids = set(), set()
    for seed in seeds:
        for teacher in ("clean", "partial_error"):
            _, demonstration = make_demo(partial_error=teacher == "partial_error")
            rule = demonstration.rule
            episodes = collect_recovery_training(seed, train_tasks, rule)
            folder = output / f"seed-{seed}-{teacher}"
            folder.mkdir()
            learner = RecoveryLearner(folder / "state.db", rule=rule)
            for episode in episodes:
                learner.feedback(episode)
                train_ids.add(episode.initial.base.task_id)
            tick = time.perf_counter()
            receipt = learner.learn()
            fit_times = {"D_recovery": time.perf_counter()-tick}
            learner.save(folder / "checkpoint.json")
            learner.store.export_jsonl(folder / "events.jsonl")
            replay, _ = learner.replay_rows()
            tick = time.perf_counter()
            memory = RecoveryMemory(replay)
            fit_times["memory"] = time.perf_counter()-tick
            policies = {"fixed_fallback": SerialPolicy(), "cost_heuristic": CostHeuristic(),
                "memory": memory, "D_recovery": learner.policy, "frozen": RecoveryLinear(),
                "initial_context": RecoveryLinear(), "no_inquiry": RecoveryLinear(),
                "LinUCB": RecoveryLinUCB()}
            initial_rows = [row for episode in episodes for row in episode_rows(episode, True)]
            for name, data in (("initial_context", initial_rows), ("no_inquiry", replay), ("LinUCB", replay)):
                tick = time.perf_counter()
                policies[name].fit(data)
                fit_times[name] = time.perf_counter()-tick
            if include_mlp:
                tick = time.perf_counter()
                policies["MLP"] = RecoveryMLP(seed)
                policies["MLP"].fit(replay)
                fit_times["MLP"] = time.perf_counter()-tick
            before = {name: p.digest for name, p in policies.items()}
            for regime in REGIMES:
                for condition in HELP:
                    for i in range(tasks_per_cell):
                        env_seed = 50000000 + seed*100000+i
                        for name, policy in policies.items():
                            env = MenuEnvironment(env_seed, regime=regime, help_condition=condition)
                            row = evaluate_policy(policy, env, rule,
                                initial_context=name == "initial_context", inquiry=name != "no_inquiry")
                            test_ids.add(row["task_id"])
                            records.append(dict(seed=seed, teacher=teacher, regime=regime,
                                help_condition=condition, agent=name, **row))
            audits.append(dict(seed=seed, teacher=teacher, training_episodes=len(episodes),
                training_transitions=len(replay), training_digest=digest_state({"rows":replay}),
                learning=receipt.model_dump(), fit_seconds=fit_times,
                evaluation_unchanged=all(before[n] == p.digest for n,p in policies.items()),
                states={n:p.state() for n,p in policies.items()}))
            learner.close()
    names = sorted({r["agent"] for r in records})
    cohorts = {name:{(r["seed"],r["teacher"],r["task_id"]) for r in records
                    if r["agent"] == name and r["failed_prefix"]} for name in names}
    if not all(cohorts[name] == cohorts[names[0]] for name in names):
        raise AssertionError("evaluation policies have different failure cohorts")
    full = {name:summarize([r for r in records if r["agent"]==name]) for name in names}
    failed = {name:summarize([r for r in records if r["agent"]==name and r["failed_prefix"]]) for name in names}
    per_seed = {str(seed):{name:summarize([r for r in records if r["agent"]==name and r["seed"]==seed and r["failed_prefix"]]) for name in names} for seed in seeds}
    comparisons = {name:{metric:interval([per_seed[str(s)]["D_recovery"][metric]-per_seed[str(s)][name][metric] for s in seeds])
        for metric in ("reward","success","operations")} for name in names if name != "D_recovery"}
    groups = defaultdict(list)
    for row in records:
        if row["failed_prefix"]:
            groups[(row["agent"],row["teacher"],row["regime"],row["help_condition"])].append(row)
    prediction_errors = {}
    for name in names:
        errors = [t["prediction"]-t["realized_future_return"] for r in records if r["agent"]==name for t in r["trace"]]
        prediction_errors[name] = dict(n=len(errors), bias=float(np.mean(errors)), rmse=float(np.sqrt(np.mean(np.square(errors)))))
    summary = dict(config=dict(seeds=list(seeds), train_tasks=train_tasks, tasks_per_cell=tasks_per_cell,
            include_mlp=include_mlp, features=list(RECOVERY_FEATURES), ridge=1,
            target="Monte Carlo return-to-go under logged uniform behavior", prefix="shared semantic attempt"),
        runtime=dict(seconds=time.perf_counter()-started, device="cpu", python=sys.version,
            numpy=np.__version__, platform=platform.platform(), api_cost_usd=0),
        all_tasks=full, failed_prefix=failed, per_seed_failed_prefix=per_seed,
        paired_recovery_minus_baseline=comparisons, prediction_error=prediction_errors,
        cells=[dict(agent=k[0],teacher=k[1],regime=k[2],help_condition=k[3],**summarize(v)) for k,v in groups.items()],
        integrity=dict(evaluation_parameters_unchanged=all(a["evaluation_unchanged"] for a in audits),
            shared_failure_cohort=True, train_test_overlap=len(train_ids & test_ids),
            training_task_ids=sorted(train_ids),test_task_ids=sorted(test_ids)))
    with (output/"episodes.jsonl").open("w",encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record,allow_nan=False)+"\n")
    write_json(output/"summary.json",summary)
    write_json(output/"parameter_audit.json",audits)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int,default=5)
    parser.add_argument("--train-tasks",type=int,default=480)
    parser.add_argument("--tasks-per-cell",type=int,default=30)
    parser.add_argument("--without-mlp",action="store_true")
    args = parser.parse_args()
    summary = run_recovery_benchmark(args.output,list(range(args.seeds)),args.train_tasks,args.tasks_per_cell,not args.without_mlp)
    print(json.dumps(summary["failed_prefix"],indent=2))


if __name__ == "__main__":
    main()
