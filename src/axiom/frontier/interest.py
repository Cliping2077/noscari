"""Frontier A: differentiation and revision of learned exploration preferences."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from axiom.benchmarks.run import interval, write_json
from axiom.frontier.motivation import MotivationCore

TOPICS = ("alpha", "beta", "gamma")
PHASES = ("initial", "reversed")
POLICIES = ("learned", "frozen", "uniform", "memory_mean")


def micro_task(seed, world, phase, topic, split, index):
    """Private synthetic environment; no world/type/solution enters the policy.

    Eight training examples fit w. Validation supplies experience feedback; test
    examples are independent of both fitting and motivation feedback.
    """
    identity = f"frontier-A/{split}/{seed}/{world}/{phase}/{topic}/{index}"
    digest = hashlib.sha256(identity.encode()).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    learnable = TOPICS[(world+phase) % 2]
    kind = "linear" if topic == learnable else ("zero" if topic == "gamma" else "noise")
    coefficient = float(rng.choice([-1, 1]) * rng.uniform(.8, 1.2))

    def examples(n):
        x = rng.normal(size=n)
        if kind == "linear":
            y = coefficient*x + rng.normal(0, .05, n)
        elif kind == "noise":
            y = rng.normal(size=n)
        else:
            y = np.zeros(n)
        return x, y

    x, y = examples(8)
    weight = float(x @ y / (x @ x + .1))
    vx, vy = examples(64)
    tx, ty = examples(128)
    before = float(np.mean(vy**2))
    after = float(np.mean((vy-weight*vx)**2))
    test_before = float(np.mean(ty**2))
    test_after = float(np.mean((ty-weight*tx)**2))
    return dict(task_id=digest, namespace=identity, topic=topic, kind=kind,
        train_x=x.tolist(), train_y=y.tolist(), predictor_before=0., predictor_after=weight,
        loss_before=before, loss_after=after,
        feedback_progress=float(np.clip((before-after)/max(1.,before),-1,1)),
        test_loss_before=test_before, test_loss_after=test_after,
        test_improvement=test_before-test_after, operations=1, api_cost_usd=0.)


def aggregate(rows):
    return dict(n=len(rows), test_improvement=float(np.mean([r["test_improvement"] for r in rows])),
        test_loss_after=float(np.mean([r["test_loss_after"] for r in rows])),
        predictor_delta=float(np.mean([abs(r["predictor_after"]) for r in rows])),
        useful_selection=float(np.mean([r["kind"] == "linear" for r in rows])),
        topic_fractions={t:sum(r["topic"]==t for r in rows)/len(rows) for t in TOPICS})


def run_interest_experiment(output, seeds=(0,1,2,3,4), train_per_topic=24, eval_tasks=60):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    if not seeds or len(set(seeds)) != len(seeds) or train_per_topic < 1 or eval_tasks < 1:
        raise ValueError("positive sample counts and distinct seeds required")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    training, records, audits, trajectories = [], [], [], []
    initial_digests, frozen_eval = set(), True
    for seed in seeds:
        for world in (0,1):
            folder = output / f"seed-{seed}-world-{world}"
            folder.mkdir()
            core = MotivationCore(folder / "state.db", budget=6*train_per_topic)
            goals = {t:core.observe(t, novelty=1) for t in TOPICS}
            initial_digests.add(core.parameter_digest)
            samples = {t:[] for t in TOPICS}
            for phase, phase_name in enumerate(PHASES):
                for cycle in range(train_per_topic):
                    for topic in TOPICS:
                        sample = micro_task(seed,world,phase,topic,"train",cycle)
                        attempt = core.reserve(goals[topic])
                        recorded_digest = core.parameter_digest
                        core.feedback(attempt,loss_before=sample["loss_before"],loss_after=sample["loss_after"],verified=True)
                        if core.parameter_digest != recorded_digest:
                            raise AssertionError("recording must not train")
                        samples[topic].append(sample["feedback_progress"])
                        training.append(dict(seed=seed,world=world,phase=phase_name,**sample))
                    receipt = core.learn()
                    trajectories.append(dict(seed=seed,world=world,phase=phase_name,cycle=cycle,
                        weights=core.weights,learning=receipt.model_dump()))
                # All comparators see the same measured training outcomes. Frozen
                # retains the same records but does not update its ordering.
                learned_topic = max(TOPICS,key=lambda t:core.weights[t])
                mean_topic = max(TOPICS,key=lambda t:float(np.mean(samples[t])))
                choices = {"learned":learned_topic,"frozen":TOPICS[0],"memory_mean":mean_topic}
                before = core.parameter_digest
                rng = np.random.default_rng(seed*10+world*2+phase)
                for i in range(eval_tasks):
                    uniform_topic = TOPICS[int(rng.integers(0,3))]
                    for policy in POLICIES:
                        topic = uniform_topic if policy == "uniform" else choices[policy]
                        # The evaluation harness owns a separate, equal eval_tasks
                        # budget per policy. It never grants new execution quota
                        # to the trained core, nor feeds test outcomes back to it.
                        sample = micro_task(seed,world,phase,topic,"test",i)
                        records.append(dict(seed=seed,world=world,phase=phase_name,policy=policy,**sample))
                frozen_eval = frozen_eval and before == core.parameter_digest
                audits.append(dict(seed=seed,world=world,phase=phase_name,
                    weights=core.weights,learning=receipt.model_dump(),selected_topic=learned_topic,
                    mean_selected_topic=mean_topic,parameter_digest=core.parameter_digest,
                    evaluation_unchanged=before==core.parameter_digest,spent=core.spent))
                core.save(folder / f"{phase_name}-checkpoint.json")
                core.store.export_jsonl(folder / f"{phase_name}-events.jsonl")
                # Reopen between phases; goal identities and preferences survive.
                core.close()
                core = MotivationCore(folder / "state.db")
                if core.parameter_digest != before or set(core.goals) != set(goals.values()):
                    raise AssertionError("reopening lost persistent goals or trained state")
            core.close()
    metrics = {phase:{p:aggregate([r for r in records if r["phase"]==phase and r["policy"]==p])
        for p in POLICIES} for phase in PHASES}
    paired = {}
    for phase in PHASES:
        paired[phase] = {}
        for policy in POLICIES[1:]:
            differences = []
            for seed in seeds:
                selected = [r for r in records if r["phase"]==phase and r["seed"]==seed]
                learned = aggregate([r for r in selected if r["policy"]=="learned"])
                comparator = aggregate([r for r in selected if r["policy"]==policy])
                differences.append(learned["test_improvement"]-comparator["test_improvement"])
            paired[phase][policy] = interval(differences)
    train_ids = {r["task_id"] for r in training}
    test_ids = {r["task_id"] for r in records}
    results = dict(classification="Frontier Lab exploratory experiment A; not confirmatory evidence",
        config=dict(seeds=list(seeds),worlds=2,topics=TOPICS,phases=PHASES,
            train_per_topic=train_per_topic,eval_tasks=eval_tasks,alpha=.25,train_examples=8,
            validation_examples=64,test_examples=128,objective="normalized observed validation-loss improvement"),
        metrics=metrics,paired_test_improvement=paired,
        selections=[dict(seed=r["seed"],world=r["world"],phase=r["phase"],topic=r["selected_topic"]) for r in audits],
        runtime=dict(seconds=time.perf_counter()-started,device="cpu",numpy=np.__version__,
            platform=platform.platform(),api_cost_usd=0,external_calls=0),
        integrity=dict(train_test_overlap=len(train_ids & test_ids),evaluation_parameters_unchanged=frozen_eval,
            identical_initial_parameters=len(initial_digests)==1,balanced_shared_training=True,
            training_tasks=len(training),evaluation_tasks=len(records),
            test_outcomes_used_for_feedback=False),
        limits=["The progress objective and question template are engineered, not learned.",
            "Balanced training is provided; only heldout opportunity selection is autonomous.",
            "Transfer is within known topic IDs; predictors restart for each micro-task.",
            "Lower selected-task error alone is not a fair comparison across domains; report improvement and task selection.",
            "Nothing in this experiment tests subjective experience or human-like development."])
    write_json(output / "summary.json",results)
    write_json(output / "parameter_audit.json",audits)
    for filename, rows in (("training.jsonl",training),("episodes.jsonl",records),("learning_trajectory.jsonl",trajectories)):
        with (output/filename).open("w",encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int,default=5)
    parser.add_argument("--train-per-topic",type=int,default=24)
    parser.add_argument("--eval-tasks",type=int,default=60)
    args = parser.parse_args()
    result = run_interest_experiment(args.output,tuple(range(args.seeds)),args.train_per_topic,args.eval_tasks)
    print(json.dumps(dict(metrics=result["metrics"],integrity=result["integrity"],runtime=result["runtime"]),indent=2))


if __name__ == "__main__":
    main()
