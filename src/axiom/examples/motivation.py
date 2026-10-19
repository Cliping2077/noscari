"""Offline demonstration of persistent questions and explicit preference learning."""
import argparse
from pathlib import Path

from axiom.benchmarks.run import write_json
from axiom.frontier.interest import TOPICS, micro_task
from axiom.frontier.motivation import MotivationCore


def run_demo(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    output.mkdir(parents=True,exist_ok=True)
    core = MotivationCore(output / "state.db",budget=16)
    goals = {t:core.observe(t,novelty=1) for t in TOPICS}
    initial_choice = core.suggest()[0]["topic"]
    initial_digest = core.parameter_digest
    for cycle in range(4):
        for topic in TOPICS:
            sample = micro_task(0,1,0,topic,"demo",cycle)
            attempt = core.reserve(goals[topic])
            core.feedback(attempt,loss_before=sample["loss_before"],loss_after=sample["loss_after"],verified=True)
    recorded_digest = core.parameter_digest
    receipt = core.learn()
    trained_choice = core.suggest()[0]["topic"]
    trained_digest = core.parameter_digest
    questions = [g.question for g in core.goals.values()]
    weights = core.weights
    core.save(output / "checkpoint.json")
    core.store.export_jsonl(output / "events.jsonl")
    core.close()
    restored = MotivationCore(output / "state.db")
    same_parameters = restored.parameter_digest == trained_digest
    same_goals = set(restored.goals) == set(goals.values())
    restored.control(paused=True)
    result = dict(questions=questions,initial_choice=initial_choice,trained_choice=trained_choice,
        recording_unchanged=initial_digest==recorded_digest,learning=receipt.model_dump(),weights=weights,
        initial_digest=initial_digest,trained_digest=trained_digest,
        restored_same_parameters=same_parameters,restored_same_goals=same_goals,
        pause_blocks_suggestions=restored.suggest()==[],spent=restored.spent,
        limitation="Template questions and a fixed progress objective; this engineering demo does not establish human curiosity.")
    write_json(output / "demo.json",result)
    restored.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    result = run_demo(parser.parse_args().output)
    print("Generated questions:")
    for question in result["questions"]:
        print(question)
    print(f"Preference: {result['initial_choice']} -> {result['trained_choice']}")
    print(f"Parameter delta: {result['learning']['parameter_delta']:.6f}")
    print(f"Restored goals and parameters: {result['restored_same_goals'] and result['restored_same_parameters']}")
    print(f"Pause enforced: {result['pause_blocks_suggestions']}")


if __name__ == "__main__":
    main()
