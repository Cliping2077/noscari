"""A witnessed recovery outcome changes an explicit internal action preference."""
import argparse
from pathlib import Path

from axiom.benchmarks.run import write_json
from axiom.core.recovery import RecoveryLearner
from axiom.environments.interaction import MenuSession
from axiom.environments.menu import MenuEnvironment


def run_demo(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    # Select a witnessed successful teacher example for an engineering demonstration.
    # This is NOT a random-sample performance estimate; the benchmark is separate.
    witness = None
    for seed in range(100):
        session = MenuSession(MenuEnvironment(seed, split="demo", regime="scrambled",
            budget=8, help_condition="cheap"))
        state = session.semantic_prefix({})
        if state.semantic_failed and session.step("ask", {}).after.success:
            witness = session
            break
    if witness is None:
        raise RuntimeError("no valid demonstration found")
    learner = RecoveryLearner(output / "state.db")
    learner.observe(witness.initial)
    initial_digest = learner.parameter_digest
    initial_choice = learner.suggest()[0].method
    learner.teach(witness.episode(verified=True, source="simulated-teacher"))
    recorded_digest = learner.parameter_digest
    recorded_choice = learner.suggest()[0].method
    receipt = learner.learn()
    trained_digest = learner.parameter_digest
    trained_choice = learner.suggest()[0].method
    learner.save(output / "checkpoint.json")
    learner.store.export_jsonl(output / "events.jsonl")
    learner.rollback()
    result = dict(initial_choice=initial_choice, recorded_choice=recorded_choice,
        trained_choice=trained_choice, initial_digest=initial_digest, recorded_digest=recorded_digest,
        trained_digest=trained_digest, rollback_digest=learner.parameter_digest,
        learning=receipt.model_dump(), episode=witness.episode(verified=True).model_dump(mode="json"),
        limitation="One selected successful teaching example proves an interface/behavior change, not general recovery improvement.")
    write_json(output / "demo.json", result)
    learner.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = run_demo(args.output)
    print(f"Recovery choice: {result['initial_choice']} -> {result['recorded_choice']} -> {result['trained_choice']}")
    print(f"Parameter delta: {result['learning']['parameter_delta']:.6f}")
    print(f"Exact rollback: {result['rollback_digest'] == result['initial_digest']}")


if __name__ == "__main__":
    main()
