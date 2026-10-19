import json

import numpy as np
import pytest

from axiom import Axiom, Demonstration, Feedback
from axiom.environments.menu import MenuEnvironment, execute_method, make_demo
from axiom.learning.policies import LinearPolicy
from axiom.memory.store import EventStore


def test_observation_contains_no_answer_and_names_are_held_out():
    train = MenuEnvironment(1, split="train")
    test = MenuEnvironment(1, split="test")
    assert train.observe().goal != test.observe().goal
    public = train.observe().model_dump()
    assert not {"target_path", "target_menu", "seed", "regime", "items"} & public.keys()
    assert MenuEnvironment(19).observe() == MenuEnvironment(19).observe()


@pytest.mark.parametrize("regime", ["semantic", "scrambled", "legacy", "renamed"])
def test_budget_is_enforced_and_old_method_remains_valid(regime):
    env = MenuEnvironment(13, regime=regime, budget=20)
    result = execute_method(env, "traverse", {})
    assert result.success
    assert result.operations <= 20
    with pytest.raises(RuntimeError):
        env.open_menu(env.observe().menus[0].id)


def test_recording_is_not_learning_and_single_demo_changes_parameters(tmp_path):
    agent = Axiom(tmp_path / "state.db")
    obs, demo = make_demo()
    agent.observe(obs)
    before = agent.parameter_digest
    assert agent.suggest()[0].method == "traverse"
    agent.teach(demo)
    assert agent.parameter_digest == before
    assert {"traverse", "semantic"} <= set(agent.skills)
    receipt = agent.learn()
    assert receipt.updated and receipt.parameter_delta > 0
    assert receipt.before_digest == before
    assert agent.suggest()[0].method == "semantic"
    again = agent.learn()
    assert not again.updated
    agent.close()


def test_unverified_demo_and_feedback_cannot_train(tmp_path):
    agent = Axiom(tmp_path / "state.db")
    obs, demo = make_demo()
    agent.observe(obs)
    demo = demo.model_copy(update={"verified": False, "source_confidence": 1.0})
    agent.teach(demo)
    before = agent.parameter_digest
    agent.feedback(Feedback(observation=obs, method="semantic", reward=1, verified=False))
    assert not agent.learn().updated
    assert agent.parameter_digest == before


def test_permissions_override_learned_preferences(tmp_path):
    agent = Axiom(tmp_path / "state.db")
    obs, demo = make_demo()
    agent.teach(demo)
    obs = obs.model_copy(update={"allow_ask": False, "allow_search": False})
    agent.observe(obs)
    assert {c.method for c in agent.suggest()} <= {"traverse", "semantic"}


def test_save_load_rollback_and_replay(tmp_path):
    agent = Axiom(tmp_path / "state.db")
    obs, demo = make_demo()
    agent.observe(obs)
    agent.teach(demo)
    initial = agent.parameter_digest
    agent.learn()
    trained = agent.parameter_digest
    checkpoint = tmp_path / "checkpoint.json"
    agent.save(checkpoint)
    restored = Axiom.load(checkpoint, tmp_path / "restored.db")
    assert restored.parameter_digest == trained
    assert restored.suggest() == agent.suggest()
    agent.rollback()
    assert agent.parameter_digest == initial
    agent.learn()
    assert agent.parameter_digest == trained
    journal = tmp_path / "events.jsonl"
    restored.store.export_jsonl(journal)
    replica = EventStore.replay(journal, tmp_path / "replay.db")
    assert replica.events() == restored.store.events()


def test_retraction_and_privacy_delete_invalidate_learning(tmp_path):
    agent = Axiom(tmp_path / "state.db")
    obs, demo = make_demo(subject_id="private-person")
    agent.observe(obs)
    initial = agent.parameter_digest
    event_id = agent.teach(demo)
    agent.learn()
    agent.retract(event_id)
    assert agent.parameter_digest == initial
    assert not agent.learn().updated
    agent.forget("private-person")
    output = tmp_path / "events.jsonl"
    agent.store.export_jsonl(output)
    assert "private-person" not in output.read_text(encoding="utf-8")
    assert not agent.history


def test_replay_regression_learns_cost_condition_and_is_deterministic():
    policy = LinearPolicy()
    x = np.zeros(policy.feature_dim)
    x[0] = 1
    rows = [(x.tolist(), "semantic", 0.95), (x.tolist(), "traverse", 0.1)]
    old = policy.digest
    policy.fit(rows)
    assert policy.digest != old
    assert policy.scores(x)["semantic"] > policy.scores(x)["traverse"]
    clone = LinearPolicy()
    clone.fit(rows)
    assert clone.digest == policy.digest


def test_checkpoint_rejects_unknown_schema(tmp_path):
    file = tmp_path / "invalid.json"
    file.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        Axiom.load(file, tmp_path / "x.db")
