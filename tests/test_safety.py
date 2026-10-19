import json
import socket

import pytest
from pydantic import ValidationError

from axiom import Axiom, Demonstration, Feedback
from axiom.environments.menu import MenuEnvironment, execute_method, make_demo
from axiom.memory.store import EventStore
from axiom.examples.teach_menu import run_demo


def test_offline_demo_does_not_open_network(monkeypatch, tmp_path):
    def deny(*args, **kwargs):
        raise AssertionError("network use is forbidden in the core demo")
    monkeypatch.setattr(socket.socket, "connect", deny)
    result = run_demo(tmp_path / "demo", tasks_per_cell=2)
    assert result["learning"]["parameter_delta"] > 0
    assert result["before_digest"] == result["rollback_digest"]


def test_teacher_cannot_supply_executable_fields():
    obs, demo = make_demo()
    payload = demo.model_dump()
    payload["system_permissions"] = ["execute_shell"]
    with pytest.raises(ValidationError):
        Demonstration.model_validate(payload)


def test_budget_hint_denial_cannot_be_bypassed_by_direct_execution():
    env = MenuEnvironment(7, help_condition="restricted")
    with pytest.raises(PermissionError):
        execute_method(env, "ask", {})
    env = MenuEnvironment(7, budget=1)
    result = execute_method(env, "semantic", {})
    assert result.operations == 1
    with pytest.raises(RuntimeError):
        env.request_hint("search")


def test_partial_teacher_error_is_observable_negative_transfer():
    _, demo = make_demo(partial_error=True)
    outcomes = []
    for seed in range(100):
        env = MenuEnvironment(seed)
        if env.observe().goal.startswith("image:"):
            outcomes.append(execute_method(env, "semantic", demo.rule))
    assert outcomes and all(r.wrong_transfer for r in outcomes)
    assert any(r.success and r.fallback_used for r in outcomes)


def test_reopen_preserves_updates_and_correction_invalidates_them(tmp_path):
    path = tmp_path / "state.db"
    agent = Axiom(path)
    obs, demo = make_demo()
    agent.observe(obs)
    event_id = agent.teach(demo)
    agent.learn()
    trained = agent.parameter_digest
    agent.close()
    agent = Axiom(path)
    assert agent.parameter_digest == trained
    assert agent.suggest()[0].method == "semantic"
    agent.correct(event_id, demo.model_copy(update={"verified": False}))
    assert not agent.learn().updated


def test_frozen_agent_gets_same_records_without_parameter_change(tmp_path):
    agent = Axiom(tmp_path / "frozen.db", frozen=True)
    obs, demo = make_demo()
    initial = agent.parameter_digest
    agent.teach(demo)
    agent.feedback(Feedback(observation=obs, method="semantic", reward=1, verified=True))
    receipt = agent.learn()
    assert receipt.eligible_records == 2 and not receipt.updated
    assert agent.parameter_digest == initial


def test_replay_atomic_failure_leaves_empty_destination(tmp_path):
    source = tmp_path / "bad.jsonl"
    good = dict(id="same", kind="reset", payload={}, active=True)
    source.write_text(json.dumps(good) + "\n" + json.dumps(good), encoding="utf-8")
    with pytest.raises(Exception):
        EventStore.replay(source, tmp_path / "state.db")
    store = EventStore(tmp_path / "state.db")
    assert store.events() == []
    store.close()
