import json

import pytest

from axiom import Axiom, Feedback
from axiom.environments.menu import make_demo
from axiom.memory.store import EventStore


def test_import_rejects_string_false_as_verification(tmp_path):
    obs, demo = make_demo()
    payload = demo.model_dump(mode="json")
    payload["verified"] = "false"
    event = dict(id="bad", kind="teach", payload=payload, active=True)
    store = EventStore(tmp_path / "bad.db")
    with pytest.raises(ValueError):
        store.import_events([event])
    assert store.events() == []


def test_invalid_correction_preserves_original_and_parameters(tmp_path):
    agent = Axiom(tmp_path / "state.db")
    _, demo = make_demo()
    event_id = agent.teach(demo)
    agent.learn()
    before = agent.parameter_digest
    with pytest.raises(ValueError):
        agent.correct(event_id, {"invalid": "replacement"})
    assert agent.parameter_digest == before
    assert next(e for e in agent.store.events() if e["id"] == event_id)["active"]


def test_invalid_checkpoint_does_not_commit_events(tmp_path):
    _, demo = make_demo()
    events = [dict(id="ok", kind="teach", payload=demo.model_dump(mode="json"), active=True),
              dict(id="bad", kind="learn", active=True, payload={"after": {"weights": []}})]
    file = tmp_path / "bad.json"
    file.write_text(json.dumps(dict(schema_version=1, config={}, events=events)), encoding="utf-8")
    with pytest.raises(ValueError):
        Axiom.load(file, tmp_path / "loaded.db")
    store = EventStore(tmp_path / "loaded.db")
    assert store.events() == []


def test_old_rule_feedback_is_excluded_after_teaching_replacement(tmp_path):
    agent = Axiom(tmp_path / "state.db")
    obs, demo = make_demo()
    old_id = agent.teach(demo)
    feedback_id = agent.feedback(Feedback(observation=obs, method="semantic", reward=0.9, verified=True))
    agent.learn()
    replacement = demo.model_copy(update={"rule": {"file": "image"}, "verified": False})
    agent.correct(old_id, replacement)
    rows, ids = agent.replay_rows()
    assert feedback_id not in ids
    assert not agent.learn().updated


def test_import_rejects_invalid_feature_dimensions(tmp_path):
    obs, _ = make_demo()
    payload = Feedback(observation=obs, method="traverse", reward=1, verified=True).model_dump(mode="json")
    payload.update(features=[1, 2], skill_id="builtin:traverse:1")
    store = EventStore(tmp_path / "bad-features.db")
    with pytest.raises(ValueError):
        store.import_events([dict(id="bad", kind="feedback", payload=payload, active=True)])
    assert store.events() == []


def test_verified_execution_can_validate_an_unverified_taught_method(tmp_path):
    agent = Axiom(tmp_path / "state.db")
    obs, demo = make_demo()
    agent.observe(obs)
    agent.teach(demo.model_copy(update={"verified": False}))
    agent.feedback(Feedback(observation=obs, method="semantic", reward=1, verified=True))
    assert agent.learn().updated
    assert agent.suggest()[0].method == "semantic"
