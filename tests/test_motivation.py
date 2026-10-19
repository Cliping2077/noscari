import json

import pytest

from axiom.frontier.motivation import MotivationCore


def create(core, topic="alpha", subject="synthetic"):
    return core.observe(topic, novelty=1, subject_id=subject)


def teach(core, goal, before=1., after=0., verified=True):
    attempt = core.reserve(goal)
    return core.feedback(attempt, loss_before=before, loss_after=after, verified=verified)


def test_observation_creates_goal_without_user_task_and_training_is_explicit(tmp_path):
    core = MotivationCore(tmp_path / "state.db", budget=10)
    assert core.observe("known", novelty=0) is None
    first, second = create(core), create(core, "beta")
    assert create(core) == first
    assert core.goals[second].question.startswith("Can I improve")
    digest = core.parameter_digest
    teach(core, second)
    assert core.parameter_digest == digest
    assert core.suggest()[0]["goal_id"] == first
    receipt = core.learn()
    assert receipt.updated and receipt.parameter_delta > 0
    assert core.suggest()[0]["goal_id"] == second
    core.rollback()
    assert core.parameter_digest == digest
    assert core.spent == 1  # rollback never refunds a tool budget


def test_pause_permission_and_budget_cannot_be_overridden_by_learning(tmp_path):
    core = MotivationCore(tmp_path / "state.db", budget=1)
    goal = create(core)
    core.control(paused=True)
    assert core.suggest() == []
    with pytest.raises(PermissionError):
        core.reserve(goal)
    core.control(paused=False, allow_experiment=False)
    with pytest.raises(PermissionError):
        core.reserve(goal)
    core.control(allow_experiment=True)
    teach(core, goal)
    core.learn()
    assert core.suggest() == []
    with pytest.raises(PermissionError):
        core.reserve(goal)


def test_unverified_duplicate_and_nonfinite_feedback(tmp_path):
    core = MotivationCore(tmp_path / "state.db", budget=4)
    goal = create(core)
    sample = teach(core, goal, verified=False)
    assert not core.learn().updated
    with pytest.raises(ValueError):
        core.feedback("missing", loss_before=1, loss_after=0, verified=True)
    attempt = core.reserve(goal)
    core.feedback(attempt, loss_before=1, loss_after=0, verified=True)
    with pytest.raises(ValueError):
        core.feedback(attempt, loss_before=1, loss_after=0, verified=True)
    with pytest.raises(ValueError):
        core.feedback(core.reserve(goal), loss_before=float("nan"), loss_after=0, verified=True)


def test_persistent_goal_status_and_parameter_recovery(tmp_path):
    path = tmp_path / "state.db"
    core = MotivationCore(path, budget=8)
    goal = create(core)
    teach(core, goal)
    core.learn()
    core.set_status(goal, "paused")
    assert core.observe("alpha", novelty=1) == goal
    assert core.suggest() == []  # observations cannot override a user pause
    core.save(tmp_path / "checkpoint.json")
    core.store.export_jsonl(tmp_path / "events.jsonl")
    digest = core.parameter_digest
    core.close()
    restored = MotivationCore(path)
    copy = MotivationCore.load(tmp_path / "checkpoint.json", tmp_path / "copy.db")
    assert restored.parameter_digest == copy.parameter_digest == digest
    assert copy.goals[goal].status == "paused"
    copy.set_status(goal, "active")
    assert copy.suggest()[0]["goal_id"] == goal
    with pytest.raises(ValueError):
        MotivationCore(path, budget=1000)


def test_delete_removes_personal_provenance_and_derivatives_without_budget_refund(tmp_path):
    path = tmp_path / "state.db"
    core = MotivationCore(path, budget=10)
    private = create(core, "private-topic", "private-person")
    public = create(core, "public")
    teach(core, private)
    teach(core, public)
    core.learn()
    core.forget("private-person")
    core.save(tmp_path / "clean.json")
    text = (tmp_path / "clean.json").read_text()
    assert "private-person" not in text and "private-topic" not in text
    assert core.spent == 2
    assert core.weights == {"public": 0.}
    core.close()
    copy = MotivationCore(path)
    assert copy.spent == 2 and copy.weights == {"public": 0.}
    assert copy.learn().updated


def test_invalid_journal_rejected_before_import(tmp_path):
    core = MotivationCore(tmp_path / "state.db", budget=1)
    goal = create(core)
    teach(core, goal)
    core.save(tmp_path / "checkpoint.json")
    data = json.loads((tmp_path / "checkpoint.json").read_text())
    attempt = next(e for e in data["events"] if e["kind"] == "motivation_attempt")
    attempt["payload"]["goal_id"] = "missing"
    (tmp_path / "invalid.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        MotivationCore.load(tmp_path / "invalid.json", tmp_path / "bad.db")
    assert not (tmp_path / "bad.db").exists()
