import json

import pytest

from axiom import Axiom
from axiom.core.models import Demonstration
from axiom.core.recovery_models import RecoveryState
from axiom.environments.classroom import ClassroomMenu
from axiom.examples.first_teacher import prepare_lesson, resume_lesson, evaluate_transfer
from axiom.learning.recovery import RecoveryLinear, recovery_features
from axiom.tools.teacher_bridge import TeacherReply, TeacherRequest


def selector():
    policy = RecoveryLinear()
    obs = ClassroomMenu(0, split="lesson").observe()
    x = recovery_features(RecoveryState(base=obs), {}).tolist()
    policy.fit([(x, "ask", .9), (x, "traverse", .1)])
    return policy.state()


def response(request, label="sound desk"):
    return TeacherReply(request_id=request["request_id"], rule={"audio":label},
        explanation="Choose the root whose functional meaning matches the audio task.")


def test_actual_ranker_emits_request_without_search_or_hidden_answer(tmp_path):
    request = prepare_lesson(tmp_path / "lesson", selector())
    assert request["status"] == "pending_teacher"
    assert request["request"]["selected_action"] == "ask"
    assert request["request"]["state"]["base"]["allow_search"] is False
    text = (tmp_path / "lesson/request.json").read_text()
    assert all(key not in text for key in ('"target"', '"target_path"', '"regime"'))
    assert "search" not in [c["method"] for c in request["request"]["candidates"]]
    assert request["spent"] == 1


def test_permission_and_budget_prevent_teacher_request(tmp_path):
    denied = prepare_lesson(tmp_path / "denied", selector(), allow_ask=False)
    tight = prepare_lesson(tmp_path / "tight", selector(), budget=1)
    assert denied["status"] == tight["status"] == "finished_without_teacher"
    assert not (tmp_path / "denied/request.json").exists()
    assert not (tmp_path / "tight/request.json").exists()


def test_untrained_selector_is_not_forced_to_ask(tmp_path):
    result = prepare_lesson(tmp_path / "prior",RecoveryLinear().state())
    assert result["status"] == "finished_without_teacher"
    assert not (tmp_path / "prior/request.json").exists()


def test_changed_frozen_test_set_blocks_resume(tmp_path):
    folder = tmp_path / "lesson"
    prepare_lesson(folder,selector())
    request = json.loads((folder/"request.json").read_text())
    (folder/"sealed-tests.json").write_text("[]")
    with pytest.raises(ValueError,match="frozen tests"):
        resume_lesson(folder,response(request),teacher_source="unit-test")


def test_teacher_reply_cannot_supply_verification_or_permissions():
    with pytest.raises(ValueError):
        TeacherReply(request_id="x",rule={"audio":"sound desk"},explanation="x",verified=True)


def test_wrong_binding_or_unknown_label_rejected_before_learning(tmp_path):
    folder = tmp_path / "lesson"
    prepare_lesson(folder, selector())
    request = json.loads((folder / "request.json").read_text())
    bad = response(request).model_copy(update={"request_id":"different"})
    with pytest.raises(ValueError):
        resume_lesson(folder, bad, teacher_source="unit-test")
    with pytest.raises(ValueError):
        resume_lesson(folder, response(request,"missing root"), teacher_source="unit-test")
    core = Axiom(folder / "learner.db")
    assert "semantic" not in core.skills
    core.close()


def test_verified_lesson_changes_parameters_retains_old_skill_and_restores(tmp_path):
    folder = tmp_path / "lesson"
    prepare_lesson(folder, selector())
    request = json.loads((folder / "request.json").read_text())
    result = resume_lesson(folder, response(request), teacher_source="unit-test")
    assert result["accepted"] and result["success"]
    assert result["before_digest"] == result["recorded_digest"]
    assert result["learning"]["parameter_delta"] > 0
    assert result["before_choice"] == result["recorded_choice"] == "traverse"
    assert result["after_choice"] == "semantic"
    assert result["traversal_retained"] and result["exact_rollback"]
    restored = Axiom.load(folder / "trained-checkpoint.json", tmp_path / "restored.db")
    assert restored.parameter_digest == result["trained_digest"]
    restored.close()
    with pytest.raises(ValueError):
        resume_lesson(folder, response(request), teacher_source="unit-test")
    report = evaluate_transfer(folder, tasks_per_cell=2)
    assert report["integrity"]["evaluation_parameters_unchanged"]
    assert report["integrity"]["train_test_overlap"] == 0
    assert report["metrics"]["learned"]["teacher_calls"] == 0


def test_failed_teacher_suggestion_does_not_become_skill_or_weights(tmp_path):
    folder = tmp_path / "lesson"
    prepare_lesson(folder, selector())
    request = json.loads((folder / "request.json").read_text())
    result = resume_lesson(folder,response(request,"canvas"),teacher_source="unit-test-faulty")
    assert not result["accepted"]
    assert result["before_digest"] == result["trained_digest"]
    core = Axiom(folder / "learner.db")
    assert "semantic" not in core.skills
    assert not core.learn().updated
    core.close()


def test_interrupted_learning_cannot_replace_the_bound_reply(tmp_path, monkeypatch):
    folder = tmp_path / "lesson"
    prepare_lesson(folder,selector())
    request = json.loads((folder/"request.json").read_text())
    original = Axiom.learn
    def interrupted(self):
        original(self)
        raise RuntimeError("injected interruption after committed learning")
    monkeypatch.setattr(Axiom,"learn",interrupted)
    with pytest.raises(RuntimeError,match="injected"):
        resume_lesson(folder,response(request),teacher_source="first-teacher")
    monkeypatch.setattr(Axiom,"learn",original)
    with pytest.raises(ValueError,match="partial"):
        resume_lesson(folder,response(request,"canvas"),teacher_source="replacement")
    assert json.loads((folder/"teacher-reply.json").read_text())["rule"] == {"audio":"sound desk"}
    assert not (folder/"lesson.json").exists()


def test_evaluation_rejects_swapped_training_checkpoint(tmp_path):
    folder = tmp_path / "lesson"
    prepare_lesson(folder,selector())
    request = json.loads((folder/"request.json").read_text())
    resume_lesson(folder,response(request),teacher_source="unit-test")
    (folder/"trained-checkpoint.json").write_bytes((folder/"recorded-checkpoint.json").read_bytes())
    with pytest.raises(ValueError,match="checkpoint"):
        evaluate_transfer(folder,tasks_per_cell=1)
