import json

import numpy as np
import pytest

from axiom.core.recovery import RecoveryLearner
from axiom.core.recovery_models import RecoveryEpisode
from axiom.environments.interaction import MenuSession
from axiom.environments.menu import MenuEnvironment
from axiom.learning.recovery import RecoveryLinear, recovery_features, allowed_actions


def failed_session():
    for seed in range(100):
        session = MenuSession(MenuEnvironment(seed, regime="scrambled", budget=8))
        session.semantic_prefix({})
        if session.observe().semantic_failed:
            return session
    raise AssertionError("no failed prefix generated")


def collect_episode(method="traverse", verified=True, subject_id="synthetic"):
    session = failed_session()
    while not session.observe().done:
        available = allowed_actions(session.observe(), {})
        session.step(method if method in available else "traverse", {})
    return session.episode(verified=verified, subject_id=subject_id)


def test_step_snapshot_contains_only_observed_feedback():
    session = failed_session()
    state = session.observe()
    assert state.spent == 1 and len(state.opened) == 1
    assert state.semantic_failed
    assert "semantic" not in allowed_actions(state, {})
    payload = state.model_dump_json()
    assert all(field not in payload for field in ["target_menu", "target_path", "regime", '"seed"'])
    old = state
    transition = session.step("traverse", {})
    assert transition.before == old
    assert transition.after.spent == old.spent + 1
    assert len(transition.after.opened) == 2


def test_step_filters_are_hard_and_budgeted():
    session = MenuSession(MenuEnvironment(3, budget=2, help_condition="restricted"))
    assert "ask" not in allowed_actions(session.observe(), {})
    with pytest.raises(PermissionError):
        session.step("ask", {})
    while not session.observe().done:
        session.step("traverse", {})
    with pytest.raises(RuntimeError):
        session.step("traverse", {})


def test_dynamic_features_change_after_actual_failure():
    env = MenuEnvironment(4, regime="scrambled", budget=8)
    session = MenuSession(env)
    initial = recovery_features(session.observe(), {})
    session.semantic_prefix({})
    current = recovery_features(session.observe(), {})
    assert not np.array_equal(initial, current)
    assert np.array_equal(initial, recovery_features(session.observe(), {}, initial_context=True))


def test_episode_requires_contiguous_transitions():
    episode = collect_episode()
    payload = episode.model_dump(mode="json")
    if len(payload["transitions"]) > 1:
        payload["transitions"][1]["before"]["spent"] = 0
        with pytest.raises(ValueError):
            RecoveryEpisode.model_validate(payload)


def test_recovery_learning_records_then_updates_and_rolls_back(tmp_path):
    learner = RecoveryLearner(tmp_path / "state.db")
    initial = learner.parameter_digest
    episode = collect_episode("ask")
    learner.feedback(episode)
    assert learner.parameter_digest == initial
    receipt = learner.learn()
    assert receipt.updated and receipt.parameter_delta > 0
    trained = learner.parameter_digest
    learner.observe(failed_session().observe())
    assert learner.suggest()
    learner.save(tmp_path / "checkpoint.json")
    loaded = RecoveryLearner.load(tmp_path / "checkpoint.json", tmp_path / "loaded.db")
    assert loaded.parameter_digest == trained
    assert loaded.suggest() == learner.suggest()
    learner.rollback()
    assert learner.parameter_digest == initial
    learner.learn()
    assert learner.parameter_digest == trained


def test_unverified_episodes_do_not_train_and_forget_removes_derivatives(tmp_path):
    learner = RecoveryLearner(tmp_path / "state.db")
    learner.feedback(collect_episode(verified=False))
    assert not learner.learn().updated
    learner.feedback(collect_episode(subject_id="private-teacher"))
    learner.learn()
    learner.forget("private-teacher")
    learner.save(tmp_path / "checkpoint.json")
    assert "private-teacher" not in (tmp_path / "checkpoint.json").read_text()
    assert learner.parameter_digest == RecoveryLinear().digest


def test_equal_visible_states_produce_equal_actions_despite_hidden_placement(tmp_path):
    learner = RecoveryLearner(tmp_path / "state.db")
    state = failed_session().observe()
    learner.observe(state)
    first = learner.suggest()
    learner.observe(type(state).model_validate_json(state.model_dump_json()))
    assert learner.suggest() == first


def test_reopen_restores_configuration_and_rejects_mismatched_features(tmp_path):
    path = tmp_path / "configured.db"
    learner = RecoveryLearner(path, initial_context=True, inquiry=False)
    learner.feedback(collect_episode())
    learner.learn()
    digest = learner.parameter_digest
    learner.close()
    reopened = RecoveryLearner(path)
    assert reopened.initial_context and not reopened.inquiry
    assert reopened.parameter_digest == digest
    reopened.close()
    with pytest.raises(ValueError, match="configuration"):
        RecoveryLearner(path, initial_context=False)


def test_checkpoint_rejects_configuration_event_mismatch_before_import(tmp_path):
    learner = RecoveryLearner(tmp_path / "source.db")
    path = tmp_path / "checkpoint.json"
    learner.save(path)
    payload = json.loads(path.read_text())
    payload["config"]["initial_context"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        RecoveryLearner.load(path, tmp_path / "destination.db")
    assert not (tmp_path / "destination.db").exists()


def test_verified_recovery_changes_choice_and_rollback_restores_it(tmp_path):
    witness = None
    for seed in range(100):
        session = MenuSession(MenuEnvironment(seed, regime="scrambled", budget=8, help_condition="cheap"))
        state = session.semantic_prefix({})
        if not state.semantic_failed:
            continue
        transition = session.step("ask", {})
        if transition.after.success:
            witness = session
            break
    assert witness is not None
    learner = RecoveryLearner(tmp_path / "state.db")
    learner.observe(witness.initial)
    assert learner.suggest()[0].method == "traverse"
    learner.feedback(witness.episode(verified=True))
    assert learner.suggest()[0].method == "traverse"
    learner.learn()
    assert learner.suggest()[0].method == "ask"
    learner.rollback()
    assert learner.suggest()[0].method == "traverse"


def test_recovery_configuration_is_read_only(tmp_path):
    learner = RecoveryLearner(tmp_path / "state.db")
    for key in ("frozen", "initial_context", "inquiry"):
        with pytest.raises(AttributeError):
            setattr(learner, key, not getattr(learner, key))
    learner.rule["image"] = "file"
    assert learner.rule == {}
    learner.save(tmp_path / "checkpoint.json")
    loaded = RecoveryLearner.load(tmp_path / "checkpoint.json", tmp_path / "copy.db")
    assert loaded.rule == {}
    learner.close()
    loaded.close()


def test_help_cannot_succeed_without_new_menu_evidence():
    from axiom.core.recovery_models import RecoveryState, RecoveryTransition
    base = MenuEnvironment(3, budget=8, help_condition="cheap").observe()
    before = RecoveryState(base=base, spent=1, opened=(base.menus[0].id,))
    after = RecoveryState(base=base, spent=2, opened=before.opened, help_used=("ask",), success=True)
    with pytest.raises(ValueError, match="new menu"):
        RecoveryTransition(before=before, action="ask", after=after)
