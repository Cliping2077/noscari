import importlib.util

import numpy as np
import pytest

pytest.importorskip('torch')


def api():
    assert importlib.util.find_spec('axiom.learning.cognitive') is not None, 'cognitive selector missing'
    from axiom.learning.cognitive import CognitiveSelector
    return CognitiveSelector


def evidence_sequence(seed=2):
    from axiom.environments.contextual import ContextMenu, make_session
    events = []
    for case in make_session('unit-learning', seed, 'reliable', tasks=4):
        env = ContextMenu(case)
        while not env.observe().done:
            action = 'semantic' if 'semantic' in env.observe().available else 'traverse'
            events.append(env.step(action))
    return events


def test_feedback_changes_fast_state_but_only_learn_changes_parameters(tmp_path):
    Agent = api()
    events = evidence_sequence()
    agent = Agent(seed=3)
    agent.observe(events[0].before)
    before = agent.parameter_digest
    hidden = agent.hidden_values
    first = agent.suggest()
    assert agent.suggest() == first  # suggestion is not itself an experience
    agent.feedback(events[0])
    assert before == agent.parameter_digest
    assert agent.hidden_values != hidden
    fast = agent.hidden_values
    agent.observe(events[-1].after)  # observation alone does not invent feedback
    assert agent.hidden_values == fast
    receipt = agent.learn([events], epochs=4)
    assert receipt['parameter_delta'] > 0 and receipt['before'] != receipt['after']
    agent.save(tmp_path/'state.json')
    restored = Agent.load(tmp_path/'state.json')
    assert restored.parameter_digest == agent.parameter_digest
    assert restored.hidden_values == agent.hidden_values
    assert restored.suggest() == agent.suggest()
    agent.rollback()
    assert agent.parameter_digest == before


def test_unverified_and_mismatched_feedback_do_not_update():
    Agent = api()
    event = evidence_sequence()[0]
    agent = Agent()
    agent.observe(event.before)
    state, digest = agent.hidden_values, agent.parameter_digest
    with pytest.raises(ValueError):
        agent.feedback(event.model_copy(update={'verified': False}))
    with pytest.raises(ValueError):
        agent.learn([[event.model_copy(update={'verified': False})]], epochs=1)
    assert state == agent.hidden_values and digest == agent.parameter_digest
    agent.observe(event.after)
    with pytest.raises(ValueError):
        agent.feedback(event)


def test_sequence_predictions_are_causal_and_match_online():
    Agent = api()
    from axiom.learning.cognitive import batch_tensors
    events = evidence_sequence()
    agent = Agent(seed=11)
    q, _ = agent.network_predictions(batch_tensors([events]))
    online = []
    for event in events:
        agent.observe(event.before)
        online.append(agent.all_predictions()[0])
        agent.feedback(event)
    assert np.allclose(q[0, :len(events)], online, atol=1e-6)
    short, _ = agent.network_predictions(batch_tensors([events[:2]]))
    assert np.allclose(q[0, :2], short[0, :2], atol=1e-6)


def test_identical_public_observation_different_experience_changes_scores():
    Agent = api()
    from axiom.environments.contextual import ContextMenu, make_session
    agent = Agent(seed=2)
    original = agent.parameter_digest
    observed = make_session('unit', 42, 'reliable', tasks=1)[0]
    env = ContextMenu(observed)
    agent.observe(env.observe())
    before = agent.all_predictions()[0]
    agent.feedback(env.step('semantic'))
    agent.observe(ContextMenu(observed).observe())
    after = agent.all_predictions()[0]
    assert not np.allclose(before, after)
    assert agent.parameter_digest == original
    agent.reset_context()
    agent.observe(ContextMenu(observed).observe())
    assert np.allclose(before, agent.all_predictions()[0])


def test_explicit_statistics_and_trainable_state_are_separate():
    Agent = api()
    from axiom.learning.cognitive import HistoryStatistics
    event = evidence_sequence()[0]
    stats = HistoryStatistics()
    before = stats.vector()
    stats.update(event)
    assert stats.vector() != before
    for kind in ('linear', 'mlp', 'context_mlp'):
        agent = Agent(kind=kind)
        agent.observe(event.before)
        digest = agent.parameter_digest
        agent.feedback(event)
        assert agent.parameter_digest == digest
        assert agent.hidden_values == [0.] * 24


def test_checkpoint_rejects_nonfinite_or_wrong_shapes(tmp_path):
    Agent = api()
    import json
    agent = Agent()
    agent.save(tmp_path/'state.json')
    data = json.loads((tmp_path/'state.json').read_text())
    key = next(iter(data['weights']))
    data['weights'][key] = [float('nan')]
    (tmp_path/'bad.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        Agent.load(tmp_path/'bad.json')


def test_neural_baselines_have_comparable_parameter_budget():
    Agent = api()
    counts = [sum(p.numel() for p in Agent(kind=k).model.parameters())
        for k in ('gru', 'mlp', 'context_mlp')]
    assert max(counts)/min(counts) < 1.02


def test_checkpoint_rejects_duplicate_experience(tmp_path):
    Agent = api()
    import json
    agent = Agent()
    event = evidence_sequence()[0]
    agent.observe(event.before)
    agent.feedback(event)
    agent.save(tmp_path/'state.json')
    data = json.loads((tmp_path/'state.json').read_text())
    data['evidence'].append(data['evidence'][0])
    (tmp_path/'duplicate.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        Agent.load(tmp_path/'duplicate.json')
