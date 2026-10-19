import json

import pytest


def api():
    import axiom.environments as package
    import importlib.util
    assert importlib.util.find_spec(package.__name__ + '.contextual') is not None, 'contextual simulator missing'
    from axiom.environments.contextual import ContextMenu, make_session
    return ContextMenu, make_session


def test_public_schema_excludes_truth_and_replay_is_identical():
    Env, make = api()
    cases = make('unit', 7, 'partial', tasks=6)
    first, second = Env(cases[0]), Env(cases[0])
    public = first.observe().model_dump(mode='json')
    assert not {'regime', 'target', 'reliability', 'teacher_accuracy', 'probability', 'change_point'} & public.keys()
    assert 'teacher_reliability' not in json.dumps(public)
    while not first.observe().done:
        assert first.step('traverse') == second.step('traverse')
    assert first.observe() == second.observe()
    assert len({c['public']['task_id'] for c in cases}) == 6


def test_hidden_truth_does_not_change_initial_features():
    Env, make = api()
    from axiom.environments.contextual import observation_vector
    import copy
    case = make('unit', 1, 'reliable', tasks=1)[0]
    changed = copy.deepcopy(case)
    changed['target'] = next(m['id'] for m in case['public']['menus'] if m['id'] != case['target'])
    changed['teacher_accuracy'] = 0
    changed['answer_probability'] = 0
    assert Env(case).observe() == Env(changed).observe()
    assert observation_vector(Env(case).observe()) == observation_vector(Env(changed).observe())


def test_budget_permissions_and_candidates_before_failure():
    Env, make = api()
    case = make('unit', 8, 'reliable', tasks=1)[0]
    case['public'].update(allow_ask=False, budget=1)
    env = Env(case)
    assert set(env.observe().available) == {'traverse', 'semantic'}
    with pytest.raises(PermissionError):
        env.step('ask')
    env.step('semantic')
    assert env.observe().done
    with pytest.raises(PermissionError):
        env.step('traverse')


def test_teacher_refusal_is_an_observation_not_false_verification():
    Env, make = api()
    case = make('unit', 9, 'reliable', tasks=1)[0]
    case['public'].update(allow_ask=True, ask_cost=1, budget=6)
    case['answer_probability'] = 0
    env = Env(case)
    result = env.step('ask')
    assert not result.responded and not result.after.success
    assert result.cost == 1 and result.after.opened == ()
    assert 'ask' not in result.after.available
    # A refusal has no permanent global permission effect on another task.
    assert 'ask' in Env(case).observe().available
