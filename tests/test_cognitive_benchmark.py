import importlib.util

import pytest

pytest.importorskip('torch')


def api():
    assert importlib.util.find_spec('axiom.benchmarks.cognitive') is not None, 'benchmark missing'
    from axiom.benchmarks.cognitive import freeze, train, evaluate, replay, read
    return freeze, train, evaluate, replay, read


def test_frozen_pipeline_restore_and_causal_trajectory(tmp_path):
    freeze, train, evaluate, replay, read = api()
    folder = tmp_path/'smoke'
    freeze(folder, seeds=1, train_a=2, train_b=3, train_tasks=3, test_sessions=4, test_tasks=4, epochs_a=2, epochs_b=2, namespace='development')
    manifest = read(folder/'manifest.json')
    assert manifest['train_test_overlap'] == 0
    train(folder)
    result = evaluate(folder)
    assert len(result['seed_metrics']) == 15  # 11 final policies + four phase-A policies
    assert all(row['n'] == 16 for row in result['seed_metrics'])
    assert result['evaluation_weights_unchanged']
    assert replay(folder)['matched_decisions'] > 0
    with pytest.raises(ValueError):
        evaluate(folder)
    with pytest.raises(ValueError):
        freeze(folder)


def test_modified_fixtures_rejected_before_training(tmp_path):
    freeze, train, _, _, _ = api()
    folder = tmp_path/'changed'
    freeze(folder, seeds=1, train_a=1, train_b=3, train_tasks=1, test_sessions=4, test_tasks=2, epochs_a=1, epochs_b=1, namespace='development')
    path = folder/'test-cases.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='hash'):
        train(folder)


def test_replay_checks_exploration_decision_not_just_feedback(tmp_path):
    freeze, train, evaluate, replay, read = api()
    import json
    folder = tmp_path/'decision'
    freeze(folder, seeds=1, train_a=1, train_b=3, train_tasks=1, test_sessions=4,
        test_tasks=2, epochs_a=1, epochs_b=1, namespace='development-decision')
    train(folder)
    evaluate(folder)
    path = folder/'seed-0/trace-gru.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['exploration'] = not rows[0]['exploration']
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match='decision'):
        replay(folder)


def test_replay_requires_all_sessions(tmp_path):
    freeze, train, evaluate, replay, read = api()
    import json
    folder = tmp_path/'coverage'
    freeze(folder, seeds=1, train_a=1, train_b=3, train_tasks=1, test_sessions=4,
        test_tasks=2, epochs_a=1, epochs_b=1, namespace='development-coverage')
    train(folder)
    evaluate(folder)
    path = folder/'seed-0/trace-gru.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['session_id'] == 'test-3'))
    with pytest.raises(ValueError, match='coverage|order'):
        replay(folder)


def test_run_config_cannot_diverge_from_sealed_protocol(tmp_path):
    freeze, train, _, _, read = api()
    import json
    folder = tmp_path/'config'
    freeze(folder, seeds=1, train_a=1, train_b=3, train_tasks=1, test_sessions=4,
        test_tasks=2, epochs_a=1, epochs_b=1, namespace='development-config')
    data = read(folder/'manifest.json')
    data['config']['epochs_b'] = 2
    (folder/'manifest.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='config'):
        train(folder)
