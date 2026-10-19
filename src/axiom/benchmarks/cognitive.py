"""Frozen, offline conditional-selection study. No oracle labels reach a policy."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from axiom.environments.contextual import ACTIONS, ContextMenu, Evidence, make_session
from axiom.learning.cognitive import CognitiveSelector, HistoryStatistics, KINDS

REGIMES = ('reliable', 'unreliable', 'partial', 'changing')
FINAL_POLICIES = ('traverse', 'rule_memory', 'ema_greedy', 'linear', 'mlp',
    'context_mlp', 'gru', 'gru_untrained', 'gru_reset_task', 'gru_no_help', 'gru_no_replay')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = ('benchmarks/cognitive.py', 'learning/cognitive.py', 'environments/contextual.py')
    return {p: sha(root/p) for p in paths}


def freeze(output, *, seeds=5, train_a=16, train_b=48, train_tasks=16,
           test_sessions=16, test_tasks=24, epochs_a=60, epochs_b=100, namespace='cognitive-v1'):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be new or empty')
    config = dict(seeds=seeds, train_a=train_a, train_b=train_b, train_tasks=train_tasks,
        test_sessions=test_sessions, test_tasks=test_tasks, epochs_a=epochs_a, epochs_b=epochs_b)
    if any(type(v) is not int or v < 1 for v in config.values()) or test_sessions % 4:
        raise ValueError('positive counts and a multiple of four test sessions required')
    if not isinstance(namespace, str) or not namespace:
        raise ValueError('namespace required')
    config.update(namespace=namespace, exploration=.05)
    output.mkdir(parents=True, exist_ok=True)
    training, testing = {}, {}
    for seed in range(seeds):
        training[str(seed)] = {}
        for phase, count in (('A', train_a), ('B', train_b)):
            sessions = []
            for i in range(count):
                regime = 'reliable' if phase == 'A' else REGIMES[1+i%3]
                sessions.append(dict(session_id=f'train-{phase}-{i}', regime=regime,
                    cases=make_session(f'train-{namespace}-{phase}', seed*10000+i, regime, tasks=train_tasks)))
            training[str(seed)][phase] = sessions
        testing[str(seed)] = [dict(session_id=f'test-{i}', regime=REGIMES[i%4],
            cases=make_session(f'test-{namespace}', seed*10000+i, REGIMES[i%4], tasks=test_tasks))
            for i in range(test_sessions)]
    train_ids = {c['public']['task_id'] for phases in training.values() for sessions in phases.values()
        for s in sessions for c in s['cases']}
    test_ids = {c['public']['task_id'] for sessions in testing.values() for s in sessions for c in s['cases']}
    assert not train_ids & test_ids
    write(output/'train-cases.json', training)
    write(output/'test-cases.json', testing)
    protocol = dict(primary='paired seed mean utility: gru minus context_mlp',
        learning='chosen-action within-task Monte Carlo MSE + 0.25 immediate-success BCE',
        optimizer='Adam lr=0.003; fresh optimizer per phase; full-sequence causal replay',
        reward='completion - 0.6*consumed/budget', comparison='same experience, tasks, operation budgets; matched neural parameter counts',
        state='reset between software sessions; persist between tasks; frozen evaluation weights',
        config=config)
    write(output/'protocol.json', protocol)
    source_doc = Path(__file__).resolve().parents[3]/'docs/cognitive-research.md'
    if source_doc.is_file():
        (output/'design-at-freeze.md').write_bytes(source_doc.read_bytes())
    sealed = ['train-cases.json', 'test-cases.json', 'protocol.json']
    if (output/'design-at-freeze.md').exists():
        sealed.append('design-at-freeze.md')
    write(output/'manifest.json', dict(config=config, created_utc=datetime.now(timezone.utc).isoformat(),
        files={name: sha(output/name) for name in sealed},
        source_hashes=source_hashes(), train_tasks=len(train_ids), test_tasks=len(test_ids),
        train_test_overlap=0, boundary='Policies receive PublicState and verified Evidence only.'))
    return read(output/'manifest.json')


def check(output):
    manifest = read(output/'manifest.json')
    if any(sha(output/name) != value for name, value in manifest['files'].items()):
        raise ValueError('frozen fixture hash changed')
    if source_hashes() != manifest['source_hashes']:
        raise ValueError('source hash changed; preserve run and create a new version')
    if json.dumps(manifest['config'], sort_keys=True) != json.dumps(read(output/'protocol.json')['config'], sort_keys=True):
        raise ValueError('runtime config differs from sealed protocol')
    return manifest


def collect(sessions, seed, path):
    sequences = []
    with path.open('x', encoding='utf-8') as stream:
        for session in sessions:
            rng = random.Random(f'behavior:{seed}:{session["session_id"]}')
            events = []
            for case in session['cases']:
                env = ContextMenu(case)
                while not env.observe().done:
                    events.append(env.step(rng.choice(env.observe().available)))
            # Evaluable labels stay in private case files, not this learner data.
            stream.write(json.dumps(dict(session_id=session['session_id'],
                events=[e.model_dump(mode='json') for e in events]), allow_nan=False)+'\n')
            sequences.append(events)
    return sequences


def train(output):
    output = Path(output)
    manifest = check(output)
    if (output/'training-started.json').exists():
        raise ValueError('training already started; preserve interrupted or completed run')
    write(output/'training-started.json', dict(utc=datetime.now(timezone.utc).isoformat()))
    config, cases = manifest['config'], read(output/'train-cases.json')
    torch.set_num_threads(1)
    started = time.perf_counter()
    audits, checkpoints = [], {}
    for seed in range(config['seeds']):
        folder = output/f'seed-{seed}'
        folder.mkdir()
        a = collect(cases[str(seed)]['A'], seed, folder/'experience-A.jsonl')
        b = collect(cases[str(seed)]['B'], seed, folder/'experience-B.jsonl')
        for kind in KINDS:
            agent = CognitiveSelector(kind=kind, seed=seed)
            initial = folder/f'{kind}-initial.json'
            agent.save(initial)
            first = agent.learn(a, epochs=config['epochs_a'])
            agent.save(folder/f'{kind}-A.json')
            second = agent.learn(a+b, epochs=config['epochs_b'])
            agent.save(folder/f'{kind}-B.json')
            if kind == 'gru':
                no_replay = CognitiveSelector.load(folder/'gru-A.json')
                no_replay_receipt = no_replay.learn(b, epochs=config['epochs_b'])
                no_replay.save(folder/'gru-no-replay.json')
                write(folder/'no-replay-training.json', no_replay_receipt)
            # Record a tested exact parameter rollback without replacing checkpoint.
            trained = agent.parameter_digest
            agent.rollback()
            assert agent.parameter_digest == first['after']
            audits.append(dict(seed=seed, kind=kind, phase_a=first, phase_b=second,
                exact_rollback=True, trained_digest=trained))
            print(f'trained seed={seed} kind={kind} transitions={second["verified_transitions"]}', flush=True)
        for path in folder.glob('*.json'):
            if '-training' not in path.name:
                checkpoints[path.relative_to(output).as_posix()] = sha(path)
    report = dict(seconds=time.perf_counter()-started, device='cpu', torch_threads=1,
        torch_version=torch.__version__, numpy_version=np.__version__, checkpoints=checkpoints, audits=audits,
        training_data_hashes={p.relative_to(output).as_posix(): sha(p) for p in output.glob('seed-*/experience-*.jsonl')},
        api_calls=0, api_cost_usd=0, test_results_inspected=False)
    write(output/'training.json', report)
    return report


def policy_agent(output, seed, name):
    folder = output/f'seed-{seed}'
    if name.startswith('phase_a_'):
        return CognitiveSelector.load(folder/f'{name.removeprefix("phase_a_")}-A.json')
    if name in KINDS:
        return CognitiveSelector.load(folder/f'{name}-B.json')
    if name in {'gru_reset_task', 'gru_no_help'}:
        return CognitiveSelector.load(folder/'gru-B.json')
    if name == 'gru_untrained':
        return CognitiveSelector.load(folder/'gru-initial.json')
    if name == 'gru_no_replay':
        return CognitiveSelector.load(folder/'gru-no-replay.json')
    return None


def run_policy(sessions, seed, name, agent, trace, epsilon):
    rows = []
    for session in sessions:
        stats = HistoryStatistics()
        if agent is not None:
            agent.reset_context()
        for index, case in enumerate(session['cases']):
            env = ContextMenu(case)
            if agent is not None and name == 'gru_reset_task':
                agent.hidden.zero_()
            events, exploration, errors, methods, state_movement = [], 0, [], [], []
            while not env.observe().done:
                state = env.observe()
                inquiry = name != 'gru_no_help'
                available = [a for a in state.available if inquiry or a != 'ask']
                hidden = None
                probability = None
                if agent is not None:
                    agent.observe(state)
                    candidates = agent.suggest(inquiry=inquiry)
                    hidden = agent.hidden_values
                    statistics = agent.statistics.vector()
                else:
                    statistics = stats.vector()
                    values = stats.scores(state) if name == 'ema_greedy' else {
                        'traverse': 1., 'semantic': 2. if name == 'rule_memory' else 0., 'ask': -1.}
                    candidates = sorted([dict(method=a, score=values[a]) for a in available],
                        key=lambda c: (-c['score'], c['method']))
                chosen = candidates[0]['method']
                rng = random.Random(f'evaluate:{seed}:{state.task_id}:{len(events)}')
                explore = name not in {'traverse', 'rule_memory'} and rng.random() < epsilon
                if explore:
                    chosen = rng.choice(available)
                if agent is not None:
                    probability = next(c['success_probability'] for c in candidates if c['method'] == chosen)
                event = env.step(chosen)
                if agent is not None:
                    agent.feedback(event)
                    after_hidden = agent.hidden_values
                    state_movement.append(float(np.linalg.norm(np.asarray(hidden)-after_hidden)))
                    errors.append((probability-float(event.after.success))**2)
                    after_stats = agent.statistics.vector()
                else:
                    stats.update(event)
                    after_hidden, after_stats = None, stats.vector()
                trace.write(json.dumps(dict(seed=seed, policy=name, session_id=session['session_id'],
                    task_index=index, step=len(events), event=event.model_dump(mode='json'),
                    candidates=candidates, exploration=explore, hidden_before=hidden, hidden_after=after_hidden,
                    statistics_before=statistics, statistics_after=after_stats), allow_nan=False)+'\n')
                events.append(event)
                exploration += int(explore)
                methods.append(chosen)
            rows.append(dict(seed=seed, policy=name, regime=session['regime'], session_id=session['session_id'],
                task_index=index, half='early' if index < len(session['cases'])//2 else 'late',
                task_id=env.observe().task_id, success=float(env.observe().success),
                utility=sum(e.reward for e in events), operations=env.observe().spent,
                teacher_calls=sum(e.action == 'ask' for e in events),
                teacher_refusals=sum(e.action == 'ask' and not e.responded for e in events),
                semantic_attempts=sum(e.action == 'semantic' for e in events),
                wrong_transfer=sum(e.action == 'semantic' and not e.after.success for e in events),
                traversal_steps=sum(e.action == 'traverse' for e in events),
                first_semantic=float(methods[0] == 'semantic'), first_ask=float(methods[0] == 'ask'),
                exploration_steps=exploration, decisions=len(events),
                exploration_cost=sum(e.cost for i, e in enumerate(events) if
                    name not in {'traverse', 'rule_memory'} and random.Random(f'evaluate:{seed}:{e.before.task_id}:{i}').random() < epsilon),
                brier=float(np.mean(errors)) if errors else None,
                mean_state_change=float(np.mean(state_movement)) if state_movement else 0.))
    return rows


METRICS = ('success', 'utility', 'operations', 'teacher_calls', 'teacher_refusals',
    'semantic_attempts', 'wrong_transfer', 'traversal_steps', 'first_semantic', 'first_ask',
    'exploration_steps', 'exploration_cost', 'decisions', 'brier', 'mean_state_change')


def aggregate(rows):
    result = dict(n=len(rows))
    for key in METRICS:
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def evaluate(output):
    output = Path(output)
    manifest = check(output)
    training = read(output/'training.json')
    if any(sha(output/name) != value for name, value in training['checkpoints'].items()):
        raise ValueError('training checkpoint hash changed')
    if (output/'evaluation-started.json').exists():
        raise ValueError('evaluation already started; preserve all outputs')
    write(output/'evaluation-started.json', dict(utc=datetime.now(timezone.utc).isoformat()))
    torch.set_num_threads(1)
    started = time.perf_counter()
    cases = read(output/'test-cases.json')
    rows, unchanged = [], True
    names = FINAL_POLICIES + tuple('phase_a_'+k for k in KINDS)
    for seed in range(manifest['config']['seeds']):
        for name in names:
            agent = policy_agent(output, seed, name)
            before = agent.parameter_digest if agent is not None else None
            with (output/f'seed-{seed}'/f'trace-{name}.jsonl').open('x', encoding='utf-8') as stream:
                rows.extend(run_policy(cases[str(seed)], seed, name, agent, stream, manifest['config']['exploration']))
            if agent is not None:
                unchanged &= before == agent.parameter_digest
                if name == 'gru':
                    agent.save(output/f'seed-{seed}'/'growth-final.json')
            print(f'evaluated seed={seed} policy={name}', flush=True)
    with (output/'episodes.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    seed_metrics = [dict(seed=seed, policy=name, **aggregate([r for r in rows if r['seed'] == seed and r['policy'] == name]))
        for seed in range(manifest['config']['seeds']) for name in names]
    differences = [next(r['utility'] for r in seed_metrics if r['seed'] == seed and r['policy'] == 'gru') -
        next(r['utility'] for r in seed_metrics if r['seed'] == seed and r['policy'] == 'context_mlp')
        for seed in range(manifest['config']['seeds'])]
    # Preregistered five-seed Student-t interval, not a per-episode independence claim.
    mean = float(np.mean(differences))
    half_width = 2.7764451052*float(np.std(differences, ddof=1))/np.sqrt(5) if len(differences) == 5 else None
    summary = dict(seed_metrics=seed_metrics,
        overall={name: aggregate([r for r in rows if r['policy'] == name]) for name in names},
        cells=[dict(policy=name, regime=regime, half=half,
            **aggregate([r for r in rows if r['policy'] == name and r['regime'] == regime and r['half'] == half]))
            for name in names for regime in REGIMES for half in ('early', 'late')],
        primary=dict(comparison='gru minus context_mlp utility', paired_seed_differences=differences,
            mean=mean, ci95=[mean-half_width, mean+half_width] if half_width is not None else None),
        evaluation_weights_unchanged=bool(unchanged), seconds=time.perf_counter()-started,
        test_tasks=manifest['test_tasks'], test_executions=len(rows), api_calls=0,
        artifacts={p.relative_to(output).as_posix(): sha(p) for p in
            [output/'episodes.jsonl', *output.glob('seed-*/trace-*.jsonl'), *output.glob('seed-*/growth-final.json')]},
        limitation='Frozen parameters, causal within-session adaptation; not open-world or independent laboratory validation.')
    write(output/'summary.json', summary)
    return summary


def replay(output):
    """Re-execute every final GRU decision and feedback against the sealed world."""
    output = Path(output)
    manifest = check(output)
    training = read(output/'training.json')
    summary = read(output/'summary.json')
    torch.set_num_threads(1)
    cases = read(output/'test-cases.json')
    matched = 0
    for seed in range(manifest['config']['seeds']):
        name = f'seed-{seed}/gru-B.json'
        if sha(output/name) != training['checkpoints'][name]:
            raise ValueError('checkpoint hash changed')
        agent = CognitiveSelector.load(output/name)
        with (output/f'seed-{seed}/trace-gru.jsonl').open(encoding='utf-8') as stream:
            for session in cases[str(seed)]:
                agent.reset_context()
                for index, case in enumerate(session['cases']):
                    env = ContextMenu(case)
                    step = 0
                    while not env.observe().done:
                        line = next(stream, None)
                        if line is None:
                            raise ValueError('replay coverage incomplete')
                        row = json.loads(line)
                        if (row['seed'], row['policy'], row['session_id'], row['task_index'], row['step']) != (
                            seed, 'gru', session['session_id'], index, step):
                            raise ValueError('replay coverage/order mismatch')
                        event = Evidence.model_validate(row['event'])
                        if env.observe() != event.before:
                            raise ValueError('replay observation mismatch')
                        agent.observe(event.before)
                        if agent.hidden_values != row['hidden_before'] or agent.suggest() != row['candidates']:
                            raise ValueError('replay state or candidate mismatch')
                        rng = random.Random(f'evaluate:{seed}:{event.before.task_id}:{step}')
                        explore = rng.random() < manifest['config']['exploration']
                        expected = rng.choice(list(event.before.available)) if explore else agent.suggest()[0]['method']
                        if event.action != expected or row['exploration'] != explore:
                            raise ValueError('replay action-selection decision mismatch')
                        if env.step(event.action) != event:
                            raise ValueError('replay simulator mismatch')
                        agent.feedback(event)
                        if agent.hidden_values != row['hidden_after']:
                            raise ValueError('replay feedback mismatch')
                        matched += 1
                        step += 1
            if next(stream, None) is not None:
                raise ValueError('replay coverage contains extra decisions')
        restored = CognitiveSelector.load(output/f'seed-{seed}/growth-final.json')
        if restored.hidden_values != agent.hidden_values or restored.parameter_digest != agent.parameter_digest:
            raise ValueError('persisted growth state mismatch')
    if any(sha(output/name) != expected for name, expected in summary['artifacts'].items()):
        raise ValueError('evaluation artifact hash changed')
    result = dict(matched_decisions=matched, environment_feedback=True,
        candidates=True, hidden_states=True, checkpoint_restoration=True)
    write(output/'replay-verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'train', 'evaluate', 'replay'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='Development execution check, not a performance study')
    args = parser.parse_args()
    if args.command == 'freeze':
        kwargs = dict(seeds=1, train_a=2, train_b=3, train_tasks=3, test_sessions=4,
            test_tasks=4, epochs_a=2, epochs_b=2, namespace='development') if args.smoke else {}
        result = freeze(args.output, **kwargs)
    else:
        result = globals()[args.command](args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in {'audits', 'cells', 'seed_metrics', 'checkpoints'}}, indent=2))


if __name__ == '__main__':
    main()
