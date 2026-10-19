"""Re-run published experiments with the installed package; never modify reference data."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

EXPECTED_CHECKS = ['round1', 'round1-parameters', 'retention', 'round2',
    'round2-parameters', 'interest', 'interest-parameters', 'one-demo',
    'round2-demo', 'motivation-demo', 'teacher-transfer', 'teacher-parameters',
    'teacher-controls', 'wrong-teacher', 'cognitive', 'cognitive-training', 'cognitive-replay']


def compare(expected, actual, path='$'):
    """Strict structure and bool checking; 1e-10 numeric tolerance, no ignored fields."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return [] if type(expected) is type(actual) and expected == actual else [path]
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return [] if math.isfinite(expected) and math.isfinite(actual) and math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-10) else [path]
    if type(expected) is not type(actual):
        return [path]
    if isinstance(expected, dict):
        errors = [] if expected.keys() == actual.keys() else [path+'.keys']
        for key in sorted(expected.keys() & actual.keys()):
            errors.extend(compare(expected[key], actual[key], path+'.'+key))
        return errors
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [path+'.length']
        return [e for i, (a, b) in enumerate(zip(expected, actual)) for e in compare(a, b, f'{path}[{i}]')]
    return [] if expected == actual else [path]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparison_status(checks, *, completed=False, expected_count=17):
    return completed and len(checks) == expected_count and all(c['passed'] for c in checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='Public experiments directory')
    parser.add_argument('--output', type=Path, required=True, help='New scratch directory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference, output = args.reference.resolve(), args.output.resolve()
    import axiom
    if 'site-packages' not in Path(axiom.__file__).parts:
        raise RuntimeError('Run from an installed wheel, with PYTHONPATH unset, outside source tree')
    import torch
    torch.set_num_threads(1)
    results = []
    started = time.perf_counter()

    def verify(name, old, new, fields):
        expected, actual = read(reference/old), read(output/new)
        missing = [f for f in fields if f not in expected or f not in actual]
        differences = [e for f in fields if f not in missing for e in compare(expected[f], actual[f], f)]
        results.append(dict(name=name, fields=fields, missing=missing, differences=differences,
            reference_sha256=sha(reference/old), reproduced_sha256=sha(output/new),
            passed=not missing and not differences))
        save()
        print(f'{name}: {"PASS" if results[-1]["passed"] else "FAIL"}', flush=True)

    def save(completed=False, error_type=None):
        report = dict(classification='Engineering reproduction of public data, not a new independent study',
            installed_package=True, package_version=axiom.__version__,
            python=sys.version.split()[0], torch=torch.__version__,
            seconds=time.perf_counter()-started, checks=results,
            completed=completed, expected_checks=17,
            passed=comparison_status(results, completed=completed),
            status='failed' if error_type else 'completed' if completed else 'in_progress',
            error_type=error_type,
            next_expected_check=EXPECTED_CHECKS[len(results)] if len(results) < len(EXPECTED_CHECKS) else None,
            numeric_tolerance=dict(relative=1e-10, absolute=1e-10),
            exclusions='Wall time, host platform, timestamps and event/request UUIDs are not performance measurements. Selected result fields are explicit per check; no global filtering.')
        (output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')

    def verify_parameters(name, old, new, timing_field=None):
        expected, actual = read(reference/old), read(output/new)
        if timing_field:
            expected = [{k: v for k, v in row.items() if k != timing_field} for row in expected]
            actual = [{k: v for k, v in row.items() if k != timing_field} for row in actual]
        differences = compare(expected, actual)
        results.append(dict(name=name, excluded_field=timing_field, differences=differences,
            reference_sha256=sha(reference/old), reproduced_sha256=sha(output/new),
            passed=not differences))
        save()
        print(f'{name}: {"PASS" if not differences else "FAIL"}', flush=True)

    previous_exception_hook = sys.excepthook
    def failed(exc_type, value, traceback):
        save(error_type=exc_type.__name__)
        previous_exception_hook(exc_type, value, traceback)
    sys.excepthook = failed
    save()

    from axiom.benchmarks.run import run_benchmark
    from axiom.benchmarks.recovery import run_recovery_benchmark as run_recovery
    from axiom.benchmarks.retention import run_retention
    from axiom.frontier.interest import run_interest_experiment as run_experiment
    from axiom.examples.teach_menu import run_demo
    from axiom.examples.recovery import run_demo as recovery_demo
    from axiom.examples.motivation import run_demo as motivation_demo
    from axiom.benchmarks import cognitive
    from axiom.examples.first_teacher import prepare_lesson, resume_lesson, evaluate_transfer, load_selector_state
    from axiom.learning.recovery import RecoveryLinear
    from axiom.tools.teacher_bridge import TeacherReply
    from axiom.environments.classroom import ClassroomMenu
    from axiom.environments.menu import execute_method

    # Original parameters, including all five seeds and all arms; no smoke substitution.
    run_benchmark(output/'validated', seeds=range(5), train_tasks=240, tasks_per_cell=20)
    verify('round1', 'validated/summary.json', 'validated/summary.json', ['config', 'aggregate', 'cells', 'per_seed', 'paired_axiom_minus_baseline', 'integrity'])
    verify_parameters('round1-parameters', 'validated/parameter_audit.json', 'validated/parameter_audit.json', 'training_seconds')
    run_retention(output/'retention')
    verify('retention', 'retention/retention.json', 'retention/retention.json', ['config', 'aggregate', 'states', 'episodes'])
    run_recovery(output/'round2', seeds=range(5), train_tasks=480, tasks_per_cell=30)
    verify('round2', 'round2/summary.json', 'round2/summary.json', ['config', 'all_tasks', 'failed_prefix', 'per_seed_failed_prefix', 'paired_recovery_minus_baseline', 'prediction_error', 'cells', 'integrity'])
    verify_parameters('round2-parameters', 'round2/parameter_audit.json', 'round2/parameter_audit.json', 'fit_seconds')
    run_experiment(output/'interest', seeds=range(5), train_per_topic=24, eval_tasks=60)
    verify('interest', 'frontier/interest-v02/summary.json', 'interest/summary.json', ['config', 'metrics', 'paired_test_improvement', 'selections', 'integrity'])
    verify_parameters('interest-parameters', 'frontier/interest-v02/parameter_audit.json', 'interest/parameter_audit.json')
    run_demo(output/'one-demo')
    verify('one-demo', 'one-demo/demo.json', 'one-demo/demo.json', ['before_digest', 'after_recording_digest', 'learning', 'before_choice', 'after_choice', 'rollback_digest', 'skills', 'aggregate', 'cells', 'episodes'])
    recovery_demo(output/'round2-demo')
    verify('round2-demo', 'round2-demo/demo.json', 'round2-demo/demo.json', ['initial_choice', 'recorded_choice', 'trained_choice', 'initial_digest', 'recorded_digest', 'trained_digest', 'rollback_digest', 'learning', 'episode'])
    motivation_demo(output/'motivation-demo')
    verify_parameters('motivation-demo', 'frontier/motivation-demo-v02/demo.json', 'motivation-demo/demo.json')

    state = load_selector_state(output/'round2/seed-0-clean/checkpoint.json')
    folder = output/'first-teacher'
    prepare_lesson(folder, state, seed=0, test_namespace='first-live-lesson-v1')
    reply = read(reference/'core/first-live-teacher/teacher-reply.json')
    reply['request_id'] = read(folder/'request.json')['request_id']
    resume_lesson(folder, TeacherReply.model_validate(reply), teacher_source='recorded-public-answer-replay')
    evaluate_transfer(folder)
    verify('teacher-transfer', 'core/first-live-teacher/transfer-summary.json', 'first-teacher/transfer-summary.json', ['metrics', 'cells', 'integrity'])
    verify('teacher-parameters', 'core/first-live-teacher/lesson.json', 'first-teacher/lesson.json', ['accepted', 'success', 'operations', 'trace', 'before_digest', 'recorded_digest', 'trained_digest', 'learning', 'traversal_retained', 'exact_rollback', 'test_sha256'])
    controls = output/'controls'
    controls.mkdir()
    summary = dict(classification='Post-lesson diagnostic and synthetic fault controls; not new live teaching or confirmatory tests')
    summary['teaching_task_traversal_counterfactual'] = execute_method(ClassroomMenu(0, split='lesson'), 'traverse', {}).model_dump()
    summary['prior_selector'] = prepare_lesson(controls/'prior', RecoveryLinear().state(), seed=0)
    summary['teacher_denied'] = prepare_lesson(controls/'denied', state, seed=0, allow_ask=False)
    pending = prepare_lesson(controls/'no_reply', state, seed=0)
    summary['no_reply'] = dict(status=pending['status'], spent=pending['spent'], note='Pending request persists; no additional action or learning occurs without a reply.')
    prepare_lesson(controls/'wrong_teacher', state, seed=0)
    reply = TeacherReply(request_id=read(controls/'wrong_teacher/request.json')['request_id'], rule={'audio': 'canvas'}, explanation='Synthetic diagnostic replay')
    summary['wrong_teacher'] = resume_lesson(controls/'wrong_teacher', reply, teacher_source='synthetic-fault-injection')
    (controls/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    verify('teacher-controls', 'core/first-teacher-controls/summary.json', 'controls/summary.json', ['teaching_task_traversal_counterfactual', 'prior_selector', 'teacher_denied', 'no_reply'])
    verify('wrong-teacher', 'core/first-teacher-controls/wrong_teacher/lesson.json', 'controls/wrong_teacher/lesson.json', ['accepted', 'success', 'operations', 'trace', 'before_digest', 'trained_digest', 'learning', 'traversal_retained', 'exact_rollback'])

    cognitive.freeze(output/'cognitive')
    cognitive.train(output/'cognitive')
    cognitive.evaluate(output/'cognitive')
    cognitive.replay(output/'cognitive')
    verify('cognitive', 'core/cognitive-v1/summary.json', 'cognitive/summary.json', ['overall', 'cells', 'seed_metrics', 'primary', 'evaluation_weights_unchanged', 'test_tasks', 'test_executions', 'api_calls', 'artifacts'])
    verify('cognitive-training', 'core/cognitive-v1/training.json', 'cognitive/training.json', ['audits', 'checkpoints', 'training_data_hashes', 'api_calls', 'api_cost_usd', 'test_results_inspected'])
    verify('cognitive-replay', 'core/cognitive-v1/replay-verification.json', 'cognitive/replay-verification.json', ['matched_decisions', 'environment_feedback', 'candidates', 'hidden_states', 'checkpoint_restoration'])
    save(completed=True)
    raise SystemExit(0 if comparison_status(results, completed=True) else 1)


if __name__ == '__main__':
    main()
