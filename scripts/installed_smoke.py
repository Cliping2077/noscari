"""Exercise the installed wheel in a fresh directory (use python -I)."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--install-wheel', type=Path)
    parser.add_argument('--without-torch', action='store_true')
    args = parser.parse_args()
    if args.install_wheel:
        import subprocess
        wheels = list(args.install_wheel.glob('*.whl'))
        if len(wheels) != 1:
            raise ValueError('CI dist directory must contain exactly one wheel')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', str(wheels[0])], check=True)
    import axiom
    if 'site-packages' not in Path(axiom.__file__).parts:
        raise RuntimeError('Smoke imported source rather than the installed package')
    args.output.mkdir(parents=True, exist_ok=False)
    from axiom.examples.teach_menu import run_demo
    from axiom.examples.recovery import run_demo as run_recovery
    from axiom import Axiom
    demo = run_demo(args.output/'demo')
    recovery = run_recovery(args.output/'recovery')
    restored = Axiom.load(args.output/'demo/checkpoint.json', args.output/'restored.db')
    assert restored.parameter_digest == demo['learning']['after_digest']
    restored.close()
    result = dict(version=axiom.__version__, imported_installed_wheel=True,
        restored_parameters=True, teaching_delta=demo['learning']['parameter_delta'],
        recovery_delta=recovery['learning']['parameter_delta'])
    if args.without_torch:
        assert importlib.util.find_spec('torch') is None, 'Core CI must genuinely omit torch'
        result['optional_torch_absent'] = True
    else:
        from axiom.benchmarks import cognitive
        folder = args.output/'cognitive'
        cognitive.freeze(folder, seeds=1, train_a=2, train_b=3, train_tasks=3,
            test_sessions=4, test_tasks=4, epochs_a=2, epochs_b=2, namespace='release-smoke')
        cognitive.train(folder)
        result['cognitive_test_executions'] = cognitive.evaluate(folder)['test_executions']
        result['replay'] = cognitive.replay(folder)
    (args.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
