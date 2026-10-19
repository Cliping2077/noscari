"""Release tools must not silently miss archived records or numerical regressions."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def tool(name):
    path = ROOT / 'scripts' / (name + '.py')
    assert path.is_file(), 'release tool has not been implemented'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_archive_secret_and_privacy_are_detected_without_values(tmp_path):
    secret = 'gh' + 'p_' + 'X' * 36
    private = 'C:' + '\\Users\\' + 'example-person' + '\\notes.txt'
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, 'w') as z:
        z.writestr('old.json', json.dumps({'key': secret, 'path': private}))
    with tarfile.open(tmp_path / 'history.tar.gz', 'w:gz') as t:
        info = tarfile.TarInfo('old.zip')
        info.size = len(nested.getvalue())
        t.addfile(info, io.BytesIO(nested.getvalue()))
    result = tool('audit_release').audit([tmp_path])
    assert {'credential', 'personal_path'} <= {f['kind'] for f in result['findings']}
    rendered = json.dumps(result)
    assert secret not in rendered and 'example-person' not in rendered
    assert 'history.tar.gz!old.zip!old.json' in rendered


def test_corrupt_archive_is_a_failure_not_clean(tmp_path):
    (tmp_path / 'old.zip').write_bytes(b'broken')
    result = tool('audit_release').audit([tmp_path])
    assert result['errors'] and not result['clean']


def test_archive_environment_file_detected_even_without_known_key(tmp_path):
    with zipfile.ZipFile(tmp_path / 'history.zip', 'w') as z:
        z.writestr('.env', 'PASSWORD=unknown-format')
    result = tool('audit_release').audit([tmp_path])
    assert any(f['kind'] == 'credential_file' for f in result['findings'])


def test_result_comparison_checks_all_cells_and_parameters():
    compare = tool('reproduce').compare
    a = {'aggregate': {'gru': {'success': .8}}, 'parameters': [1., 2.]}
    assert compare(a, a) == []
    assert compare(a, {'aggregate': {'gru': {'success': .81}}, 'parameters': [1., 2.]})
    assert compare(a, {'aggregate': {'gru': {'success': .8}}, 'parameters': [1., 3.]})
    assert compare(a, {'aggregate': {}})
    assert compare({'success': True}, {'success': 1})


def test_compare_tolerance_does_not_hide_large_or_nonfinite_values():
    compare = tool('reproduce').compare
    assert compare({'x': .1}, {'x': .1 + 1e-13}) == []
    assert compare({'x': .1}, {'x': .1001})
    assert compare({'x': float('nan')}, {'x': float('nan')})


def test_partial_or_empty_reproduction_is_never_a_pass():
    status = tool('reproduce').comparison_status
    assert not status([{'passed': True}], completed=False, expected_count=1)
    assert not status([], completed=True, expected_count=1)
    assert not status([{'passed': False}], completed=True, expected_count=1)
    assert status([{'passed': True}], completed=True, expected_count=1)
