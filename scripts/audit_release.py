"""Read-only, bounded archive-aware credential/privacy inventory; never emits values.

This is a screening tool, not proof of privacy or a substitute for human review.
Pass the release tree and every retained historical archive explicitly. No extraction.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import zipfile

IGNORE = {'.git', '.venv', '__pycache__', '.pytest_cache', 'build', 'dist', 'work'}
PATTERNS = {
    'credential': re.compile(r'\bsk-[A-Za-z0-9_-]{24,}|\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}|\bAKIA[0-9A-Z]{16}\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'personal_path': re.compile(r'(?:[A-Za-z]:[/\\]+Users[/\\]+[^/\\\s"<>]+|/(?:home|Users)/[^/\s"<>]+)'),
    'email_review': re.compile(r'\b[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    'authorization_review': re.compile(r'(?i)(?:bearer\s+[A-Za-z0-9_.-]{20,}|[?&](?:access_token|api_key|token|signature)=[A-Za-z0-9%_.-]{16,})'),
}


def audit(paths):
    result = dict(files=0, bytes=0, archives=[], inputs=[], findings=[], errors=[],
                  excluded=[], binary_unread=[], git_history='not scanned by this file/archive tool')
    limit = 512 * 1024 * 1024

    def inspect(name, data, depth=0):
        result['bytes'] += len(data)
        if len(data) > limit or result['bytes'] > 4 * 1024**3 or depth > 4:
            result['errors'].append(dict(path=name, kind='size_or_nesting_limit'))
            return
        base = name.split('!')[-1].replace('\\', '/').rsplit('/', 1)[-1]
        if base == '.env' or (base.startswith('.env.') and not base.endswith('.example')):
            result['findings'].append(dict(path=name, kind='credential_file', line=0))
        archive_kind = 'zip' if name.endswith(('.zip', '.whl')) else 'tar' if name.endswith(('.tar.gz', '.tgz', '.tar')) else None
        if archive_kind:
            result['archives'].append(dict(path=name, sha256=hashlib.sha256(data).hexdigest()))
            try:
                if archive_kind == 'zip':
                    with zipfile.ZipFile(io.BytesIO(data)) as arc:
                        for item in arc.infolist():
                            if item.is_dir():
                                continue
                            if item.file_size > limit:
                                raise ValueError('oversize member')
                            inspect(name+'!'+item.filename, arc.read(item), depth+1)
                else:
                    with tarfile.open(fileobj=io.BytesIO(data)) as arc:
                        for item in arc:
                            if item.isdir():
                                continue
                            if not item.isfile() or item.size > limit:
                                raise ValueError('unsupported member')
                            inspect(name+'!'+item.name, arc.extractfile(item).read(), depth+1)
            except (OSError, ValueError, RuntimeError, EOFError, zipfile.BadZipFile, tarfile.TarError):
                result['errors'].append(dict(path=name, kind='unreadable_archive'))
            return
        result['files'] += 1
        try:
            text = data.decode('utf-8-sig')
        except UnicodeError:
            result['binary_unread'].append(name)
            return
        for number, line in enumerate(text.splitlines(), 1):
            for kind, pattern in PATTERNS.items():
                if pattern.search(line):
                    result['findings'].append(dict(path=name, line=number, kind=kind))

    for value in paths:
        path = Path(value)
        result['inputs'].append(path.name)
        if not path.exists():
            result['errors'].append(dict(path=path.name, kind='missing_input'))
            continue
        candidates = sorted(path.rglob('*')) if path.is_dir() else [path]
        for file in candidates:
            if not file.is_file():
                continue
            relative = file.relative_to(path) if path.is_dir() else Path(file.name)
            if path.is_dir() and (any(p in IGNORE or p.endswith('.egg-info') for p in relative.parts)
                                  or file.suffix in {'.pyc', '.db'} or file.name.endswith('.db-journal')):
                result['excluded'].append(relative.as_posix())
                continue
            if file.stat().st_size > limit:
                result['errors'].append(dict(path=relative.as_posix(), kind='size_limit'))
                continue
            inspect(relative.as_posix(), file.read_bytes())
    result['clean'] = not (result['findings'] or result['errors'] or result['binary_unread'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: len(result[k]) if isinstance(result[k], list) else result[k]
                      for k in ('files', 'archives', 'findings', 'errors', 'binary_unread', 'clean')}))
    raise SystemExit(0 if result['clean'] else 1)


if __name__ == '__main__':
    main()
