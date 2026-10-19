"""Conservative local source scanner; reports paths/line numbers, never values."""
import re
from pathlib import Path
import sys

PATTERNS = [re.compile(p) for p in (
    r"\bsk-[A-Za-z0-9_-]{24,}", r"\bgh[pousr]_[A-Za-z0-9]{30,}",
    r"\bgithub_pat_[A-Za-z0-9_]{30,}", r"\bAKIA[0-9A-Z]{16}\b",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
)]
IGNORE = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist", "work"}


def scan(root):
    root = Path(root)
    findings = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in IGNORE for part in relative.parts):
            continue
        if path.name == ".env" or (path.name.startswith(".env.") and not path.name.endswith(".example")):
            findings.append(f"{relative}: environment credential file")
            continue
        if path.suffix in {".db", ".pyc", ".png", ".zip", ".whl"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if any(pattern.search(line) for pattern in PATTERNS):
                findings.append(f"{relative}:{number}: credential-like value")
    return findings


if __name__ == "__main__":
    findings = scan(sys.argv[1] if len(sys.argv) > 1 else ".")
    print("\n".join(findings) if findings else "No credential patterns found.")
    raise SystemExit(bool(findings))
