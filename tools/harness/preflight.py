"""Small, offline repository checks for rag-qdrant."""

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "rag-qdrant"
FORBIDDEN_TRACKED = re.compile(
    r"(^|/)(?:\.env(?:\..*)?|qdrant_storage|[^/]*_rag_cache\.json|[^/]*\.pid|[^/]*\.log)(?:/|$)",
    re.IGNORECASE,
)
PORT = re.compile(r'^\s*-\s*["\']?([^\s"\']+:[0-9]+:[0-9]+)["\']?\s*$')
POLISH_DIACRITIC = re.compile("[\u0105\u0107\u0119\u0142\u0144\u00f3\u015b\u017a\u017c]", re.IGNORECASE)
LOCAL_PATH = re.compile(
    r"file:" r"//|(?<![A-Za-z0-9])[A-Za-z]:[\\/](?:Users|AI|Programowanie)[\\/]"
    r"|/(?:home|Users)/[^/\s<>]+",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {".md", ".py", ".ps1", ".bat", ".sh", ".json", ".toml", ".yml", ".yaml", ".txt"}


def tracked_paths():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "-z"], cwd=ROOT, capture_output=True, check=True
    )
    return [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]


def check_tracked_state():
    bad = [path for path in tracked_paths() if FORBIDDEN_TRACKED.search(path.replace("\\", "/"))]
    if bad:
        raise ValueError("Local state is tracked or staged: " + ", ".join(bad))


def check_compose_ports():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    ports = [match.group(1) for line in compose.splitlines() if (match := PORT.match(line))]
    if not ports:
        raise ValueError("No published Docker ports found; review the Compose port format")
    if any(not port.startswith("127.0.0.1:") for port in ports):
        raise ValueError("Qdrant Docker ports must publish on 127.0.0.1 only")


def check_python_syntax():
    for path in sorted(PACKAGE.rglob("*.py")):
        if any(part in {".venv", "venv", "__pycache__"} for part in path.parts):
            continue
        compile(path.read_bytes(), str(path), "exec")


def scan_added_lines(diff):
    """Find narrow language and local-path violations in an added-lines diff."""
    path = None
    line_number = 0
    violations = []
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@ "):
            match = re.search(r"\+(\d+)", line)
            if match:
                line_number = int(match.group(1))
        elif path and line.startswith("+"):
            content = line[1:]
            if Path(path).suffix.lower() in TEXT_SUFFIXES:
                if POLISH_DIACRITIC.search(content):
                    violations.append(f"{path}:{line_number}: Polish diacritic in added text")
                if LOCAL_PATH.search(content):
                    violations.append(f"{path}:{line_number}: local absolute path in added text")
            line_number += 1
        elif path and line.startswith(" "):
            line_number += 1
    return violations


def check_staged_text():
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "diff", "--cached", "--no-ext-diff", "--unified=0", "--no-color", "--text", "--"],
        cwd=ROOT, capture_output=True, check=True
    )
    violations = scan_added_lines(result.stdout.decode("utf-8", errors="replace"))
    if violations:
        raise ValueError("Staged content violations:\n" + "\n".join(violations))


def run_full_tests():
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-q"],
        cwd=PACKAGE,
    )
    if result.returncode:
        raise ValueError(f"unit tests failed (exit {result.returncode})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also run unit tests (requires project dependencies)")
    args = parser.parse_args()
    try:
        check_tracked_state()
        check_compose_ports()
        check_python_syntax()
        check_staged_text()
        if args.full:
            run_full_tests()
    except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError) as exc:
        print(f"preflight: FAIL: {exc}", file=sys.stderr)
        return 1
    print("preflight: PASS (tracked state, loopback ports, Python syntax, staged text" + (", unit tests" if args.full else "") + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
