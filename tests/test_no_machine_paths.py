"""No file of the repository may contain a path of a particular machine: every location comes from
configs/paths.yaml (git-ignored) or from the command line."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATTERN = re.compile(r"(/mnt/|/home/[a-z]|/media/|/Users/[A-Za-z]|[A-Za-z]:[\\/]Users)")  # noqa: path-scan
SKIP_DIRS = {".git", "__pycache__", "build", "dist", ".pytest_cache"}
TEXT = {".py", ".sh", ".md", ".yaml", ".yml", ".json", ".toml", ".txt", ".cfg", ".cpp", ".h", ".tex"}


ROOTS = ["gencpm", "configs", "reproduce", "tests", "docs", "README.md", "pyproject.toml", "setup.py",
         "environment.yml", ".gitignore"]


def candidate_files():
    files = []
    for r in ROOTS:
        p = REPO / r
        files += [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and not SKIP_DIRS & set(f.parts)]
    return [f for f in files if (f.suffix in TEXT or f.name == ".gitignore") and f.name != "paths.yaml"]


def test_no_machine_paths():
    hits = []
    for f in candidate_files():
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if PATTERN.search(line) and "noqa: path-scan" not in line:
                hits.append(f"{f.relative_to(REPO)}:{i}: {line.strip()[:120]}")
    assert not hits, "machine paths found:\n" + "\n".join(hits)
