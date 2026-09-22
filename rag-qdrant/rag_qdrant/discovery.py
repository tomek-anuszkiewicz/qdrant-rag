"""Markdown discovery rules for RAG source directories."""

from pathlib import Path
from typing import List


BASE_IGNORED_PARTS = {
    ".git",
    ".antigravity",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".system_generated",
    ".obsidian",
    ".smart-env",
    ".trash",
}


def _is_ignored(path: Path) -> bool:
    """Return whether a path is excluded by built-in system or privacy rules."""
    for part in path.parts:
        part_lower = part.lower()
        if part_lower in BASE_IGNORED_PARTS:
            return True
        if part_lower == "_private" or "private" in part_lower:
            return True
    return False


def discover_markdown_files(directory: Path) -> List[Path]:
    """Find every indexable Markdown file at or below the selected root."""
    markdown_files = []
    for path in directory.rglob("*.md"):
        if _is_ignored(path):
            continue
        markdown_files.append(path)
    return sorted(markdown_files)
