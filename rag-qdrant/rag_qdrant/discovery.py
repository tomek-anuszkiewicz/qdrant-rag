"""Markdown discovery rules for RAG source directories."""

import fnmatch
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


def is_included_path(path: Path, directory: Path, include_dirs: List[str]) -> bool:
    """Return whether a path is below one of the selected top-level directories."""
    relative_path = path.relative_to(directory)
    if len(relative_path.parts) < 2:
        return False

    top_level_dir = relative_path.parts[0]
    return any(
        top_level_dir.lower() == included.lower()
        or fnmatch.fnmatch(top_level_dir.lower(), included.lower())
        for included in include_dirs
    )


def discover_markdown_files(directory: Path, include_dirs: List[str]) -> List[Path]:
    """Find indexable Markdown files only below selected top-level directories."""
    if not include_dirs:
        raise ValueError("include_dirs must contain at least one top-level directory pattern")

    markdown_files = []
    for path in directory.rglob("*.md"):
        if _is_ignored(path):
            continue

        if not is_included_path(path, directory, include_dirs):
            continue

        markdown_files.append(path)
    return markdown_files
