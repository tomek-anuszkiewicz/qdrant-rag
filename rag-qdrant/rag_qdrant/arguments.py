"""Command-line argument construction for the RAG indexer."""

import argparse
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Build the supported rag_qdrant command-line interface."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("path", nargs="?", default=None, help="Directory path to index")
    parser.add_argument("-s", "--source", default=None, help="Source tag (e.g. 'project-a', 'engineering-notes')")
    parser.add_argument("-l", "--list-sources", action="store_true", help="List all indexed sources")
    parser.add_argument("--status", action="store_true", help="Show database connection and status")
    parser.add_argument("--reindex", action="store_true", help="Force re-index ignoring hash cache")
    parser.add_argument(
        "--include-dirs",
        nargs="+",
        default=None,
        help="Required top-level subdirectories to index (e.g. '01*' '02*')",
    )
    parser.add_argument("-h", "--help", action="store_true", help="Show help")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON for status or source listing")
    return parser


def create_search_parser() -> argparse.ArgumentParser:
    """Build the machine-readable semantic search command interface."""
    parser = argparse.ArgumentParser(prog="rag_qdrant search")
    parser.add_argument("query", help="Semantic query text")
    parser.add_argument("-s", "--source", default=None, help="Optional source tag or comma-separated source tags")
    parser.add_argument("--limit", type=int, default=5, help="Maximum result count")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def validate_indexing_arguments(args: argparse.Namespace) -> Optional[str]:
    """Return an actionable error when an indexing command lacks required scope."""
    if not args.source:
        return "The --source (-s) option is required."
    if not args.include_dirs:
        return "The --include-dirs option is required and must name one or more top-level directories."
    return None
