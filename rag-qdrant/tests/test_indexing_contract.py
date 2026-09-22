"""Regression tests for the public RAG indexing interface."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SERVER_ROOT = PACKAGE_ROOT.parent / "amiga-rag-mcp-server"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(SERVER_ROOT))

from rag_qdrant.arguments import create_parser, create_search_parser, validate_indexing_arguments
from rag_qdrant.cache import normalize_cache
from rag_qdrant.discovery import discover_markdown_files
from rag_qdrant_command import run_rag_qdrant_json


class CliContractTests(unittest.TestCase):
    def test_removed_filter_options_are_reported_as_unknown(self):
        parser = create_parser()

        _, unknown = parser.parse_known_args(
            [".", "--source", "amiga", "--exclude", "private", "--no-root-notes", "--reindex"]
        )

        self.assertEqual(unknown, ["--exclude", "private", "--no-root-notes", "--reindex"])

    def test_index_json_is_required_for_indexing(self):
        args = create_parser().parse_args(
            [".", "--source", "amiga", "--index-json", "rag-index.json"]
        )

        self.assertEqual(args.path, ".")
        self.assertEqual(args.source, "amiga")
        self.assertEqual(args.index_json, "rag-index.json")
        self.assertIsNone(validate_indexing_arguments(args))

    def test_indexing_without_index_json_is_rejected(self):
        args = create_parser().parse_args([".", "--source", "amiga"])

        self.assertEqual(
            validate_indexing_arguments(args),
            "The --index-json option is required and must name the JSON index file.",
        )


class DiscoveryContractTests(unittest.TestCase):
    def test_ragignore_does_not_affect_discovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".ragignore").write_text("included\n", encoding="utf-8")
            indexed_note = root / "note.md"
            indexed_note.write_text("# Indexed\n", encoding="utf-8")

            files = discover_markdown_files(root)

        self.assertEqual(files, [indexed_note])

    def test_discovery_returns_all_markdown_below_the_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            root_note = root / "root.md"
            root_note.write_text("# Root\n", encoding="utf-8")
            (root / "01 Design").mkdir()
            selected_note = root / "01 Design" / "selected.md"
            selected_note.write_text("# Selected\n", encoding="utf-8")
            (root / "02 Other").mkdir()
            other_note = root / "02 Other" / "other.md"
            other_note.write_text("# Other\n", encoding="utf-8")
            (root / "_Private").mkdir()
            (root / "_Private" / "secret.md").write_text("# Secret\n", encoding="utf-8")

            files = discover_markdown_files(root)

        self.assertEqual(files, [root_note, selected_note, other_note])


class LauncherContractTests(unittest.TestCase):
    def test_rag_qdrant_launchers_replace_the_legacy_names(self):
        launchers_dir = PACKAGE_ROOT / "bin"

        self.assertTrue((launchers_dir / "rag_qdrant.ps1").is_file())
        self.assertTrue((launchers_dir / "rag_qdrant.bat").is_file())
        self.assertFalse((launchers_dir / "amiga_rag.ps1").exists())
        self.assertFalse((launchers_dir / "amiga_rag.bat").exists())

    def test_cli_help_uses_the_rag_qdrant_command_name(self):
        cli_source = (PACKAGE_ROOT / "rag_qdrant" / "cli.py").read_text(encoding="utf-8")

        self.assertIn("rag_qdrant PATH --source NAME --index-json FILE [OPTIONS]", cli_source)
        self.assertNotIn("amiga_rag [PATH] --source NAME [OPTIONS]", cli_source)


class CacheContractTests(unittest.TestCase):
    def test_collection_change_clears_file_hashes_and_drops_legacy_image_descriptions(self):
        cache, requires_flush = normalize_cache(
            {
                "sources": {"amiga": {"C:/docs/note.md": {"hash": "old"}}},
                "image_descriptions": {"C:/docs/diagram.png": "Timing diagram"},
            },
            "projects_docs",
        )

        self.assertTrue(requires_flush)
        self.assertEqual(cache["collection_name"], "projects_docs")
        self.assertEqual(cache["sources"], {})
        self.assertNotIn("image_descriptions", cache)

    def test_matching_collection_retains_incremental_hashes(self):
        original = {
            "collection_name": "projects_docs",
            "sources": {"project-a": {"C:/docs/note.md": {"hash": "current"}}},
        }

        cache, requires_flush = normalize_cache(original, "projects_docs")

        self.assertFalse(requires_flush)
        self.assertIs(cache, original)
        self.assertEqual(cache["sources"], original["sources"])


class CommandContractTests(unittest.TestCase):
    def test_server_and_cli_have_separate_tool_roots(self):
        tools_root = PACKAGE_ROOT.parent

        self.assertTrue((tools_root / "rag-qdrant" / "README.md").is_file())
        self.assertTrue((SERVER_ROOT / "README.md").is_file())
        self.assertTrue((SERVER_ROOT / "amiga_rag_mcp_server.py").is_file())
        self.assertFalse((tools_root / "rag").exists())

    def test_search_command_accepts_machine_readable_source_filter(self):
        args = create_search_parser().parse_args(
            ["Copper timing", "--source", "amiga,obsidian", "--index-json", "rag-index.json", "--limit", "3", "--json"]
        )

        self.assertEqual(args.query, "Copper timing")
        self.assertEqual(args.source, "amiga,obsidian")
        self.assertEqual(args.index_json, "rag-index.json")
        self.assertEqual(args.limit, 3)
        self.assertTrue(args.json)

    def test_path_command_runner_decodes_json_response(self):
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(returncode=0, stdout='[{"source": "amiga"}]', stderr="")

        result = run_rag_qdrant_json(
            ["search", "Copper timing", "--index-json", "rag-index.json", "--json"], runner=runner
        )

        self.assertEqual(result, [{"source": "amiga"}])
        self.assertEqual(
            calls[0][0],
            ["rag_qdrant", "search", "Copper timing", "--index-json", "rag-index.json", "--json"],
        )
        self.assertTrue(calls[0][1]["capture_output"])

    def test_mcp_server_uses_path_command_and_renamed_entrypoint(self):
        server_source = (SERVER_ROOT / "amiga_rag_mcp_server.py").read_text(encoding="utf-8")
        config_source = (PACKAGE_ROOT.parents[1] / ".agents" / "mcp_config.json").read_text(
            encoding="utf-8"
        )

        self.assertIn("run_rag_qdrant_json", server_source)
        self.assertNotIn("KnowledgeIndexer", server_source)
        self.assertIn("tools/amiga-rag-mcp-server/amiga_rag_mcp_server.py", config_source)
        self.assertFalse((SERVER_ROOT / "amiga_mcp_server.py").exists())
