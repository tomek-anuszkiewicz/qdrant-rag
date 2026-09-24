"""Regression checks for the staged-content gate."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import preflight


class StagedTextTests(unittest.TestCase):
    def test_only_added_text_lines_are_checked(self):
        diff = (
            "+++ b/README.md\n"
            "@@ -1 +1 @@\n"
            "-" + "Za\u017c\u00f3\u0142\u0107" + "\n"
            "+A portable English sentence.\n"
            "+++ b/image.png\n"
            "@@ -0,0 +1 @@\n"
            "+" + "Za\u017c\u00f3\u0142\u0107" + "\n"
        )
        self.assertEqual(preflight.scan_added_lines(diff), [])

    def test_staged_invalid_then_corrected_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
            note = root / "note.md"
            note.write_text("Za\u017c\u00f3\u0142\u0107\n" + "C:" + "\\Users\\Alice\\repo\n", encoding="utf-8")
            subprocess.run(["git", "add", "note.md"], cwd=root, check=True)
            with patch.object(preflight, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "Polish diacritic"):
                    preflight.check_staged_text()
                note.write_text("A portable English note.\n", encoding="utf-8")
                subprocess.run(["git", "add", "note.md"], cwd=root, check=True)
                preflight.check_staged_text()


if __name__ == "__main__":
    unittest.main()
