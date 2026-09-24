# Small repository harness

`AGENTS.md` contains the standing project boundaries. Focused rules live in `.agents/rules/`; Codex also sees their short summaries in `AGENTS.md`. Three repository skills live in `.agents/skills/` and apply to indexing contract changes, service boundary reviews, and requested documentation parity audits.

## Local workflow

From the repository root:

```powershell
python tools/harness/preflight.py
python tools/harness/preflight.py --full
```

The quick check uses only Git and the Python standard library. It rejects tracked local state, public Docker port bindings, Python syntax errors, and staged added text containing Polish diacritics or common machine-specific paths. Text checks apply only to added lines so existing Polish documentation does not create a blanket failure. They do not prove that every new sentence is English or every path is portable; review the diff as well. The full mode additionally runs the existing unit tests after `rag-qdrant/requirements.txt` is installed. Neither mode starts Docker or indexes documents.

The optional Git hook runs the quick check before commits:

```powershell
git config --local core.hooksPath .githooks
```

This setting affects only this checkout. Git for Windows executes the tracked shell hook; Python must be on PATH. Remove the setting with `git config --local --unset core.hooksPath`. The hook examines staged added text and paths but reads current working-tree Python and Compose files. Finish editing before committing.

## Design choices

The language and path boundaries adapt devnotes' `notes-language`, `no-absolute-paths`, `check-polish`, and `check-paths` to a code repository. The indexing and service skills adapt Amiga's focused regression and review procedures to this package. The documentation parity skill is an on-demand review, not a mandatory whole-repository gate. No live RAG operation, specialist agent, or Codex lifecycle hook is required for normal code changes.
