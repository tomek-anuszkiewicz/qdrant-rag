---
name: rag-contract-change
description: Change or fix rag_qdrant CLI, Markdown discovery, cache, or indexing behavior while preserving the public incremental-index contract.
---

# RAG contract change

Use for a behavior change in `arguments.py`, `cli.py`, `discovery.py`, `cache.py`, `indexer.py`, or the indexing path in `core.py` and `service.py`. Ordinary documentation edits and read-only reviews do not need this workflow.

1. Identify the user-visible contract and affected callers. Read the relevant implementation and tests, then reproduce a reported defect with an isolated fixture when practical.
2. Preserve explicit `--source` and `--index-json` scope. Check collection ownership of file hashes, replacement of changed vectors, and removal of missing files when the change touches those paths.
3. Test with temporary directories and fakes or mocks. Do not use the live collection, canonical JSON state, or model download as a regression fixture.
4. Run the focused contract tests and `tools/harness/preflight.py`; run full unit tests when dependencies are present. Update CLI help and README only when behavior actually changes.
5. Report the behavior verified and any live Qdrant behavior that remains untested.
