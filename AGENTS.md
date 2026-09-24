# Agent instructions for rag-qdrant

This repository contains a local Python RAG service and its Qdrant Docker configuration. Keep changes small and tied to the requested behavior.

## Boundaries

- Write new repository content in English, including documentation, code comments, and agent files. Reply to the user in their language. See [.agents/rules/repository-language.md](.agents/rules/repository-language.md) for the scoped language rule.
- Use portable relative paths in committed documentation and examples. See [.agents/rules/portable-paths.md](.agents/rules/portable-paths.md) for the path boundary.
- Treat `.env`, `qdrant_storage/`, `*_rag_cache.json`, service logs, and tokens as local state. Do not add their contents to Git, examples, logs, or responses.
- Keep Qdrant and the RAG service bound to loopback unless the task explicitly changes the network and authentication design. Preserve Host/Origin checks and profile-scoped search/index permissions when editing the service.
- Indexing changes must preserve explicit `--source` and `--index-json` scope, incremental state, and deletion behavior. Do not use a live collection or local cache as a test fixture.
- Do not start, stop, reset, or reindex a live Qdrant instance to verify an unrelated code change. Use isolated fixtures for tests.
- Do not commit unless the user asks. Inspect the diff and keep unrelated work untouched.

## Verification

- Run `python tools/harness/preflight.py` after changing Python code, Docker configuration, or agent policy. It needs only the Python standard library and Git.
- Run `python tools/harness/preflight.py --full` when dependencies from `rag-qdrant/requirements.txt` are installed and service behavior changed. This runs the repository's existing unit tests without starting Docker.
- If a check cannot run, report the exact missing dependency or environment condition. Do not describe a mock or static check as live Qdrant verification.

Use the repository skills only for their matching work: `rag-contract-change` for indexing or CLI behavior, `service-boundary-review` for security or exposure changes, and `docs-code-parity` for a requested audit or a behavior change that affects published instructions.

The repository workflow and optional Git hook are documented in [tools/harness/README.md](tools/harness/README.md).
