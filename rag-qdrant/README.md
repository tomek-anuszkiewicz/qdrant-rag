# rag_qdrant

`rag_qdrant` is a standalone local, multi-source Qdrant indexer and semantic-search CLI. It is independent of the Amiga project: callers choose the indexed path, source tag, and directory scope.

## Installation

Install the CLI dependencies from this directory:

```powershell
pip install -r requirements.txt
```

Set `RAG_CACHE_FILE` in the environment or an adjacent `.env` file. The cache tracks SHA-256 hashes for incremental indexing.

## Commands

Use the launchers in `bin/` or expose the `rag_qdrant` command on `PATH`.

```powershell
# Inspect the shared Qdrant collection
rag_qdrant --status
rag_qdrant --list-sources

# Index selected top-level directories under a source tag
rag_qdrant <PATH_TO_DOCUMENTS> --source project-a --include-dirs "docs" "design"

# Query one or more source tags with a machine-readable response
rag_qdrant search "DMA arbitration" --source project-a,engineering-notes --limit 5 --json
rag_qdrant --list-sources --json
rag_qdrant --status --json
```

`--include-dirs` is mandatory for indexing. Only Markdown files below matching top-level directories are scanned; root-level notes and built-in system/privacy directories are excluded.

## Sidecars

`generate_all_sidecars.py <DOCUMENT_ROOT>` creates technical text sidecars for unindexed images. It is optional and intended for image-heavy document sets; ordinary Markdown-only indexing does not require it.

## Layout

```text
rag-qdrant/
├── README.md
├── requirements.txt
├── generate_all_sidecars.py
├── bin/                    # PATH launchers
├── rag_qdrant/             # CLI implementation
└── tests/                  # CLI regression tests
```
