# rag_qdrant

`rag_qdrant` indexes Markdown documentation in a local [Qdrant](https://qdrant.tech/) database and provides fast semantic search.

Since version 2.0, the architecture uses **one persistent background process** that keeps the FastEmbed model (`BAAI/bge-base-en-v1.5`) and Qdrant connection in RAM/VRAM. Both the thin `rag_qdrant` CLI client and the MCP server for AI agents use the same engine, avoiding repeated library imports and model loading on each request.

---

## Main features

- **Fast search**: CLI response time dropped from ~2.1 s to ~0.15–0.20 s.
- **Full backward compatibility**: Existing commands, flags (`--status`, `--list-sources`, `search`, `--json`, `--index-json`), and the `rag_qdrant.bat` and `rag_qdrant.ps1` launchers continue to work the same way.
- **Autostart on demand**: If the background service is not running, the CLI starts it, waits until it is ready, and then sends the request.
- **Two interfaces**: Local REST API and an MCP server (Streamable HTTP / SSE) on the shared `127.0.0.1:6335` port.
- **Security**: The service and Qdrant listen only on loopback (`127.0.0.1`), validate `Host` and `Origin` headers against DNS rebinding, and require a token for data access.
- **Permission profiles**: Client profiles (`admin`, `amiga`, `devnotes`) limit visible and indexable sources and directories.

---

## Installation

In the `rag-qdrant` directory:

```powershell
pip install -r requirements.txt
```

If other local projects need to call the CLI without a full path, **manually** add `D:\AI\qdrant\rag-qdrant\bin` to your Windows user `Path` environment variable (System Settings → Environment Variables → `Path` → Edit → New). Add the `bin` directory, not the launcher file. If the repository is elsewhere, use the path to its `bin` directory instead.

Open a new terminal and verify:

```powershell
Get-Command rag_qdrant.bat
rag_qdrant.bat --help
```

When configuring tools launched by other processes on Windows, use `rag_qdrant.bat`. You can also specify the launcher's full path, in which case no `Path` entry is needed. This setting applies only to the computer running the CLI.

Optional NVIDIA RTX / CUDA acceleration:
```powershell
pip install onnxruntime-gpu nvidia-cublas-cu12
```

---

## Configuration (`.env`)

Create or update `.env` in the repository root or `rag-qdrant/`:

```env
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_URL=http://127.0.0.1:6333

RAG_SERVICE_HOST=127.0.0.1
RAG_SERVICE_PORT=6335

RAG_ADMIN_TOKEN=random-admin-token
RAG_AMIGA_TOKEN=random-amiga-profile-token
RAG_DEVNOTES_TOKEN=random-devnotes-profile-token

RAG_CANONICAL_INDEX_JSON=d:\AI\qdrant\amiga_rag_cache.json
```

---

## CLI commands

Run the CLI through `bin\rag_qdrant.ps1` or `bin\rag_qdrant.bat`, or as a Python module: `python -m rag_qdrant.cli`.

### 1. Semantic search

```powershell
rag_qdrant search "DMA arbitration" --source amiga --limit 5 --index-json D:\AI\qdrant\amiga_rag_cache.json --json
```

| Parameter | Description |
| --- | --- |
| `QUERY` | Natural-language query text |
| `-s TAGS`, `--source TAGS` | Optional source tag or comma-separated tags (for example, `amiga,devnotes`) |
| `--limit N` | Maximum result count (default 5, minimum 1, maximum 50) |
| `--index-json FILE` | Required JSON index state file |
| `--json` | Required. Returns an array of objects with `score`, `source`, `file_path`, `relative_path`, `header`, `content`, and `images` fields. |

### 2. Collection and source status

```powershell
# Database and collection status
rag_qdrant --status --index-json D:\AI\qdrant\amiga_rag_cache.json [--json]

# List indexed sources
rag_qdrant --list-sources --index-json D:\AI\qdrant\amiga_rag_cache.json [--json]
```

### 3. Indexing a directory

```powershell
rag_qdrant D:\Docs\Amiga --source amiga --index-json D:\AI\qdrant\amiga_rag_cache.json
```

- Scans Markdown files under the selected directory and calculates SHA-256 hashes.
- Splits new and changed files into chunks, embeds them, and writes vectors to Qdrant.
- Skips unchanged files without embedding them again.
- Removes deleted files from Qdrant and the JSON state file.
- Streams the indexing plan and live progress over SSE from the service.

### 4. Managing the background service

```powershell
rag_qdrant service status   # Check status and statistics
rag_qdrant service start    # Start the service manually in the background
rag_qdrant service stop     # Stop the running service
```

---

## Local REST API (`http://127.0.0.1:6335`)

All endpoints except `/v1/health` and `/v1/ready` require an `Authorization: Bearer <TOKEN>` or `X-API-Key: <TOKEN>` header.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/v1/health` | Service liveness check (`{"status": "ok"}`) |
| `GET` | `/v1/ready` | Model and collection readiness check |
| `GET` | `/v1/status` | Qdrant and collection statistics |
| `GET` | `/v1/sources` | Sources, file counts, and chunk counts |
| `POST` | `/v1/search` | JSON: `{"query": "...", "source": "...", "limit": 5}` |
| `POST` | `/v1/index` | JSON: `{"path": "...", "source": "...", "stream": false/true}` |
| `POST` | `/v1/service/stop` | Graceful shutdown (requires the admin profile) |

---

## MCP server (`http://127.0.0.1:6335/mcp`)

The service includes an MCP server using Streamable HTTP / SSE.

### Available tools:
- `search(query, sources=None, limit=5)`: Semantic search within the token profile's permissions.
- `status()`: Read-only collection status.
- `list_sources()`: Source statistics from the cache.
- `index(path, source)`: Incremental indexing (only for tokens with write permission).

### MCP client configuration (for example, Antigravity IDE, Claude Desktop, Cursor):

In `mcp_config.json`:
```json
{
  "mcpServers": {
    "qdrant-rag": {
      "url": "http://127.0.0.1:6335/mcp?token=YOUR_PROFILE_TOKEN",
      "transport": "sse"
    }
  }
}
```

---

## Architecture and files

```
rag-qdrant/
├── bin/
│   ├── rag_qdrant.bat       # Windows CMD launcher
│   └── rag_qdrant.ps1       # Windows PowerShell launcher
├── rag_qdrant/
│   ├── arguments.py         # CLI argument parsing
│   ├── cache.py             # JSON index state normalization and schema
│   ├── chunker.py           # Splits Markdown into logical sections
│   ├── client.py            # Thin HTTP client with autostart and SSE
│   ├── cli.py               # CLI entry point
│   ├── config.py            # Settings, environment, and hardware parameters
│   ├── core.py              # Central RagEngine (Qdrant + FastEmbed + lock)
│   ├── discovery.py         # Markdown discovery with private-directory exclusions
│   ├── indexer.py           # Compatibility KnowledgeIndexer class
│   ├── security.py          # Host/Origin validation, authentication, and profiles
│   └── service.py           # Starlette/Uvicorn service and FastMCP
└── tests/
    ├── test_indexing_contract.py  # CLI and cache contract tests
    └── test_service.py            # REST API, security, and MCP tests
```
