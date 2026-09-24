# Qdrant Vector Database & Persistent RAG Service

This directory contains the Docker Compose configuration for the [Qdrant](https://qdrant.tech/) vector database, vector storage, and the central `rag_qdrant` service for semantic search, documentation indexing, a REST API, and an MCP server.

Agent instructions are in [AGENTS.md](AGENTS.md). The local preflight check and optional Git hook are documented in [tools/harness/README.md](tools/harness/README.md).

---

## 1. System architecture

```text
rag_qdrant CLI ── HTTP/JSON ──┐
                              ├── service process (127.0.0.1:6335) ── Qdrant Docker (127.0.0.1:6333)
MCP client ── Streamable HTTP ┘       │
                                      └── FastEmbed model and index state in RAM/VRAM
```

- **One background process**: Keeps the embedding model (`BAAI/bge-base-en-v1.5`, with CUDA/CPU acceleration) and Qdrant client in memory.
- **Two interfaces**:
  1. Local HTTP/JSON REST API for the thin `rag_qdrant` CLI client.
  2. MCP server (Streamable HTTP / SSE) at `/mcp` for AI agents.
- **Thin CLI client**: Does not load the model or heavy libraries for every request. Search response time dropped from ~2.1 s to ~0.15–0.20 s. The client starts the background service on demand if needed.

---

## 2. Security and ports

1. **Network isolation (loopback only)**:
   - Qdrant listens only on `127.0.0.1:6333` (REST) and `127.0.0.1:6334` (gRPC).
   - The `rag_qdrant` service listens only on `127.0.0.1:6335`.
2. **Qdrant API key**:
   - The Qdrant container requires `QDRANT__SERVICE__API_KEY`, loaded from `.env`.
   - Requests without the key to `http://127.0.0.1:6333/collections` return HTTP 401 Unauthorized.
3. **RAG service authentication**:
   - All data endpoints (REST and `/mcp`) require an authorization token (`Authorization: Bearer <TOKEN>` or `?token=<TOKEN>`).
   - Client profiles (`admin`, `amiga`, `devnotes`) define permitted search sources, indexing sources, and directories.
4. **DNS rebinding protection**:
   - The service validates the `Host` and `Origin` headers. Requests with foreign values, such as browser-based attack attempts, return HTTP 403 Forbidden.

---

## 3. Configuration (`.env`)

Create a local `.env` file (ignored by `.gitignore`):

```bash
# Qdrant Docker
QDRANT_API_KEY=your-random-qdrant-api-key
QDRANT_URL=http://127.0.0.1:6333

# rag-qdrant service
RAG_SERVICE_HOST=127.0.0.1
RAG_SERVICE_PORT=6335

# Authentication tokens
RAG_ADMIN_TOKEN=random-token-for-cli-and-admin
RAG_AMIGA_TOKEN=random-token-for-amiga-profile
RAG_DEVNOTES_TOKEN=random-token-for-devnotes-profile

# Shared local collection state
RAG_CANONICAL_INDEX_JSON=d:\AI\qdrant\amiga_rag_cache.json
```

---

## 4. Managing Qdrant

Run these commands from the project root:

```powershell
# Start the container in the background
docker compose up -d

# View container logs
docker compose logs -f

# Check readiness
curl http://127.0.0.1:6333/readyz

# Stop the container
docker compose down
```

---

## 5. Managing the RAG service

The service starts **automatically on demand** on the first `rag_qdrant` call. You can also manage it directly:

```powershell
# Service status
rag_qdrant service status

# Start manually in the background
rag_qdrant service start

# Stop the service
rag_qdrant service stop
```

---

## 6. MCP server for AI agents

The service exposes an MCP server at:
`http://127.0.0.1:6335/mcp?token=<RAG_ADMIN_OR_PROFILE_TOKEN>`

### MCP tools:
- `search(query: str, sources: Optional[str] = None, limit: int = 5)`: Semantic documentation search within the profile's permissions.
- `status()`: Database status, the `projects_docs` collection, and vector count.
- `list_sources()`: Indexed sources and their file and chunk counts.
- `index(path: str, source: str)`: Incremental directory indexing (requires write permission).

### Configuration in `mcp_config.json`:

For clients supporting SSE/HTTP transport:
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

For existing MCP adapters that launch the CLI: Existing `rag_qdrant.bat` and `rag_qdrant.ps1` commands continue to work, with queries running 10–20 times faster through the persistent service.

---

## 7. Directory structure

```
.
├── README.md                 # This documentation
├── docker-compose.yml        # Qdrant container configuration, bound to 127.0.0.1
├── .env                      # Local configuration and secrets (ignored by Git)
├── amiga_rag_cache.json      # Shared index state file
├── qdrant_storage/           # Qdrant data volume
└── rag-qdrant/               # Service and CLI package
    ├── bin/                  # rag_qdrant.bat and rag_qdrant.ps1 launchers
    ├── rag_qdrant/           # core, service, client, cli, security implementation
    └── tests/                # Unit and contract regression tests
```
