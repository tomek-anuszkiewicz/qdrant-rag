# Qdrant Vector Database

This directory contains the Docker Compose configuration, vector storage, and incremental cache for a local [Qdrant](https://qdrant.tech/) vector database instance.

---

## 1. Quick Overview

- **Vector Database**: Qdrant running via Docker Compose (`qdrant/qdrant:latest`).
- **Web Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard) (inspect collections, points, vectors, and payloads).
- **REST API**: `http://localhost:6333`
- **gRPC API**: `localhost:6334`
- **Persistent Storage**: `./qdrant_storage/` (mounted directly to `/qdrant/storage:z` inside the container).
- **Incremental Hash Cache**: `./amiga_rag_cache.json` (tracks SHA256 file hashes for incremental RAG indexing).

---

## 2. Managing the Docker Container

Run these commands from this directory:

```powershell
# Start Qdrant in background
docker compose up -d

# View container logs
docker compose logs -f

# Check health endpoint
curl http://localhost:6333/readyz

# Stop Qdrant
docker compose down
```

---

## 3. Directory Structure

```
.
├── README.md                 # This documentation
├── docker-compose.yml        # Docker service configuration
├── amiga_rag_cache.json      # File hashes for incremental indexing
└── qdrant_storage/           # Persisted database files (WAL & vector storage)
```
