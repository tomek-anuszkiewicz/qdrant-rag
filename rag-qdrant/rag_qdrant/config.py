import os
from pathlib import Path


def _load_env_file():
    """Discover and load .env file from workspace or parent directories."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        env_path = parent / ".env"
        if env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip().strip("'\"")
                            if key and key not in os.environ:
                                os.environ[key] = val
            except Exception:
                pass
            break


_load_env_file()

# Qdrant settings
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_TIMEOUT = float(os.environ.get("QDRANT_TIMEOUT", "60.0"))

COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "projects_docs")

# Embedding settings: local FastEmbed with automatic CUDA GPU acceleration and CPU fallback
EMBEDDING_PROVIDER = "fastembed"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))

# Concurrency & Parallelism settings (calculated from hardware)
NUM_WORKERS = os.cpu_count() or 4

# Differentiated batch sizes for GPU vs CPU
GPU_EMBEDDING_BATCH_SIZE = 128
CPU_EMBEDDING_BATCH_SIZE = 64
DEFAULT_EMBEDDING_BATCH_SIZE = 64

# Bulk Qdrant upsert batch size
UPSERT_BATCH_SIZE = 256

# Deferred cache checkpointing: flush cache every N files or K chunks
CACHE_FLUSH_INTERVAL_FILES = 10
CACHE_FLUSH_INTERVAL_CHUNKS = 250

# Service settings
SERVICE_HOST = os.environ.get("RAG_SERVICE_HOST", "127.0.0.1")
SERVICE_PORT = int(os.environ.get("RAG_SERVICE_PORT", "6335"))
SERVICE_URL = f"http://{SERVICE_HOST}:{SERVICE_PORT}"

# Canonical index state JSON path
CANONICAL_INDEX_JSON = os.environ.get("RAG_CANONICAL_INDEX_JSON", "")

# PID and log files for background service
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SERVICE_PID_FILE = os.environ.get(
    "RAG_SERVICE_PID_FILE", str(PACKAGE_ROOT.parent / "rag_qdrant_service.pid")
)
SERVICE_LOG_FILE = os.environ.get(
    "RAG_SERVICE_LOG_FILE", str(PACKAGE_ROOT.parent / "rag_qdrant_service.log")
)

# Authentication & Profiles
ADMIN_TOKEN = os.environ.get("RAG_ADMIN_TOKEN", "")
AMIGA_TOKEN = os.environ.get("RAG_AMIGA_TOKEN", "")
DEVNOTES_TOKEN = os.environ.get("RAG_DEVNOTES_TOKEN", "")

# Limits
MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024  # 16 MB
MAX_SEARCH_LIMIT = 50
DEFAULT_SEARCH_LIMIT = 5
SCORE_THRESHOLD = 0.50
