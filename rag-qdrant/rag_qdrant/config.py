import os
from pathlib import Path
from dotenv import load_dotenv

# Try loading .env from the current working directory first, then the tool's parent project.
cwd_env = Path.cwd() / ".env"
project_env = Path(__file__).resolve().parents[3] / ".env"

if cwd_env.is_file():
    load_dotenv(dotenv_path=cwd_env)
elif project_env.is_file():
    load_dotenv(dotenv_path=project_env)
else:
    load_dotenv()

# Qdrant settings
QDRANT_URL = "http://localhost:6333"
QDRANT_TIMEOUT = 60.0

COLLECTION_NAME = "projects_docs"

cache_env = os.getenv("RAG_CACHE_FILE")
if not cache_env or not cache_env.strip():
    raise RuntimeError("RAG_CACHE_FILE environment variable is mandatory and must be defined in .env")
CACHE_FILE = Path(cache_env.strip('"\''))

# Embedding settings: local FastEmbed with automatic CUDA GPU acceleration and CPU fallback
EMBEDDING_PROVIDER = "fastembed"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

VISION_MODEL = "gemini-flash-latest"

# Concurrency & Parallelism settings (calculated from hardware)
NUM_WORKERS = os.cpu_count() or 4
VISION_MAX_WORKERS = os.cpu_count() or 4

# Differentiated batch sizes for GPU vs CPU
GPU_EMBEDDING_BATCH_SIZE = 128
CPU_EMBEDDING_BATCH_SIZE = 64
DEFAULT_EMBEDDING_BATCH_SIZE = 64

# Bulk Qdrant upsert batch size
UPSERT_BATCH_SIZE = 256

# Deferred cache checkpointing: flush cache every N files or K chunks to avoid Google Drive thrashing
CACHE_FLUSH_INTERVAL_FILES = 10
CACHE_FLUSH_INTERVAL_CHUNKS = 250
