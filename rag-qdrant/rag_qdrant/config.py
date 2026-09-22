import os

# Qdrant settings
QDRANT_URL = "http://localhost:6333"
QDRANT_TIMEOUT = 60.0

COLLECTION_NAME = "projects_docs"

# Embedding settings: local FastEmbed with automatic CUDA GPU acceleration and CPU fallback
EMBEDDING_PROVIDER = "fastembed"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768

# Concurrency & Parallelism settings (calculated from hardware)
NUM_WORKERS = os.cpu_count() or 4

# Differentiated batch sizes for GPU vs CPU
GPU_EMBEDDING_BATCH_SIZE = 128
CPU_EMBEDDING_BATCH_SIZE = 64
DEFAULT_EMBEDDING_BATCH_SIZE = 64

# Bulk Qdrant upsert batch size
UPSERT_BATCH_SIZE = 256

# Deferred cache checkpointing: flush cache every N files or K chunks to avoid Google Drive thrashing
CACHE_FLUSH_INTERVAL_FILES = 10
CACHE_FLUSH_INTERVAL_CHUNKS = 250
