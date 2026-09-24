"""Core RAG engine managing embeddings, Qdrant client, cache, and operations."""

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .cache import new_cache, normalize_cache
from .chunker import MarkdownChunker
from .config import (
    CACHE_FLUSH_INTERVAL_CHUNKS,
    CACHE_FLUSH_INTERVAL_FILES,
    CANONICAL_INDEX_JSON,
    COLLECTION_NAME,
    CPU_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    GPU_EMBEDDING_BATCH_SIZE,
    NUM_WORKERS,
    QDRANT_API_KEY,
    QDRANT_TIMEOUT,
    QDRANT_URL,
    SCORE_THRESHOLD,
    UPSERT_BATCH_SIZE,
)
from .discovery import discover_markdown_files
from .security import (
    ClientProfile,
    validate_index_authorization,
    validate_search_sources,
)


def _setup_cuda_dll_paths():
    """Discovers NVIDIA CUDA and cuDNN DLL directories and registers them with Windows DLL search path."""
    import sys
    if sys.platform != "win32":
        return
    import site
    search_dirs = []
    try:
        site_dirs = list(site.getsitepackages())
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            site_dirs.append(user_site)
        for base in site_dirs:
            nv_dir = Path(base) / "nvidia"
            if nv_dir.is_dir():
                for bin_dir in nv_dir.glob("**/bin"):
                    if bin_dir.is_dir():
                        search_dirs.append(bin_dir)
                        search_dirs.extend(
                            child for child in bin_dir.iterdir() if child.is_dir()
                        )
    except Exception:
        pass

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        p = Path(cuda_path) / "bin"
        if p.is_dir():
            search_dirs.append(p)

    for d in search_dirs:
        try:
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass


_setup_cuda_dll_paths()


class ConcurrencyError(RuntimeError):
    """Raised when an indexing operation is requested while another is in progress."""
    pass


class DimensionMismatchError(RuntimeError):
    """Raised when existing collection vector dimension does not match embedder dimension."""
    pass


class RagEngine:
    """Persistent in-memory engine holding QdrantClient, FastEmbed model, and index cache."""

    def __init__(self, index_json: Optional[Union[str, Path]] = None):
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            timeout=QDRANT_TIMEOUT,
        )
        self.chunker = MarkdownChunker()

        # Determine index JSON file path
        if index_json:
            self.index_json = Path(index_json).expanduser().resolve()
        elif CANONICAL_INDEX_JSON:
            self.index_json = Path(CANONICAL_INDEX_JSON).expanduser().resolve()
        else:
            # Default fallback to workspace root shared state
            workspace_root = Path(__file__).resolve().parents[2]
            self.index_json = (workspace_root / "amiga_rag_cache.json").resolve()

        self.cache, self.dirty_cache = self._load_cache()
        self.host_gpu = self.detect_host_gpu()
        self.fastembed_model = None
        self.active_provider = "CPU"
        self.active_batch_size = CPU_EMBEDDING_BATCH_SIZE
        self.indexing_lock = threading.Lock()
        self._init_embedder()

    @staticmethod
    def detect_host_gpu() -> Optional[str]:
        """Probes nvidia-smi for discrete GPU details."""
        try:
            import subprocess
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=1.0
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip().split("\n")[0]
        except Exception:
            pass
        return None

    def _init_embedder(self):
        # 1. Attempt CUDA first
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                from fastembed import TextEmbedding
                model = TextEmbedding(
                    model_name=EMBEDDING_MODEL,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                # Verify forward pass on tiny probe
                list(model.embed(["probe"], batch_size=1))

                active_providers = []
                if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "get_providers"):
                    active_providers = model.model.model.get_providers()
                elif hasattr(model, "model") and hasattr(model.model, "providers"):
                    active_providers = model.model.providers

                if "CUDAExecutionProvider" in active_providers:
                    self.fastembed_model = model
                    self.active_provider = "CUDA"
                    self.active_batch_size = GPU_EMBEDDING_BATCH_SIZE
                    return
        except Exception:
            pass

        # 2. Fallback to CPU
        try:
            from fastembed import TextEmbedding
            self.fastembed_model = TextEmbedding(
                model_name=EMBEDDING_MODEL,
                threads=NUM_WORKERS,
                providers=["CPUExecutionProvider"]
            )
            self.active_provider = "CPU"
            self.active_batch_size = CPU_EMBEDDING_BATCH_SIZE
        except Exception as e:
            self.active_provider = "None"
            self.active_batch_size = CPU_EMBEDDING_BATCH_SIZE

    def _load_cache(self) -> Tuple[Dict[str, Any], bool]:
        if self.index_json.is_file():
            try:
                with open(self.index_json, "r", encoding="utf-8") as f:
                    return normalize_cache(json.load(f), COLLECTION_NAME)
            except Exception:
                pass
        return new_cache(COLLECTION_NAME), False

    def flush_cache(self):
        """Flushes dirty cache to disk using atomic rename."""
        if getattr(self, "dirty_cache", False):
            self.index_json.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.index_json.parent / f"{self.index_json.name}.tmp.{os.getpid()}"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.index_json)
            self.dirty_cache = False

    def _save_cache(self, force: bool = False):
        self.dirty_cache = True
        if force:
            self.flush_cache()

    def _file_hash(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def ensure_collection(self):
        """Creates collection if not exists. Throws error if existing dimension mismatches."""
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME in collections:
            info = self.client.get_collection(COLLECTION_NAME)
            current_dim = info.config.params.vectors.size
            if current_dim != EMBEDDING_DIM:
                if info.points_count == 0:
                    self.client.delete_collection(COLLECTION_NAME)
                    collections.remove(COLLECTION_NAME)
                else:
                    raise DimensionMismatchError(
                        f"Existing collection '{COLLECTION_NAME}' has dimension {current_dim}, "
                        f"which mismatches required embedder dimension {EMBEDDING_DIM}."
                    )

        if COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE
                )
            )
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if not self.fastembed_model:
            raise RuntimeError("FastEmbed model failed to initialize or is unavailable.")
        embeddings = [
            v.tolist() for v in self.fastembed_model.embed(
                texts,
                batch_size=self.active_batch_size,
                parallel=None
            )
        ]
        return embeddings

    def delete_file_points(self, file_path_str: str):
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=file_path_str)
                        )
                    ]
                )
            )
        )

    def get_sources_stats(self, profile: Optional[ClientProfile] = None) -> List[Dict[str, Any]]:
        """Returns statistics of indexed sources, respecting client profile restrictions."""
        # Reload cache if file was modified externally
        self.cache, cache_requires_flush = self._load_cache()
        self.dirty_cache = self.dirty_cache or cache_requires_flush
        sources = self.cache.get("sources", {})
        result = []
        for name, files in sources.items():
            if profile and "*" not in profile.allowed_search_sources:
                if name.lower() not in [a.lower() for a in profile.allowed_search_sources]:
                    continue
            total_files = len(files)
            total_chunks = sum(f.get("chunks", 0) for f in files.values())
            timestamps = [f.get("last_indexed", "") for f in files.values() if f.get("last_indexed")]
            last_updated = max(timestamps) if timestamps else "N/A"
            result.append({
                "source": name,
                "files_count": total_files,
                "chunks_count": total_chunks,
                "last_updated": last_updated
            })
        return result

    def get_qdrant_sources_stats(self) -> Dict[str, Any]:
        """Queries Qdrant live database points count."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if COLLECTION_NAME not in collections:
                return {"total_points": 0, "total_files": 0, "sources": {}}
            info = self.client.get_collection(COLLECTION_NAME)
            total_points = info.points_count
        except Exception:
            total_points = 0

        sources = sorted(self.cache.get("sources", {}).keys())
        result = {"total_points": total_points, "total_files": 0, "sources": {}}
        total_cached_files = 0
        for s in sources:
            try:
                cnt = self.client.count(
                    collection_name=COLLECTION_NAME,
                    count_filter=models.Filter(
                        must=[models.FieldCondition(key="source", match=models.MatchValue(value=s))]
                    ),
                    exact=True
                ).count
            except Exception:
                cnt = 0
            cached_files = len(self.cache.get("sources", {}).get(s, {}))
            total_cached_files += cached_files
            result["sources"][s] = {
                "vectors": cnt,
                "cached_files": cached_files
            }
        result["total_files"] = total_cached_files
        return result

    def get_status_payload(self, profile: Optional[ClientProfile] = None) -> Dict[str, Any]:
        """Purely read-only status inspection without mutating collections or files."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if COLLECTION_NAME not in collections:
                return {
                    "qdrant_url": QDRANT_URL,
                    "collection": COLLECTION_NAME,
                    "health_status": "not_created",
                    "total_vectors": 0,
                    "total_files": 0,
                    "source_count": 0,
                }
            info = self.client.get_collection(COLLECTION_NAME)
            sources = self.get_sources_stats(profile=profile)
            return {
                "qdrant_url": QDRANT_URL,
                "collection": COLLECTION_NAME,
                "health_status": str(info.status),
                "total_vectors": info.points_count,
                "total_files": sum(source["files_count"] for source in sources),
                "source_count": len(sources),
            }
        except Exception as error:
            raise RuntimeError(f"Failed to inspect Qdrant collection: {error}") from error

    def search(
        self,
        query: str,
        sources: Optional[Union[str, List[str]]] = None,
        limit: int = 5,
        score_threshold: float = SCORE_THRESHOLD,
        profile: Optional[ClientProfile] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search in Qdrant with profile-enforced multi-source filtering."""
        # Convert sources argument to list of strings
        raw_sources: Optional[List[str]] = None
        if sources:
            if isinstance(sources, str):
                raw_sources = [s.strip().lower() for s in sources.split(",") if s.strip()]
            elif isinstance(sources, list):
                raw_sources = [s.strip().lower() for s in sources if isinstance(s, str) and s.strip()]

        if profile:
            effective_sources, error = validate_search_sources(raw_sources, profile)
            if error:
                raise PermissionError(error)
        else:
            effective_sources = raw_sources

        # Check if collection exists without creating it
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            return []

        query_embeddings = self.get_embeddings([query])
        if not query_embeddings:
            return []
        query_vector = query_embeddings[0]

        query_filter = None
        if effective_sources:
            if len(effective_sources) == 1:
                query_filter = models.Filter(
                    must=[models.FieldCondition(key="source", match=models.MatchValue(value=effective_sources[0]))]
                )
            else:
                query_filter = models.Filter(
                    must=[models.FieldCondition(key="source", match=models.MatchAny(any=effective_sources))]
                )

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold
        )

        formatted = []
        for hit in results.points:
            payload = hit.payload or {}
            formatted.append({
                "score": round(hit.score, 4),
                "source": payload.get("source"),
                "file_path": payload.get("file_path"),
                "relative_path": payload.get("relative_path"),
                "header": payload.get("header"),
                "content": payload.get("content"),
                "images": payload.get("images", [])
            })
        return formatted

    def index_directory(
        self,
        directory: Path,
        source_name: str,
        profile: Optional[ClientProfile] = None,
        progress_cb: Optional[Callable[..., Any]] = None,
        plan_cb: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, Any]:
        """Indexes directory with single-process concurrency locking and profile validation."""
        if not self.indexing_lock.acquire(blocking=False):
            raise ConcurrencyError("An indexing operation is already in progress.")

        try:
            if profile:
                allowed, err = validate_index_authorization(directory, source_name, profile)
                if not allowed:
                    raise PermissionError(err)

            self.ensure_collection()
            dir_path = directory.resolve()
            source_name = source_name.strip().lower()

            # Ensure latest cache state
            self.cache, _ = self._load_cache()
            source_cache = self.cache.setdefault("sources", {}).setdefault(source_name, {})

            md_files = discover_markdown_files(dir_path)
            active_paths = {str(p.resolve()) for p in md_files}

            stats = {
                "scanned": len(md_files),
                "indexed": 0,
                "updated": 0,
                "skipped": 0,
                "deleted": 0,
                "total_points": 0,
            }

            # 1. Clean up deleted files under dir_path
            cached_paths = list(source_cache.keys())
            for old_path in cached_paths:
                old_p = Path(old_path)
                try:
                    is_under_dir = old_p.is_relative_to(dir_path)
                except (ValueError, AttributeError):
                    is_under_dir = str(old_p).startswith(str(dir_path))

                if is_under_dir and old_path not in active_paths:
                    self.delete_file_points(old_path)
                    del source_cache[old_path]
                    stats["deleted"] += 1

            if not md_files:
                self._save_cache(force=True)
                return stats

            # 2. Check hashes
            def _check_file(file_p: Path):
                p_str = str(file_p.resolve())
                h = self._file_hash(file_p)
                size = file_p.stat().st_size
                cached_entry = source_cache.get(p_str)
                needs_idx = (cached_entry is None) or (cached_entry.get("hash") != h)
                is_upd = cached_entry is not None and needs_idx
                return file_p, p_str, h, needs_idx, is_upd, size

            total_scanned_bytes = sum(f.stat().st_size for f in md_files)
            total_files = len(md_files)
            files_to_index = []
            total_to_index_bytes = 0
            total_skipped_bytes = 0
            scanned_bytes = 0
            scanned_count = 0

            if progress_cb:
                progress_cb(0, total_scanned_bytes, f"Checking {total_files} files across {NUM_WORKERS} workers...", "hashing", True, f"({0:>{len(str(total_files))}}/{total_files})")

            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {executor.submit(_check_file, p): p for p in md_files}
                for future in as_completed(futures):
                    file_p, p_str, h, needs_idx, is_upd, size = future.result()
                    scanned_count += 1
                    scanned_bytes += size
                    if needs_idx:
                        files_to_index.append((file_p, p_str, h, is_upd, size))
                        total_to_index_bytes += size
                    else:
                        stats["skipped"] += 1
                        total_skipped_bytes += size

                    if progress_cb:
                        count_tag = f"({scanned_count:>{len(str(total_files))}}/{total_files})"
                        progress_cb(scanned_bytes, total_scanned_bytes, file_p.name, "hashing", True, count_tag)

            if plan_cb:
                plan_cb({
                    "scanned_files": total_files,
                    "scanned_bytes": total_scanned_bytes,
                    "to_index_files": len(files_to_index),
                    "to_index_bytes": total_to_index_bytes,
                    "skipped_files": stats["skipped"],
                    "skipped_bytes": total_skipped_bytes,
                    "cpu_workers": NUM_WORKERS,
                })

            if not files_to_index:
                if progress_cb:
                    progress_cb(total_scanned_bytes, total_scanned_bytes, "All files up to date", "skipped", True, f"({total_files}/{total_files})")
                self._save_cache(force=True)
                return stats

            # 3. Chunking
            chunked_results = []

            def _chunk_worker(item):
                file_p, p_str, h, is_upd, size = item
                chunks = self.chunker.chunk_markdown(file_p, dir_path, source_name)
                return file_p, p_str, h, is_upd, size, chunks

            processed_bytes = 0
            chunked_count = 0
            total_to_chunk = len(files_to_index)
            chunk_digits = len(str(total_to_chunk))
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = [executor.submit(_chunk_worker, item) for item in files_to_index]
                for future in as_completed(futures):
                    file_p, p_str, h, is_upd, size, chunks = future.result()
                    chunked_count += 1
                    processed_bytes += size
                    chunked_results.append((file_p, p_str, h, is_upd, chunks))
                    if progress_cb:
                        count_tag = f"({chunked_count:>{chunk_digits}}/{total_to_chunk})"
                        progress_cb(processed_bytes, total_to_index_bytes, file_p.name, "chunked", True, count_tag)

            # 4. Batched Embedding and Upserting
            total_chunks_to_index = sum(len(chunks) for _, _, _, _, chunks in chunked_results)
            total_files_to_index = len(chunked_results)
            completed_chunks = 0
            completed_files = 0
            f_digits = len(str(total_files_to_index))

            if progress_cb and total_chunks_to_index > 0:
                progress_cb(0, total_chunks_to_index, f"Starting incremental indexing across {total_files_to_index} files...", "indexing", False, f"({0:>{f_digits}}/{total_files_to_index})")

            file_chunk_counts: Dict[str, int] = {}
            file_points_accum: Dict[str, int] = {}
            file_meta: Dict[str, tuple] = {}

            all_chunk_items = []
            for file_p, p_str, h, is_upd, chunks in chunked_results:
                file_meta[p_str] = (file_p, h, is_upd)
                if not chunks:
                    source_cache[p_str] = {
                        "hash": h,
                        "chunks": 0,
                        "last_indexed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self._save_cache()
                    completed_files += 1
                    stats["skipped"] += 1
                    if progress_cb:
                        count_tag = f"({completed_files:>{f_digits}}/{total_files_to_index})"
                        progress_cb(completed_chunks, total_chunks_to_index, file_p.name, "skipped", False, count_tag)
                    continue

                if is_upd:
                    self.delete_file_points(p_str)

                file_chunk_counts[p_str] = len(chunks)
                file_points_accum[p_str] = 0

                for chunk in chunks:
                    full_embed_text = chunk["content"]
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_name}:{chunk['chunk_id']}"))
                    payload = {
                        "source": source_name,
                        "file_path": chunk["file_path"],
                        "relative_path": chunk["relative_path"],
                        "header": chunk["header"],
                        "content": chunk["content"],
                        "full_context": full_embed_text,
                        "images": chunk["images"],
                        "chunk_id": chunk["chunk_id"]
                    }
                    all_chunk_items.append((p_str, point_id, full_embed_text, payload))

            pending_points: List[models.PointStruct] = []
            files_completed_since_flush = 0
            chunks_since_flush = 0

            def _flush_pending_points():
                nonlocal pending_points
                if pending_points:
                    for i in range(0, len(pending_points), UPSERT_BATCH_SIZE):
                        self.client.upsert(
                            collection_name=COLLECTION_NAME,
                            points=pending_points[i:i + UPSERT_BATCH_SIZE]
                        )
                    pending_points = []

            try:
                for b_idx in range(0, len(all_chunk_items), self.active_batch_size):
                    batch_items = all_chunk_items[b_idx:b_idx + self.active_batch_size]
                    batch_texts = [item[2] for item in batch_items]
                    batch_embeddings = self.get_embeddings(batch_texts)

                    for item, emb in zip(batch_items, batch_embeddings):
                        p_str, point_id, _, payload = item
                        pending_points.append(
                            models.PointStruct(id=point_id, vector=emb, payload=payload)
                        )
                        file_points_accum[p_str] += 1
                        completed_chunks += 1
                        chunks_since_flush += 1

                        if file_points_accum[p_str] == file_chunk_counts[p_str]:
                            file_p, h, is_upd = file_meta[p_str]
                            source_cache[p_str] = {
                                "hash": h,
                                "chunks": file_chunk_counts[p_str],
                                "last_indexed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            self._save_cache()
                            completed_files += 1
                            files_completed_since_flush += 1
                            stats["total_points"] += file_chunk_counts[p_str]
                            if is_upd:
                                stats["updated"] += 1
                            else:
                                stats["indexed"] += 1

                            if progress_cb:
                                count_tag = f"({completed_files:>{f_digits}}/{total_files_to_index})"
                                progress_cb(completed_chunks, total_chunks_to_index, file_p.name, "indexing", False, count_tag)

                    while len(pending_points) >= UPSERT_BATCH_SIZE:
                        upsert_chunk = pending_points[:UPSERT_BATCH_SIZE]
                        self.client.upsert(
                            collection_name=COLLECTION_NAME,
                            points=upsert_chunk
                        )
                        pending_points = pending_points[UPSERT_BATCH_SIZE:]

                    if (files_completed_since_flush >= CACHE_FLUSH_INTERVAL_FILES or
                            chunks_since_flush >= CACHE_FLUSH_INTERVAL_CHUNKS):
                        _flush_pending_points()
                        self.flush_cache()
                        files_completed_since_flush = 0
                        chunks_since_flush = 0

                _flush_pending_points()
                self.flush_cache()

            finally:
                try:
                    _flush_pending_points()
                except Exception:
                    pass
                self.flush_cache()

            return stats
        finally:
            self.indexing_lock.release()
