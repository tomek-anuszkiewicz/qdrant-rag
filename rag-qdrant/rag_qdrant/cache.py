"""Collection-scoped cache schema for incremental RAG indexing."""

from typing import Any, Dict, Tuple


def new_cache(collection_name: str) -> Dict[str, Any]:
    """Create an empty cache owned by one Qdrant collection."""
    return {
        "collection_name": collection_name,
        "sources": {},
    }


def normalize_cache(cache: Any, collection_name: str) -> Tuple[Dict[str, Any], bool]:
    """Return a usable cache and whether it must be written back.

    File hashes are valid only for the Qdrant collection that received their
    vectors. A collection change therefore clears those hashes.
    """
    if not isinstance(cache, dict):
        return new_cache(collection_name), False

    if cache.get("collection_name") != collection_name:
        return {
            "collection_name": collection_name,
            "sources": {},
        }, True

    changed = False
    sources = cache.get("sources")
    if not isinstance(sources, dict):
        cache["sources"] = {}
        changed = True
    if "image_descriptions" in cache:
        del cache["image_descriptions"]
        changed = True

    return cache, changed
