"""Collection-scoped cache schema for incremental RAG indexing."""

from typing import Any, Dict, Tuple


def new_cache(collection_name: str) -> Dict[str, Any]:
    """Create an empty cache owned by one Qdrant collection."""
    return {
        "collection_name": collection_name,
        "sources": {},
        "image_descriptions": {},
    }


def normalize_cache(cache: Any, collection_name: str) -> Tuple[Dict[str, Any], bool]:
    """Return a usable cache and whether it must be written back.

    File hashes are valid only for the Qdrant collection that received their
    vectors. A collection change therefore clears those hashes while retaining
    reusable image descriptions.
    """
    if not isinstance(cache, dict):
        return new_cache(collection_name), False

    image_descriptions = cache.get("image_descriptions")
    if not isinstance(image_descriptions, dict):
        image_descriptions = {}

    if cache.get("collection_name") != collection_name:
        return {
            "collection_name": collection_name,
            "sources": {},
            "image_descriptions": image_descriptions,
        }, True

    changed = False
    sources = cache.get("sources")
    if not isinstance(sources, dict):
        cache["sources"] = {}
        changed = True
    if cache.get("image_descriptions") is not image_descriptions:
        cache["image_descriptions"] = image_descriptions
        changed = True

    return cache, changed
