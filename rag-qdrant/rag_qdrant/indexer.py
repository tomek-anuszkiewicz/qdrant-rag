"""Backward-compatible KnowledgeIndexer interface preserving existing class contract."""

from pathlib import Path
from typing import Union

from .core import RagEngine


class KnowledgeIndexer(RagEngine):
    """Backward-compatible KnowledgeIndexer extending RagEngine."""

    def __init__(self, index_json: Union[str, Path]):
        super().__init__(index_json=index_json)
