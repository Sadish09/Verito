"""
BaseVectorStore: contract for all vector storage backends.

Subclassing checklist:
  - upsert()          — insert or replace chunks by ID
  - query()           — ANN search, returns scored results
  - delete_file()     — remove all chunks for a file path
  - get_file_mtimes() — fetch the mtime map for incremental indexing
  - set_file_mtime()  — persist a single file's mtime after indexing
  - stats()           — counts for /status endpoint
  - aclose()          — release resources on shutdown

All methods are async.  Synchronous backends should
run blocking calls in the default executor inside their implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):

    @abstractmethod
    async def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        #Insert or replace chunks. ids, vectors, texts, metadatas are parallel lists.
        ...

    @abstractmethod
    async def query(
        self,
        vector: list[float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Return up to top_k results ordered by descending similarity.
        Each result dict must contain at minimum:
          id, chunk_text, score (float in [0, 1])
        plus any stored metadata fields.
        """
        ...

    @abstractmethod
    async def delete_file(self, file_path: str) -> None:
        """Delete all chunks associated with file_path."""
        ...

    @abstractmethod
    async def get_file_mtimes(self) -> dict[str, float]:
        """
        Return a mapping of file_path → last-indexed mtime.
        Used by the Indexer to skip unchanged files.
        """
        ...

    @abstractmethod
    async def set_file_mtime(self, file_path: str, mtime: float) -> None:
        #Persist the mtime for file_path after successful indexing.
        ...

    @abstractmethod
    async def stats(self) -> dict[str, Any]:
        """
        Return store statistics for the /status endpoint.
        Must include at minimum: total_chunks, indexed_files.
        """
        ...

    @abstractmethod
    async def aclose(self) -> None:
        #Release any held resources on shutdown.
        ...