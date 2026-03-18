"""
ChromaDB implementation of BaseVectorStore.

Collections:
  - obsidian_chunks : chunk embeddings + metadata (cosine distance)
  - obsidian_mtimes : file_path → mtime string (lightweight, no embeddings)

score = 1.0 - cosine_distance  →  always in [0, 1].

ChromaDB is synchronous — every call is dispatched to the default
asyncio executor via _run() so we never block the event loop.

Large upserts are batched in _UPSERT_BATCH_SIZE chunks to avoid
hitting ChromaDB's internal limits on very large vaults.
"""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from loguru import logger

from base import BaseVectorStore

_CHUNKS_COLLECTION = "obsidian_chunks"
_MTIMES_COLLECTION = "obsidian_mtimes"
_UPSERT_BATCH_SIZE = 512   # max items per single ChromaDB upsert call


class ChromaStore(BaseVectorStore):

    def __init__(self, persist_dir: str) -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._chunks = self._client.get_or_create_collection(
            _CHUNKS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._mtimes = self._client.get_or_create_collection(_MTIMES_COLLECTION)
        logger.info(
            f"ChromaStore ready at {persist_dir!r} "
            f"({self._chunks.count()} chunks, {self._mtimes.count()} files)"
        )

    # Executor helper — keeps all blocking Chroma calls off the event loop

    @staticmethod
    async def _run(fn, *args, **kwargs) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(fn, *args, **kwargs))

    # BaseVectorStore interface

    async def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert in batches to avoid ChromaDB's per-call size limits."""
        for start in range(0, len(ids), _UPSERT_BATCH_SIZE):
            end = start + _UPSERT_BATCH_SIZE
            await self._run(
                self._chunks.upsert,
                ids=ids[start:end],
                embeddings=vectors[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

    async def query(
        self,
        vector: list[float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        total = await self._run(self._chunks.count)
        if total == 0:
            return []

        results = await self._run(
            self._chunks.query,
            query_embeddings=[vector],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict[str, Any]] = []
        ids       = results.get("ids",       [[]])[0]
        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            output.append({
                "id":         chunk_id,
                "chunk_text": doc,
                "score":      round(1.0 - dist, 4),
                **(meta or {}),
            })

        return output

    async def delete_file(self, file_path: str) -> None:
        #Delete all chunks for file_path using the metadata where filter.
        try:
            await self._run(
                self._chunks.delete,
                where={"file_path": {"$eq": file_path}},
            )
            # Also remove the mtime entry
            await self._run(
                self._mtimes.delete,
                ids=[file_path],
            )
        except Exception as exc:
            logger.warning(f"delete_file({file_path!r}) error: {exc}")

    async def get_file_mtimes(self) -> dict[str, float]:
        try:
            result = await self._run(
                self._mtimes.get,
                include=["documents"],
            )
            return {
                path: float(mtime)
                for path, mtime in zip(result["ids"], result["documents"])
            }
        except Exception:
            return {}

    async def set_file_mtime(self, file_path: str, mtime: float) -> None:
        await self._run(
            self._mtimes.upsert,
            ids=[file_path],
            documents=[str(mtime)],
        )

    async def stats(self) -> dict[str, Any]:
        chunk_count = await self._run(self._chunks.count)
        file_count  = await self._run(self._mtimes.count)
        return {
            "total_chunks":  chunk_count,
            "indexed_files": file_count,
        }

    async def aclose(self) -> None:
        # ChromaDB PersistentClient has no explicit close — data is flushed
        # on each write.  Method present to satisfy the ABC contract.
        pass