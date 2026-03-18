"""
Indexer: vault crawl -> parse -> chunk -> embed -> store.

This file implements incremental indexing, ChromaDB collection (obsidian_mtimes) maps
file paths -> last-seen mtime. Only new or changed files are processed.
Chunk IDs are constructed as  file_path::chunk::N  so all chunks for a
file can be deleted before re-indexing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import frontmatter
from loguru import logger

from chunker import ParagraphChunker
from md_parser import MarkdownParser

if TYPE_CHECKING:
    from backend.embeddings.base import BaseEmbedder
    from backend.storage.base import BaseVectorStore
    from backend.utils.config import SemanticSearchConfig


class Indexer:
    def __init__(
        self,
        config: "SemanticSearchConfig",
        embedder: "BaseEmbedder",
        store: "BaseVectorStore",
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._store = store
        self._chunker = ParagraphChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.is_indexing = False
        self._indexed_count = 0
        self._total_files = 0


    # Public API

    async def index_vault(self, vault_path: str, watch: bool = False) -> dict:
        """
        Crawl vault_path, embed changed files, store chunks.
        Returns summary statistics.
        """
        self.is_indexing = True
        self._indexed_count = 0
        start = time.monotonic()

        try:
            result = await self._run_index(Path(vault_path))
        finally:
            self.is_indexing = False

        elapsed = round(time.monotonic() - start, 2)
        logger.info(
            f"Indexing complete: {result['indexed']} new/changed, "
            f"{result['skipped']} unchanged, {elapsed}s"
        )
        return {**result, "elapsed_seconds": elapsed}

    @property
    def indexed_count(self) -> int:
        return self._indexed_count

    @property
    def total_files(self) -> int:
        return self._total_files

    # Internal

    async def _run_index(self, vault: Path) -> dict:
        md_files = sorted(vault.rglob("*.md"))
        self._total_files = len(md_files)

        if not md_files:
            return {"indexed": 0, "skipped": 0, "errors": 0, "total_files": 0}

        # Fetch stored mtimes once for O(1) lookup
        stored_mtimes = await self._store.get_file_mtimes()

        indexed = skipped = errors = 0

        for md_path in md_files:
            path_str = str(md_path)
            try:
                current_mtime = md_path.stat().st_mtime
                stored_mtime = stored_mtimes.get(path_str)

                if stored_mtime is not None and abs(current_mtime - float(stored_mtime)) < 0.01:
                    skipped += 1
                    continue

                # Delete stale chunks before re-indexing
                if stored_mtime is not None:
                    await self._store.delete_file(path_str)

                n_chunks = await self._embed_file(md_path)
                if n_chunks > 0:
                    await self._store.set_file_mtime(path_str, current_mtime)
                    indexed += 1
                    self._indexed_count += 1

            except Exception as exc:
                logger.warning(f"Error indexing {path_str}: {exc}")
                errors += 1

        return {
            "indexed": indexed,
            "skipped": skipped,
            "errors": errors,
            "total_files": self._total_files,
        }

    async def _embed_file(self, path: Path) -> int:
        """Parse, chunk, embed, and store a single file. Returns chunk count."""
        raw = path.read_text(encoding="utf-8", errors="ignore")

        # Separate frontmatter from body
        post = frontmatter.loads(raw)
        implicit_title = str(post.metadata.get("title", ""))

        sections = MarkdownParser.parse(post.content, implicit_title=implicit_title)
        if not sections:
            return 0

        # Frontmatter fields stored on every chunk for future filtering
        fm_meta = {
            k: str(v)
            for k, v in post.metadata.items()
            if isinstance(v, (str, int, float, bool))
        }

        # Collect all chunks across all sections before any I/O
        all_ids:       list[str]       = []
        all_texts:     list[str]       = []  # raw body text stored in DB
        all_embed:     list[str]       = []  # breadcrumb-prefixed text sent to embedder
        all_metadatas: list[dict]      = []

        global_index = 0
        for section in sections:
            chunks = self._chunker.chunk(section.body)
            for chunk in chunks:
                chunk_id = f"{path}::chunk::{global_index}"
                all_ids.append(chunk_id)
                all_texts.append(chunk.text)
                all_embed.append(
                    f"{section.breadcrumb}\n\n{chunk.text}"
                    if section.breadcrumb else chunk.text
                )
                all_metadatas.append({
                    "file_path":      str(path),
                    "file_name":      path.name,
                    "chunk_index":    global_index,
                    "heading_path":   section.breadcrumb,
                    "heading_levels": " > ".join(str(l) for l in section.heading_levels),
                    **fm_meta,
                })
                global_index += 1

        if not all_ids:
            return 0

        # Patch total now that we know it
        for m in all_metadatas:
            m["chunk_total"] = global_index

        # Embed all chunks — embed_batch handles concurrency internally
        vectors = await self._embedder.embed_batch(all_embed)

        await self._store.upsert(
            ids=all_ids,
            vectors=list(vectors),
            texts=all_texts,
            metadatas=all_metadatas,
        )
        return global_index