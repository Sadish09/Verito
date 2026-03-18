"""
OllamaEmbedder: persistent httpx.AsyncClient → Ollama /api/embeddings.

Design notes:
  - Single AsyncClient instance reused across all embed() calls — avoids
    TCP reconnect overhead during bulk indexing.
  - embed_batch() fans out concurrently via asyncio.gather with a semaphore
    to avoid overwhelming Ollama's request queue.
  - Exponential backoff retry on transient network errors (3 attempts).
  - is_available() checks /api/tags and matches model name by prefix so
    "nomic-embed-text" matches "nomic-embed-text:latest".
"""

from __future__ import annotations

import asyncio

import httpx
from loguru import logger

from base import BaseEmbedder

# Max concurrent embed requests to Ollama
_BATCH_CONCURRENCY = 8
_MAX_RETRIES = 3


class OllamaEmbedder(BaseEmbedder):
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
        )
        self._semaphore = asyncio.Semaphore(_BATCH_CONCURRENCY)

    # ------------------------------------------------------------------ #
    # BaseEmbedder interface

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        """
        Embed a single text with exponential-backoff retry on transient errors.
        Raises RuntimeError after _MAX_RETRIES failed attempts.
        """
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post(
                    "/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding")
                if not embedding:
                    raise ValueError(
                        f"Ollama returned no 'embedding' field for model "
                        f"{self._model!r}. Response: {data}"
                    )
                return embedding

            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                wait = 2 ** attempt   # 1s → 2s → 4s
                logger.warning(
                    f"Ollama embed attempt {attempt + 1}/{_MAX_RETRIES} failed "
                    f"({exc}); retrying in {wait}s"
                )
                await asyncio.sleep(wait)

            except Exception:
                raise   # non-transient — propagate immediately

        raise RuntimeError(
            f"Ollama embed failed after {_MAX_RETRIES} attempts"
        ) from last_exc

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts concurrently, bounded by _BATCH_CONCURRENCY.
        Preserves input order in the returned list.
        """
        async def _bounded(text: str) -> list[float]:
            async with self._semaphore:
                return await self.embed(text)

        return list(await asyncio.gather(*[_bounded(t) for t in texts]))

    async def is_available(self) -> bool:
        """
        Return True if Ollama is reachable and the configured model exists.
        Short 5s timeout so status checks never block the UI.
        """
        try:
            r = await self._client.get("/api/tags", timeout=5.0)
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
            prefix = self._model.split(":")[0]
            return any(m.startswith(prefix) for m in models)
        except Exception as exc:
            logger.debug(f"Ollama availability check failed: {exc}")
            return False

    async def aclose(self) -> None:
        await self._client.aclose()