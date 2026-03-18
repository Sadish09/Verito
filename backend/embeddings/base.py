from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """
    Contract for all embedding backends:
    Implementations must be async safe.
    The server wires one embedder instance at startup; it lives for the
    process lifetime and is closed via aclose() on shutdown.

    Subclassing checklist:
      - embed()        - single-text vector
      - embed_batch()  - multi-text vectors (may delegate to embed)
      - is_available() - fast liveness check
      - model property - currently active model name
      - aclose()       - release connections / resources
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        #Return the embedding vector for text.
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Return embedding vectors for a list of texts.

        Implementations may send texts concurrently, in serial, or in
        backend-native batches.  The returned list must be the same
        length and order as *texts*.
        """
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        #Return True if the embedding backend is reachable.
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        #The model name currently in use.
        ...

    @abstractmethod
    async def aclose(self) -> None:
        #Release any held connections or resources.
        ...