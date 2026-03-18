"""
Vector store factory.

Adding a new backend:
  1. Implement BaseVectorStore in a new file under storage/
  2. Register it in _REGISTRY below — no other file needs to change.

_REGISTRY maps the OSS_VECTOR_BACKEND config value to a zero-arg
callable that returns a constructed BaseVectorStore.  Imports are
deferred inside the lambdas so unused backends add zero import cost.
"""

from __future__ import annotations

from typing import Callable

from base import BaseVectorStore
from backend.utils.config import SemanticSearchConfig

# Registry: backend name -> factory callable(config) -> BaseVectorStore
_REGISTRY: dict[str, Callable[[SemanticSearchConfig], BaseVectorStore]] = {}


def _register(name: str):
    #Decorator to register a backend factory under name.
    def decorator(fn: Callable[[SemanticSearchConfig], BaseVectorStore]):
        _REGISTRY[name] = fn
        return fn
    return decorator


@_register("chroma")
def _chroma(config: SemanticSearchConfig) -> BaseVectorStore:
    from chroma_store import ChromaStore
    return ChromaStore(persist_dir=config.chroma_persist_dir)


def create_store(config: SemanticSearchConfig) -> BaseVectorStore:
    backend = config.vector_store_backend.lower()
    factory = _REGISTRY.get(backend)

    if factory is None:
        supported = ", ".join(f"'{k}'" for k in _REGISTRY)
        raise ValueError(
            f"Unknown vector store backend: {backend!r}. "
            f"Supported: {supported}"
        )

    return factory(config)

#Return all the registered backends
def registered_backends() -> list[str]:
    return list(_REGISTRY.keys())