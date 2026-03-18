"""GET /status - index stats, Ollama reachability, is_indexing flag."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/status")
async def get_status(request: Request):
    indexer  = request.app.state.indexer
    embedder = request.app.state.embedder
    store    = request.app.state.store
    config   = request.app.state.config

    store_stats = await store.stats()
    ollama_ok   = await embedder.is_available()

    return {
        "is_indexing":       indexer.is_indexing,
        "indexed_count":     indexer.indexed_count,
        "total_files":       indexer.total_files,
        "ollama_reachable":  ollama_ok,
        # None signals "not configured" — the UI shows the model picker
        "embedding_model":   config.embedding_model,
        "model_configured":  config.embedding_model is not None,
        "vector_backend":    config.vector_store_backend,
        **store_stats,
    }