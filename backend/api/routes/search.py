"""POST /search - query embedding -> vector search → filtered ranked results."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    embedding_model: str | None = None   # per-request override


class SearchResult(BaseModel):
    id: str
    file_path: str
    file_name: str
    chunk_text: str
    chunk_index: int
    chunk_total: int
    score: float
    heading_path: str = ""


@router.post("/search", response_model=list[SearchResult])
async def search(body: SearchRequest, request: Request):
    embedder = request.app.state.embedder
    store    = request.app.state.store
    config   = request.app.state.config

    if not body.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    # Resolve the model to use — per-request override takes priority,
    # then the configured model, then fail clearly.
    active_model = body.embedding_model or config.embedding_model
    if not active_model:
        raise HTTPException(
            status_code=422,
            detail="No embedding model selected. Choose a model from Settings before searching."
        )

    # Use a temporary embedder if the requested model differs from the default
    if active_model != embedder.model:
        from backend.embeddings.ollama_embedder import OllamaEmbedder
        tmp = OllamaEmbedder(
            base_url=config.ollama_url,
            model=active_model,
            timeout=60.0,
        )
        try:
            query_vector = await tmp.embed(body.query)
        finally:
            await tmp.aclose()
    else:
        query_vector = await embedder.embed(body.query)

    raw_results = await store.query(vector=query_vector, top_k=body.top_k)

    results: list[SearchResult] = []
    for r in raw_results:
        try:
            results.append(SearchResult(
                id=r["id"],
                file_path=r.get("file_path", ""),
                file_name=r.get("file_name", ""),
                chunk_text=r.get("chunk_text", ""),
                chunk_index=int(r.get("chunk_index", 0)),
                chunk_total=int(r.get("chunk_total", 0)),
                score=float(r.get("score", 0.0)),
                heading_path=r.get("heading_path", ""),
            ))
        except Exception:
            continue

    return results