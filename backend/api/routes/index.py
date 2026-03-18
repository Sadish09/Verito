"""POST /index - trigger vault crawl -> chunk -> embed -> store."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class IndexRequest(BaseModel):
    vault_path: str
    watch: bool = False


@router.post("/index")
async def trigger_index(
    body: IndexRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    config  = request.app.state.config
    indexer = request.app.state.indexer

    if not config.embedding_model:
        raise HTTPException(
            status_code=422,
            detail="No embedding model selected. Choose a model from Settings before indexing."
        )

    if indexer.is_indexing:
        raise HTTPException(status_code=409, detail="Indexing already in progress.")

    background_tasks.add_task(
        indexer.index_vault,
        vault_path=body.vault_path,
        watch=body.watch,
    )

    return {"status": "indexing_started", "vault_path": body.vault_path}