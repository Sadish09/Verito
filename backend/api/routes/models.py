"""
GET  /models  - list all embedding models installed in Ollama.
PATCH /config - persist a user config change.

The UI calls GET /models to populate the model dropdown, then PATCH /config
with { "embedding_model": "<selected>" } when the user picks one.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


@router.get("/models")
async def list_models(request: Request):
    """
    Query Ollama /api/tags and return all installed model names.
    Returns an empty list if Ollama is unreachable rather than 500-ing —
    the UI handles the empty state by showing 'Ollama unreachable'.
    """
    embedder = request.app.state.embedder

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{embedder._base_url}/api/tags")
            r.raise_for_status()
            raw = r.json().get("models", [])
    except Exception:
        return {"models": [], "ollama_reachable": False}

    names = [m["name"] for m in raw if m.get("name")]
    current = request.app.state.config.embedding_model

    return {
        "models":           names,
        "selected":         current,       # None if not yet configured
        "ollama_reachable": True,
    }


class ConfigPatch(BaseModel):
    embedding_model: str | None = None


@router.patch("/config")
async def patch_config(body: ConfigPatch, request: Request):
    """
    Apply a partial config update and persist it to disk.
    Currently only embedding_model is patchable from the UI.
    """
    config  = request.app.state.config
    embedder = request.app.state.embedder

    if body.embedding_model is not None:
        model = body.embedding_model.strip()
        if not model:
            raise HTTPException(status_code=422, detail="embedding_model must not be empty.")

        # Confirm the model is actually installed before accepting it
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{embedder._base_url}/api/tags")
                r.raise_for_status()
                installed = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            raise HTTPException(status_code=503, detail="Ollama is unreachable.")

        if not any(m.startswith(model.split(":")[0]) for m in installed):
            raise HTTPException(
                status_code=404,
                detail=f"Model {model!r} is not installed in Ollama. "
                       f"Run: ollama pull {model}"
            )

        config.embedding_model = model
        # Hot-swap the embedder's active model so searches use it immediately
        embedder._model = model

    # Persist whatever changed
    from backend.utils.config import save_config
    save_config(config)

    return {"status": "ok", "embedding_model": config.embedding_model}