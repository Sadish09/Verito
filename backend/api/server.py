"""
create_app(config) wires all services and returns the FastAPI instance.
Services live on app.state so routes can access them via request.app.state

embedding_model may be None at startup if the user has not yet selected
one.  The embedder is still constructed;
the index and search routes guard against None before calling Ollama.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from routes import index as index_router
from routes import models as models_router
from routes import search as search_router
from routes import status as status_router
from backend.core.indexer import Indexer
from backend.embeddings.ollama_embedder import OllamaEmbedder
from backend.storage.store_factory import create_store
from backend.utils.config import SemanticSearchConfig

# Used as the OllamaEmbedder model placeholder when none has been selected yet
_NO_MODEL = ""


def create_app(config: SemanticSearchConfig) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        #start
        #embedding_model is None until the user selects one from the UI.
        #OllamaEmbedder is constructed regardless so the httpx client is ready to respond
        #operations that call Ollama check config.embedding_model first.
        embedder = OllamaEmbedder(
            base_url=config.ollama_url,
            model=config.embedding_model or _NO_MODEL,
        )
        store   = create_store(config)
        indexer = Indexer(config=config, embedder=embedder, store=store)

        app.state.config   = config
        app.state.embedder = embedder
        app.state.store    = store
        app.state.indexer  = indexer

        model_label = config.embedding_model or "not selected"
        logger.info(
            f"Backend started — model={model_label!r} "
            f"store={config.vector_store_backend!r} "
            f"port={config.port}"
        )
        yield

        #shutdown
        await embedder.aclose()
        await store.aclose()
        logger.info("Backend shutdown complete.")

    app = FastAPI(
        title="Obsidian Semantic Search",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["app://obsidian.md", "http://localhost"],
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.include_router(status_router.router)
    app.include_router(models_router.router)
    app.include_router(index_router.router)
    app.include_router(search_router.router)

    return app