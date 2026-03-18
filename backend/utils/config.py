"""
Configuration chain: env vars (OSS_ prefix) > JSON file > dataclass defaults.

Adding a new config field:
  1. Add the field to SemanticSearchConfig with a default value.
  2. Add an entry to _ENV_MAP with its env var suffix and coerce type.
  That's it — JSON file loading and validation pick it up automatically.

embedding_model is intentionally None by default. The user must select a
model from the UI dropdown (populated by GET /models, which queries Ollama
for what is actually installed). The backend refuses to index or search
until a model has been selected and persisted.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

CONFIG_DIR  = Path.home() / ".obsidian-semantic-search"
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class SemanticSearchConfig:
    # Server
    host: str = "127.0.0.1"
    port: int = 8765

    # Ollama
    ollama_url:      str       = "http://localhost:11434"
    embedding_model: str | None = None   # None until user selects from UI

    # Logging
    log_level: str = "INFO"

    # Vector store
    vector_store_backend: str = "chroma"
    chroma_persist_dir:   str = str(CONFIG_DIR / "chroma")

    # Chunking
    chunk_size:    int = 256   # tokens per chunk
    chunk_overlap: int = 32    # overlap between consecutive chunks


# Maps OSS_<SUFFIX> → (dataclass field name, coerce type)
_ENV_MAP: dict[str, tuple[str, type]] = {
    "HOST":            ("host",                 str),
    "PORT":            ("port",                 int),
    "OLLAMA_URL":      ("ollama_url",            str),
    "EMBEDDING_MODEL": ("embedding_model",       str),
    "LOG_LEVEL":       ("log_level",             str),
    "VECTOR_BACKEND":  ("vector_store_backend",  str),
    "CHROMA_DIR":      ("chroma_persist_dir",    str),
    "CHUNK_SIZE":      ("chunk_size",            int),
    "CHUNK_OVERLAP":   ("chunk_overlap",         int),
}

_VALID_LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}


def load_config() -> SemanticSearchConfig:
    """
    Build a SemanticSearchConfig by layering:
      defaults → JSON file → environment variables
    Invalid values are logged as warnings rather than crashing, with the
    exception of clearly broken combinations caught by _validate().
    """
    cfg = SemanticSearchConfig()

    # Layer 1: JSON file (skipped if absent or malformed)
    if CONFIG_FILE.exists():
        try:
            _apply_dict(cfg, json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            import warnings
            warnings.warn(f"Config file {CONFIG_FILE} is malformed, using defaults: {exc}")

    # Layer 2: Environment variables (highest priority)
    for suffix, (attr, coerce) in _ENV_MAP.items():
        raw = os.environ.get(f"OSS_{suffix}")
        if raw is not None:
            try:
                setattr(cfg, attr, coerce(raw))
            except (ValueError, TypeError) as exc:
                import warnings
                warnings.warn(
                    f"OSS_{suffix}={raw!r} could not be coerced to "
                    f"{coerce.__name__}: {exc}"
                )

    _validate(cfg)
    return cfg


def save_config(cfg: SemanticSearchConfig) -> None:
    """
    Persist the current config to the JSON file.
    Called by the /config PATCH route when the user selects a model.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _apply_dict(cfg: SemanticSearchConfig, data: dict[str, Any]) -> None:
    """Apply key/value pairs from *data* to *cfg*, coercing to the field's type."""
    field_types = {f.name: f.type for f in fields(cfg)}
    for key, value in data.items():
        if hasattr(cfg, key):
            try:
                expected = field_types.get(key)
                if expected in ("int", int) and not isinstance(value, int):
                    value = int(value)
                elif expected in ("str", str) and not isinstance(value, str):
                    value = str(value)
                setattr(cfg, key, value)
            except (ValueError, TypeError):
                pass


def _validate(cfg: SemanticSearchConfig) -> None:
    """Raise ValueError for obviously invalid config combinations."""
    if cfg.chunk_overlap >= cfg.chunk_size:
        raise ValueError(
            f"chunk_overlap ({cfg.chunk_overlap}) must be less than "
            f"chunk_size ({cfg.chunk_size})"
        )
    if cfg.port < 1 or cfg.port > 65535:
        raise ValueError(f"port must be 1–65535, got {cfg.port}")
    if cfg.log_level.upper() not in _VALID_LOG_LEVELS:
        raise ValueError(
            f"log_level {cfg.log_level!r} is not valid. "
            f"Choose from: {', '.join(sorted(_VALID_LOG_LEVELS))}"
        )
    # embedding_model is allowed to be None at startup — validated at use time
    # by the index and search routes, not here.