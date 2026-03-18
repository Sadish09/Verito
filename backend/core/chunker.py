"""
Chunker: paragraph-first sliding-window chunking with token counting.

Strategy:
  - Split on blank lines to get paragraphs.
  - Accumulate paragraphs into a token window of `chunk_size` tokens.
  - Slide forward by (chunk_size - chunk_overlap) tokens when the window fills.
  - Oversized single paragraphs are hard-split at sentence boundaries.

Token counting uses tiktoken cl100k_base, accurate for nomic embedding models; more to be added soon
BaseChunker ABC allows alternative strategies (fixed-char, heading-aware)
to be added as subclasses without touching the Indexer.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _token_count(text: str) -> int:
    return len(ENCODING.encode(text))


def _split_sentences(text: str) -> list[str]:
    #Split text into sentences on .!? boundaries.
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p for p in parts if p]


@dataclass
class Chunk:
    text: str
    index: int          # 0-based position within the file
    token_count: int


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        ...


class ParagraphChunker(BaseChunker):
    """
    Params
    chunk_size : int
        max tokens per chunk
    chunk_overlap : int
        token overlap between consecutive chunks
    """

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._stride = chunk_size - chunk_overlap


    def chunk(self, text: str) -> list[Chunk]:
        if not text.strip():
            return []

        paragraphs = self._split_paragraphs(text)
        # Flatten paragraphs → list of atomic units (sentences for big ones)
        units = self._to_units(paragraphs)

        chunks: list[Chunk] = []
        window: list[str] = []
        window_tokens = 0

        for unit in units:
            unit_tokens = _token_count(unit)

            # Edge case: single unit bigger than chunk_size
            if unit_tokens > self.chunk_size:
                # Flush current window first
                if window:
                    chunks.append(self._make_chunk(window, len(chunks)))
                    window, window_tokens = self._trim_overlap(window)

                # Force-emit the oversized unit as its own chunk
                chunks.append(Chunk(
                    text=unit,
                    index=len(chunks),
                    token_count=unit_tokens,
                ))
                continue

            if window_tokens + unit_tokens > self.chunk_size:
                # Emit current window
                chunks.append(self._make_chunk(window, len(chunks)))
                # Retain overlap
                window, window_tokens = self._trim_overlap(window)

            window.append(unit)
            window_tokens += unit_tokens

        # Emit any remaining units
        if window:
            chunks.append(self._make_chunk(window, len(chunks)))

        return chunks

    # Private helpers

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    def _to_units(self, paragraphs: list[str]) -> list[str]:
        """
        Return paragraphs as-is if they fit within chunk_size,
        otherwise split into sentences
        """
        units: list[str] = []
        for para in paragraphs:
            if _token_count(para) <= self.chunk_size:
                units.append(para)
            else:
                units.extend(_split_sentences(para))
        return units

    def _trim_overlap(self, window: list[str]) -> tuple[list[str], int]:
        """
        Remove units from the front of the window until remaining tokens
        are <= chunk_overlap, keeping the tail for the next chunk
        """
        tokens = sum(_token_count(u) for u in window)
        while window and tokens > self.chunk_overlap:
            tokens -= _token_count(window[0])
            window = window[1:]
        return window, tokens

    @staticmethod
    def _make_chunk(units: list[str], index: int) -> Chunk:
        text = " ".join(units)
        return Chunk(text=text, index=index, token_count=_token_count(text))