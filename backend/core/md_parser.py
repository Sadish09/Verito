"""
Each Section represents a heading-bounded region of the document:
  - heading_path   : breadcrumb of ancestor headings  ["Security", "Auth"]
  - heading_levels : corresponding ATX levels          [2, 3]
  - body           : cleaned prose paragraphs joined by double newline

The AST walk preserves document structure so that chunks know which section
they came from, breadcrumbs are prepended to chunks to preserve data.

Notes with no headings produce a single Section whose heading_path is either
["<frontmatter title>"] or [] if no title is available.  The Indexer passes
the frontmatter title in via the `implicit_title` parameter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import mistune


# Inline cleaning — applied to extracted text, not raw markdown

_WIKI_IMAGE   = re.compile(r"!\[\[([^\]]+)\]\]")
_WIKI_LINK    = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_EXCESS_BLANK = re.compile(r"\n{3,}")
_CALLOUT_HDR  = re.compile(r"^\[![^\]]+\]\s*", re.MULTILINE)  # [!NOTE] inside blockquote


def _clean_inline(text: str) -> str:
    """Strip Obsidian-specific syntax from already-extracted plain text."""
    text = _WIKI_IMAGE.sub("", text)
    text = _WIKI_LINK.sub(lambda m: m.group(2) or m.group(1), text)
    text = _CALLOUT_HDR.sub("", text)
    return text.strip()

# Section dataclass


@dataclass
class Section:
    heading_path:   list[str] = field(default_factory=list)
    heading_levels: list[int] = field(default_factory=list)
    body:           str       = ""

    @property
    def breadcrumb(self) -> str:
        """Human-readable heading path, e.g. 'Security > Authentication'."""
        return " > ".join(self.heading_path)

    @property
    def embed_text(self) -> str:
        """
        Text to embed: breadcrumb prepended to body so the vector captures
        section context.  If there is no heading, body is returned as-is.
        """
        if self.heading_path:
            return f"{self.breadcrumb}\n\n{self.body}"
        return self.body

    def is_empty(self) -> bool:
        return not self.body.strip()


# AST walker

class _SectionBuilder:
    """
    Stateful walker over a mistune AST token list.

    Maintains a heading stack so every paragraph knows its full ancestor path.
    Produces one Section per heading-bounded region.
    """

    def __init__(self) -> None:
        self._sections: list[Section] = []
        # Stack entries: (level: int, text: str)
        self._heading_stack: list[tuple[int, str]] = []
        self._current_paragraphs: list[str] = []


    def walk(self, tokens: list[dict]) -> list[Section]:
        for token in tokens:
            t = token.get("type")
            if t == "heading":
                self._flush()
                self._push_heading(token)
            elif t in ("paragraph", "block_text"):
                self._current_paragraphs.append(self._extract_text(token))
            elif t == "block_quote":
                # Recurse into blockquote children — callout headers are stripped
                # by _clean_inline; body paragraphs are kept
                for child in token.get("children", []):
                    if child.get("type") in ("paragraph", "block_text"):
                        self._current_paragraphs.append(self._extract_text(child))
            elif t == "list":
                self._current_paragraphs.append(self._extract_list(token))
            # code_block, thematic_break, html_block → intentionally ignored

        self._flush()
        return self._sections

    # Private

    def _push_heading(self, token: dict) -> None:
        level = token.get("attrs", {}).get("level", 1)
        text  = _clean_inline(self._inline_text(token.get("children", [])))

        # Pop any headings at same or deeper level
        self._heading_stack = [
            (lvl, t) for lvl, t in self._heading_stack if lvl < level
        ]
        self._heading_stack.append((level, text))

    def _flush(self) -> None:
        """Emit the current paragraph buffer as a Section."""
        body = _EXCESS_BLANK.sub(
            "\n\n",
            "\n\n".join(p for p in self._current_paragraphs if p.strip()),
        ).strip()

        if body:
            levels = [lvl for lvl, _ in self._heading_stack]
            path   = [t   for _,   t in self._heading_stack]
            self._sections.append(
                Section(heading_path=path, heading_levels=levels, body=body)
            )

        self._current_paragraphs = []

    # Text extraction from inline token trees

    def _extract_text(self, token: dict) -> str:
        raw = self._inline_text(token.get("children", []))
        return _clean_inline(raw)

    def _inline_text(self, children: list[dict]) -> str:
        parts: list[str] = []
        for child in children:
            t = child.get("type")
            if t == "raw":
                parts.append(child.get("raw", ""))
            elif t == "softlinebreak":
                parts.append(" ")
            elif t == "linebreak":
                parts.append("\n")
            elif t in ("strong", "emphasis", "codespan"):
                # Unwrap — keep inner text, discard markers
                parts.append(self._inline_text(child.get("children", [])))
            elif t == "link":
                # [text](url) → text
                parts.append(self._inline_text(child.get("children", [])))
            elif t == "image":
                # ![alt](url) → discard entirely
                pass
            else:
                # Unknown inline node — recurse defensively
                parts.append(self._inline_text(child.get("children", [])))
        return "".join(parts)

    def _extract_list(self, token: dict) -> str:
        """Flatten list items into prose lines."""
        lines: list[str] = []
        for item in token.get("children", []):
            for child in item.get("children", []):
                text = self._extract_text(child).strip()
                if text:
                    lines.append(text)
        return "\n".join(lines)


# Public interface

# Build the mistune parser once at import time — it's stateless
_MD = mistune.create_markdown(renderer=None)   # renderer=None → AST mode


class MarkdownParser:

    @staticmethod
    def parse(text: str, implicit_title: str = "") -> list[Section]:
        """
        Parse *text* (post-frontmatter body) into a list of Sections.

        Params
        text :
            Raw markdown body with frontmatter already stripped.
        implicit_title :
            Frontmatter title to use as the heading path for headingless notes.
            Ignored if the document contains at least one heading.

        Returns
        list[Section]
            One Section per heading-bounded region, in document order.
            Never returns an empty list for non-empty input — at minimum
            one Section with an empty heading_path is returned.
        """
        if not text.strip():
            return []

        tokens = _MD(text)  # list[dict] AST
        sections = _SectionBuilder().walk(tokens)

        # Filter empty sections
        sections = [s for s in sections if not s.is_empty()]

        if not sections:
            return []

        # If no headings were found, apply implicit_title to all sections
        has_headings = any(s.heading_path for s in sections)
        if not has_headings and implicit_title:
            for s in sections:
                s.heading_path   = [implicit_title]
                s.heading_levels = [1]

        return sections