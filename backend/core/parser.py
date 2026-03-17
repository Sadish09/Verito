"""
Parser crawls a Markdown file and extract all the headings, subheadings, paragraph content and so on.
Fancy artifacts may not be stripped.
"""

from __future__ import annotations

import re


class MarkdownParser:

    _FENCED_CODE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    _INLINE_CODE = re.compile(r"`[^`\n]+`")
    _WIKI_IMAGE = re.compile(r"!\[\[([^\]]+)\]\]")
    _WIKI_LINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
    _MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^\)]*\)")
    _MD_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
    _HTML_TAG = re.compile(r"<[^>]+>")
    _HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
    _BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")
    _HR = re.compile(r"^(\*{3,}|-{3,}|_{3,})\s*$", re.MULTILINE)
    _BLOCKQUOTE_CALLOUT = re.compile(r"^>\s*\[![^\]]+\][^\n]*", re.MULTILINE)
    _BLOCKQUOTE_PREFIX = re.compile(r"^>\s?", re.MULTILINE)
    _EXCESS_BLANK = re.compile(r"\n{3,}")

    @classmethod
    def clean(cls, text: str) -> str:
        """Return cleaned prose ready for chunking."""
        t = text

        # remove fenced code blocks entirely
        t = cls._FENCED_CODE.sub("", t)

        # remove inline code
        t = cls._INLINE_CODE.sub("", t)

        # remove image wiki links  ![[image.png]]
        t = cls._WIKI_IMAGE.sub("", t)

        # resolve wiki links  [[Target|Alias]] → Alias,  [[Target]] → Target
        t = cls._WIKI_LINK.sub(lambda m: m.group(2) or m.group(1), t)

        # remove markdown images  ![alt](url) → ""
        t = cls._MD_IMAGE.sub("", t)

        # resolve markdown links  [text](url) → text
        t = cls._MD_LINK.sub(r"\1", t)

        # strip HTML tags
        t = cls._HTML_TAG.sub("", t)

        # strip heading markers (keep the heading text)
        t = cls._HEADING.sub("", t)

        # unwrap bold/italic (keep inner text)
        t = cls._BOLD_ITALIC.sub(r"\2", t)

        # remove horizontal rules
        t = cls._HR.sub("", t)

        # strip Obsidian callout headers  > [!NOTE] Title
        t = cls._BLOCKQUOTE_CALLOUT.sub("", t)

        # remove block-quote `>` prefix
        t = cls._BLOCKQUOTE_PREFIX.sub("", t)

        # collapse 3+ blank lines to 2
        t = cls._EXCESS_BLANK.sub("\n\n", t)

        return t.strip()
