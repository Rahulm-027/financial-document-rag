"""
chunker.py
==========
Converts RawElements into Chunks — the unit that gets embedded and stored.

Three distinct strategies per modality:

TEXT   → semantic / heading-aware splitting at ~512 tokens, with 64-token
         overlap.  We do NOT blindly split at fixed byte boundaries; instead
         we respect sentence and paragraph boundaries.

TABLE  → kept whole where possible (up to max_tokens).  A *textual
         representation* of the table is also generated (pipe-delimited
         markdown-style) because the embedding model works on text, not grids.

CHART  → the VLM-generated structured description is the chunk content.
         We add the raw caption and section as extra context.

Every chunk carries full provenance metadata so that downstream retrieval
can filter by company, year, page, or document type.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

# We use a simple heuristic token counter to avoid a hard transformers dep here.
# (Actual embedding uses the sentence-transformer tokenizer.)
def _approx_tokens(text: str) -> int:
    return len(text.split())


# ─────────────────────────────────────────────
# Chunk dataclass
# ─────────────────────────────────────────────

@dataclass
class Chunk:
    """
    The atomic unit of the RAG system.

    Fields
    ------
    chunk_id      : unique identifier (uuid4)
    chunk_type    : "text" | "table" | "chart"
    content       : the textual content that will be embedded
    company       : e.g. "Tesla"
    document      : e.g. "Tesla 10-K 2023"
    year          : integer year
    page          : 1-indexed page number in the source PDF
    section       : nearest heading/section name
    caption       : table or chart caption if available
    raw_table     : list[list[str]] for table chunks (for display)
    """

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chunk_type: str = ""
    content: str = ""
    company: str = ""
    document: str = ""
    year: int = 0
    page: int = 0
    section: str = ""
    caption: str = ""
    raw_table: Optional[list] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────
# Table → textual representation
# ─────────────────────────────────────────────

def table_to_text(table: list[list[str]], caption: str = "", section: str = "") -> str:
    """
    Convert a 2-D table (list of rows) into a readable text representation.

    Format:
        Section: <section>
        Caption: <caption>
        | Col1 | Col2 | ... |
        |------|------|-----|
        | val  | val  | ... |
    """
    lines = []
    if section:
        lines.append(f"Section: {section}")
    if caption:
        lines.append(f"Caption: {caption}")

    if not table:
        return "\n".join(lines)

    # Normalise rows to same width
    max_cols = max(len(row) for row in table)
    padded = [row + [""] * (max_cols - len(row)) for row in table]

    # Header row
    header = padded[0]
    lines.append("| " + " | ".join(str(c) for c in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    # Data rows
    for row in padded[1:]:
        # Skip completely empty rows
        if all(not str(c).strip() for c in row):
            continue
        lines.append("| " + " | ".join(str(c) for c in row) + " |")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Text splitting
# ─────────────────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_PARA_SPLIT_RE = re.compile(r'\n{2,}')


def _split_text(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[str]:
    """
    Split text into chunks of at most max_tokens, respecting paragraph
    and sentence boundaries.  Adjacent chunks share overlap_tokens of text.
    """
    # First split at paragraph boundaries
    paragraphs = _PARA_SPLIT_RE.split(text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[str] = []
    current_tokens = 0
    current_parts: list[str] = []

    def flush():
        nonlocal current_tokens, current_parts
        if current_parts:
            chunks.append(" ".join(current_parts))
        current_parts = []
        current_tokens = 0

    for para in paragraphs:
        para_tokens = _approx_tokens(para)

        if para_tokens > max_tokens:
            # Split paragraph further at sentence boundaries
            sentences = _SENTENCE_SPLIT_RE.split(para)
            for sent in sentences:
                st = _approx_tokens(sent)
                if current_tokens + st > max_tokens:
                    flush()
                    # overlap: keep last overlap_tokens worth of previous sentences
                    if chunks:
                        last_words = chunks[-1].split()[-overlap_tokens:]
                        current_parts = [" ".join(last_words)]
                        current_tokens = len(last_words)
                current_parts.append(sent)
                current_tokens += st
        else:
            if current_tokens + para_tokens > max_tokens:
                flush()
                if chunks:
                    last_words = chunks[-1].split()[-overlap_tokens:]
                    current_parts = [" ".join(last_words)]
                    current_tokens = len(last_words)
            current_parts.append(para)
            current_tokens += para_tokens

    flush()
    return [c for c in chunks if c.strip()]


# ─────────────────────────────────────────────
# Main Chunker
# ─────────────────────────────────────────────

class Chunker:
    """
    Converts a list of RawElement dicts (from pdf_parser + chart_describer)
    into a flat list of Chunk objects.

    Args
    ----
    text_max_tokens   : max tokens per text chunk
    text_overlap      : token overlap between text chunks
    table_max_tokens  : if a table text repr exceeds this, split by row groups
    min_content_len   : skip chunks shorter than this (characters)
    """

    def __init__(
        self,
        text_max_tokens: int = 512,
        text_overlap: int = 64,
        table_max_tokens: int = 1024,
        min_content_len: int = 30,
    ) -> None:
        self.text_max_tokens = text_max_tokens
        self.text_overlap = text_overlap
        self.table_max_tokens = table_max_tokens
        self.min_content_len = min_content_len

    # ------------------------------------------------------------------

    def chunk(self, elements: list[dict]) -> list[Chunk]:
        """
        Process all elements and return chunks.

        Args
        ----
        elements : list of dicts from pdf_parser / chart_describer.
                   Each must have keys: element_type, content, company,
                   document, year, page, section, caption.
                   Chart elements must additionally have 'description'.
        """
        all_chunks: list[Chunk] = []
        for elem in elements:
            etype = elem.get("element_type", "")
            if etype == "text":
                all_chunks.extend(self._chunk_text(elem))
            elif etype == "table":
                all_chunks.extend(self._chunk_table(elem))
            elif etype == "chart":
                chunk = self._chunk_chart(elem)
                if chunk:
                    all_chunks.append(chunk)

        return [c for c in all_chunks if len(c.content) >= self.min_content_len]

    # ------------------------------------------------------------------

    def _base_meta(self, elem: dict) -> dict:
        return {
            "company": elem.get("company", ""),
            "document": elem.get("document", ""),
            "year": elem.get("year", 0),
            "page": elem.get("page", 0),
            "section": elem.get("section", ""),
            "caption": elem.get("caption", ""),
        }

    # ------------------------------------------------------------------

    def _chunk_text(self, elem: dict) -> list[Chunk]:
        raw = elem.get("content", "")
        if isinstance(raw, str):
            text = raw
        else:
            text = str(raw)

        meta = self._base_meta(elem)
        splits = _split_text(text, self.text_max_tokens, self.text_overlap)
        chunks = []
        for split in splits:
            # Prepend section as context so the embedding captures topic
            full_content = (
                f"[{meta['company']} | {meta['document']} | "
                f"Page {meta['page']} | {meta['section']}]\n{split}"
                if meta["section"]
                else f"[{meta['company']} | {meta['document']} | "
                     f"Page {meta['page']}]\n{split}"
            )
            chunks.append(
                Chunk(chunk_type="text", content=full_content, **meta)
            )
        return chunks

    # ------------------------------------------------------------------

    def _chunk_table(self, elem: dict) -> list[Chunk]:
        raw_table = elem.get("content", [])
        if not raw_table:
            return []

        meta = self._base_meta(elem)
        text_repr = table_to_text(
            raw_table, caption=meta["caption"], section=meta["section"]
        )

        # Prepend provenance header
        header = (
            f"[TABLE | {meta['company']} | {meta['document']} | "
            f"Page {meta['page']}]"
        )
        full_content = f"{header}\n{text_repr}"

        # If table is small enough, return as single chunk
        if _approx_tokens(full_content) <= self.table_max_tokens:
            return [
                Chunk(
                    chunk_type="table",
                    content=full_content,
                    raw_table=raw_table,
                    **meta,
                )
            ]

        # Otherwise, split into row groups of ~20 rows each
        chunks = []
        header_row = raw_table[0] if raw_table else []
        data_rows = raw_table[1:] if len(raw_table) > 1 else []
        group_size = 20

        for start in range(0, len(data_rows), group_size):
            group = data_rows[start: start + group_size]
            sub_table = [header_row] + group
            sub_repr = table_to_text(
                sub_table,
                caption=f"{meta['caption']} (rows {start+1}–{start+len(group)})",
                section=meta["section"],
            )
            sub_content = f"{header} (rows {start+1}–{start+len(group)})\n{sub_repr}"
            chunks.append(
                Chunk(
                    chunk_type="table",
                    content=sub_content,
                    raw_table=sub_table,
                    **meta,
                )
            )
        return chunks

    # ------------------------------------------------------------------

    def _chunk_chart(self, elem: dict) -> Optional[Chunk]:
        description = elem.get("description", "").strip()
        caption = elem.get("caption", "").strip()

        # Combine description + caption if both present
        if description and caption:
            content = f"{description}\nAdditional context: {caption}"
        elif description:
            content = description
        elif caption:
            # Fallback: only a caption, no VLM description
            content = f"[CHART/FIGURE]\nCaption: {caption}"
        else:
            return None  # No usable content

        meta = self._base_meta(elem)
        header = (
            f"[CHART | {meta['company']} | {meta['document']} | "
            f"Page {meta['page']}]"
        )
        full_content = f"{header}\n{content}"

        return Chunk(chunk_type="chart", content=full_content, **meta)
