"""
bm25_index.py
=============
Builds and queries a BM25 keyword index (rank_bm25) over all chunks.

Why BM25 alongside dense embeddings?
  Financial documents are full of exact identifiers — ticker symbols,
  specific dollar amounts like "$4.2B", acronyms like "EBITDA" or "FCF",
  and period labels like "Q3 FY2024".  Semantic embeddings can miss
  exact-match queries for these terms; BM25 retrieves them reliably.

Index is serialised with pickle for fast reloads.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import string
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Tokeniser
# ─────────────────────────────────────────────

_STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "be", "was", "were", "are", "has", "have",
    "had", "this", "that", "with", "from", "as", "by", "its",
}

# Unit multipliers for canonical normalisation
_CANON_UNIT: dict[str, str] = {
    "billion": "b", "bn": "b",
    "million": "m", "mn": "m",
    "thousand": "k",
}

# Regex sub-patterns
_DOLLAR_WORD  = r'\$\s*[\d,.]+\s*(?:billion|bn|million|mn|thousand)'
_DOLLAR_ABBR  = r'\$[\d,.]+[bmk]'
_PCT_WORD     = r'[\d,.]+\s*(?:percent|percentage\s+points?)'
_PCT_ABBR     = r'[\d,.]+%'
_FY           = r'fy\d{4}'
_QFY          = r'q[1-4]\s*fy\d{4}'
_QUARTER      = r'q[1-4]'
_WORD         = r'[\w]+'

_TOKEN_RE = re.compile(
    r'|'.join([_DOLLAR_WORD, _DOLLAR_ABBR, _PCT_WORD, _PCT_ABBR,
               _QFY, _FY, _QUARTER, _WORD]),
    re.IGNORECASE,
)

_NUM_CLEAN_RE = re.compile(r'[,\s]')


def _canonicalise_financial_token(token: str) -> str:
    """
    Normalise financial expressions to a canonical form so that
    "$4.2 billion", "$4.2B", "$4.2bn" all become "$4.2b".

    Examples
    --------
    "$4.2 billion" → "$4.2b"
    "$4.2B"        → "$4.2b"
    "23.5 percent" → "23.5%"
    "18.5%"        → "18.5%"
    """
    t = _NUM_CLEAN_RE.sub("", token.lower())

    # Dollar + word unit
    m = re.match(r'^\$([\d.]+)(billion|bn|million|mn|thousand)$', t)
    if m:
        return f"${m.group(1)}{_CANON_UNIT[m.group(2)]}"

    # Dollar + letter unit (already short — just normalise case)
    m = re.match(r'^\$([\d.]+)([bmk])$', t)
    if m:
        return f"${m.group(1)}{m.group(2)}"

    # Percent word forms
    m = re.match(r'^([\d.]+)percent.*$', t)
    if m:
        return f"{m.group(1)}%"

    return t


def tokenize(text: str) -> list[str]:
    """
    Financial-domain tokeniser:

    - Preserves and canonicalises financial expressions:
        "$4.2 billion" = "$4.2B" = "$4.2bn" → all become "$4.2b"
        "23.5 percent" = "23.5%" → "23.5%"
        "FY2024" / "Q3 FY2024" → preserved as single tokens
    - Lowercases everything else
    - Removes stop words and single-character tokens
    """
    tokens = []
    for match in _TOKEN_RE.finditer(text.lower()):
        raw = match.group(0).strip()
        canon = _canonicalise_financial_token(raw)
        if canon and canon not in _STOP_WORDS and len(canon) > 1:
            tokens.append(canon)
    return tokens


# ─────────────────────────────────────────────
# BM25Index
# ─────────────────────────────────────────────

class BM25Index:
    """
    Wraps rank_bm25's BM25Okapi with persistence and metadata tracking.

    Usage
    -----
    idx = BM25Index(index_path="data/indexes/bm25.pkl")
    idx.build(chunks)
    idx.save()

    # Later:
    idx.load()
    results = idx.search("revenue Q3 2023", k=20)
    """

    def __init__(self, index_path: str = "data/indexes/bm25.pkl") -> None:
        self.index_path = Path(index_path)
        self._bm25: Optional[BM25Okapi] = None
        self._metadata: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []

    # ------------------------------------------------------------------

    def build(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunk dicts (must have 'content' key)."""
        logger.info(f"Tokenising {len(chunks)} chunks for BM25…")
        self._metadata = [c.copy() for c in chunks]
        self._tokenized_corpus = [tokenize(c["content"]) for c in chunks]

        logger.info("Fitting BM25Okapi…")
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index built.")

    # ------------------------------------------------------------------

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bm25": self._bm25,
            "metadata": self._metadata,
            "corpus": self._tokenized_corpus,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"BM25 index saved → {self.index_path}")

    # ------------------------------------------------------------------

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {self.index_path}. "
                "Run build() + save() first."
            )
        with open(self.index_path, "rb") as f:
            payload = pickle.load(f)
        self._bm25 = payload["bm25"]
        self._metadata = payload["metadata"]
        self._tokenized_corpus = payload["corpus"]
        logger.info(
            f"BM25 index loaded ({len(self._metadata)} documents)."
        )

    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 20) -> list[tuple[dict, float]]:
        """
        Retrieve top-k chunks by BM25 score.

        Returns
        -------
        list of (chunk_dict, score) sorted by score descending
        """
        if self._bm25 is None:
            self.load()

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:   # only include non-zero scoring docs
                chunk = self._metadata[idx].copy()
                results.append((chunk, float(scores[idx])))

        return results
