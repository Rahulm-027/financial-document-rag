"""
reranker.py
===========
Cross-encoder reranking pass over the hybrid retrieval results.

Why cross-encoders?
  Bi-encoder models (like BGE-base) encode query and document independently.
  This is fast but less accurate — the model never "sees" them together.

  A cross-encoder takes (query, document) as a PAIR and produces a single
  relevance score.  This is slower (can't precompute doc embeddings) but
  dramatically more accurate for identifying the most relevant passage.

  Typical pipeline:
    retrieve 20 candidates (fast) → rerank top 5 (slow but only 20 pairs)

Model
-----
  BAAI/bge-reranker-base — strong open-source cross-encoder trained for
  retrieval tasks, works well on English financial text.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks a list of (chunk_dict, score) pairs using a cross-encoder.

    Args
    ----
    model_name : HuggingFace model name (default BAAI/bge-reranker-base)
    device     : "cpu" or "cuda"
    top_k      : number of chunks to return after reranking
    max_length : max token length for query + passage pair
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        top_k: int = 5,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self.max_length = max_length

        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is None:
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.eval()
            self._model.to(self.device)
            logger.info("Cross-encoder loaded.")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """Score a list of (query, passage) pairs and return floats."""
        self._load()

        pairs = [[query, p] for p in passages]

        # Tokenise
        encoded = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self._model(**encoded).logits

        # BGE reranker outputs single logit — sigmoid for [0,1] or raw for ranking
        if logits.shape[1] == 1:
            scores = torch.sigmoid(logits.squeeze(-1)).cpu().tolist()
        else:
            scores = logits[:, 1].cpu().tolist()   # positive class

        return scores

    # ------------------------------------------------------------------
    # Main rerank API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[tuple[dict, float]],
    ) -> list[tuple[dict, float]]:
        """
        Rerank a list of (chunk_dict, retrieval_score) pairs.

        Returns
        -------
        list of (chunk_dict, reranker_score) sorted by reranker_score desc,
        limited to self.top_k items.
        """
        if not candidates:
            return []

        passages = [chunk["content"] for chunk, _ in candidates]
        rerank_scores = self._score_pairs(query, passages)

        # Pair chunks with reranker scores
        reranked = [
            (chunk, float(score))
            for (chunk, _), score in zip(candidates, rerank_scores)
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            f"Reranker: {len(reranked)} candidates → top {self.top_k} selected"
        )
        return reranked[: self.top_k]
