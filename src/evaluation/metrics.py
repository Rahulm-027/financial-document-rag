"""Evaluation metrics for the financial RAG benchmark.

The evaluator uses page-level gold relevance labels (expanded to chunk IDs)
and combines deterministic retrieval/citation metrics with an LLM judge for
faithfulness and answer correctness.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


def _binary_relevance(retrieved_ids: list[str], gold_ids: set[str], k: int) -> list[int]:
    return [1 if cid in gold_ids else 0 for cid in retrieved_ids[:k]]


def _dcg(rels: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def compute_retrieval_metrics(
    retrieved_ids: list[str], gold_ids: set[str], ks: tuple[int, ...] = (1, 3, 5, 10)
) -> dict[str, float]:
    """Compute Recall@K, Precision@K, NDCG@K and MRR.

    Gold relevance is a set of chunk IDs. If there are no gold IDs, retrieval
    metrics are returned as 0 rather than producing misleading perfect scores.
    """
    out: dict[str, float] = {}
    if not gold_ids:
        for k in ks:
            out[f"recall@{k}"] = 0.0
            out[f"precision@{k}"] = 0.0
            out[f"ndcg@{k}"] = 0.0
        out["mrr"] = 0.0
        return out

    for k in ks:
        rels = _binary_relevance(retrieved_ids, gold_ids, k)
        hits = sum(rels)
        out[f"recall@{k}"] = hits / len(gold_ids)
        out[f"precision@{k}"] = hits / k
        ideal_hits = min(len(gold_ids), k)
        ideal = _dcg([1] * ideal_hits)
        out[f"ndcg@{k}"] = _dcg(rels) / ideal if ideal else 0.0

    rr = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_ids:
            rr = 1.0 / rank
            break
    out["mrr"] = rr
    return out


def citation_accuracy(citations: list[Any], evidence_pages: list[int]) -> float:
    """Fraction of cited pages that match the supplied evidence pages.

    Supports citations represented as dicts (page/page_number), integers,
    or strings containing a page number. Empty citation/evidence lists score 0.
    """
    if not citations or not evidence_pages:
        return 0.0
    gold = {int(p) for p in evidence_pages}
    correct = 0
    for c in citations:
        page = None
        if isinstance(c, dict):
            page = c.get("page", c.get("page_number"))
        elif isinstance(c, int):
            page = c
        elif isinstance(c, str):
            import re
            m = re.search(r"(?:page|p\.)\s*(\d+)", c, flags=re.I)
            if m:
                page = int(m.group(1))
            else:
                m = re.search(r"\b(\d+)\b", c)
                if m:
                    page = int(m.group(1))
        try:
            if page is not None and int(page) in gold:
                correct += 1
        except (TypeError, ValueError):
            pass
    return correct / len(citations)


def aggregate_metrics(per_query: list[dict[str, Any]]) -> dict[str, float]:
    """Arithmetic mean of numeric per-query metrics."""
    if not per_query:
        return {}
    keys = set().union(*(m.keys() for m in per_query))
    out: dict[str, float] = {}
    for key in sorted(keys):
        values = [m[key] for m in per_query if isinstance(m.get(key), (int, float))]
        if values:
            out[key] = round(sum(values) / len(values), 4)
    return out


class LLMJudge:
    """Small deterministic JSON-scoring judge backed by an OpenAI client."""

    def __init__(self, client: Any, model: str = "gpt-4o-mini") -> None:
        self.client = client
        self.model = model

    def _score(self, system: str, user: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return {"score": 0.0, "reason": "Judge returned invalid JSON."}
        try:
            data["score"] = max(0.0, min(1.0, float(data.get("score", 0.0))))
        except (TypeError, ValueError):
            data["score"] = 0.0
        return data

    def faithfulness(self, answer: str, evidence_texts: list[str]) -> dict[str, Any]:
        evidence = "\n\n--- EVIDENCE ---\n".join(evidence_texts)
        return self._score(
            "You are a strict RAG evaluation judge. Return JSON with score (0 to 1) and reason. "
            "Score how fully the answer is supported by the supplied evidence. Do not reward outside knowledge.",
            f"ANSWER:\n{answer}\n\nEVIDENCE:\n{evidence}",
        )

    def answer_correctness(self, question: str, answer: str, gold_answer: str) -> dict[str, Any]:
        return self._score(
            "You are a strict QA judge. Return JSON with score (0 to 1) and reason. "
            "Score factual agreement with the reference answer while requiring the answer to address the question.",
            f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{gold_answer}\n\nMODEL ANSWER:\n{answer}",
        )
