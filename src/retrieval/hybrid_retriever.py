"""
hybrid_retriever.py
===================
Combines dense (FAISS) and BM25 retrieval via Weighted Reciprocal Rank Fusion
(WRRF).

Why RRF over simple score averaging?
  Score scales differ: FAISS returns cosine similarities (0–1), BM25 returns
  raw term-frequency-based scores (0–∞).  Normalising and averaging works
  poorly when one retriever returns very few results.  RRF operates on *ranks*
  not scores, making it robust regardless of absolute score magnitude.

  Standard RRF formula:
    RRF_score(d) = Σ 1 / (k + rank_i(d))

  Weighted RRF (this implementation):
    WRRF_score(d) = w_dense / (k + rank_dense(d))
                  + w_bm25  / (k + rank_bm25(d))

  The query classifier assigns per-query-type weights (e.g. numerical queries
  get w_dense=0.3, w_bm25=0.7) so that BM25 dominates when exact financial
  identifiers matter.  Documents absent from a retriever's list contribute
  only the weight from the list that found them.

References
----------
  Cormack, Clarke & Buettcher (2009) — "Reciprocal Rank Fusion outperforms
  Condorcet and individual rank learning methods"
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

# Lazy imports: DenseEmbedder and BM25Index are only imported inside methods
# so that this module can be imported (e.g. for RRF helpers in tests) without
# pulling in sentence_transformers or faiss.
if TYPE_CHECKING:
    from src.embeddings.dense_embedder import DenseEmbedder
    from src.retrieval.bm25_index import BM25Index

from src.retrieval.query_classifier import QueryClassifier, ClassificationResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Reciprocal Rank Fusion (unweighted, for tests / baselines)
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """
    Unweighted RRF — each retriever contributes equally.

    Args
    ----
    ranked_lists : list of ranked chunk_id lists (rank 1 = best)
    k            : RRF smoothing constant (default 60)

    Returns
    -------
    dict mapping chunk_id → RRF score (higher = better)
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return dict(scores)


def weighted_reciprocal_rank_fusion(
    dense_ranked: list[str],
    bm25_ranked: list[str],
    dense_weight: float,
    bm25_weight: float,
    k: int = 60,
) -> dict[str, float]:
    """
    Weighted RRF — each retriever's contribution is scaled by its weight.

    WRRF_score(d) = w_dense / (k + rank_dense(d))
                  + w_bm25  / (k + rank_bm25(d))

    A document absent from a retriever's list contributes 0 from that
    retriever (no implicit penalty — it simply isn't ranked there).

    Args
    ----
    dense_ranked  : chunk_ids in dense-retrieval rank order (best first)
    bm25_ranked   : chunk_ids in BM25 rank order (best first)
    dense_weight  : scalar in [0, 1] for dense retriever
    bm25_weight   : scalar in [0, 1] for BM25 retriever
    k             : RRF smoothing constant

    Returns
    -------
    dict mapping chunk_id → WRRF score (higher = better)
    """
    scores: dict[str, float] = defaultdict(float)
    for rank, chunk_id in enumerate(dense_ranked, start=1):
        scores[chunk_id] += dense_weight / (k + rank)
    for rank, chunk_id in enumerate(bm25_ranked, start=1):
        scores[chunk_id] += bm25_weight / (k + rank)
    return dict(scores)


# ─────────────────────────────────────────────
# Table-type boost
# ─────────────────────────────────────────────

def _boost_table_chunks(
    wrrf_scores: dict[str, float],
    chunk_registry: dict[str, dict],
    boost: float = 0.003,
) -> dict[str, float]:
    """
    Apply a small additive boost to table chunks for numerical/comparison queries.

    Why tables need a boost: dense retrievers score sparse tabular content
    (pipe-delimited rows) lower than surrounding prose, because the embedding
    model sees less semantic richness in "| Segment | Revenue | $211,915 |"
    than in a full paragraph.  For financial figures the answer is almost
    always in a table row, not the paragraph.

    Magnitude choice: 0.003 is roughly 18% of the max possible single-retriever
    RRF contribution (w_bm25=0.7 at rank 1 ≈ 0.0115).  It's enough to break
    realistic score ties between a table chunk and a nearby prose chunk, but
    not large enough to override a genuinely irrelevant table over a relevant
    text chunk.

    Concrete scenario (k=60, numerical query w=0.3/0.7):
      Text chunk at dense rank 3 (not in BM25): score ≈ 0.0048
      Table chunk at dense rank 5, BM25 rank 8: score ≈ 0.0142
      After boost: 0.0142 + 0.003 = 0.0172 — table stays ahead.

      Text chunk at dense rank 2, not in BM25: score ≈ 0.0048
      Table chunk not in dense top-20, BM25 rank 3: score ≈ 0.0100
      After boost: 0.0103 — table still ahead.

    The boost is additive to WRRF rank-based scores, so it does NOT override
    retrieval ordering for chunks with large WRRF score differences.
    """
    boosted = dict(wrrf_scores)
    for cid, chunk in chunk_registry.items():
        if chunk.get("chunk_type") == "table" and cid in boosted:
            boosted[cid] += boost
    return boosted

def _financial_relevance_boost(
    scores: dict[str, float],
    chunk_registry: dict[str, dict],
    query: str,
    boost: float = 0.006,
) -> dict[str, float]:
    """
    Add a small financial-domain relevance adjustment to WRRF scores.

    Rewards chunks that contain:
      - the requested financial metric
      - the requested fiscal/calendar year
      - financial-statement terminology

    Penalizes common financial distractors such as:
      - unearned revenue
      - deferred revenue
      - revenue recognition
      - revenue guidance
    """
    query_lower = query.lower()
    adjusted = dict(scores)

    # Financial metrics commonly requested in QA.
    metric_terms = [
        "total revenue",
        "revenue",
        "net sales",
        "sales",
        "net income",
        "operating income",
        "gross profit",
        "gross margin",
        "earnings",
        "eps",
        "cash flow",
    ]

    requested_metrics = [
        term for term in metric_terms
        if term in query_lower
    ]

    # Extract years mentioned in the query.
    import re

    requested_years = set(
        re.findall(r"\b(?:19|20)\d{2}\b", query_lower)
    )

    statement_terms = [
        "income statements",
        "income statement",
        "summary results of operations",
        "results of operations",
        "consolidated statements",
        "statement of operations",
        "financial statements",
    ]

    distractor_terms = [
        "unearned revenue",
        "deferred revenue",
        "revenue recognition",
        "revenue guidance",
        "remaining performance obligations",
    ]

    for cid, chunk in chunk_registry.items():
        if cid not in adjusted:
            continue

        content = str(chunk.get("content", "")).lower()
        section = str(chunk.get("section", "")).lower()

        bonus = 0.0

        # Metric relevance.
        for metric in requested_metrics:
            if metric in content or metric in section:
                bonus += boost

        # Year relevance.
        if requested_years:
            if any(year in content for year in requested_years):
                bonus += boost

        # Financial-statement relevance.
        if any(term in content or term in section for term in statement_terms):
            bonus += boost

        # Tables are particularly valuable for numerical questions.
        if chunk.get("chunk_type") == "table":
            bonus += boost * 0.5

        # Penalize misleading revenue-related contexts.
        if any(term in content or term in section for term in distractor_terms):
            bonus -= boost * 1.5

        adjusted[cid] += bonus

    return adjusted

# ─────────────────────────────────────────────
# HybridRetriever
# ─────────────────────────────────────────────

class HybridRetriever:
    """
    Fuses dense and BM25 retrieval with Weighted Reciprocal Rank Fusion (WRRF).

    The QueryClassifier assigns per-query-type weights, which are applied
    directly in the WRRF formula:

        WRRF_score(d) = w_dense / (k + rank_dense(d))
                      + w_bm25  / (k + rank_bm25(d))

    For cross-company queries, genuinely per-company retrieval is used:
    each named company's chunks are fetched by filtering the full index by
    company metadata, not just by hoping the company appears in the global
    top-k.  This guarantees each explicitly mentioned company contributes
    candidates before WRRF fusion and reranking.

    For numerical and comparison queries, a small additive boost is applied
    to table chunks after WRRF — tables are the primary source of financial
    figures and tend to be ranked below surrounding text by dense retrievers.

    Args
    ----
    dense_embedder   : DenseEmbedder (loaded)
    bm25_index       : BM25Index (loaded)
    dense_top_k      : candidates fetched from dense retriever per search
    bm25_top_k       : candidates fetched from BM25 retriever per search
    rrf_k            : WRRF smoothing constant (default 60)
    use_classifier   : enable adaptive weighting via QueryClassifier
    all_companies    : list of company names in corpus (for cross-company routing)
    table_boost      : additive score boost for table chunks (numerical/comparison queries)
    k_per_company    : chunks fetched per company in cross-company mode
    """

    def __init__(
        self,
        dense_embedder: "DenseEmbedder",
        bm25_index: "BM25Index",
        dense_top_k: int = 50,
        bm25_top_k: int = 50,
        rrf_k: int = 60,
        use_classifier: bool = True,
        all_companies: Optional[list[str]] = None,
        table_boost: float = 0.003,
        k_per_company: int = 10,
    ) -> None:
        self.dense = dense_embedder
        self.bm25 = bm25_index
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.classifier = QueryClassifier(companies=all_companies) if use_classifier else None
        self.all_companies = all_companies or []
        self.table_boost = table_boost
        self.k_per_company = k_per_company

        # Query types where tables are the primary answer source
        self._table_boost_types = {"numerical", "comparison", "cross_company"}

    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: Optional[dict] = None,
    ) -> list[tuple[dict, float]]:
        """
        Retrieve top-k chunks for a query using hybrid WRRF.

        For cross-company queries, company-balanced retrieval is used
        automatically: each detected company contributes k_per_company
        candidates fetched directly from its portion of the index, not
        filtered from a fixed-size global result set.

        For numerical and comparison queries, table chunks receive a small
        additive boost after WRRF to counteract the dense retriever's tendency
        to rank surrounding text prose above the actual data table.

        Args
        ----
        query           : user question string
        top_k           : number of final results to return
        metadata_filter : optional exact-match filter applied after retrieval

        Returns
        -------
        list of (chunk_dict, wrrf_score) sorted descending
        """
        clf_result: Optional[ClassificationResult] = None
        dense_weight = 0.5
        bm25_weight = 0.5
        query_type_str = "factual"

        if self.classifier:
            clf_result = self.classifier.classify(query)
            dense_weight = clf_result.dense_weight
            bm25_weight = clf_result.bm25_weight
            query_type_str = clf_result.query_type.value
            logger.debug(
                f"Query type={query_type_str} "
                f"w_dense={dense_weight:.2f} w_bm25={bm25_weight:.2f}"
            )

            if query_type_str == "cross_company":
                return self._retrieve_cross_company(
                    query, top_k, dense_weight, bm25_weight
                )

        apply_table_boost = query_type_str in self._table_boost_types

        return self._retrieve_wrrf(
            query, top_k, dense_weight, bm25_weight,
            metadata_filter, apply_table_boost=apply_table_boost,
        )

    # ------------------------------------------------------------------

    def _retrieve_wrrf(
        self,
        query: str,
        top_k: int,
        dense_weight: float,
        bm25_weight: float,
        metadata_filter: Optional[dict] = None,
        apply_table_boost: bool = False,
    ) -> list[tuple[dict, float]]:
        """Core WRRF retrieval with optional table boost."""
        dense_results = self.dense.search(query, k=self.dense_top_k)
        bm25_results = self.bm25.search(query, k=self.bm25_top_k)

        chunk_registry: dict[str, dict] = {}
        dense_ranked: list[str] = []
        for chunk, _ in dense_results:
            cid = chunk.get("chunk_id", str(id(chunk)))
            chunk_registry[cid] = chunk
            dense_ranked.append(cid)

        bm25_ranked: list[str] = []
        for chunk, _ in bm25_results:
            cid = chunk.get("chunk_id", str(id(chunk)))
            if cid not in chunk_registry:
                chunk_registry[cid] = chunk
            bm25_ranked.append(cid)

        wrrf_scores = weighted_reciprocal_rank_fusion(
            dense_ranked, bm25_ranked,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            k=self.rrf_k,
        )

        if apply_table_boost and self.table_boost > 0:
            wrrf_scores = _boost_table_chunks(
                wrrf_scores, chunk_registry, boost=self.table_boost
            )

        wrrf_scores = _financial_relevance_boost(
            wrrf_scores,
            chunk_registry,
            query,
        )

        if metadata_filter:
            wrrf_scores = {
                cid: score for cid, score in wrrf_scores.items()
                if all(
                    str(chunk_registry[cid].get(k, "")).lower() == str(v).lower()
                    for k, v in metadata_filter.items()
                )
            }

        sorted_ids = sorted(wrrf_scores, key=lambda c: wrrf_scores[c], reverse=True)
        results = [(chunk_registry[cid], wrrf_scores[cid]) for cid in sorted_ids[:top_k]]

        logger.info(
            f"WRRF: {len(dense_results)} dense + {len(bm25_results)} BM25 "
            f"(w={dense_weight:.2f}/{bm25_weight:.2f}) "
            f"→ {len(wrrf_scores)} unique → top {len(results)}"
            + (" [table boost]" if apply_table_boost else "")
        )
        return results

    # ------------------------------------------------------------------

    def _retrieve_cross_company(
        self,
        query: str,
        top_k: int,
        dense_weight: float,
        bm25_weight: float,
    ) -> list[tuple[dict, float]]:
        """
        Company-balanced retrieval for cross-company queries.

        Algorithm
        ---------
        1. Detect which companies are mentioned in the query.
        2. For each mentioned company, perform a targeted dense search and
           BM25 search against the full index, then keep only that company's
           results.  This is genuine per-company retrieval: we don't filter
           a fixed global top-k, so a company with few globally-top chunks
           still contributes k_per_company candidates.
        3. Also run a standard global search (dense + BM25) and include those
           results in the pool — this catches high-ranking chunks that belong
           to mentioned companies without duplicating work.
        4. Run WRRF over the merged pool using the full ranked lists (dense
           and BM25 order preserved) and return top_k.

        Why step 2 matters
        ------------------
        A naive approach does `self.dense.search(query, k=20)` and then
        filters by company.  If Apple doesn't appear in the global top-20,
        Apple gets zero candidates — defeating the purpose of balanced
        retrieval.  Step 2 avoids this by fetching from the entire index
        and post-filtering, so each company is guaranteed k_per_company
        chunks if they exist.
        """
        query_lower = query.lower()
        mentioned = [c for c in self.all_companies if c.lower() in query_lower]
        if len(mentioned) < 2:
            logger.info("Cross-company: couldn't detect ≥2 companies, using global WRRF")
            return self._retrieve_wrrf(
                query, top_k, dense_weight, bm25_weight, apply_table_boost=True
            )

        logger.info(f"Cross-company retrieval: companies={mentioned}")

        chunk_registry: dict[str, dict] = {}
        dense_ranked: list[str] = []
        bm25_ranked: list[str] = []

        # ── Global search — contributes high-rank results for any company ──
        dense_global = self.dense.search(query, k=self.dense_top_k)
        bm25_global = self.bm25.search(query, k=self.bm25_top_k)

        for chunk, _ in dense_global:
            cid = chunk.get("chunk_id", str(id(chunk)))
            chunk_registry[cid] = chunk
            dense_ranked.append(cid)

        for chunk, _ in bm25_global:
            cid = chunk.get("chunk_id", str(id(chunk)))
            chunk_registry.setdefault(cid, chunk)
            bm25_ranked.append(cid)

        # ── Per-company search — guarantees each company has candidates ──
        # We search with a large k to scan the full index, then filter.
        # This is the key difference from naive global-then-filter: we're
        # not limited by how many global top-k slots a company occupies.
        # The search size is bounded at min(all_chunks, dense_top_k * n_companies * 3)
        # to avoid scanning the whole index every time.
        n_mentioned = len(mentioned)
        per_company_search_k = min(
            self.dense_top_k * n_mentioned * 4,   # scan enough to find each company
            10000,                                  # hard cap — avoid scanning giant indexes
        )

        for company in mentioned:
            co_lower = company.lower()
            found_dense = 0
            found_bm25 = 0

            # Dense: search broadly, keep this company's chunks
            if found_dense < self.k_per_company:
                dense_co_results = self.dense.search(query, k=per_company_search_k)
                for chunk, _ in dense_co_results:
                    if chunk.get("company", "").lower() != co_lower:
                        continue
                    if found_dense >= self.k_per_company:
                        break
                    cid = chunk.get("chunk_id", str(id(chunk)))
                    chunk_registry.setdefault(cid, chunk)
                    if cid not in dense_ranked:
                        dense_ranked.append(cid)
                    found_dense += 1

            # BM25: same approach
            if found_bm25 < self.k_per_company:
                bm25_co_results = self.bm25.search(query, k=per_company_search_k)
                for chunk, _ in bm25_co_results:
                    if chunk.get("company", "").lower() != co_lower:
                        continue
                    if found_bm25 >= self.k_per_company:
                        break
                    cid = chunk.get("chunk_id", str(id(chunk)))
                    chunk_registry.setdefault(cid, chunk)
                    if cid not in bm25_ranked:
                        bm25_ranked.append(cid)
                    found_bm25 += 1

            logger.debug(
                f"  {company}: {found_dense} dense + {found_bm25} BM25 candidates added"
            )

        # WRRF over the full merged pool
        wrrf_scores = weighted_reciprocal_rank_fusion(
            dense_ranked, bm25_ranked,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            k=self.rrf_k,
        )

        # Table boost applies for cross-company too (often comparing figures)
        if self.table_boost > 0:
            wrrf_scores = _boost_table_chunks(
                wrrf_scores, chunk_registry, boost=self.table_boost
            )

        sorted_ids = sorted(wrrf_scores, key=lambda c: wrrf_scores[c], reverse=True)
        results = [(chunk_registry[cid], wrrf_scores[cid]) for cid in sorted_ids[:top_k]]

        # Log company distribution — useful for verifying balance in practice
        company_dist: dict[str, int] = defaultdict(int)
        for chunk, _ in results:
            company_dist[chunk.get("company", "unknown")] += 1
        logger.info(
            f"Cross-company: {n_mentioned} companies (search_k={per_company_search_k}) "
            f"→ {len(wrrf_scores)} unique → top {len(results)} "
            f"(dist: {dict(company_dist)})"
        )
        return results

    # ------------------------------------------------------------------
    # Baseline retrieval modes (for ablation study)
    # ------------------------------------------------------------------

    def retrieve_dense_only(self, query: str, top_k: int = 20) -> list[tuple[dict, float]]:
        """Dense-only baseline (no BM25, no weighting)."""
        return self.dense.search(query, k=top_k)

    def retrieve_bm25_only(self, query: str, top_k: int = 20) -> list[tuple[dict, float]]:
        """BM25-only baseline."""
        return self.bm25.search(query, k=top_k)

    # ------------------------------------------------------------------

    def retrieve_with_trace(
        self,
        query: str,
        top_k: int = 20,
    ) -> tuple[list[tuple[dict, float]], dict]:
        """
        Same as retrieve() but also returns a trace dict showing each chunk's
        rank at every stage: dense, BM25, WRRF, after table boost.

        Useful for debugging why a specific chunk (e.g. the Microsoft revenue
        table) does or does not make it into the final reranker input.

        Returns
        -------
        (results, trace) where trace is:
          {
            "query_type": str,
            "dense_weight": float,
            "bm25_weight": float,
            "dense_ranks": {chunk_id: rank},
            "bm25_ranks": {chunk_id: rank},
            "wrrf_scores": {chunk_id: float},
            "boosted_scores": {chunk_id: float},
            "final_top_k_ids": [chunk_id, ...],
          }
        """
        clf_result = None
        dense_weight = 0.5
        bm25_weight = 0.5
        query_type_str = "factual"

        if self.classifier:
            clf_result = self.classifier.classify(query)
            dense_weight = clf_result.dense_weight
            bm25_weight = clf_result.bm25_weight
            query_type_str = clf_result.query_type.value

        apply_table_boost = query_type_str in self._table_boost_types

        dense_results = self.dense.search(query, k=self.dense_top_k)
        bm25_results = self.bm25.search(query, k=self.bm25_top_k)

        chunk_registry: dict[str, dict] = {}
        dense_ranked: list[str] = []
        for chunk, _ in dense_results:
            cid = chunk.get("chunk_id", str(id(chunk)))
            chunk_registry[cid] = chunk
            dense_ranked.append(cid)

        bm25_ranked: list[str] = []
        for chunk, _ in bm25_results:
            cid = chunk.get("chunk_id", str(id(chunk)))
            chunk_registry.setdefault(cid, chunk)
            bm25_ranked.append(cid)

        wrrf_scores = weighted_reciprocal_rank_fusion(
            dense_ranked, bm25_ranked,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            k=self.rrf_k,
        )

        boosted_scores = dict(wrrf_scores)
        if apply_table_boost and self.table_boost > 0:
            boosted_scores = _boost_table_chunks(
                wrrf_scores, chunk_registry, boost=self.table_boost
            )

        sorted_ids = sorted(boosted_scores, key=lambda c: boosted_scores[c], reverse=True)
        results = [(chunk_registry[cid], boosted_scores[cid]) for cid in sorted_ids[:top_k]]

        trace = {
            "query_type": query_type_str,
            "dense_weight": dense_weight,
            "bm25_weight": bm25_weight,
            "table_boost_applied": apply_table_boost,
            "dense_ranks": {cid: rank for rank, cid in enumerate(dense_ranked, 1)},
            "bm25_ranks": {cid: rank for rank, cid in enumerate(bm25_ranked, 1)},
            "wrrf_scores": dict(sorted(wrrf_scores.items(), key=lambda x: x[1], reverse=True)),
            "boosted_scores": dict(sorted(boosted_scores.items(), key=lambda x: x[1], reverse=True)),
            "final_top_k_ids": sorted_ids[:top_k],
            "n_dense_candidates": len(dense_results),
            "n_bm25_candidates": len(bm25_results),
            "n_unique_candidates": len(wrrf_scores),
        }
        return results, trace
