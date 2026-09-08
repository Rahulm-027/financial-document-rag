"""
evaluator.py
============
Runs the complete evaluation pipeline, including the ablation study.

Research question
-----------------
  How much does each retrieval enhancement contribute to financial document QA?

Experimental configurations
----------------------------
  A  — Text-only Dense RAG           (text chunks only, FAISS)
  B  — Multimodal Dense RAG          (text + table + chart chunks, FAISS)
  C  — Multimodal Hybrid + Reranker  (full system)

  NOTE on A vs B: these two configurations MUST use separate FAISS indexes:
    - text_faiss_index:       data/indexes/faiss_text.index
    - multimodal_faiss_index: data/indexes/faiss_multimodal.index
  The pipeline builds both during ingestion.
  Comparing them fairly requires that only the index differs — all other
  hyperparameters stay identical.

Ablation study (retrieval only, no generation)
-----------------------------------------------
  1. Dense only          (multimodal index)
  2. BM25 only
  3. Dense + BM25        (unweighted RRF)
  4. Dense + BM25        (weighted RRF, query-adaptive)
  5. Weighted + Reranker (full retrieval stack)

Gold relevance proxy
--------------------
  Ground truth is defined at page-level: all chunks whose (company, page)
  matches the question's evidence_pages are considered relevant.

  Limitation: page-level labels introduce false positives (a page may contain
  irrelevant text alongside the evidence table).  This is a standard trade-off
  in financial QA benchmarking given annotation cost.  The evaluation code
  explicitly records this limitation.

Benchmark
---------
  65 queries total: 60 in-scope + 5 out-of-scope abstention cases.

Results are saved to data/results/eval_results.json and a summary
table is printed to stdout.

Usage
-----
  python -m src.evaluation.evaluator --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import openai
import yaml

from src.embeddings.dense_embedder import DenseEmbedder
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.grounded_generator import GroundedGenerator
from src.evaluation.metrics import (
    compute_retrieval_metrics,
    LLMJudge,
    citation_accuracy,
    aggregate_metrics,
)
from src.evaluation.test_set import TEST_SET, get_out_of_scope

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _get_retrieved_ids(results: list[tuple[dict, float]]) -> list[str]:
    return [chunk.get("chunk_id", "") for chunk, _ in results]


def _gold_ids_from_pages(
    chunks: list[dict],
    company: str,
    pages: list[int],
) -> set[str]:
    """
    Find chunk IDs whose (company, page) matches gold evidence pages.

    This is our ground-truth relevance proxy.  Note that page-level matching
    may include irrelevant chunks from the same page — this is accepted as a
    known limitation of not having chunk-level gold labels.
    """
    if not pages:
        return set()
    return {
        c.get("chunk_id", "")
        for c in chunks
        if c.get("company", "").lower() == company.lower()
        and c.get("page", -1) in pages
    }


def _build_openai_client(config: dict) -> Optional[openai.OpenAI]:
    """
    Build the API client.  Prefers Gemini (GEMINI_API_KEY) over OpenAI.
    Returns None if neither key is set — evaluation will skip generation.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if gemini_key:
        base_url = config.get("openai", {}).get(
            "base_url",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        logger.info(f"Using Gemini API at {base_url}")
        return openai.OpenAI(api_key=gemini_key, base_url=base_url)
    elif openai_key:
        logger.info("Using OpenAI API (GEMINI_API_KEY not set)")
        return openai.OpenAI(api_key=openai_key)
    else:
        logger.warning(
            "Neither GEMINI_API_KEY nor OPENAI_API_KEY is set. "
            "Generation and LLM-judge evaluation will be skipped."
        )
        return None


# ─────────────────────────────────────────────
# Evaluator class
# ─────────────────────────────────────────────

class Evaluator:
    """
    Evaluator for the Financial RAG system.

    Loads two separate FAISS indexes:
      - text_dense  : text-only chunks (for baseline A)
      - multimodal  : text + table + chart chunks (for baselines B, C)

    This ensures the A vs B comparison is scientifically valid — only the
    chunk modalities differ, not the retriever or generation setup.

    Cost tracking
    -------------
    The evaluator records approximate API token usage across all generation
    and judge calls so you can report cost/query alongside quality metrics.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        embed_cfg = config["embeddings"]
        bm25_cfg = config["bm25"]
        rerank_cfg = config["reranker"]
        ret_cfg = config["retrieval"]

        # ── Text-only dense embedder (for baseline A) ─────────────────
        text_faiss_cfg = config.get("faiss_text", config["faiss"])
        logger.info("Loading text-only dense embedder…")
        self.dense_text = DenseEmbedder(
            model_name=embed_cfg["model"],
            device=embed_cfg["device"],
            index_path=text_faiss_cfg.get(
                "text_index_path", "data/indexes/faiss_text.index"
            ),
            metadata_path=text_faiss_cfg.get(
                "text_metadata_path", "data/indexes/faiss_text_metadata.jsonl"
            ),
        )
        self.dense_text.load_index()

        # ── Multimodal dense embedder (for baselines B and C) ─────────
        mm_faiss_cfg = config.get("faiss_multimodal", config["faiss"])
        logger.info("Loading multimodal dense embedder…")
        self.dense_mm = DenseEmbedder(
            model_name=embed_cfg["model"],
            device=embed_cfg["device"],
            index_path=mm_faiss_cfg.get(
                "mm_index_path", "data/indexes/faiss_multimodal.index"
            ),
            metadata_path=mm_faiss_cfg.get(
                "mm_metadata_path", "data/indexes/faiss_multimodal_metadata.jsonl"
            ),
        )
        self.dense_mm.load_index()

        logger.info("Loading BM25 index (multimodal)…")
        self.bm25 = BM25Index(index_path=bm25_cfg["index_path"])
        self.bm25.load()

        logger.info("Loading cross-encoder reranker…")
        self.reranker = CrossEncoderReranker(
            model_name=rerank_cfg["model"],
            device=rerank_cfg["device"],
            top_k=rerank_cfg["top_k"],
        )

        # Collect all company names from corpus for cross-company detection
        all_companies = list({
            c["company"] for c in config.get("documents", [])
        })

        # ── Text-only retriever ───────────────────────────────────────
        self.hybrid_text = HybridRetriever(
            dense_embedder=self.dense_text,
            bm25_index=self.bm25,
            dense_top_k=ret_cfg["dense_top_k"],
            bm25_top_k=ret_cfg["bm25_top_k"],
            rrf_k=ret_cfg["rrf_k"],
            use_classifier=True,
            all_companies=all_companies,
        )

        # ── Multimodal retriever ──────────────────────────────────────
        self.hybrid_mm = HybridRetriever(
            dense_embedder=self.dense_mm,
            bm25_index=self.bm25,
            dense_top_k=ret_cfg["dense_top_k"],
            bm25_top_k=ret_cfg["bm25_top_k"],
            rrf_k=ret_cfg["rrf_k"],
            use_classifier=True,
            all_companies=all_companies,
        )

        # ── LLM + Judge ───────────────────────────────────────────────
        openai_client = _build_openai_client(config)
        gen_cfg = config["generation"]
        self.generator: Optional[GroundedGenerator] = None
        self.judge: Optional[LLMJudge] = None

        if openai_client is not None:
            self.generator = GroundedGenerator(
                openai_client=openai_client,
                model=config["openai"]["model"],
                max_tokens=config["openai"]["max_tokens"],
                evidence_score_filter=gen_cfg.get("evidence_score_filter", gen_cfg.get("confidence_threshold", 0.0)),
            )
            self.judge = LLMJudge(openai_client, model=config["openai"]["model"])
        else:
            logger.warning("Skipping generation/judge initialisation (no API key).")

        # Chunk registries for gold-id lookup
        self.text_chunks: list[dict] = self.dense_text._metadata
        self.mm_chunks: list[dict] = self.dense_mm._metadata

        # Cost / token tracking
        self._total_tokens = 0
        self._total_api_calls = 0

    # ------------------------------------------------------------------

    def run_full_evaluation(self, output_path: str = "data/results/eval_results.json") -> dict:
        """
        Run all configurations and save results.

        Experimental design
        -------------------
        A — Text-only Dense RAG      : only text chunks indexed
        B — Multimodal Dense RAG     : text + table + chart chunks
        C — Multimodal Hybrid+Rerank : full system (B + BM25 + WRRF + CE)

        Ablation (retrieval only, multimodal index)
        -------------------------------------------
        1. dense_only
        2. bm25_only
        3. dense_bm25_unweighted     (equal weights)
        4. dense_bm25_weighted       (query-adaptive WRRF)
        5. weighted_plus_reranker    (full retrieval stack)
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        in_scope = [q for q in TEST_SET if q["in_scope"]]
        out_of_scope = get_out_of_scope()

        results = {
            "benchmark": {
                "total_queries": len(TEST_SET),
                "in_scope": len(in_scope),
                "out_of_scope": len(out_of_scope),
            },
            "gold_relevance_note": (
                "Relevance is defined at page-level: all chunks from the "
                "specified evidence_pages are treated as relevant.  "
                "This may introduce false positives where a page contains "
                "irrelevant text alongside the target evidence.  "
                "Recall/Precision metrics are therefore conservative upper bounds."
            ),
            "configurations": {},
            "ablation": {},
            "abstention_results": {},
            "cost": {},
        }

        # ── Main system configurations ────────────────────────────────
        configs_to_run = {
            "A_text_dense": {
                "retriever": "text_dense_only",
                "reranker": False,
                "description": "Text-only Dense RAG",
                "chunk_registry": self.text_chunks,
            },
            "B_multimodal_dense": {
                "retriever": "mm_dense_only",
                "reranker": False,
                "description": "Multimodal Dense RAG",
                "chunk_registry": self.mm_chunks,
            },
            "C_multimodal_hybrid_rerank": {
                "retriever": "mm_hybrid",
                "reranker": True,
                "description": "Multimodal Hybrid + Reranker",
                "chunk_registry": self.mm_chunks,
            },
        }

        for cfg_name, cfg in configs_to_run.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {cfg['description']}")
            logger.info(f"{'='*60}")
            results["configurations"][cfg_name] = self._run_config(in_scope, cfg, cfg_name)

        # ── Ablation study ────────────────────────────────────────────
        ablation_configs = {
            "dense_only": {
                "retriever": "mm_dense_only", "reranker": False, "weighted": False},
            "bm25_only": {
                "retriever": "bm25_only", "reranker": False, "weighted": False},
            "dense_bm25_unweighted": {
                "retriever": "mm_hybrid_unweighted", "reranker": False, "weighted": False},
            "dense_bm25_weighted": {
                "retriever": "mm_hybrid", "reranker": False, "weighted": True},
            "weighted_plus_reranker": {
                "retriever": "mm_hybrid", "reranker": True, "weighted": True},
        }

        for abl_name, abl_cfg in ablation_configs.items():
            logger.info(f"Ablation: {abl_name}")
            results["ablation"][abl_name] = self._run_retrieval_only(in_scope, abl_cfg)

        # ── Abstention / out-of-scope test ────────────────────────────
        logger.info("Running abstention test…")
        results["abstention_results"] = self._run_abstention_test(out_of_scope)

        # ── Cost summary ──────────────────────────────────────────────
        results["cost"] = {
            "total_api_calls": self._total_api_calls,
            "total_tokens_approx": self._total_tokens,
            "note": "Token counts are approximate; cost depends on the model used.",
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved → {output_path}")

        self._print_summary_table(results)
        return results

    # ------------------------------------------------------------------

    def _run_config(self, questions: list[dict], cfg: dict, cfg_name: str) -> dict:
        """Retrieval + generation + full metric suite for one configuration."""
        if self.generator is None or self.judge is None:
            logger.warning(f"Skipping generation for {cfg_name} (no API client).")
            return self._run_retrieval_only(questions, cfg)

        per_query_metrics = []
        chunk_registry = cfg.get("chunk_registry", self.mm_chunks)

        for q in questions:
            t0 = time.time()

            candidates = self._retrieve(q["question"], cfg)
            retrieved_ids = _get_retrieved_ids(candidates)
            gold_ids = _gold_ids_from_pages(
                chunk_registry, q.get("company", ""), q.get("evidence_pages", [])
            )
            ret_metrics = compute_retrieval_metrics(retrieved_ids, gold_ids)

            final_chunks = (
                self.reranker.rerank(q["question"], candidates)
                if cfg.get("reranker")
                else candidates[:5]
            )

            gen_result = self.generator.generate(q["question"], final_chunks)
            self._total_api_calls += 1

            evidence_texts = [c["content"] for c, _ in final_chunks]
            faith = self.judge.faithfulness(gen_result.answer, evidence_texts)
            correctness = self.judge.answer_correctness(
                q["question"], gen_result.answer, q["gold_answer"]
            )
            cit_acc = citation_accuracy(gen_result.citations, q.get("evidence_pages", []))
            self._total_api_calls += 2

            elapsed = time.time() - t0

            per_query_metrics.append({
                **ret_metrics,
                "faithfulness": faith["score"],
                "answer_correctness": correctness["score"],
                "citation_accuracy": cit_acc,
                "abstained": int(gen_result.abstained),
                "latency_s": round(elapsed, 2),
                "numerical_verified": int(
                    gen_result.verification.verified if gen_result.verification else True
                ),
            })

        return {
            "description": cfg.get("description", cfg_name),
            "n_questions": len(questions),
            "aggregated": aggregate_metrics(per_query_metrics),
        }

    # ------------------------------------------------------------------

    def _run_retrieval_only(self, questions: list[dict], cfg: dict) -> dict:
        """Retrieval-only (no generation) for ablation study."""
        per_query = []
        for q in questions:
            candidates = self._retrieve(q["question"], cfg)
            if cfg.get("reranker"):
                candidates = self.reranker.rerank(q["question"], candidates)
            retrieved_ids = _get_retrieved_ids(candidates)
            gold_ids = _gold_ids_from_pages(
                self.mm_chunks, q.get("company", ""), q.get("evidence_pages", [])
            )
            per_query.append(compute_retrieval_metrics(retrieved_ids, gold_ids))
        return aggregate_metrics(per_query)

    # ------------------------------------------------------------------

    def _retrieve(self, query: str, cfg: dict) -> list[tuple[dict, float]]:
        k = self.config["retrieval"]["final_top_k"]
        mode = cfg.get("retriever", "mm_hybrid")
        if mode == "text_dense_only":
            return self.hybrid_text.retrieve_dense_only(query, top_k=k)
        elif mode == "mm_dense_only":
            return self.hybrid_mm.retrieve_dense_only(query, top_k=k)
        elif mode == "bm25_only":
            return self.hybrid_mm.retrieve_bm25_only(query, top_k=k)
        elif mode == "mm_hybrid_unweighted":
            # Force equal weights by bypassing the classifier
            from src.retrieval.hybrid_retriever import weighted_reciprocal_rank_fusion
            dense_r = self.hybrid_mm.retrieve_dense_only(query, top_k=k)
            bm25_r = self.hybrid_mm.retrieve_bm25_only(query, top_k=k)
            registry: dict[str, dict] = {}
            dense_ranked, bm25_ranked = [], []
            for chunk, _ in dense_r:
                cid = chunk.get("chunk_id", str(id(chunk)))
                registry[cid] = chunk
                dense_ranked.append(cid)
            for chunk, _ in bm25_r:
                cid = chunk.get("chunk_id", str(id(chunk)))
                registry.setdefault(cid, chunk)
                bm25_ranked.append(cid)
            scores = weighted_reciprocal_rank_fusion(
                dense_ranked, bm25_ranked, 0.5, 0.5,
                k=self.config["retrieval"]["rrf_k"],
            )
            sorted_ids = sorted(scores, key=lambda c: scores[c], reverse=True)
            return [(registry[cid], scores[cid]) for cid in sorted_ids[:k]]
        else:  # mm_hybrid (weighted)
            return self.hybrid_mm.retrieve(query, top_k=k)

    # ------------------------------------------------------------------

    def _run_abstention_test(self, out_of_scope: list[dict]) -> dict:
        """
        Abstention evaluation on out-of-scope queries.

        A question is considered correctly abstained if the generator's
        abstained flag is True.  We do NOT use raw reranker scores as a
        calibrated confidence signal — they are not calibrated probabilities.
        """
        if self.generator is None:
            logger.warning("Skipping abstention test (no API client).")
            return {"n_questions": len(out_of_scope), "skipped": True}

        correct_abstentions = 0
        details = []
        for q in out_of_scope:
            candidates = self.hybrid_mm.retrieve(q["question"], top_k=5)
            final = self.reranker.rerank(q["question"], candidates)
            result = self.generator.generate(q["question"], final)
            self._total_api_calls += 1

            abstained = result.abstained
            if abstained:
                correct_abstentions += 1

            details.append({
                "question": q["question"],
                "abstained": abstained,
                "answer_preview": (result.answer or "")[:120],
            })

        rate = correct_abstentions / len(out_of_scope) if out_of_scope else 0.0
        logger.info(
            f"Abstention accuracy: {correct_abstentions}/{len(out_of_scope)} = {rate:.2%}"
        )
        return {
            "n_questions": len(out_of_scope),
            "correct_abstentions": correct_abstentions,
            "abstention_accuracy": round(rate, 4),
            "details": details,
        }

    # ------------------------------------------------------------------

    def _print_summary_table(self, results: dict) -> None:
        b = results.get("benchmark", {})
        print("\n" + "=" * 90)
        print("EVALUATION SUMMARY")
        print(f"Benchmark: {b.get('total_queries','?')} queries "
              f"({b.get('in_scope','?')} in-scope, {b.get('out_of_scope','?')} OOS)")
        print("=" * 90)

        print(
            f"\n{'Configuration':<34} {'R@5':>5} {'NDCG@5':>7} "
            f"{'Faithful':>9} {'Correct':>8} {'Cite':>6} {'Lat(s)':>7}"
        )
        print("-" * 80)
        for name, cfg_data in results["configurations"].items():
            agg = cfg_data.get("aggregated", {})
            print(
                f"{cfg_data.get('description', name)[:34]:<34} "
                f"{agg.get('recall@5', 0):>5.3f} "
                f"{agg.get('ndcg@5', 0):>7.3f} "
                f"{agg.get('faithfulness', 0):>9.3f} "
                f"{agg.get('answer_correctness', 0):>8.3f} "
                f"{agg.get('citation_accuracy', 0):>6.3f} "
                f"{agg.get('latency_s', 0):>7.2f}"
            )

        print("\nABLATION — Retrieval Only (multimodal index)")
        print(f"\n{'Configuration':<32} {'R@1':>5} {'R@3':>5} {'R@5':>5} {'NDCG@5':>7} {'MRR':>6}")
        print("-" * 60)
        for name, abl in results["ablation"].items():
            print(
                f"{name:<32} "
                f"{abl.get('recall@1', 0):>5.3f} "
                f"{abl.get('recall@3', 0):>5.3f} "
                f"{abl.get('recall@5', 0):>5.3f} "
                f"{abl.get('ndcg@5', 0):>7.3f} "
                f"{abl.get('mrr', 0):>6.3f}"
            )

        ab = results.get("abstention_results", {})
        if not ab.get("skipped"):
            print(
                f"\nAbstention: {ab.get('correct_abstentions','?')}/{ab.get('n_questions','?')} "
                f"= {ab.get('abstention_accuracy', 0):.1%}"
            )
        print("=" * 90)


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Financial RAG evaluation")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--output", type=str, default="data/results/eval_results.json"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ev = Evaluator(config)
    ev.run_full_evaluation(output_path=args.output)


if __name__ == "__main__":
    main()
