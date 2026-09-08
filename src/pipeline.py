"""
pipeline.py
===========
End-to-end pipeline orchestrator.

This script runs the full ingestion → embedding → indexing sequence.
Run it once per document set (or when new documents are added).

Usage
-----
  python -m src.pipeline --config configs/config.yaml

Stages
------
  1. PDF ingestion      (pdf_parser.py)
  2. Chart description  (chart_describer.py, calls VLM via Gemini or OpenAI)
  3. Chunking           (chunker.py, modality-aware)
  4a. Text-only FAISS   (dense_embedder.py) — for baseline A in evaluation
  4b. Multimodal FAISS  (dense_embedder.py) — for baselines B and C
  5. BM25 indexing      (bm25_index.py, multimodal chunks)

Two separate FAISS indexes are built so that the evaluation can fairly
compare text-only vs multimodal retrieval without any other variable
changing.  Both indexes use the same BGE model and hyperparameters.

Output
------
  data/processed/chunks.jsonl              — all chunks (multimodal)
  data/processed/chunks_text.jsonl         — text-only chunks
  data/indexes/faiss_multimodal.index      — multimodal FAISS
  data/indexes/faiss_multimodal_metadata.jsonl
  data/indexes/faiss_text.index            — text-only FAISS
  data/indexes/faiss_text_metadata.jsonl
  data/indexes/bm25.pkl                    — BM25 (multimodal)

Cost notes
----------
  Stage 2 (chart VLM description) is the only API-cost step at ingestion
  time.  Stages 1/3/4/5 run locally with no API calls.
  Typical cost: ~$0.005–0.02 per chart image (Gemini Flash / gpt-4o-mini vision).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import yaml

from src.ingestion.pdf_parser import parse_all_documents, RawElement
from src.ingestion.chart_describer import ChartDescriber
from src.ingestion.chunker import Chunker
from src.embeddings.dense_embedder import DenseEmbedder
from src.retrieval.bm25_index import BM25Index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Minimum yield expected from a successfully parsed PDF.
# If a document produces fewer elements than this it's likely malformed.
_MIN_ELEMENTS_PER_DOC = 10


def _get_openai_client(config: dict):
    """
    Build the OpenAI-compatible client.

    The system is configured to use Gemini through Google's OpenAI-compatible
    endpoint.  Set GEMINI_API_KEY in the environment and the config's
    openai.base_url field.  If only OPENAI_API_KEY is set (and no Gemini key),
    we fall back to the standard OpenAI endpoint so the pipeline still works
    for users with OpenAI access.

    Never prints or logs the API key value.
    """
    import openai

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
        return None


def _elem_to_dict(e) -> dict:
    """Convert RawElement (dataclass) or plain dict to dict."""
    if hasattr(e, "__dict__"):
        return e.__dict__.copy()
    return dict(e)


def run_pipeline(config: dict, skip_charts: bool = False) -> None:
    """
    Full ingestion + indexing pipeline.

    Args
    ----
    config      : loaded config.yaml dict
    skip_charts : if True, skip VLM chart description (faster, no API cost,
                  but chart chunks will have no description and are dropped)
    """

    # ── Stage 1: PDF Ingestion ────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: PDF Ingestion")
    logger.info("=" * 60)

    processed_dir = Path(config["ingestion"]["processed_dir"])
    raw_elements = parse_all_documents(config, output_dir=processed_dir)

    # Validate per-document yields — surface malformed PDFs early
    # rather than silently producing an empty or near-empty index.
    _validate_document_yields(raw_elements, config)

    # Split by element type (normalise to dicts for downstream stages)
    text_elems  = [_elem_to_dict(e) for e in raw_elements
                   if _get_etype(e) == "text"]
    table_elems = [_elem_to_dict(e) for e in raw_elements
                   if _get_etype(e) == "table"]
    chart_elems = [_elem_to_dict(e) for e in raw_elements
                   if _get_etype(e) == "chart"]

    logger.info(
        f"Extracted: {len(text_elems)} text | "
        f"{len(table_elems)} tables | "
        f"{len(chart_elems)} charts"
    )

    # ── Stage 2: Chart Description ────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Chart Description (Vision LLM)")
    logger.info("=" * 60)

    if skip_charts:
        logger.info("--skip_charts flag set: skipping chart description.")
        logger.info(
            "Chart chunks will be excluded from the index. "
            "Re-run without --skip_charts to include chart evidence."
        )
        chart_elems = []  # drop charts entirely — no description, no chunk
    elif not chart_elems:
        logger.info("No chart images detected — skipping chart description.")
    else:
        client = _get_openai_client(config)
        if client is None:
            logger.warning(
                "Neither GEMINI_API_KEY nor OPENAI_API_KEY is set. "
                "Chart descriptions will be skipped. "
                "Set GEMINI_API_KEY to enable multimodal chart indexing."
            )
            chart_elems = []
        else:
            vision_model = config.get("openai", {}).get("vision_model", "gemini-1.5-flash")
            describer = ChartDescriber(
                openai_client=client,
                model=vision_model,
            )
            chart_elems = describer.describe_batch(chart_elems)
            described = sum(1 for e in chart_elems if e.get("description", "").strip())
            logger.info(f"Described {described}/{len(chart_elems)} charts successfully.")

    # ── Stage 3: Chunking ─────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: Modality-Aware Chunking")
    logger.info("=" * 60)

    chunk_cfg = config["chunking"]
    chunker = Chunker(
        text_max_tokens=chunk_cfg["text"]["chunk_size"],
        text_overlap=chunk_cfg["text"]["chunk_overlap"],
        table_max_tokens=chunk_cfg["table"]["max_tokens"],
    )

    all_elements = text_elems + table_elems + chart_elems
    chunks = chunker.chunk(all_elements)

    n_text  = sum(1 for c in chunks if c.chunk_type == "text")
    n_table = sum(1 for c in chunks if c.chunk_type == "table")
    n_chart = sum(1 for c in chunks if c.chunk_type == "chart")
    logger.info(
        f"Created {len(chunks)} chunks: "
        f"{n_text} text | {n_table} table | {n_chart} chart"
    )

    if n_table == 0:
        logger.warning(
            "No table chunks were created. "
            "Numerical queries will rely entirely on text chunks. "
            "Check that pdfplumber can extract tables from the PDFs."
        )

    # ── Split chunks by modality ──────────────────────────────────────
    all_chunks_dicts = [c.to_dict() for c in chunks]
    text_only_chunks = [c for c in all_chunks_dicts if c["chunk_type"] == "text"]
    mm_chunks = all_chunks_dicts  # all modalities

    # Save all chunks for inspection / debugging
    chunks_path = Path(config["ingestion"]["chunks_path"])
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w") as f:
        for chunk in mm_chunks:
            f.write(json.dumps(chunk) + "\n")
    logger.info(f"Multimodal chunks saved → {chunks_path}")

    text_chunks_path = chunks_path.parent / "chunks_text.jsonl"
    with open(text_chunks_path, "w") as f:
        for chunk in text_only_chunks:
            f.write(json.dumps(chunk) + "\n")
    logger.info(f"Text-only chunks saved → {text_chunks_path}")

    # ── Stage 4a: Text-only FAISS (for baseline A) ───────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4a: Text-Only Dense Embeddings (FAISS)")
    logger.info("=" * 60)

    embed_cfg = config["embeddings"]
    text_faiss_cfg = config.get("faiss_text", {})
    text_index_path = text_faiss_cfg.get(
        "text_index_path", "data/indexes/faiss_text.index"
    )
    text_meta_path = text_faiss_cfg.get(
        "text_metadata_path", "data/indexes/faiss_text_metadata.jsonl"
    )

    embedder_text = DenseEmbedder(
        model_name=embed_cfg["model"],
        device=embed_cfg["device"],
        batch_size=embed_cfg["batch_size"],
        index_path=text_index_path,
        metadata_path=text_meta_path,
    )
    embedder_text.build_index(text_only_chunks)
    logger.info(
        f"Text-only FAISS built: {len(text_only_chunks)} chunks → {text_index_path}"
    )

    # ── Stage 4b: Multimodal FAISS (for baselines B and C) ───────────
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4b: Multimodal Dense Embeddings (FAISS)")
    logger.info("=" * 60)

    mm_faiss_cfg = config.get("faiss_multimodal", config["faiss"])
    mm_index_path = mm_faiss_cfg.get(
        "mm_index_path", config["faiss"]["index_path"]
    )
    mm_meta_path = mm_faiss_cfg.get(
        "mm_metadata_path", config["faiss"]["metadata_path"]
    )

    embedder_mm = DenseEmbedder(
        model_name=embed_cfg["model"],
        device=embed_cfg["device"],
        batch_size=embed_cfg["batch_size"],
        index_path=mm_index_path,
        metadata_path=mm_meta_path,
    )
    embedder_mm.build_index(mm_chunks)
    logger.info(
        f"Multimodal FAISS built: {len(mm_chunks)} chunks → {mm_index_path}"
    )

    # ── Stage 5: BM25 Index (multimodal) ─────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 5: BM25 Keyword Index (multimodal)")
    logger.info("=" * 60)

    bm25 = BM25Index(index_path=config["bm25"]["index_path"])
    bm25.build(mm_chunks)
    bm25.save()

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE ✓")
    logger.info(
        f"  {len(text_only_chunks)} text-only chunks | "
        f"{len(mm_chunks)} multimodal chunks\n"
        f"  Text FAISS:  {text_index_path}\n"
        f"  MM FAISS:    {mm_index_path}\n"
        f"  BM25:        {config['bm25']['index_path']}"
    )
    logger.info("=" * 60)


def _get_etype(e) -> str:
    """Get element_type from either a RawElement dataclass or dict."""
    if hasattr(e, "element_type"):
        return e.element_type
    return e.get("element_type", "")


def _validate_document_yields(raw_elements: list, config: dict) -> None:
    """
    Check per-document element counts and warn about suspiciously low yields.

    A real financial 10-K should produce hundreds of text elements.
    If a document produces fewer than _MIN_ELEMENTS_PER_DOC elements,
    it's likely malformed, encrypted, or a scanned PDF without a text layer.
    """
    doc_counts: dict[str, int] = {}
    for e in raw_elements:
        doc = _elem_to_dict(e).get("document", "unknown")
        doc_counts[doc] = doc_counts.get(doc, 0) + 1

    for doc_cfg in config.get("documents", []):
        doc_label = f"{doc_cfg['company']} {doc_cfg['type']} {doc_cfg['year']}"
        count = doc_counts.get(doc_label, 0)
        pdf_path = Path(config["ingestion"]["raw_dir"]) / doc_cfg["filename"]

        if not pdf_path.exists():
            # Already warned in parse_all_documents
            continue

        if count == 0:
            logger.error(
                f"MALFORMED PDF: '{doc_cfg['filename']}' produced 0 elements. "
                f"The file may be encrypted, image-only (scanned), or corrupted. "
                f"Verify with: pdfinfo {pdf_path}"
            )
        elif count < _MIN_ELEMENTS_PER_DOC:
            logger.warning(
                f"LOW YIELD: '{doc_cfg['filename']}' produced only {count} elements "
                f"(expected >{_MIN_ELEMENTS_PER_DOC} for a real 10-K). "
                f"Check if the PDF has a text layer. URL: {doc_cfg.get('url', 'N/A')}"
            )
        else:
            logger.info(f"  {doc_label}: {count} elements ✓")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Financial RAG ingestion pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--skip_charts",
        action="store_true",
        help=(
            "Skip VLM chart description. "
            "Saves API cost but chart evidence is excluded from the index."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_pipeline(config, skip_charts=args.skip_charts)


if __name__ == "__main__":
    main()
