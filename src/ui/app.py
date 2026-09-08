"""
app.py
======
Streamlit application for the Financial Document Intelligence RAG system.

Features:
  - Natural language financial Q&A over indexed 10-Ks / earnings reports
  - Query type display (numerical / comparison / chart_trend / cross_company / factual)
  - Answer with inline citations + page references
  - Numerical verification warnings
  - Evidence panel showing retrieved chunks with scores
  - System config toggle (text-only vs multimodal, dense vs hybrid)

Run:
  streamlit run src/ui/app.py

API:
  Set GEMINI_API_KEY in your environment or enter it in the sidebar.
  The system uses Google Gemini through an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st
import yaml

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Financial Document Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
  .main-header {
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 0.25rem;
  }
  .sub-header {
    color: #555;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
  }
  .answer-box {
    background: #f8f9fa;
    border-left: 4px solid #0066cc;
    padding: 1rem 1.25rem;
    border-radius: 4px;
    font-size: 0.97rem;
    line-height: 1.6;
  }
  .evidence-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
  }
  .evidence-header {
    font-weight: 600;
    color: #333;
    margin-bottom: 0.3rem;
  }
  .evidence-meta {
    color: #666;
    font-size: 0.78rem;
    margin-bottom: 0.4rem;
  }
  .badge {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 12px;
    font-size: 0.73rem;
    font-weight: 600;
    margin-right: 0.3rem;
  }
  .badge-text    { background: #e3f2fd; color: #1565c0; }
  .badge-table   { background: #e8f5e9; color: #2e7d32; }
  .badge-chart   { background: #fff3e0; color: #e65100; }
  .badge-high    { background: #e8f5e9; color: #2e7d32; }
  .badge-medium  { background: #fff8e1; color: #f57f17; }
  .badge-low     { background: #fce4ec; color: #c62828; }
  .warning-box {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    font-size: 0.87rem;
    margin-top: 0.75rem;
  }
  .query-type-chip {
    display: inline-block;
    background: #ede7f6;
    color: #4527a0;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }
  .retrieval-debug {
    font-family: monospace;
    font-size: 0.78rem;
    color: #444;
    background: #f5f5f5;
    padding: 0.5rem 0.75rem;
    border-radius: 4px;
    margin-top: 0.5rem;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────

def init_session():
    defaults = {
        "query_history": [],
        "prefill_query": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ─────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Gemini API key — this is the intended API for this project
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help=(
            "Google Gemini API key. "
            "Get one at https://aistudio.google.com/app/apikey"
        ),
    )
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    st.divider()

    st.markdown("### Retrieval Mode")
    retrieval_mode = st.selectbox(
        "System configuration",
        options=[
            "Multimodal Hybrid + Reranker  (full system)",
            "Dense only  (baseline)",
            "BM25 only  (baseline)",
            "Dense + BM25 (no reranker)",
        ],
        index=0,
    )
    use_classifier = st.toggle("Adaptive query routing", value=True)
    show_debug = st.toggle("Show retrieval debug info", value=False)

    st.divider()

    st.markdown("### Index")
    config_path = st.text_input("Config path", value="configs/config.yaml")
    load_btn = st.button("🔄 Load Indexes", use_container_width=True)

    if load_btn:
        # Clear the cache so indexes reload with the new key/config
        _load_indexes.clear()

    st.divider()
    st.markdown("### Example Questions")
    examples = [
        "What was Microsoft's revenue in FY2023?",
        "What was Tesla's gross margin in FY2023?",
        "Compare Apple and Microsoft revenue growth.",
        "How did Nvidia's data center revenue trend?",
        "What was Amazon's AWS operating income?",
        "Which company had the highest gross margin?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["prefill_query"] = ex
            st.rerun()


# ─────────────────────────────────────────────
# Index loading (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading indexes…")
def _load_indexes(config_path: str, mode: str, use_clf: bool):
    """
    Load FAISS, BM25, reranker, and generator.  Cached so the heavy
    models only load once per session.  Call _load_indexes.clear() to
    force a reload (e.g. after changing the API key or config path).
    """
    import openai as oai
    from src.embeddings.dense_embedder import DenseEmbedder
    from src.retrieval.bm25_index import BM25Index
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.query_classifier import QueryClassifier
    from src.generation.grounded_generator import GroundedGenerator

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        return {"error": f"Config not found: {config_path}"}

    try:
        mm_faiss_cfg = cfg.get("faiss_multimodal", cfg["faiss"])
        dense = DenseEmbedder(
            model_name=cfg["embeddings"]["model"],
            device=cfg["embeddings"]["device"],
            index_path=mm_faiss_cfg.get("mm_index_path", cfg["faiss"]["index_path"]),
            metadata_path=mm_faiss_cfg.get("mm_metadata_path", cfg["faiss"]["metadata_path"]),
        )
        dense.load_index()

        bm25 = BM25Index(index_path=cfg["bm25"]["index_path"])
        bm25.load()

        reranker = CrossEncoderReranker(
            model_name=cfg["reranker"]["model"],
            device=cfg["reranker"]["device"],
            top_k=cfg["reranker"]["top_k"],
        )

        all_companies = list({doc["company"] for doc in cfg.get("documents", [])})
        retriever = HybridRetriever(
            dense_embedder=dense,
            bm25_index=bm25,
            dense_top_k=cfg["retrieval"]["dense_top_k"],
            bm25_top_k=cfg["retrieval"]["bm25_top_k"],
            rrf_k=cfg["retrieval"]["rrf_k"],
            use_classifier=use_clf,
            all_companies=all_companies,
        )

        # Build Gemini client — fall back to OpenAI if only that key is set
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")

        if gemini_key:
            base_url = cfg.get("openai", {}).get("base_url", GEMINI_BASE_URL)
            client = oai.OpenAI(api_key=gemini_key, base_url=base_url)
        elif openai_key:
            client = oai.OpenAI(api_key=openai_key)
        else:
            return {"error": "No API key set. Enter your Gemini API Key in the sidebar."}

        gen_cfg = cfg["generation"]
        generator = GroundedGenerator(
            openai_client=client,
            model=cfg["openai"]["model"],
            max_tokens=cfg["openai"]["max_tokens"],
            # Read new key name, fall back to old key for compatibility
            evidence_score_filter=gen_cfg.get(
                "evidence_score_filter",
                gen_cfg.get("confidence_threshold", 0.0),
            ),
        )

        return {
            "retriever": retriever,
            "reranker": reranker,
            "generator": generator,
            "classifier": QueryClassifier(companies=all_companies) if use_clf else None,
            "mode": mode,
            "cfg": cfg,
            "error": None,
        }

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────

st.markdown(
    '<div class="main-header">📊 Financial Document Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Evidence-grounded Q&amp;A over 10-Ks and earnings reports'
    ' | Tesla · Apple · Microsoft · Nvidia · Amazon</div>',
    unsafe_allow_html=True,
)

# Question input
prefill = st.session_state.pop("prefill_query", "")
query = st.text_input(
    "Ask a question:",
    value=prefill,
    placeholder="e.g. What was Microsoft's revenue in FY2023?",
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.session_state["query_history"] = []
    st.rerun()


# ─────────────────────────────────────────────
# Query execution
# ─────────────────────────────────────────────

if ask_btn and query.strip():
    # API key check before trying to load indexes
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        st.error("Gemini API Key required. Enter it in the sidebar.")
        st.stop()

    resources = _load_indexes(config_path, retrieval_mode, use_classifier)

    if resources is None or resources.get("error"):
        st.error(f"Failed to load indexes: {resources.get('error') if resources else 'Unknown error'}")
        st.stop()

    retriever = resources["retriever"]
    reranker = resources["reranker"]
    generator = resources["generator"]
    classifier = resources["classifier"]
    mode = resources["mode"]

    with st.spinner("Retrieving and generating…"):
        t0 = time.time()

        # Classify query and log it
        clf_result = None
        if classifier:
            clf_result = classifier.classify(query)

        # Retrieve
        if "Dense only" in mode:
            candidates = retriever.retrieve_dense_only(query, top_k=20)
        elif "BM25 only" in mode:
            candidates = retriever.retrieve_bm25_only(query, top_k=20)
        else:
            candidates = retriever.retrieve(query, top_k=20)

        # Rerank if full system
        if "Reranker" in mode:
            final_chunks = reranker.rerank(query, candidates)
        else:
            final_chunks = candidates[:5]

        result = generator.generate(query, final_chunks)
        elapsed = time.time() - t0

    # ── Display results ────────────────────────────────────────────

    # Query type chip
    if clf_result:
        st.markdown(
            f'<div class="query-type-chip">🏷️ Query type: '
            f'{clf_result.query_type.value} · '
            f'Dense {clf_result.dense_weight:.0%} / '
            f'BM25 {clf_result.bm25_weight:.0%}</div>',
            unsafe_allow_html=True,
        )

    # Answer
    st.markdown("### 💬 Answer")
    conf_class = f"badge-{result.confidence}"
    st.markdown(
        f'<div class="answer-box">{result.answer}</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            f'<span class="badge {conf_class}">Confidence: {result.confidence}</span>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.caption(f"⏱ {elapsed:.1f}s")
    with col_c:
        st.caption(f"📄 {len(final_chunks)} evidence chunks")

    # Numerical verification
    if result.verification and not result.verification.verified:
        mismatch_txt = "<br>".join(
            f"• {m.message}" for m in result.verification.mismatches
        )
        st.markdown(
            f'<div class="warning-box">⚠️ <b>Numerical Verification Alert</b><br>'
            f'{mismatch_txt}</div>',
            unsafe_allow_html=True,
        )
    elif result.verification and result.verification.verified:
        st.success(
            f"✅ Numerical check passed — "
            f"{len(result.verification.matched)} value(s) verified against evidence."
        )

    # Citations
    if result.citations:
        st.markdown("### 📎 Citations")
        seen: set = set()
        for c in result.citations:
            key = (c.get("company"), c.get("document"), c.get("page"))
            if key in seen:
                continue
            seen.add(key)
            st.markdown(
                f"- **{c.get('company', '?')}** — "
                f"{c.get('document', '?')} | "
                f"Page {c.get('page', '?')}"
                + (f" | *{c['section']}*" if c.get("section") else "")
            )

    # Evidence panel + optional retrieval debug
    with st.expander(f"📚 Evidence Chunks ({len(final_chunks)})"):
        if show_debug and clf_result:
            company_dist: dict[str, int] = {}
            for chunk, _ in final_chunks:
                co = chunk.get("company", "unknown")
                company_dist[co] = company_dist.get(co, 0) + 1
            type_dist: dict[str, int] = {}
            for chunk, _ in final_chunks:
                ct = chunk.get("chunk_type", "?")
                type_dist[ct] = type_dist.get(ct, 0) + 1

            debug_lines = [
                f"Query type: {clf_result.query_type.value}  |  "
                f"Dense weight: {clf_result.dense_weight:.0%}  |  "
                f"BM25 weight: {clf_result.bm25_weight:.0%}",
                f"Candidates retrieved: {len(candidates)}  →  "
                f"After reranker: {len(final_chunks)}",
                f"Chunk types: {type_dist}",
                f"Company distribution: {company_dist}",
            ]
            st.markdown(
                '<div class="retrieval-debug">' +
                "<br>".join(debug_lines) +
                "</div>",
                unsafe_allow_html=True,
            )

        for i, (chunk, score) in enumerate(final_chunks, 1):
            ctype = chunk.get("chunk_type", "text")
            badge_class = f"badge-{ctype}"
            st.markdown(
                f'<div class="evidence-card">'
                f'<div class="evidence-header">Evidence {i}</div>'
                f'<div class="evidence-meta">'
                f'<span class="badge {badge_class}">{ctype}</span>'
                f' {chunk.get("company", "")} | {chunk.get("document", "")} | '
                f'Page {chunk.get("page", "?")} | Score: {score:.4f}'
                f'</div>'
                f'<div>{chunk.get("content", "")[:500]}…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Save to history
    st.session_state["query_history"].append({
        "query": query,
        "answer": result.answer,
        "abstained": result.abstained,
    })


# ─────────────────────────────────────────────
# Query history
# ─────────────────────────────────────────────

if st.session_state["query_history"]:
    with st.expander(f"📜 Query History ({len(st.session_state['query_history'])})"):
        for i, h in enumerate(reversed(st.session_state["query_history"]), 1):
            flag = "🚫" if h["abstained"] else "✅"
            st.markdown(f"**{i}. {flag} {h['query']}**")
            st.caption(
                h["answer"][:200] + "…" if len(h["answer"]) > 200 else h["answer"]
            )
            st.divider()
