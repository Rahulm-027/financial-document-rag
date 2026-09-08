# Financial Document Intelligence — Multimodal RAG

A financial question-answering system built on top of annual reports (10-Ks)
from five public companies: Tesla, Apple, Microsoft, Nvidia, Amazon.

The system extracts text, tables, and chart descriptions from PDFs, indexes
them with both dense (FAISS) and lexical (BM25) retrieval, applies a
cross-encoder reranker, and generates grounded answers via a Gemini LLM.
Every numerical claim in the answer is verified against the retrieved evidence.

---

## Architecture

```
Financial Reports (PDF)
        ↓
  PDF Parsing             pdfplumber (tables) + PyMuPDF (text + images)
        ↓
  Text / Table / Chart    modality-aware extraction
        ↓
  Chunking                semantic text splits, whole-table preservation,
                          VLM chart descriptions
        ↓
  Dense Embeddings        BAAI/bge-base-en-v1.5 → FAISS FlatIP
  + BM25 Index            financial-aware tokeniser (canonicalises $4.2B = $4.2bn)
        ↓
  Hybrid Retrieval        Weighted Reciprocal Rank Fusion (WRRF)
  + Query Classification  rule-based: numerical / comparison / chart_trend /
                          cross_company / factual
        ↓
  Table Boost             small additive score bonus for table chunks on
                          numerical + comparison queries
        ↓
  Cross-Encoder Reranker  BAAI/bge-reranker-base — scores (query, passage) pairs
        ↓
  Grounded Generation     Gemini (via OpenAI-compatible endpoint)
                          evidence-only system prompt + JSON output
        ↓
  Citation Validation     citations must match retrieved (company, page) pairs
  Numerical Verification  claim-aware: requires value + metric context match
        ↓
  Streamlit UI
```

---

## Retrieval Design

### Hybrid retrieval (WRRF)

Dense retrieval (BGE bi-encoder) captures semantic similarity — useful for
factual and trend questions.  BM25 captures exact financial identifiers:
ticker symbols, dollar amounts like `$211,915`, acronyms like `EBITDA`,
fiscal period labels like `FY2023`.

These are fused with Weighted RRF rather than score averaging because FAISS
returns cosine similarities (0–1) and BM25 returns raw term-frequency scores
(0–∞) — scales are incomparable.  RRF operates on ranks instead.

### Query classification

A rule-based classifier assigns retrieval weights per query type:

| Type | w_dense | w_bm25 | Rationale |
|------|---------|--------|-----------|
| numerical | 0.30 | 0.70 | Exact values, fiscal years → BM25 |
| chart_trend | 0.70 | 0.30 | Conceptual trajectory → dense |
| factual | 0.50 | 0.50 | Balanced |
| comparison | 0.50 | 0.50 | Balanced |
| cross_company | 0.50 | 0.50 | Balanced + per-company balancing |

### Table boost

Table chunks tend to rank below surrounding text prose in dense retrieval
because the embedding model sees terse `| col | val |` content as less
semantically rich.  For numerical and comparison queries, a small additive
score bonus (0.005) is applied after WRRF to give tables a fair chance before
reranking.  The boost only breaks ties — it doesn't override strong
semantic mismatches.

### Cross-company retrieval

For queries mentioning two or more companies (detected dynamically from the
corpus), the retriever runs separate dense and BM25 searches per company
in addition to a global search.  This prevents a single dominant company
from filling all top-k slots before reranking.

---

## Multimodal Processing

### Chart pipeline

1. PyMuPDF detects image regions on each page (filtered by area ≥ 2% of page).
2. Each region is rasterised and passed to the VLM (Gemini) with a structured
   prompt requesting: chart title, type, axes, key values, trend, notable points.
3. The structured description becomes a `chart` chunk, embedded and indexed
   alongside text and table chunks.

**Limitation:** the area-based heuristic may flag logos or photos as charts.
This can generate spurious VLM calls and low-quality chart chunks.  Chart
descriptions that return empty or fail gracefully produce no chunk.

**Known data issue:** Tesla's 10-K URL may point to an image-only (scanned)
PDF with no extractable text layer.  The pipeline logs a clear warning and
produces 0 text elements for that document.  See the `notes` field in
`configs/config.yaml` for the workaround.

### Separate indexes for fair evaluation

Two FAISS indexes are built during ingestion:

- `faiss_text.index` — text chunks only (ablation baseline A)
- `faiss_multimodal.index` — text + table + chart chunks (baselines B, C)

This ensures the A vs B comparison changes only the chunk modalities, nothing
else.

---

## Numerical Verification

The `NumericalVerifier` checks every number in the generated answer against
the retrieved evidence.  It is **claim-aware** — not just value-matching:

- A number's metric context (surrounding 60 chars) is extracted and compared
  against the evidence number's context.
- `$25B revenue` is not verified against `$25B net income` even though the
  values match, because the metric keywords don't overlap.
- Plural forms are stemmed (`revenues` → `revenue`) to avoid false negatives.
- Parenthesised values — `(9%)` and `($1.2B)` — are correctly extracted as
  negative values (standard financial table notation).
- Derived percentages (e.g. 7% YoY from `$211,915` → `$198,270`) are
  classified as `DERIVED` rather than `UNVERIFIABLE`.

**Important:** the verifier reports mismatches but does not suppress the
answer.  Mismatches are surfaced as warnings in the UI.

---

## Citation Validation

After generation, citations are validated against retrieved evidence:
only (company, page) pairs that appear in the actual retrieved chunks are
kept.  This prevents the LLM from hallucinating citation metadata.

---

## Abstention

The system abstains when the LLM determines that the retrieved evidence is
insufficient (via the structured JSON output with `abstained: true`).

The `confidence_threshold` in config (default 0.0) is a reranker-score
filter applied *before* generation — setting it above 0.0 requires careful
empirical calibration since reranker logits are not calibrated probabilities.

---

## Evaluation

65-query benchmark: 60 in-scope + 5 out-of-scope (abstention test).

Query categories: factual (15), numerical (15), comparison (10), chart_trend
(10), cross_company (10).

**Gold relevance:** defined at page-level — all chunks from the specified
`evidence_pages` are treated as relevant.  This is a conservative proxy
that may include irrelevant chunks from the same page.  The eval results
JSON documents this limitation.

Metrics reported: Recall@1/3/5, NDCG@5, MRR, faithfulness (LLM judge),
answer correctness (LLM judge), citation accuracy, abstention accuracy,
latency.

Ablation configurations:
1. Dense-only
2. BM25-only
3. Dense + BM25 (equal weights)
4. Dense + BM25 (query-adaptive WRRF)
5. Weighted WRRF + cross-encoder reranker (full system)

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### API key

The system uses Google Gemini through an OpenAI-compatible endpoint.
Set `GEMINI_API_KEY` in your environment or in a `.env` file (not committed):

```bash
export GEMINI_API_KEY=your_key_here
```

If only `OPENAI_API_KEY` is set, the system falls back to OpenAI.

### Download PDFs

```bash
mkdir -p data/raw
# Download each PDF listed in configs/config.yaml under documents.url
# Example:
curl -L "https://www.sec.gov/Archives/edgar/data/789019/000078901923000039/msft-20230630.pdf" \
     -o data/raw/Microsoft_10K_2023.pdf
```

### Run ingestion pipeline

```bash
python -m src.pipeline --config configs/config.yaml

# Skip chart VLM description (faster, saves API cost):
python -m src.pipeline --config configs/config.yaml --skip_charts
```

### Run the UI

```bash
streamlit run src/ui/app.py
```

### Run evaluation

```bash
python -m src.evaluation.evaluator --config configs/config.yaml
```

### Run tests

```bash
pytest tests/ -v
```

Tests cover: query classifier, RRF formulas, table boost, numerical verifier
(including parenthesised negatives, plural stemming, derived percentages),
chunker, BM25 tokeniser, cross-company classification.

Tests do **not** require GPU, FAISS, or sentence-transformers — those deps
are only needed for pipeline execution.

---

## Limitations

- **Chart detection heuristic:** area-based filtering may misclassify logos
  or decorative images as charts.
- **Tesla PDF:** the EDGAR URL may return an image-only PDF.  The pipeline
  logs a warning if 0 elements are extracted.
- **Page-level gold labels:** evaluation precision/recall metrics are
  approximate because relevance is assigned at page granularity.
- **Reranker scores are not probabilities:** the `confidence_threshold`
  is not a calibrated confidence score.  Default 0.0 is recommended.
- **No cross-document coreference:** if an answer requires combining
  information from two separate documents (e.g. a table on page 42 and
  a footnote on page 78), the system may retrieve one but not the other.
- **Financial unit parsing:** table values like `$211,915` (in millions)
  are extracted as raw numbers.  The verifier treats them as `211,915`,
  not `211,915,000,000`.  Answers that convert to billions (e.g. `$211.9B`)
  are verified correctly because the tolerance handles the scaling mismatch
  only if the evidence also contains the scaled form.

---

## Configuration

All hyperparameters are in `configs/config.yaml`.  Key settings:

| Setting | Default | Notes |
|---------|---------|-------|
| `retrieval.dense_top_k` | 20 | Candidates from dense search |
| `retrieval.bm25_top_k` | 20 | Candidates from BM25 |
| `retrieval.rrf_k` | 60 | RRF smoothing constant |
| `reranker.top_k` | 5 | Chunks passed to LLM |
| `generation.confidence_threshold` | 0.0 | Pre-generation reranker filter |
| `numerical_verification.tolerance` | 0.02 | 2% relative tolerance |
| `embeddings.device` | cpu | Change to `cuda` if GPU available |

---

## Project structure

```
configs/config.yaml          central configuration
src/
  pipeline.py                ingestion orchestrator (run once)
  ingestion/
    pdf_parser.py            text / table / chart extraction
    chunker.py               modality-aware chunking
    chart_describer.py       VLM chart → structured description
  embeddings/
    dense_embedder.py        BGE embeddings + FAISS index
  retrieval/
    bm25_index.py            BM25 with financial tokeniser
    query_classifier.py      rule-based query type detection
    hybrid_retriever.py      WRRF fusion + cross-company routing
    reranker.py              BGE cross-encoder reranker
  generation/
    grounded_generator.py    evidence-grounded Gemini generation
    numerical_verifier.py    claim-aware numerical hallucination check
  evaluation/
    evaluator.py             full eval + ablation runner
    metrics.py               Recall@K, NDCG, MRR, citation accuracy
    test_set.py              65-query benchmark
  ui/
    app.py                   Streamlit interface
tests/
  test_retrieval.py          unit tests (58 tests, no GPU required)
```
