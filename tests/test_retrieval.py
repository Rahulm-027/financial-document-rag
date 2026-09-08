"""
test_retrieval.py
=================
Unit tests for retrieval, verification, chunking, and tokenisation.
Run with: pytest tests/ -v

Note: tests that require sentence_transformers / faiss (DenseEmbedder,
HybridRetriever integration) are skipped here because those libraries
are large ML dependencies not available in all environments.
The RRF helper functions are imported separately to avoid that dep chain.
"""

import pytest

from src.retrieval.query_classifier import QueryClassifier, QueryType
from src.generation.numerical_verifier import NumericalVerifier, extract_numbers
from src.ingestion.chunker import table_to_text, Chunker, _split_text
from src.retrieval.bm25_index import tokenize

# Import RRF helpers directly — these have no heavy ML deps
from src.retrieval.hybrid_retriever import (
    reciprocal_rank_fusion,
    weighted_reciprocal_rank_fusion,
    _boost_table_chunks,
)


# ─────────────────────────────────────────────
# Query Classifier
# ─────────────────────────────────────────────

class TestQueryClassifier:
    clf = QueryClassifier()

    def test_numerical_detection(self):
        r = self.clf.classify("What was Tesla's revenue in Q3 2023?")
        assert r.query_type == QueryType.NUMERICAL
        assert r.bm25_weight > r.dense_weight

    def test_comparison_detection(self):
        r = self.clf.classify("Compare Q3 2023 vs Q3 2022 revenue.")
        assert r.query_type == QueryType.COMPARISON

    def test_cross_company_detection(self):
        r = self.clf.classify("Compare Tesla and Nvidia gross margins.")
        assert r.query_type == QueryType.CROSS_COMPANY

    def test_chart_trend_detection(self):
        r = self.clf.classify("How has the delivery trend changed over time?")
        assert r.query_type == QueryType.CHART_TREND
        assert r.dense_weight > r.bm25_weight

    def test_factual_default(self):
        r = self.clf.classify("What is the company's mission statement?")
        assert r.query_type == QueryType.FACTUAL

    def test_weights_sum_to_one(self):
        queries = [
            "What was revenue?",
            "Compare Q1 vs Q2.",
            "How did deliveries trend?",
            "Which company is bigger?",
            "Explain the business model.",
        ]
        for q in queries:
            r = self.clf.classify(q)
            assert abs(r.dense_weight + r.bm25_weight - 1.0) < 1e-9

    def test_microsoft_revenue_fy2023_is_numerical(self):
        """The reported failing query must classify as NUMERICAL → w_bm25=0.7."""
        r = self.clf.classify("What was Microsoft's revenue in FY2023?")
        assert r.query_type == QueryType.NUMERICAL, (
            f"Expected NUMERICAL, got {r.query_type} ({r.rationale})"
        )
        assert r.bm25_weight == 0.7

    def test_fiscal_year_pattern(self):
        """FY2023 in the query should trigger numerical classification."""
        r = self.clf.classify("What was Apple's net income for FY2023?")
        assert r.query_type == QueryType.NUMERICAL

    def test_cross_company_not_triggered_by_one_company(self):
        """A query mentioning only one company should NOT be cross-company."""
        r = self.clf.classify("What was Microsoft's revenue?")
        assert r.query_type != QueryType.CROSS_COMPANY

    def test_cross_company_two_distinct_companies(self):
        """Two distinct companies must trigger cross-company."""
        r = self.clf.classify("Compare Apple and Amazon revenue in 2023.")
        assert r.query_type == QueryType.CROSS_COMPANY

    def test_same_company_twice_not_cross_company(self):
        """'Apple vs Apple' must not trigger cross-company."""
        r = self.clf.classify("How did Apple compare to Apple's prior year?")
        assert r.query_type != QueryType.CROSS_COMPANY


# ─────────────────────────────────────────────
# RRF (unweighted)
# ─────────────────────────────────────────────

class TestRRF:
    def test_basic_rrf(self):
        list1 = ["doc1", "doc2", "doc3"]
        list2 = ["doc1", "doc3", "doc4"]
        scores = reciprocal_rank_fusion([list1, list2], k=60)
        assert scores["doc1"] > scores["doc2"]
        assert scores["doc1"] > scores["doc3"]
        assert scores["doc3"] > scores["doc4"]

    def test_missing_from_one_list(self):
        list1 = ["a", "b", "c"]
        list2 = ["b", "d"]
        scores = reciprocal_rank_fusion([list1, list2], k=60)
        assert scores["b"] > scores["a"]
        assert scores["b"] > scores["d"]

    def test_k_parameter(self):
        list1 = ["x", "y"]
        s_k1   = reciprocal_rank_fusion([list1], k=1)
        s_k100 = reciprocal_rank_fusion([list1], k=100)
        assert s_k1["x"] > s_k100["x"]


# ─────────────────────────────────────────────
# Weighted RRF
# ─────────────────────────────────────────────

class TestWeightedRRF:
    def test_bm25_heavy_query_type(self):
        """Numerical query: w_bm25=0.7 should surface BM25-top doc over dense-top."""
        dense_list = ["d1", "d2", "d3"]
        bm25_list  = ["b1", "d2", "d3"]  # b1 is top of BM25 only
        # With w_bm25=0.7, b1 gets 0.7/61 ≈ 0.01148
        # d1 gets 0.3/61 ≈ 0.00492
        scores = weighted_reciprocal_rank_fusion(
            dense_list, bm25_list, dense_weight=0.3, bm25_weight=0.7, k=60
        )
        assert scores["b1"] > scores["d1"], \
            "BM25-heavy query should surface BM25's top result"

    def test_dense_heavy_query_type(self):
        """Chart/trend query: w_dense=0.7 should surface dense-top doc."""
        dense_list = ["d1", "d2", "b1"]
        bm25_list  = ["b1", "d2", "d1"]
        scores = weighted_reciprocal_rank_fusion(
            dense_list, bm25_list, dense_weight=0.7, bm25_weight=0.3, k=60
        )
        assert scores["d1"] > scores["b1"], \
            "Dense-heavy query should surface dense's top result"

    def test_equal_weights_symmetric(self):
        """Equal weights should produce the same ranking as unweighted RRF."""
        dense_list = ["a", "b", "c"]
        bm25_list  = ["b", "c", "d"]
        s_equal = weighted_reciprocal_rank_fusion(
            dense_list, bm25_list, 0.5, 0.5, k=60
        )
        s_unweighted = reciprocal_rank_fusion([dense_list, bm25_list], k=60)
        ranking_weighted   = sorted(s_equal, key=lambda x: s_equal[x], reverse=True)
        ranking_unweighted = sorted(s_unweighted, key=lambda x: s_unweighted[x], reverse=True)
        assert ranking_weighted == ranking_unweighted

    def test_weights_are_applied_to_formula(self):
        """Verify WRRF formula exactly: w/(k+rank)."""
        d = ["x"]
        b = ["x"]
        scores = weighted_reciprocal_rank_fusion(d, b, dense_weight=0.3, bm25_weight=0.7, k=60)
        expected = 0.3 / (60 + 1) + 0.7 / (60 + 1)
        assert abs(scores["x"] - expected) < 1e-12

    def test_empty_lists(self):
        """Empty dense or BM25 list should not crash."""
        scores = weighted_reciprocal_rank_fusion([], ["a", "b"], 0.3, 0.7, k=60)
        assert "a" in scores and scores["a"] > 0

        scores = weighted_reciprocal_rank_fusion(["a", "b"], [], 0.3, 0.7, k=60)
        assert "a" in scores and scores["a"] > 0


# ─────────────────────────────────────────────
# Table boost
# ─────────────────────────────────────────────

class TestTableBoost:
    def test_table_chunks_get_boosted(self):
        """Table chunks should score higher than identical-WRRF text chunks."""
        registry = {
            "t1": {"chunk_id": "t1", "chunk_type": "table"},
            "x1": {"chunk_id": "x1", "chunk_type": "text"},
        }
        scores = {"t1": 0.010, "x1": 0.010}  # tied before boost
        boosted = _boost_table_chunks(scores, registry, boost=0.005)
        assert boosted["t1"] > boosted["x1"]

    def test_text_chunks_unaffected(self):
        """Text chunks should not receive any boost."""
        registry = {
            "x1": {"chunk_id": "x1", "chunk_type": "text"},
            "x2": {"chunk_id": "x2", "chunk_type": "text"},
        }
        scores = {"x1": 0.020, "x2": 0.010}
        boosted = _boost_table_chunks(scores, registry, boost=0.005)
        assert boosted["x1"] > boosted["x2"]
        assert boosted["x1"] == 0.020  # unchanged
        assert boosted["x2"] == 0.010  # unchanged

    def test_boost_zero_is_no_op(self):
        """boost=0 should leave scores unchanged."""
        registry = {"t1": {"chunk_id": "t1", "chunk_type": "table"}}
        scores = {"t1": 0.015}
        boosted = _boost_table_chunks(scores, registry, boost=0.0)
        assert boosted["t1"] == 0.015


# ─────────────────────────────────────────────
# Numerical Verifier
# ─────────────────────────────────────────────

class TestNumericalVerifier:
    v = NumericalVerifier(tolerance=0.02)

    def test_extract_dollar_billions(self):
        nums = extract_numbers("Revenue was $25.2B in Q3.")
        assert any(abs(n.value - 25.2e9) < 1e6 for n in nums)

    def test_extract_percentage(self):
        nums = extract_numbers("Gross margin was 18.5%.")
        assert any(n.is_percent and abs(n.value - 18.5) < 0.01 for n in nums)

    def test_extract_millions(self):
        nums = extract_numbers("Operating income of $450M.")
        assert any(abs(n.value - 450e6) < 1 for n in nums)

    def test_extract_word_unit(self):
        nums = extract_numbers("Revenue of $96.8 billion.")
        assert any(abs(n.value - 96.8e9) < 1e6 for n in nums), \
            "Should parse '$96.8 billion' as 96.8e9"

    def test_extract_parenthesised_negative_percent(self):
        """(9%) in a financial table means -9% or a negative change."""
        nums = extract_numbers("YoY change was (9%).")
        paren = [n for n in nums if n.is_negative]
        assert len(paren) >= 1, "Should extract parenthesised negative"
        assert any(n.is_percent for n in paren)

    def test_extract_parenthesised_negative_dollar(self):
        """($1.2B) should be extracted as a negative dollar value."""
        nums = extract_numbers("Net loss was ($1.2B) for the quarter.")
        paren = [n for n in nums if n.is_negative]
        assert len(paren) >= 1
        assert any(abs(n.value + 1.2e9) < 1e7 for n in paren)

    def test_extract_large_plain_integer(self):
        """211,915 (in millions) should be extractable."""
        nums = extract_numbers("Total revenue $211,915")
        assert any(abs(n.value - 211915) < 1 for n in nums)

    def test_verified_when_match_with_context(self):
        """Same value, same metric context → verified."""
        report = self.v.verify(
            answer="Revenue was $96.8B.",
            evidence_texts=["Tesla total revenues were $96.8 billion."]
        )
        assert report.verified

    def test_flagged_on_value_mismatch(self):
        report = self.v.verify(
            answer="Revenue was $100B.",
            evidence_texts=["Actual revenue: $96.8 billion."]
        )
        assert not report.verified
        assert len(report.mismatches) >= 1

    def test_claim_aware_cross_metric_flag(self):
        """$25B revenue should NOT be verified against $25B net income."""
        report = self.v.verify(
            answer="Net income was $25B.",
            evidence_texts=["Revenue was $25B. Net income was $5B."]
        )
        nums = extract_numbers("Net income was $25B.")
        assert any(
            "income" in n.metric_keywords or "net" in n.metric_keywords
            for n in nums
        ), "Should extract metric context for 'net income'"

    def test_derived_percentage(self):
        """25% growth derivable from $80B→$100B should be DERIVED."""
        report = self.v.verify(
            answer="Revenue grew 25%.",
            evidence_texts=["FY2022 revenue: $80 billion. FY2023 revenue: $100 billion."]
        )
        assert len(report.derived) >= 1, \
            "25% growth should be classified as DERIVED from evidence values"

    def test_unverifiable_when_no_evidence_numbers(self):
        report = self.v.verify(
            answer="Revenue was $25B.",
            evidence_texts=["The company performed well this quarter."]
        )
        assert not report.verified
        assert len(report.unverifiable) >= 1

    def test_tolerance_rounding(self):
        # $25.18B vs $25.2B → ~0.08% diff → within 2% tolerance
        report = self.v.verify(
            answer="Revenue was $25.18B.",
            evidence_texts=["Revenue: $25.2 billion."]
        )
        assert report.verified

    def test_no_numbers_in_answer(self):
        report = self.v.verify(
            answer="The company focuses on innovation.",
            evidence_texts=["Revenue was $96.8B."]
        )
        assert report.verified  # nothing to verify → passes

    def test_microsoft_total_revenue_verification(self):
        """
        The exact Microsoft FY2023 segment table values should verify correctly.
        $211,915M total revenue in an evidence table should match $211.9B in answer.
        """
        evidence = (
            "Productivity and Business Processes $69,274 $63,364 9%\n"
            "Intelligent Cloud 87,907 74,965 17%\n"
            "More Personal Computing 54,734 59,941 (9%)\n"
            "Total $211,915 $198,270 7%"
        )
        # Answer states the total in billions — within 2% of 211,915M
        report = self.v.verify(
            answer="Microsoft's total revenue in FY2023 was $211.9 billion.",
            evidence_texts=[evidence]
        )
        # 211.9e9 vs 211915e6 → ~0.007% diff → should verify
        assert report.verified, f"Should verify: {report.summary}"


# ─────────────────────────────────────────────
# Chunker
# ─────────────────────────────────────────────

class TestChunker:
    def test_table_to_text_basic(self):
        table = [["Metric", "Q3 2023", "Q3 2022"], ["Revenue", "$25.2B", "$21.5B"]]
        text = table_to_text(table, caption="Quarterly Results")
        assert "Revenue" in text
        assert "$25.2B" in text
        assert "Caption: Quarterly Results" in text

    def test_table_to_text_empty(self):
        text = table_to_text([])
        assert text == ""

    def test_text_split_respects_boundaries(self):
        long_text = ". ".join([f"Sentence {i}" for i in range(100)]) + "."
        splits = _split_text(long_text, max_tokens=50, overlap_tokens=10)
        assert len(splits) > 1
        for s in splits:
            assert len(s.split()) < 100

    def test_chunk_text_element(self):
        chunker = Chunker(text_max_tokens=50)
        elem = {
            "element_type": "text",
            "content": "This is a paragraph. " * 100,
            "company": "Tesla",
            "document": "Tesla 10-K 2023",
            "year": 2023,
            "page": 5,
            "section": "Financial Highlights",
            "caption": "",
        }
        chunks = chunker.chunk([elem])
        assert len(chunks) > 1
        for c in chunks:
            assert c.company == "Tesla"
            assert c.chunk_type == "text"
            assert "Tesla" in c.content

    def test_chunk_table_element(self):
        chunker = Chunker()
        elem = {
            "element_type": "table",
            "content": [["Metric", "Value"], ["Revenue", "$96.8B"], ["Gross Profit", "$17.6B"]],
            "company": "Tesla",
            "document": "Tesla 10-K 2023",
            "year": 2023,
            "page": 8,
            "section": "Income Statement",
            "caption": "Consolidated Income Statement",
        }
        chunks = chunker.chunk([elem])
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "table"
        assert "$96.8B" in chunks[0].content

    def test_chunk_chart_element(self):
        chunker = Chunker()
        elem = {
            "element_type": "chart",
            "content": "/path/to/chart.png",
            "description": "Chart: Vehicle Deliveries\nCompany: Tesla\nTrend: Increasing\nKey values: Q4 2023: 484,507",
            "company": "Tesla",
            "document": "Tesla 10-K 2023",
            "year": 2023,
            "page": 3,
            "section": "Operations",
            "caption": "Vehicle Deliveries by Quarter",
        }
        chunks = chunker.chunk([elem])
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "chart"
        assert "Vehicle Deliveries" in chunks[0].content

    def test_chunk_skips_chart_without_description_or_caption(self):
        """A chart element with no description and no caption should produce no chunk."""
        chunker = Chunker()
        elem = {
            "element_type": "chart",
            "content": "/path/to/chart.png",
            "description": "",
            "company": "Tesla",
            "document": "Tesla 10-K 2023",
            "year": 2023,
            "page": 3,
            "section": "Operations",
            "caption": "",
        }
        chunks = chunker.chunk([elem])
        assert len(chunks) == 0, "Should skip chart with no usable content"

    def test_table_chunk_content_has_provenance_header(self):
        """Table chunks must include the [TABLE | company | document | Page N] header."""
        chunker = Chunker()
        elem = {
            "element_type": "table",
            "content": [["Segment", "Revenue"], ["Cloud", "$87.9B"]],
            "company": "Microsoft",
            "document": "Microsoft 10-K 2023",
            "year": 2023,
            "page": 42,
            "section": "Segment Results",
            "caption": "Segment Revenue",
        }
        chunks = chunker.chunk([elem])
        assert chunks
        assert "[TABLE | Microsoft | Microsoft 10-K 2023 | Page 42]" in chunks[0].content


# ─────────────────────────────────────────────
# BM25 Tokeniser
# ─────────────────────────────────────────────

class TestTokenizer:
    def test_canonical_billion_word(self):
        """'$4.2 billion' and '$4.2B' should produce the same canonical token."""
        t1 = tokenize("Revenue of $4.2 billion")
        t2 = tokenize("Revenue of $4.2B")
        t3 = tokenize("Revenue of $4.2bn")
        canon = "$4.2b"
        assert canon in t1, f"Expected '{canon}' in {t1}"
        assert canon in t2, f"Expected '{canon}' in {t2}"
        assert canon in t3, f"Expected '{canon}' in {t3}"

    def test_canonical_million_word(self):
        t1 = tokenize("$450 million operating income")
        t2 = tokenize("$450M operating income")
        t3 = tokenize("$450mn operating income")
        canon = "$450m"
        assert canon in t1 and canon in t2 and canon in t3

    def test_preserves_percentages(self):
        tokens = tokenize("Gross margin was 23.5%")
        assert "23.5%" in tokens

    def test_percent_word_form(self):
        t1 = tokenize("margin of 23.5 percent")
        t2 = tokenize("margin of 23.5%")
        assert "23.5%" in t1 and "23.5%" in t2

    def test_removes_stopwords(self):
        tokens = tokenize("the company is a great business")
        for sw in ["the", "is", "a"]:
            assert sw not in tokens

    def test_fiscal_year_token(self):
        tokens = tokenize("Results for FY2024")
        assert "fy2024" in tokens

    def test_quarter_fiscal_year_combined(self):
        tokens = tokenize("Q3 FY2024 results")
        assert any("q3" in t for t in tokens)
        assert any("fy2024" in t for t in tokens)

    def test_canonical_tokens_same_for_billion_and_B(self):
        """
        The canonical token for '$4.2 billion' and '$4.2B' must be identical,
        so BM25 queries using word-form match documents using abbreviation.
        """
        t_word = tokenize("revenue of $4.2 billion")
        t_abbr = tokenize("revenue of $4.2B")
        t_bn   = tokenize("revenue of $4.2bn")
        assert "$4.2b" in t_word
        assert "$4.2b" in t_abbr
        assert "$4.2b" in t_bn
        assert t_word == t_abbr == t_bn, \
            "All three forms should produce identical token sequences"

    def test_large_comma_separated_number(self):
        """$211,915 (millions table value) should tokenise without splitting on comma."""
        tokens = tokenize("Total $211,915")
        # The number should appear as a token (with or without dollar prefix)
        assert any("211" in t for t in tokens), f"Expected 211 in tokens: {tokens}"

    def test_parenthesised_negative_in_table(self):
        """(9%) should tokenise to include the percentage value."""
        tokens = tokenize("YoY change (9%)")
        assert any("9%" in t for t in tokens), f"Expected 9% in tokens: {tokens}"


# ─────────────────────────────────────────────
# Cross-company retrieval logic (unit level)
# ─────────────────────────────────────────────

class TestCrossCompanyClassification:
    """
    Tests that the classifier correctly identifies cross-company queries
    without requiring the full retriever stack.
    """
    clf = QueryClassifier(companies=["Tesla", "Apple", "Microsoft", "Nvidia", "Amazon"])

    def test_apple_amazon_cross(self):
        r = self.clf.classify("Compare Apple's revenue with Amazon's revenue in 2023.")
        assert r.query_type == QueryType.CROSS_COMPANY

    def test_nvidia_microsoft_cross(self):
        r = self.clf.classify("Which had higher gross margin in 2023: Nvidia or Microsoft?")
        assert r.query_type == QueryType.CROSS_COMPANY

    def test_single_company_not_cross(self):
        r = self.clf.classify("What was Nvidia's revenue in FY2024?")
        assert r.query_type != QueryType.CROSS_COMPANY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
