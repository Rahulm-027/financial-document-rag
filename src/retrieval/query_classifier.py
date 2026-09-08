"""
query_classifier.py
===================
Classifies incoming financial queries into one of five types and returns
the optimal retrieval weight configuration for dense vs. BM25.

Query types
-----------
  factual      — "What factors contributed to revenue growth?"
  numerical    — "What was revenue in Q3 2023?"
  comparison   — "Compare Q3 2023 and Q3 2022 revenue."
  chart_trend  — "How did vehicle deliveries change over time?"
  cross_company— "Which company had higher gross margin?"

Classification is rule-based (regex + keyword matching) — fast, transparent,
and requires no additional API calls.  This is intentional for a placement
project: you can explain exactly why a query was classified a certain way.

For each type, we return [dense_weight, bm25_weight] (sum = 1.0).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QueryType(str, Enum):
    FACTUAL = "factual"
    NUMERICAL = "numerical"
    COMPARISON = "comparison"
    CHART_TREND = "chart_trend"
    CROSS_COMPANY = "cross_company"


@dataclass
class ClassificationResult:
    query_type: QueryType
    dense_weight: float
    bm25_weight: float
    confidence: str   # "high" | "medium" | "low"
    rationale: str    # human-readable explanation (useful for debugging)


# ─────────────────────────────────────────────
# Pattern libraries
# ─────────────────────────────────────────────

_DEFAULT_COMPANIES = [
    "tesla", "apple", "microsoft", "nvidia", "amazon",
    "google", "meta", "alphabet", "netflix", "salesforce",
]

_CROSS_COMPANY_GENERIC = [
    re.compile(p, re.IGNORECASE) for p in [
        r'\bwhich company\b',
        r'\bacross companies\b',
        r'\ball companies\b',
        r'\b(highest|lowest|best|worst)\b.{0,40}\b(company|companies|among)\b',
        r'\bcompare\b.{0,80}\band\b.{0,80}\b(revenue|income|profit|margin|sales)\b',
    ]
]

_COMPARISON_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'\bcompare\b.{0,60}\b(q[1-4]|quarter|fy|year|annual|h[12])\b',
        r'\b(vs\.?|versus)\b',
        r'\byear.{0,10}over.{0,10}year\b',
        r'\bq[1-4]\s+\d{4}.{0,40}q[1-4]\s+\d{4}\b',
        r'\b(changed|change|differ|difference|growth|decline)\b.{0,40}'
        r'\b(compared|relative|against|from|since)\b',
        r'\b(q[1-4]|quarter|fy|fiscal\s+year).{0,20}(q[1-4]|quarter|fy)\b',
    ]
]

# Numerical: asking for a specific value or metric.
# The "what was ... revenue" pattern is the primary signal for the
# Microsoft FY2023 query — it's matched here and gets w_bm25=0.7,
# which helps BM25 find the table with the exact "Total" row.
_NUMERICAL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'\bhow much\b',
        r'\bwhat\s+(was|is|were)\b.{0,80}'
        r'\b(revenue|income|profit|loss|margin|eps|ebitda|sales|cost|expense|cash|total)\b',
        r'\b(total|net|gross|operating)\b.{0,20}'
        r'\b(revenue|income|profit|loss|margin|eps|ebitda)\b',
        r'\$\s*[\d,]+',
        r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:billion|million|thousand|bn|mn)\b',
        r'\bpercentage\b|\bpercent\b|\b%\b',
        r'\bfigure\b|\bnumber\b|\bvalue\b|\bamount\b',
        r'\b(fy\d{4}|q[1-4]\s*\d{4})\b',   # fiscal year / quarter patterns
        r'\bfiscal\s+year\b',
    ]
]

_CHART_TREND_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'\b(trend|trajectory|pattern|over time)\b',
        r'\b(chart|graph|figure|plot|visual)\b',
        r'\b(grew|declined|increased|decreased|rose|fell)\b.{0,30}'
        r'\b(over|through|from|since|during)\b',
        r'\bhow\b.{0,30}\b(changed|evolved|trended|performed)\b',
        r'\b(delivery|deliveries)\b',
        r'\b(seasonality|seasonal|cycle)\b',
    ]
]

# ─────────────────────────────────────────────
# Weight lookup
# ─────────────────────────────────────────────

_WEIGHTS: dict[QueryType, tuple[float, float]] = {
    QueryType.FACTUAL:       (0.50, 0.50),
    QueryType.NUMERICAL:     (0.30, 0.70),  # BM25 strong on exact numbers/terms
    QueryType.COMPARISON:    (0.50, 0.50),
    QueryType.CHART_TREND:   (0.70, 0.30),  # Dense better for trend concepts
    QueryType.CROSS_COMPANY: (0.50, 0.50),
}


# ─────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────

class QueryClassifier:
    """
    Rule-based financial query classifier.

    Priority order (higher priority checked first):
      1. cross_company  — mentions multiple companies
      2. comparison     — explicit period-to-period comparison
      3. numerical      — asking for a specific value/metric
      4. chart_trend    — asks about trajectory or visual data
      5. factual        — default

    Args
    ----
    companies : optional list of company names present in the corpus.
                Used to detect cross-company queries dynamically rather
                than relying solely on hard-coded patterns.

    Example
    -------
    >>> clf = QueryClassifier()
    >>> clf.classify("What was Tesla's net income in Q3 2023?")
    ClassificationResult(query_type=<QueryType.NUMERICAL: 'numerical'>, ...)
    """

    def __init__(self, companies: Optional[list[str]] = None) -> None:
        co_list = [c.lower() for c in (companies or _DEFAULT_COMPANIES)]
        # Regex that detects two different company names in the same query
        if len(co_list) >= 2:
            co_alts = "|".join(re.escape(c) for c in co_list)
            self._cross_company_named: Optional[re.Pattern] = re.compile(
                rf'(?:{co_alts}).{{0,80}}(?:{co_alts})',
                re.IGNORECASE,
            )
        else:
            self._cross_company_named = None
        self._co_list = co_list

    def classify(self, query: str) -> ClassificationResult:
        query_stripped = query.strip()

        # 1. Cross-company
        is_cross = any(p.search(query_stripped) for p in _CROSS_COMPANY_GENERIC)
        if not is_cross and self._cross_company_named:
            m = self._cross_company_named.search(query_stripped)
            if m:
                # Verify the match actually contains two *different* companies
                matched_text = m.group(0).lower()
                co_hits = list({c for c in self._co_list if c in matched_text})
                if len(co_hits) >= 2:
                    is_cross = True

        if is_cross:
            qt = QueryType.CROSS_COMPANY
            confidence = "high"
            rationale = "Detected multiple company names or 'which company' phrasing."

        # 2. Comparison
        elif any(p.search(query_stripped) for p in _COMPARISON_PATTERNS):
            qt = QueryType.COMPARISON
            confidence = "high"
            rationale = "Detected comparison keywords (vs., compare, YoY, two periods)."

        # 3. Numerical
        elif any(p.search(query_stripped) for p in _NUMERICAL_PATTERNS):
            qt = QueryType.NUMERICAL
            confidence = "high"
            rationale = "Detected financial metric / numerical value request."

        # 4. Chart/trend
        elif any(p.search(query_stripped) for p in _CHART_TREND_PATTERNS):
            qt = QueryType.CHART_TREND
            confidence = "medium"
            rationale = "Detected trend/chart/trajectory language."

        # 5. Factual (default)
        else:
            qt = QueryType.FACTUAL
            confidence = "low"
            rationale = "No strong signal found — defaulting to balanced retrieval."

        dw, bw = _WEIGHTS[qt]
        return ClassificationResult(
            query_type=qt,
            dense_weight=dw,
            bm25_weight=bw,
            confidence=confidence,
            rationale=rationale,
        )

    def describe(self, result: ClassificationResult) -> str:
        return (
            f"Type: {result.query_type.value} "
            f"(confidence: {result.confidence}) | "
            f"Dense: {result.dense_weight:.0%}, BM25: {result.bm25_weight:.0%} | "
            f"{result.rationale}"
        )
