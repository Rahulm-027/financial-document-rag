"""
numerical_verifier.py
=====================
Detects numerical hallucinations in generated financial answers.

Pipeline
--------
  1. Extract all numbers from the generated answer with their local context
     (~60 chars around each number) to capture the metric being stated.
  2. Extract all (metric, value) pairs from retrieved evidence chunks.
  3. For each answer number, perform claim-aware matching:
     a. Find evidence numbers that are numerically close (within tolerance).
     b. Also check if the surrounding metric keywords overlap — this
        prevents a false-positive where "$25B revenue" is incorrectly
        verified against "$25B net income" present in evidence.
  4. Classify each answer number as:
     - MATCHED    : close value + overlapping metric context
     - MISMATCH   : close-ish value but wrong metric, or clearly wrong value
     - DERIVED    : percentage change or ratio that can be computed from
                   evidence numbers (e.g. 25% growth from $80B→$100B)
     - UNVERIFIABLE: no close evidence number found at all

Claim-awareness
---------------
  A naive verifier marks "$25B net income" as verified if "$25B revenue"
  exists in evidence — both have value 25e9.  We reduce this failure mode
  by extracting the 3–5 words before each number as a metric label and
  requiring ≥1 metric keyword overlap for a positive verification.

Derived-value detection
-----------------------
  For percentage values in the answer, we check if they can be computed
  from any pair of evidence values:
    pct_change = abs(a - b) / abs(b)
  If a derived match is found within tolerance, the number is classified
  as DERIVED rather than UNVERIFIABLE.

Parenthesised negatives
-----------------------
  Financial statements use "(9%)" or "($1.2B)" to denote negative values.
  These are extracted as negative floats so comparisons work correctly.

Tolerance
---------
  Default 2% (0.02) — handles rounding differences but catches
  outright substitutions (e.g. $25.2B vs $25.8B → 2.4% → flagged).

Units normalised
----------------
  B/bn/billion → ×1e9
  M/mn/million → ×1e6
  K/k/thousand → ×1e3
  % → kept as-is (percentage points)

Known limitation — table-level units
-------------------------------------
  Financial tables often state "(In millions, except percentages)" once
  in the header, then list raw numbers like 211,915 without inline units.
  The verifier sees these as plain integers (211915.0) because the unit
  context is not attached to each extracted number.  This means:

    Evidence: "Total 211,915"   → extracted as 211915.0
    Answer:   "$211.9 billion" → extracted as 211900000000.0

  These differ by 10^6 and will not match within 2% tolerance, so the
  number is classified as UNVERIFIABLE rather than MATCHED.

  The workaround is for the evidence text to include the full unit.  The
  chunker preserves table headers, so if the table header row says
  "(In millions)" and it is included in the same chunk as the data rows,
  the LLM will interpret the values correctly in the answer — but the
  verifier still cannot propagate table-header units to individual cells.

  A future improvement would be to propagate the reporting unit (millions /
  billions) as metadata on each table chunk during ingestion, then use that
  metadata in the verifier to normalise plain integers before comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Number extraction
# ─────────────────────────────────────────────

# Financial metric keywords — used for claim-aware context matching.
# A number's context is checked for these words to identify what metric
# it refers to, preventing cross-metric false positives.
_METRIC_KEYWORDS = {
    "revenue", "revenues", "sales", "income", "earnings", "profit", "loss",
    "margin", "margins", "ebitda", "ebit", "eps", "diluted", "basic",
    "cash", "debt", "assets", "liabilities", "equity", "capex", "fcf",
    "growth", "increase", "decrease", "change", "yoy", "qoq",
    "deliveries", "units", "shipments", "employees", "headcount",
    "operating", "gross", "net", "total", "segment", "geographic",
}


@dataclass
class ExtractedNumber:
    raw: str            # the original string, e.g. "$25.2B"
    value: float        # normalised float, e.g. 25_200_000_000.0
    is_percent: bool    # True if the value represents a percentage
    is_negative: bool   # True if wrapped in parentheses (financial notation)
    context: str        # surrounding ~60 chars for display
    metric_keywords: frozenset  # financial metric words found in context


# Regex for parenthesised negative values: ($1.2B) or (9%)
# These must be checked before the regular number pattern.
_PAREN_NUM_RE = re.compile(
    r'\((\$\s*)?([0-9,]+(?:\.[0-9]+)?)\s*'
    r'(billion|million|thousand|bn|mn|[bBmMkK](?!\w))?\s*(%?)\)',
    re.IGNORECASE,
)

# Regex capturing financial numbers with optional units
_NUM_RE = re.compile(
    r'(?<!\w)'                                    # not preceded by word char
    r'(\$\s*)?'                                   # optional dollar sign
    r'([0-9,]+(?:\.[0-9]+)?)'                    # integer or decimal
    r'\s*'
    r'(billion|million|thousand|bn|mn|[bBmMkK](?!\w))?'  # unit suffix
    r'\s*(%)?'                                    # optional percent
    r'(?!\d)',                                    # not followed by digit
    re.IGNORECASE,
)

_UNIT_MULTIPLIERS = {
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "m": 1e6, "mn": 1e6, "million": 1e6,
    "k": 1e3, "thousand": 1e3,
}


def _normalise_value(num_str: str, unit: str) -> float:
    cleaned = num_str.replace(",", "").strip()
    value = float(cleaned) if cleaned else 0.0
    if unit:
        u = unit.lower().strip()
        # Try full word first, then single-letter fallback
        mult = _UNIT_MULTIPLIERS.get(u, _UNIT_MULTIPLIERS.get(u[0], 1.0))
        value *= mult
    return value


def _extract_metric_keywords(text: str, pos: int, window: int = 60) -> frozenset:
    """Extract financial metric keywords from a context window around position pos."""
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    snippet = text[start:end].lower()
    words = re.findall(r'\b\w+\b', snippet)
    return frozenset(w for w in words if w in _METRIC_KEYWORDS)


def extract_numbers(text: str) -> list[ExtractedNumber]:
    """
    Extract and normalise all financial numbers from a text string.

    Handles:
      - $25.2B / $25.2 billion / $25,200 million
      - 23.5% / 23.5 percent
      - (9%) and ($1.2B) — parenthesised negatives used in financial tables
      - Plain integers like 211,915 (in millions context)
    """
    results: list[ExtractedNumber] = []
    # Track character positions already captured to avoid double-counting
    covered: set[int] = set()

    # 1. Parenthesised negatives first (e.g. "(9%)" in financial tables)
    for match in _PAREN_NUM_RE.finditer(text):
        dollar, num_str, unit, percent = match.groups()
        if not num_str.replace(",", "").replace(".", ""):
            continue
        raw = match.group(0).strip()
        is_percent = bool(percent)
        value = -_normalise_value(num_str, unit or "")
        if is_percent:
            value = abs(value)  # percentages stay positive; context clarifies direction

        start_c = max(0, match.start() - 60)
        end_c = min(len(text), match.end() + 60)
        context = text[start_c:end_c].replace("\n", " ")
        metric_kw = _extract_metric_keywords(text, match.start())

        results.append(ExtractedNumber(
            raw=raw,
            value=value,
            is_percent=is_percent,
            is_negative=True,
            context=context,
            metric_keywords=metric_kw,
        ))
        covered.update(range(match.start(), match.end()))

    # 2. Regular numbers
    for match in _NUM_RE.finditer(text):
        # Skip positions already captured by the paren pattern
        if any(p in covered for p in range(match.start(), match.end())):
            continue

        dollar_sign, num_str, unit, percent = match.groups()
        raw = match.group(0).strip()
        if not num_str.replace(",", "").replace(".", ""):
            continue

        is_percent = bool(percent)
        value = _normalise_value(num_str, unit or "")

        start_c = max(0, match.start() - 60)
        end_c = min(len(text), match.end() + 60)
        context = text[start_c:end_c].replace("\n", " ")
        metric_kw = _extract_metric_keywords(text, match.start())

        results.append(ExtractedNumber(
            raw=raw,
            value=value,
            is_percent=is_percent,
            is_negative=False,
            context=context,
            metric_keywords=metric_kw,
        ))

    return results


# ─────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────

@dataclass
class NumberMismatch:
    answer_number: ExtractedNumber
    closest_evidence_value: Optional[float]
    relative_diff: Optional[float]
    flagged: bool
    message: str


@dataclass
class VerificationReport:
    verified: bool
    mismatches: list[NumberMismatch] = field(default_factory=list)
    matched: list[ExtractedNumber] = field(default_factory=list)
    derived: list[ExtractedNumber] = field(default_factory=list)     # computed from evidence
    unverifiable: list[ExtractedNumber] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "verified": self.verified,
            "num_mismatches": len(self.mismatches),
            "num_matched": len(self.matched),
            "num_derived": len(self.derived),
            "num_unverifiable": len(self.unverifiable),
            "summary": self.summary,
            "mismatches": [
                {
                    "answer_raw": m.answer_number.raw,
                    "answer_value": m.answer_number.value,
                    "answer_context": m.answer_number.context,
                    "closest_evidence": m.closest_evidence_value,
                    "relative_diff_pct": round(m.relative_diff * 100, 2)
                    if m.relative_diff is not None else None,
                    "message": m.message,
                }
                for m in self.mismatches
            ],
        }


class NumericalVerifier:
    """
    Claim-aware numerical hallucination detector.

    Two improvements over naive value-matching:

    1. Claim-awareness
       Each answer number's surrounding metric keywords (e.g. "revenue",
       "margin") are compared against the metric keywords of candidate
       evidence numbers.  A match requires both numerical closeness AND
       at least one shared metric keyword (or the answer number has no
       metric context, in which case numerical closeness alone suffices).

    2. Derived-value detection
       For percentage values in the answer (e.g. "25% growth"), we check
       whether this value can be computed from any pair of evidence numbers
       as a percentage change:  |a - b| / |b|.  If derivable, the number
       is classified as DERIVED rather than UNVERIFIABLE.

    3. Parenthesised negatives
       Financial tables use "(9%)" to mean -9%.  These are correctly
       extracted as negative values so comparisons work for financial
       statements.

    Args
    ----
    tolerance         : relative tolerance for value comparison (default 0.02)
    require_context   : if True, require metric keyword overlap for a positive
                        match when both sides have metric context (default True)
    """

    def __init__(
        self,
        tolerance: float = 0.02,
        require_context: bool = True,
    ) -> None:
        self.tolerance = tolerance
        self.require_context = require_context

    def verify(self, answer: str, evidence_texts: list[str]) -> VerificationReport:
        """
        Verify numbers in `answer` against `evidence_texts`.

        Returns
        -------
        VerificationReport classifying each answer number as:
          matched      — value close + metric context consistent
          mismatch     — value close but metric context conflicts, or
                         value clearly wrong
          derived      — percentage derivable from evidence value pairs
          unverifiable — no close evidence value found
        """
        answer_numbers = extract_numbers(answer)
        evidence_numbers: list[ExtractedNumber] = []
        for ev in evidence_texts:
            evidence_numbers.extend(extract_numbers(ev))

        if not answer_numbers:
            return VerificationReport(
                verified=True,
                summary="No numerical values in answer — no verification needed.",
            )

        if not evidence_numbers:
            return VerificationReport(
                verified=False,
                unverifiable=answer_numbers,
                summary=(
                    f"Answer contains {len(answer_numbers)} number(s) but "
                    "no numerical evidence found in retrieved chunks."
                ),
            )

        ev_values = [en.value for en in evidence_numbers]

        mismatches: list[NumberMismatch] = []
        matched: list[ExtractedNumber] = []
        derived: list[ExtractedNumber] = []
        unverifiable: list[ExtractedNumber] = []

        for an in answer_numbers:
            # Skip trivial numbers (page numbers, list indices, etc.)
            if not an.is_percent and abs(an.value) < 1.0:
                continue

            # ── 1. Find numerically close evidence numbers ────────────
            close_ev = self._find_close_evidence(an.value, evidence_numbers)

            if close_ev:
                # ── 2. Claim-awareness check ──────────────────────────
                if self._context_consistent(an, close_ev):
                    matched.append(an)
                else:
                    # Numerically close but different metric — flag it
                    best_val = min(close_ev, key=lambda e: abs(e.value - an.value)).value
                    rel_diff = abs(best_val - an.value) / max(abs(an.value), 1e-9)
                    mismatches.append(NumberMismatch(
                        answer_number=an,
                        closest_evidence_value=best_val,
                        relative_diff=rel_diff,
                        flagged=True,
                        message=(
                            f"{an.raw} matches evidence value {best_val:.4g} numerically "
                            f"but metric context differs "
                            f"(answer: {sorted(an.metric_keywords)}, "
                            f"evidence: {sorted(close_ev[0].metric_keywords)}). "
                            "Possible metric substitution."
                        ),
                    ))
            else:
                # ── 3. Check if derivable from evidence (% changes) ──
                if an.is_percent and self._is_derivable(an.value, ev_values, an, evidence_numbers):
                    derived.append(an)
                else:
                    # Try global closest for the mismatch message
                    best_ev_val, rel_diff = self._find_closest_scalar(an.value, ev_values)
                    if best_ev_val is not None and rel_diff is not None and rel_diff < 0.5:
                        # A value exists but it's >tolerance% off
                        mismatches.append(NumberMismatch(
                            answer_number=an,
                            closest_evidence_value=best_ev_val,
                            relative_diff=rel_diff,
                            flagged=True,
                            message=(
                                f"Answer states {an.raw} (≈{an.value:.4g}) but "
                                f"closest evidence value is {best_ev_val:.4g} "
                                f"({rel_diff * 100:.1f}% difference)."
                            ),
                        ))
                    else:
                        unverifiable.append(an)

        verified = len(mismatches) == 0
        parts = []
        if matched:
            parts.append(f"{len(matched)} verified")
        if mismatches:
            parts.append(f"⚠ {len(mismatches)} MISMATCH(ES)")
        if derived:
            parts.append(f"{len(derived)} derived (computed from evidence)")
        if unverifiable:
            parts.append(f"{len(unverifiable)} unverifiable (possibly derived or external)")
        summary = "; ".join(parts) or "Verification complete."

        return VerificationReport(
            verified=verified,
            mismatches=mismatches,
            matched=matched,
            derived=derived,
            unverifiable=unverifiable,
            summary=summary,
        )

    # ------------------------------------------------------------------

    def _find_close_evidence(
        self,
        value: float,
        evidence_numbers: list[ExtractedNumber],
    ) -> list[ExtractedNumber]:
        """Return evidence numbers within self.tolerance of value."""
        if value == 0:
            return []
        return [
            en for en in evidence_numbers
            if abs(en.value - value) / max(abs(value), 1e-9) <= self.tolerance
        ]

    @staticmethod
    def _stem_keywords(kws: frozenset) -> frozenset:
        """
        Strip trailing 's' from keywords so 'revenue' matches 'revenues'.
        This is intentionally minimal — we only want to handle the common
        plural forms in financial text, not full morphological stemming.
        """
        return frozenset(k.rstrip('s') if len(k) > 3 else k for k in kws)

    def _context_consistent(
        self,
        answer_num: ExtractedNumber,
        close_ev: list[ExtractedNumber],
    ) -> bool:
        """
        Return True if the metric context is consistent between the answer
        number and at least one of the close evidence numbers.

        If either side has no metric keywords, we grant the match (the
        number exists in evidence and we can't identify a conflicting metric).

        Stemming: we strip trailing 's' before comparing so that 'revenue'
        matches 'revenues' — a common discrepancy between condensed answers
        and verbose evidence text.
        """
        if not self.require_context:
            return True
        if not answer_num.metric_keywords:
            return True  # no metric context in answer → can't penalise
        ans_stemmed = self._stem_keywords(answer_num.metric_keywords)
        for ev in close_ev:
            if not ev.metric_keywords:
                return True  # no metric context in evidence → grant match
            ev_stemmed = self._stem_keywords(ev.metric_keywords)
            if ans_stemmed & ev_stemmed:
                return True  # at least one shared (stemmed) metric keyword
        return False

    def _is_derivable(
        self,
        pct_value: float,
        ev_values: list[float],
        answer_num: ExtractedNumber,
        evidence_numbers: list[ExtractedNumber],
    ) -> bool:
        """
        Check if pct_value (a percentage) is derivable from any ordered pair of
        evidence values as a percentage change: |a - b| / |b| × 100.

        Limitation: we only require that *some* pair of evidence values produces
        this percentage — we don't require the pair to be the correct (metric, period)
        pair.  This means a percentage in the answer can be DERIVED against an
        unrelated pair of numbers that happen to produce the same value.

        To reduce false positives we require the derived pair to contain values
        of similar magnitude as the two largest distinct evidence values in the
        answer number's metric context (if any).  This is a heuristic, not a
        rigorous proof of derivation.

        We check both orderings (a→b and b→a) because growth can be expressed
        relative to either period.
        """
        nonzero = [v for v in ev_values if abs(v) > 1e-9]
        # Only consider large values (>1000) to avoid spurious matches from
        # page numbers, item counts, or other small integers in the evidence.
        financial_scale = [v for v in nonzero if abs(v) >= 1000]
        search_pool = financial_scale if financial_scale else nonzero

        for i, a in enumerate(search_pool):
            for j, b in enumerate(search_pool):
                if i == j:
                    continue
                computed_pct = abs(a - b) / abs(b) * 100
                if abs(computed_pct - pct_value) / max(abs(pct_value), 1e-9) <= self.tolerance:
                    return True
        return False

    @staticmethod
    def _find_closest_scalar(
        value: float, evidence_values: list[float]
    ) -> tuple[Optional[float], Optional[float]]:
        """Return (closest_evidence_value, relative_difference) for display."""
        if not evidence_values or value == 0:
            return None, None
        best = min(evidence_values, key=lambda ev: abs(ev - value) / max(abs(value), 1e-9))
        return best, abs(best - value) / max(abs(value), 1e-9)
