"""
grounded_generator.py
=====================
Evidence-grounded generation using the OpenAI Chat API.

Key design principles:
  1. Evidence-only: the system prompt FORBIDS using information not in
     the provided chunks.  This is essential for financial accuracy.
  2. Explicit citation: the LLM must cite company + document + page for
     each claim.  We then verify these citations exist in the metadata.
  3. Abstention: if evidence is insufficient, the model returns a
     standardised "insufficient evidence" message rather than guessing.
  4. Numerical grounding: the NumericalVerifier is run post-generation
     and any mismatches are surfaced in the response.

Response format
--------------
The model is instructed to return JSON with keys:
  answer      : str  — the answer text (may be multi-paragraph)
  citations   : list of {company, document, page, section, quote}
  confidence  : "high" | "medium" | "low"
  abstained   : bool
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from src.generation.numerical_verifier import NumericalVerifier, VerificationReport

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = textwrap.dedent("""
You are a precise financial analyst assistant.  You answer questions ONLY using
the evidence passages provided — you MUST NOT use any external knowledge or
prior training data about specific financial figures.

Instructions:
1. Read the evidence passages carefully.
2. If the evidence contains sufficient information, produce a concise, accurate answer.
3. For EVERY factual claim, cite the source using the metadata provided.
4. Use exact numbers from the evidence — do NOT estimate, interpolate, or round
   unless the evidence itself is rounded.
5. If the evidence is insufficient, respond with abstained=true and a clear
   explanation of what information is missing.

Output ONLY valid JSON with exactly this schema (no markdown, no preamble):
{
  "answer": "...",
  "citations": [
    {
      "company": "...",
      "document": "...",
      "page": ...,
      "section": "...",
      "quote": "..."
    }
  ],
  "confidence": "high|medium|low",
  "abstained": false
}

If abstaining:
{
  "answer": "Insufficient evidence found in the indexed documents to answer this question. [Explain what's missing]",
  "citations": [],
  "confidence": "low",
  "abstained": true
}
""").strip()


@dataclass
class GenerationResult:
    """Structured result from the grounded generator."""
    answer: str
    citations: list[dict] = field(default_factory=list)
    confidence: str = "low"
    abstained: bool = False
    verification: Optional[VerificationReport] = None
    raw_evidence_chunks: list[dict] = field(default_factory=list, repr=False)

    def format_for_display(self) -> str:
        """Human-readable formatted result."""
        lines = [self.answer, ""]

        if self.citations:
            lines.append("**Sources:**")
            seen = set()
            for c in self.citations:
                citation_str = (
                    f"  • {c.get('company', 'Unknown')} — "
                    f"{c.get('document', '')} | Page {c.get('page', '?')}"
                    + (f" | {c['section']}" if c.get('section') else "")
                )
                if citation_str not in seen:
                    lines.append(citation_str)
                    seen.add(citation_str)

        if self.verification and not self.verification.verified:
            lines.append("")
            lines.append("⚠️ **Numerical Verification Warnings:**")
            for m in self.verification.mismatches:
                lines.append(f"  • {m.message}")

        lines.append(f"\n*Confidence: {self.confidence}*")
        return "\n".join(lines)


class GroundedGenerator:
    """
    LLM generation grounded in retrieved evidence chunks.

    Args
    ----
    openai_client       : openai.OpenAI() instance
    model               : chat model (default gpt-4o-mini)
    max_tokens          : max tokens for generation
    confidence_threshold: chunks with reranker score below this are not
                          included in evidence (helps abstention accuracy)
    enable_verification : run NumericalVerifier on the output
    """

    def __init__(
        self,
        openai_client,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        evidence_score_filter: float = 0.0,
        enable_verification: bool = True,
    ) -> None:
        self.client = openai_client
        self.model = model
        self.max_tokens = max_tokens
        self.evidence_score_filter = evidence_score_filter
        self.verifier = NumericalVerifier() if enable_verification else None

    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        retrieved_chunks: list[tuple[dict, float]],  # (chunk_dict, score)
    ) -> GenerationResult:
        """
        Generate a grounded answer for the query given retrieved chunks.

        Args
        ----
        query            : user question
        retrieved_chunks : list of (chunk_dict, score) from retriever/reranker

        Returns
        -------
        GenerationResult with answer, citations, and verification
        """
        # Filter by confidence threshold
        filtered = [
            (chunk, score)
            for chunk, score in retrieved_chunks
            if score >= self.evidence_score_filter
        ]

        if not filtered:
            return GenerationResult(
                answer=(
                    "Insufficient evidence found in the indexed documents "
                    "to answer this question. No relevant passages met the "
                    "confidence threshold."
                ),
                abstained=True,
                confidence="low",
            )

        # Build the evidence block
        evidence_block = self._build_evidence_block(filtered)
        user_message = f"Question: {query}\n\nEvidence:\n{evidence_block}"

        # Call the LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.max_tokens,
            )
            raw_content = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed")

            print("\n" + "=" * 70)
            print("GEMINI API ERROR:")
            print(repr(e))
            print("=" * 70 + "\n")

            return GenerationResult(
                answer="Generation failed due to an API error.",
                abstained=True,
                confidence="low",
            )

        # Parse JSON response
        result = self._parse_response(raw_content)
        result.raw_evidence_chunks = [chunk for chunk, _ in filtered]

        # Numerical verification
        if self.verifier and not result.abstained:
            evidence_texts = [chunk["content"] for chunk, _ in filtered]
            result.verification = self.verifier.verify(result.answer, evidence_texts)

        # Validate citations exist in our evidence
        result.citations = self._validate_citations(result.citations, filtered)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_evidence_block(chunks: list[tuple[dict, float]]) -> str:
        """Format retrieved chunks as numbered evidence passages."""
        lines = []
        for i, (chunk, score) in enumerate(chunks, start=1):
            meta = (
                f"[Evidence {i}] "
                f"Company: {chunk.get('company', 'Unknown')} | "
                f"Document: {chunk.get('document', 'Unknown')} | "
                f"Page: {chunk.get('page', '?')} | "
                f"Section: {chunk.get('section', 'N/A')} | "
                f"Type: {chunk.get('chunk_type', 'text')} | "
                f"Score: {score:.3f}"
            )
            content = chunk.get("content", "")
            lines.append(f"{meta}\n{content}\n")
        return "\n---\n".join(lines)

    @staticmethod
    def _parse_response(raw: str) -> GenerationResult:
        """Parse the LLM's JSON response into a GenerationResult."""
        # Strip markdown code fences if present
        raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
        try:
            data = json.loads(raw)
            return GenerationResult(
                answer=data.get("answer", ""),
                citations=data.get("citations", []),
                confidence=data.get("confidence", "low"),
                abstained=data.get("abstained", False),
            )
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON; returning raw text.")
            return GenerationResult(
                answer=raw,
                confidence="low",
            )

    @staticmethod
    def _validate_citations(
        citations: list[dict],
        retrieved_chunks: list[tuple[dict, float]],
    ) -> list[dict]:
        """
        Keep only citations whose (company, page) pair appears in the
        retrieved evidence.  This prevents the model from hallucinating
        citation metadata.
        """
        valid_pairs = {
            (str(chunk.get("company", "")).lower(), chunk.get("page", -999))
            for chunk, _ in retrieved_chunks
        }
        validated = []
        for c in citations:
            pair = (str(c.get("company", "")).lower(), c.get("page", -1))
            if pair in valid_pairs:
                validated.append(c)
            else:
                logger.debug(
                    f"Citation removed (not in evidence): "
                    f"{c.get('company')} p{c.get('page')}"
                )
        return validated
