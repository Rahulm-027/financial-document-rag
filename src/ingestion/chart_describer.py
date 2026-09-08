"""
chart_describer.py
==================
Converts chart/figure images from financial PDFs into structured textual
descriptions using a Vision LLM (GPT-4o-mini).

The output is a *structured* description — not free prose — so that the
embedding model has dense, queryable text about the chart's content,
values, trends, and axes.  This is the key difference from simply storing
raw image bytes or generic "this is a chart" placeholders.

Output format:
--------------
Chart: <title if visible>
Company: <company>
Period: <e.g. Q1 2022 – Q3 2023>
Type: <bar | line | pie | table | scatter | other>
X-axis: <label>
Y-axis: <label and unit>
Key values: <bullet list>
Trend: <1–2 sentence description>
Notable: <any anomaly, peak, trough>
"""

from __future__ import annotations

import base64
import logging
import textwrap
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
You are a financial analyst expert at reading charts, graphs, and figures
from corporate financial reports.

Your task is to produce a STRUCTURED DESCRIPTION of the provided chart image.
Follow this exact template and fill in every field.  If a field is not visible
or not applicable write "N/A".

Chart: <chart title or inferred topic>
Company: <company name if visible, else {company}>
Period: <date range or quarter covered>
Type: <bar | line | pie | stacked_bar | scatter | waterfall | table | other>
X-axis: <label and categories/dates>
Y-axis: <label and unit, e.g. "Revenue ($ billions)">
Key values:
  - <metric>: <value> (<period>)
  - <metric>: <value> (<period>)
  (list 3–6 of the most important data points)
Trend: <1–2 sentences describing the overall trajectory or pattern>
Notable: <any peaks, troughs, inflection points, or anomalies worth highlighting>

Do NOT add any text outside the template.
""").strip()


# ─────────────────────────────────────────────
# Helper: image → base64
# ─────────────────────────────────────────────

def _image_to_base64(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ─────────────────────────────────────────────
# ChartDescriber
# ─────────────────────────────────────────────

class ChartDescriber:
    """
    Uses GPT-4o-mini vision to convert chart images → structured text.

    Args
    ----
    openai_client : an openai.OpenAI() client (already initialised)
    model         : vision model name (default gpt-4o-mini)
    """

    def __init__(
        self,
        openai_client,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.client = openai_client
        self.model = model

    def describe(
        self,
        image_path: str | Path,
        company: str = "",
        caption: str = "",
        page: int = 0,
        section: str = "",
    ) -> str:
        """
        Send the chart image to the VLM and return its structured description.

        Returns an empty string on failure (callers should handle gracefully).
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"Chart image not found: {image_path}")
            return ""

        b64 = _image_to_base64(image_path)

        # Build the user message including any caption/context we already have
        user_content = []
        if caption or section:
            context_hint = ""
            if section:
                context_hint += f"Section: {section}\n"
            if caption:
                context_hint += f"Caption/nearby text: {caption}\n"
            user_content.append(
                {"type": "text", "text": context_hint.strip()}
            )

        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            }
        )

        system = _SYSTEM_PROMPT.replace("{company}", company or "Unknown")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=512,
                temperature=0.0,
            )
            description = response.choices[0].message.content.strip()
            logger.debug(
                f"Described chart on page {page}: {description[:80]}…"
            )
            return description
        except Exception as e:
            logger.error(f"Chart description failed for {image_path}: {e}")
            return ""

    def describe_batch(
        self,
        chart_elements: list[dict],
    ) -> list[dict]:
        """
        Describe a list of chart element dicts (from pdf_parser output).
        Modifies each dict in-place, adding a 'description' key.
        Returns the updated list.

        Each element dict must have at minimum:
            content   : file path to the PNG
            company   : str
            page      : int
            caption   : str
            section   : str
        """
        for i, elem in enumerate(chart_elements):
            logger.info(
                f"Describing chart {i + 1}/{len(chart_elements)} "
                f"({elem.get('company', '')} p{elem.get('page', '')})"
            )
            desc = self.describe(
                image_path=elem["content"],
                company=elem.get("company", ""),
                caption=elem.get("caption", ""),
                page=elem.get("page", 0),
                section=elem.get("section", ""),
            )
            elem["description"] = desc
        return chart_elements
