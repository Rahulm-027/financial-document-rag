"""
pdf_parser.py
=============
Extracts text blocks, tables, and chart/figure regions from financial PDFs.

Produces raw "elements" — dicts with content + metadata — that are
then passed to chunker.py and chart_describer.py.

Modalities extracted:
  - TEXT  : paragraphs, headings, footnotes
  - TABLE : pdfplumber table detection → list[list[str]]
  - CHART : page image crop around detected figure bounding boxes
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import fitz          # PyMuPDF
import pdfplumber
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class RawElement:
    """One extracted unit from a PDF (text block, table, or chart image)."""
    element_type: str          # "text" | "table" | "chart"
    content: Any               # str for text; list[list] for table; PIL.Image for chart
    page: int
    company: str
    document: str
    year: int
    section: str = ""
    caption: str = ""
    bbox: tuple = field(default_factory=tuple)  # (x0, y0, x1, y1)

    def to_dict(self) -> dict:
        d = asdict(self)
        # PIL Images and raw table grids aren't JSON-serialisable — handled downstream
        if self.element_type == "chart":
            d["content"] = "<image>"
        return d


# ─────────────────────────────────────────────
# Heading / section detection helpers
# ─────────────────────────────────────────────

_HEADING_RE = re.compile(
    r'^(?:(?:\d+[\.\)]?\s+)?(?:'
    r'results\s+of\s+operations|financial\s+(highlights|results|statements)|'
    r'revenue|net\s+income|gross\s+profit|operating|risk\s+factors|'
    r'liquidity|management.{0,30}discussion|notes?\s+to|segment|'
    r'overview|outlook|guidance|cash\s+flow'
    r')[^\n]{0,120})',
    re.IGNORECASE,
)

def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > 200:
        return False
    if _HEADING_RE.match(stripped):
        return True
    # All-caps short lines
    if stripped.isupper() and 5 < len(stripped) < 120:
        return True
    return False


def _detect_section(spans: list[dict]) -> str:
    """Walk up through recent spans to find last heading text."""
    for span in reversed(spans):
        if span.get("is_heading"):
            return span["text"][:80]
    return ""


# ─────────────────────────────────────────────
# Figure / chart bounding-box detection
# ─────────────────────────────────────────────

def _get_chart_regions(page: fitz.Page) -> list[fitz.Rect]:
    """
    Heuristic: find image rects on the page.  PyMuPDF can enumerate
    images; for each, we get its bounding rect.  We filter out
    tiny decorative images (logos, bullets) below a minimum area.
    """
    MIN_AREA_FRAC = 0.02      # must be > 2% of page area
    page_area = page.rect.width * page.rect.height
    chart_rects: list[fitz.Rect] = []

    for img_info in page.get_image_info(xrefs=True):
        rect = fitz.Rect(img_info["bbox"])
        area = rect.width * rect.height
        if area / page_area >= MIN_AREA_FRAC:
            chart_rects.append(rect)

    return chart_rects


# ─────────────────────────────────────────────
# Main parser class
# ─────────────────────────────────────────────

class FinancialPDFParser:
    """
    Parses a single financial PDF into RawElements.

    Args
    ----
    pdf_path  : path to the PDF
    company   : company name string (e.g. "Tesla")
    document  : document label (e.g. "10-K 2023")
    year      : integer year
    dpi       : resolution for page rasterisation (chart capture)
    """

    def __init__(
        self,
        pdf_path: str | Path,
        company: str,
        document: str,
        year: int,
        dpi: int = 150,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.company = company
        self.document = document
        self.year = year
        self.dpi = dpi

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

    # ------------------------------------------------------------------
    def parse(self) -> list[RawElement]:
        """Main entry point.  Returns all elements across all pages."""
        elements: list[RawElement] = []

        # pdfplumber for table extraction (more robust)
        with pdfplumber.open(self.pdf_path) as plumber_doc:
            plumber_pages = plumber_doc.pages

            # PyMuPDF for text + images
            with fitz.open(self.pdf_path) as fitz_doc:
                n = len(fitz_doc)
                logger.info(
                    f"Parsing '{self.pdf_path.name}' ({n} pages) — "
                    f"{self.company} {self.year}"
                )

                recent_headings: list[dict] = []  # rolling buffer for section detection

                for page_num in range(n):
                    fitz_page = fitz_doc[page_num]
                    plumber_page = plumber_pages[page_num]
                    page_label = page_num + 1

                    # 1. Tables (via pdfplumber — handles merged cells better)
                    table_elems = self._extract_tables(
                        plumber_page, page_label, recent_headings
                    )
                    elements.extend(table_elems)

                    # Keep track of table bboxes so text extraction skips them
                    table_bboxes = [
                        e.bbox for e in table_elems
                    ]

                    # 2. Text (via PyMuPDF)
                    text_elems = self._extract_text(
                        fitz_page, page_label, table_bboxes, recent_headings
                    )
                    elements.extend(text_elems)

                    # 3. Charts / figures (images on the page)
                    chart_elems = self._extract_charts(
                        fitz_page, page_label, recent_headings
                    )
                    elements.extend(chart_elems)

        logger.info(
            f"Extracted {len(elements)} elements from '{self.pdf_path.name}' "
            f"({sum(1 for e in elements if e.element_type=='text')} text, "
            f"{sum(1 for e in elements if e.element_type=='table')} tables, "
            f"{sum(1 for e in elements if e.element_type=='chart')} charts)"
        )
        return elements

    # ------------------------------------------------------------------
    def _extract_text(
        self,
        page: fitz.Page,
        page_num: int,
        skip_bboxes: list[tuple],
        recent_headings: list[dict],
    ) -> list[RawElement]:
        elements = []
        blocks = page.get_text("blocks", sort=True)  # (x0,y0,x1,y1,text,block_no,block_type)

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()
            if not text or len(text) < 10:
                continue

            # Skip regions covered by detected tables
            if self._overlaps_any(x0, y0, x1, y1, skip_bboxes):
                continue

            is_heading = _looks_like_heading(text)
            span_info = {"text": text, "is_heading": is_heading}

            if is_heading:
                recent_headings.append(span_info)
                if len(recent_headings) > 10:
                    recent_headings.pop(0)

            section = _detect_section(recent_headings)
            elements.append(
                RawElement(
                    element_type="text",
                    content=text,
                    page=page_num,
                    company=self.company,
                    document=self.document,
                    year=self.year,
                    section=section,
                    bbox=(x0, y0, x1, y1),
                )
            )
        return elements

    # ------------------------------------------------------------------
    def _extract_tables(
        self,
        page: pdfplumber.page.Page,
        page_num: int,
        recent_headings: list[dict],
    ) -> list[RawElement]:
        elements = []
        tables = page.extract_tables()
        table_objects = page.find_tables()

        for i, (table, table_obj) in enumerate(
            zip(tables, table_objects)
        ):
            if not table or len(table) < 2:
                continue

            # Clean None cells
            cleaned = [
                [str(cell).strip() if cell else "" for cell in row]
                for row in table
            ]

            # Try to grab a caption from nearby text (above the table)
            bbox = table_obj.bbox  # (x0, top, x1, bottom) in pdfplumber coords
            caption = self._find_caption_near_table(page, bbox)

            section = _detect_section(recent_headings)

            elements.append(
                RawElement(
                    element_type="table",
                    content=cleaned,
                    page=page_num,
                    company=self.company,
                    document=self.document,
                    year=self.year,
                    section=section,
                    caption=caption,
                    bbox=bbox,
                )
            )
        return elements

    # ------------------------------------------------------------------
    def _extract_charts(
        self,
        page: fitz.Page,
        page_num: int,
        recent_headings: list[dict],
    ) -> list[RawElement]:
        elements = []
        chart_rects = _get_chart_regions(page)

        for rect in chart_rects:
            # Render only the chart region at target DPI
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            clip = fitz.Rect(rect)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            # Look for a caption near the figure
            section = _detect_section(recent_headings)
            caption = self._find_chart_caption(page, rect)

            elements.append(
                RawElement(
                    element_type="chart",
                    content=img,
                    page=page_num,
                    company=self.company,
                    document=self.document,
                    year=self.year,
                    section=section,
                    caption=caption,
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                )
            )
        return elements

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _overlaps_any(
        x0: float, y0: float, x1: float, y1: float, bboxes: list[tuple]
    ) -> bool:
        for bx0, by0, bx1, by1 in bboxes:
            if x0 < bx1 and x1 > bx0 and y0 < by1 and y1 > by0:
                return True
        return False

    @staticmethod
    def _find_caption_near_table(
        page: pdfplumber.page.Page, table_bbox: tuple, look_above_px: float = 30.0
    ) -> str:
        """Looks for a short text line just above the table."""
        x0, top, x1, _ = table_bbox
        search_area = page.within_bbox((x0, max(0, top - look_above_px), x1, top))
        words = search_area.extract_text() or ""
        words = words.strip()
        if words and len(words) < 200:
            return words
        return ""

    @staticmethod
    def _find_chart_caption(
        page: fitz.Page, rect: fitz.Rect, look_below_px: float = 40.0
    ) -> str:
        """Looks for a short text block just below the image rect."""
        search_rect = fitz.Rect(
            rect.x0, rect.y1, rect.x1, rect.y1 + look_below_px
        )
        text = page.get_text("text", clip=search_rect).strip()
        if text and len(text) < 200:
            return text
        return ""


# ─────────────────────────────────────────────
# Batch ingestion helper
# ─────────────────────────────────────────────

def parse_all_documents(
    config: dict,
    output_dir: str | Path,
) -> list[RawElement]:
    """
    Parse all documents listed in config['documents'].
    Saves per-doc element lists as JSON (without PIL Images — those are
    saved separately as PNG files for the chart describer).
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_elements: list[RawElement] = []

    for doc_cfg in config.get("documents", []):
        pdf_path = Path(config["ingestion"]["raw_dir"]) / doc_cfg["filename"]
        if not pdf_path.exists():
            logger.warning(f"PDF not found, skipping: {pdf_path}")
            continue

        parser = FinancialPDFParser(
            pdf_path=pdf_path,
            company=doc_cfg["company"],
            document=f"{doc_cfg['company']} {doc_cfg['type']} {doc_cfg['year']}",
            year=doc_cfg["year"],
            dpi=config["ingestion"]["image_resolution_dpi"],
        )
        elements = parser.parse()
        all_elements.extend(elements)

        # Save chart images to disk so chart_describer can read them
        chart_dir = output_dir / "chart_images"
        chart_dir.mkdir(exist_ok=True)
        for elem in elements:
            if elem.element_type == "chart" and isinstance(elem.content, Image.Image):
                img_name = (
                    f"{doc_cfg['company']}_{doc_cfg['year']}_p{elem.page}"
                    f"_bbox{int(elem.bbox[0])}.png"
                )
                elem.content.save(chart_dir / img_name)
                # Replace PIL object with file path for serialisation
                elem.content = str(chart_dir / img_name)

    return all_elements
