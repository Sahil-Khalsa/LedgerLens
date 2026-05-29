"""
HTML → PDF → per-page PNG rendering pipeline.

Filings are HTML. ColPali retrieves over page images, so we need per-page PNGs.
Render pipeline: Playwright (HTML→PDF) → PyMuPDF (PDF→PNG per page).

Also detects scanned pages (low OCR text density) and flags them so the eval
can separate visual-only from text-extractable pages.
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)

DEFAULT_DPI = 150  # good quality/size balance; raise to 200 for footnote-heavy filings


def html_to_pdf(html_path: str, pdf_path: str) -> None:
    """Render an HTML filing to PDF using a headless Chromium browser."""
    from playwright.sync_api import sync_playwright  # lazy import — slow to start

    Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
    abs_html = str(Path(html_path).resolve())

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(f"file://{abs_html}", wait_until="networkidle", timeout=60_000)
            page.pdf(path=pdf_path, format="A4", print_background=True)
        finally:
            browser.close()

    logger.info("Rendered PDF: %s", pdf_path)


def pdf_to_pngs(pdf_path: str, out_dir: str, dpi: int = DEFAULT_DPI) -> list[str]:
    """
    Rasterize each page of a PDF to a PNG file.
    Returns list of paths ordered by page index.
    Files: page_0000.png, page_0001.png, ...
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    paths: list[str] = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        out_path = str(Path(out_dir) / f"page_{i:04d}.png")
        pix.save(out_path)
        paths.append(out_path)

    doc.close()
    logger.info("Rendered %d pages from %s", len(paths), pdf_path)
    return paths


def is_scanned_page(png_path: str) -> bool:
    """
    Heuristic: a page with very little extractable text is probably a scanned image.
    Uses pytesseract for OCR text density check.
    Falls back to False if pytesseract is not installed.
    """
    try:
        import pytesseract

        img = Image.open(png_path)
        text = pytesseract.image_to_string(img)
        return len(text.strip()) < 50
    except Exception:
        return False  # if OCR unavailable, assume readable


def render_filing(
    html_path: str,
    accn: str,
    dpi: int = DEFAULT_DPI,
) -> list[dict[str, object]]:
    """
    Full render pipeline for one filing.
    Returns list of page metadata dicts:
        {page_idx, png_path, is_scanned}

    Stores files under:
        data/pdf/{accn_nodashes}/primary.pdf
        data/pages/{accn_nodashes}/page_NNNN.png
    """
    accn_nodashes = accn.replace("-", "")
    pdf_path = str(Path(settings.pdf_dir) / accn_nodashes / "primary.pdf")
    pages_dir = str(Path(settings.pages_dir) / accn_nodashes)

    # HTML → PDF
    try:
        html_to_pdf(html_path, pdf_path)
    except Exception as exc:
        logger.error("HTML→PDF failed for accn=%s: %s", accn, exc)
        return []

    # PDF → PNGs
    png_paths = pdf_to_pngs(pdf_path, pages_dir, dpi=dpi)
    if not png_paths:
        logger.warning("Zero pages rendered for accn=%s", accn)
        return []

    # Scan detection (warn if majority look scanned)
    pages = []
    scanned_count = 0
    for idx, png_path in enumerate(png_paths):
        scanned = is_scanned_page(png_path)
        if scanned:
            scanned_count += 1
        pages.append({"page_idx": idx, "png_path": png_path, "is_scanned": scanned})

    scanned_ratio = scanned_count / len(pages)
    if scanned_ratio > 0.5:
        logger.warning(
            "Filing %s appears to be a scanned document (%.0f%% scanned pages). "
            "VLM extraction may be degraded.",
            accn,
            scanned_ratio * 100,
        )

    return pages


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Render a filing HTML to page PNGs")
    parser.add_argument("--accn", required=True)
    parser.add_argument("--html", required=True, help="Path to the downloaded HTML file")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pages = render_filing(args.html, args.accn, dpi=args.dpi)
    print(json.dumps(pages, indent=2))


if __name__ == "__main__":
    _cli()
