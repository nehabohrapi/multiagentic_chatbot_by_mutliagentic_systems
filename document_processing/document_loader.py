import json
import csv
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docx import Document
from pypdf import PdfReader

from document_processing.text_cleaner import clean_document_text

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


class DocumentExtractionError(ValueError):
    """Raised when a supported document cannot provide usable text."""


@dataclass(frozen=True)
class ExtractedPage:
    """Extracted text and its optional original page number."""

    page_number: int | None
    text: str
    structured_data: tuple[dict[str, Any], ...] = ()


def _table_block(
    rows: list[list[str]],
    *,
    source: str,
    page_number: int | None,
) -> dict[str, Any] | None:
    normalized = [
        [str(cell).strip() for cell in row]
        for row in rows
        if any(str(cell).strip() for cell in row)
    ]
    if not normalized:
        return None
    width = max(len(row) for row in normalized)
    padded = [row + [""] * (width - len(row)) for row in normalized]
    return {
        "kind": "table",
        "source": source,
        "page_number": page_number,
        "headers": padded[0],
        "rows": padded[1:] if len(padded) > 1 else [],
    }


def _structured_text(block: dict[str, Any]) -> str:
    if block.get("kind") == "figure":
        caption = block.get("description", "Figure")
        return f"\n\n[Figure on Page {block.get('page_number')}: {caption}]\n\n"
    if block.get("kind") != "table":
        return json.dumps(block, ensure_ascii=False)
    headers = block.get("headers", [])
    rows = block.get("rows", [])
    if not headers:
        return ""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_pdf(path: Path) -> list[ExtractedPage]:
    pages: list[ExtractedPage] = []

    # Step 1: High-speed primary extraction with PyMuPDF (fitz)
    try:
        import fitz

        with fitz.open(str(path)) as doc:
            for index, page in enumerate(doc, start=1):
                page_text = clean_document_text(page.get_text("text") or "")
                blocks = []
                if page_text and any(c in page_text for c in ("|", "\t", "  ")):
                    try:
                        tables = page.find_tables()
                        for table in tables:
                            df_rows = table.extract()
                            if df_rows and len(df_rows) >= 2:
                                block = _table_block(df_rows, source=path.name, page_number=index)
                                if block:
                                    blocks.append(block)
                    except Exception:
                        pass
                if blocks:
                    table_md = "\n\n".join(_structured_text(b) for b in blocks)
                    page_text = clean_document_text(f"{page_text}\n\n{table_md}")

                pages.append(ExtractedPage(page_number=index, text=page_text, structured_data=tuple(blocks)))
    except Exception:
        pages = []

    # Step 2: Fallback to pypdf / pdfplumber if PyMuPDF was unavailable or empty
    if not pages or not any(p.text.strip() for p in pages):
        import logging
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        try:
            reader = PdfReader(str(path))
            pages = [
                ExtractedPage(
                    page_number=index,
                    text=clean_document_text(page.extract_text() or ""),
                )
                for index, page in enumerate(reader.pages, start=1)
            ]
        except Exception as exc:
            raise DocumentExtractionError(f"Could not extract PDF '{path.name}': {exc}") from exc

        try:
            import pdfplumber

            with pdfplumber.open(str(path)) as pdf:
                extracted: list[ExtractedPage] = []
                for index, page in enumerate(pdf.pages, start=1):
                    current = pages[index - 1] if index <= len(pages) else ExtractedPage(index, "")
                    blocks = []
                    for table in page.extract_tables() or []:
                        block = _table_block(table, source=path.name, page_number=index)
                        if block:
                            blocks.append(block)
                    text = current.text
                    if blocks:
                        text = clean_document_text(
                            "\n\n".join([text, *(_structured_text(block) for block in blocks)])
                        )
                    extracted.append(ExtractedPage(index, text, tuple(blocks)))
                pages = extracted
        except Exception:
            pass


    # Extract images and figures if enabled
    try:
        from config import (
            ENABLE_IMAGE_EXTRACTION,
            EXTRACTED_IMAGES_DIR,
            MAX_EXTRACTED_VISUALS_PER_DOCUMENT,
        )

        if ENABLE_IMAGE_EXTRACTION:
            import fitz
            from PIL import Image

            doc = fitz.open(str(path))
            visual_count = 0
            for page_idx, page in enumerate(doc, start=1):
                image_list = page.get_images(full=True)
                for img_idx, img_info in enumerate(
                    image_list[: min(5, MAX_EXTRACTED_VISUALS_PER_DOCUMENT - visual_count)],
                    start=1,
                ):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image")
                    image_ext = base_image.get("ext", "png")
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    if width >= 120 and height >= 120 and image_bytes:
                        safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", path.stem)
                        path_hash = hashlib.sha256(
                            str(path.resolve()).encode("utf-8")
                        ).hexdigest()[:8]
                        image_hash = hashlib.sha256(image_bytes).hexdigest()[:8]
                        img_filename = (
                            f"{safe_stem}_{path_hash}_p{page_idx}_img{img_idx}_"
                            f"{image_hash}.{image_ext}"
                        )
                        img_path = EXTRACTED_IMAGES_DIR / img_filename
                        if not img_path.exists():
                            img_path.write_bytes(image_bytes)

                        ocr_text = ""
                        try:
                            import pytesseract

                            ocr_text = pytesseract.image_to_string(
                                Image.open(io.BytesIO(image_bytes))
                            ).strip()
                        except Exception:
                            ocr_text = ""

                        desc = f"Image/Chart ({width}x{height}px)"
                        if ocr_text:
                            first_ocr = " ".join(ocr_text.split()[:40])
                            desc += f" containing text: {first_ocr}"

                        fig_block = {
                            "kind": "figure",
                            "source": path.name,
                            "page_number": page_idx,
                            "image_path": str(img_path),
                            "description": desc,
                        }

                        if page_idx <= len(pages):
                            cur = pages[page_idx - 1]
                            fig_text = f"\n\n[Figure on Page {page_idx}: {desc}]\n\n"
                            new_text = cur.text + fig_text
                            new_blocks = cur.structured_data + (fig_block,)
                            pages[page_idx - 1] = ExtractedPage(page_idx, new_text, new_blocks)
                        visual_count += 1

                # Vector charts and diagrams are not returned by get_images().
                # Render complex drawing pages so they remain available to vision.
                if (
                    visual_count < MAX_EXTRACTED_VISUALS_PER_DOCUMENT
                    and len(page.get_drawings()) >= 12
                ):
                    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", path.stem)
                    path_hash = hashlib.sha256(
                        str(path.resolve()).encode("utf-8")
                    ).hexdigest()[:8]
                    rendered = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                    image_bytes = rendered.tobytes("png")
                    image_hash = hashlib.sha256(image_bytes).hexdigest()[:8]
                    img_path = EXTRACTED_IMAGES_DIR / (
                        f"{safe_stem}_{path_hash}_p{page_idx}_chart_{image_hash}.png"
                    )
                    if not img_path.exists():
                        img_path.write_bytes(image_bytes)
                    figure = {
                        "kind": "figure",
                        "source": path.name,
                        "page_number": page_idx,
                        "image_path": str(img_path),
                        "description": "Rendered PDF page with vector graphics; inspect for charts or diagrams.",
                    }
                    if page_idx <= len(pages):
                        cur = pages[page_idx - 1]
                        new_text = (
                            cur.text
                            + f"\n\n[Figure on Page {page_idx}: {figure['description']}]\n\n"
                        )
                        pages[page_idx - 1] = ExtractedPage(
                            page_idx,
                            new_text,
                            cur.structured_data + (figure,),
                        )
                    visual_count += 1
            doc.close()
    except Exception:
        pass

    if not pages or not any(page.text for page in pages):
        try:
            import fitz
            import pytesseract

            with fitz.open(str(path)) as document:
                pages = []
                for index, page in enumerate(document, start=1):
                    image = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    from PIL import Image

                    text = pytesseract.image_to_string(
                        Image.open(io.BytesIO(image.tobytes("png")))
                    )
                    pages.append(ExtractedPage(index, clean_document_text(text)))
        except ImportError as exc:
            raise DocumentExtractionError(
                f"PDF '{path.name}' has no extractable text; install OCR dependencies "
                "(PyMuPDF, Pillow, pytesseract, and Tesseract) to process scanned PDFs."
            ) from exc
        except Exception as exc:
            raise DocumentExtractionError(
                f"Could not OCR scanned PDF '{path.name}': {exc}"
            ) from exc
    return pages


def _json_page(value: Any, default_page_number: int | None) -> ExtractedPage:
    if isinstance(value, str):
        return ExtractedPage(default_page_number, clean_document_text(value))
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        raise DocumentExtractionError(
            "Each JSON page must be a text string or an object with a string 'text' field."
        )

    page_number = value.get("page_number", default_page_number)
    if page_number is not None and (
        isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1
    ):
        raise DocumentExtractionError("JSON page_number values must be positive integers.")
    return ExtractedPage(page_number, clean_document_text(value["text"]))


def _extract_json(path: Path) -> list[ExtractedPage]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DocumentExtractionError(
            f"Could not read valid JSON from '{path.name}': {exc}"
        ) from exc

    if isinstance(payload, list):
        rows = [
            [json.dumps(value, ensure_ascii=False)]
            if isinstance(value, dict)
            else [str(value)]
            for value in payload
        ]
        block = _table_block(rows, source=path.name, page_number=None)
        return [ExtractedPage(None, _structured_text(block), (block,))] if block else []
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
        headers = sorted({key for record in records if isinstance(record, dict) for key in record})
        rows = [[str(record.get(header, "")) for header in headers] for record in records]
        block = _table_block([headers, *rows], source=path.name, page_number=None)
        return [ExtractedPage(None, _structured_text(block), (block,))] if block else []
    if isinstance(payload, dict) and isinstance(payload.get("pages"), list):
        pages = [
            _json_page(value, default_page_number=index)
            for index, value in enumerate(payload["pages"], start=1)
        ]
    elif isinstance(payload, dict) and isinstance(payload.get("text"), str):
        pages = [_json_page(payload, default_page_number=None)]
    else:
        raise DocumentExtractionError(
            "JSON document must contain a string 'text' field or a 'pages' array."
        )

    if not pages or not any(page.text for page in pages):
        raise DocumentExtractionError(f"JSON document '{path.name}' contains no usable text.")
    return pages


def _extract_docx(path: Path) -> list[ExtractedPage]:
    try:
        document = Document(str(path))
        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]
        blocks = []
        for table in document.tables:
            block = _table_block(
                [[cell.text for cell in row.cells] for row in table.rows],
                source=path.name,
                page_number=None,
            )
            if block:
                blocks.append(block)
    except Exception as exc:
        raise DocumentExtractionError(
            f"Could not extract DOCX '{path.name}': {exc}"
        ) from exc

    text = clean_document_text(
        "\n".join(paragraphs + [_structured_text(block) for block in blocks])
    )
    if not text:
        raise DocumentExtractionError(f"DOCX '{path.name}' contains no usable text.")
    return [ExtractedPage(page_number=None, text=text, structured_data=tuple(blocks))]


def _extract_csv(path: Path) -> list[ExtractedPage]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.reader(source))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DocumentExtractionError(f"Could not read CSV '{path.name}': {exc}") from exc
    block = _table_block(rows, source=path.name, page_number=None)
    if not block:
        raise DocumentExtractionError(f"CSV '{path.name}' contains no usable rows.")
    return [ExtractedPage(None, _structured_text(block), (block,))]


def _extract_text_file(path: Path) -> list[ExtractedPage]:
    try:
        text = clean_document_text(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError) as exc:
        raise DocumentExtractionError(
            f"Could not read text document '{path.name}': {exc}"
        ) from exc
    if not text:
        raise DocumentExtractionError(f"Document '{path.name}' contains no usable text.")
    return [ExtractedPage(page_number=None, text=text)]


def _extract_image(path: Path) -> list[ExtractedPage]:
    """Create searchable image evidence while retaining the original image."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or path.suffix.lstrip(".").upper()
            image.verify()
    except Exception as exc:
        raise DocumentExtractionError(
            f"Could not read image '{path.name}': {exc}"
        ) from exc

    ocr_text = ""
    try:
        import pytesseract

        with Image.open(path) as image:
            ocr_text = pytesseract.image_to_string(image).strip()
    except Exception:
        # OCR is an optional indexing aid; the original image is analyzed on demand.
        pass

    description = f"Image asset ({image_format}, {width}x{height}px)"
    if ocr_text:
        description += f" with visible text: {' '.join(ocr_text.split())[:1000]}"
    figure = {
        "kind": "figure",
        "source": path.name,
        "page_number": None,
        "image_path": str(path.resolve()),
        "description": description,
    }
    text = f"[Image: {description}]"
    return [ExtractedPage(None, text, (figure,))]


def extract_document_pages(path: str | Path) -> list[ExtractedPage]:
    """Extract PDF, JSON, DOCX, TXT, or Markdown text with source page metadata.

    JSON accepts either {"text": "..."} or
    {"pages": [{"page_number": 1, "text": "..."}]}.
    """
    document_path = Path(path)
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(document_path)
    if suffix == ".json":
        return _extract_json(document_path)
    if suffix == ".docx":
        return _extract_docx(document_path)
    if suffix == ".csv":
        return _extract_csv(document_path)
    if suffix in {".txt", ".md"}:
        return _extract_text_file(document_path)
    if suffix in IMAGE_EXTENSIONS:
        return _extract_image(document_path)
    raise DocumentExtractionError(
        f"Unsupported extraction format '{suffix or 'no extension'}'; "
        "expected PDF, JSON, DOCX, TXT, or Markdown."
    )
