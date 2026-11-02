"""Structure-preserving, table-aware text chunking for local document RAG."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Any

from config import DOCUMENT_CHUNK_OVERLAP, DOCUMENT_CHUNK_SIZE
from document_processing.document_loader import ExtractedPage


@dataclass(frozen=True)
class DocumentChunk:
    """A text chunk with stable order and its optional source page."""

    chunk_index: int
    page_number: int | None
    text: str
    structured_data: tuple[dict[str, Any], ...] = ()


def _is_table_line(line: str) -> bool:
    """Check if a line looks like a markdown or pipe-delimited table row."""
    stripped = line.strip()
    return bool(stripped and "|" in stripped and (stripped.startswith("|") or stripped.count("|") >= 2))


def _split_table(table_text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a long table by rows while repeating the header row on each split chunk."""
    lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
    if not lines:
        return []
    if len(table_text) <= chunk_size:
        return [table_text.strip()]

    # First row is header, second might be markdown divider |---|---|
    header_lines = [lines[0]]
    if len(lines) > 1 and all(c in "-|: " for c in lines[1]):
        header_lines.append(lines[1])
        data_rows = lines[2:]
    else:
        data_rows = lines[1:]

    header_block = "\n".join(header_lines)
    header_len = len(header_block)

    chunks: list[str] = []
    current_rows: list[str] = []
    current_len = header_len

    for row in data_rows:
        row_len = len(row) + 1  # newline
        if current_rows and (current_len + row_len > chunk_size):
            chunk_content = header_block + "\n" + "\n".join(current_rows)
            chunks.append(chunk_content.strip())
            # Overlap: keep last row if it fits
            overlap_rows: list[str] = []
            overlap_len = header_len
            for r in reversed(current_rows):
                if overlap_len + len(r) + 1 <= min(overlap + header_len, chunk_size // 2):
                    overlap_rows.insert(0, r)
                    overlap_len += len(r) + 1
                else:
                    break
            current_rows = list(overlap_rows)
            current_len = overlap_len
        current_rows.append(row)
        current_len += row_len

    if current_rows:
        chunk_content = header_block + "\n" + "\n".join(current_rows)
        chunks.append(chunk_content.strip())

    return chunks or [table_text.strip()]


def _split_prose(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text hierarchically: paragraphs -> newlines -> sentences -> words."""
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text.strip()]

    paragraphs = text.split("\n\n")
    pieces: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            pieces.append(para)
        else:
            lines = para.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if len(line) <= chunk_size:
                    pieces.append(line)
                else:
                    sentences = re.split(r"(?<=[.!?])\s+", line)
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                        if len(sent) <= chunk_size:
                            pieces.append(sent)
                        else:
                            start = 0
                            while start < len(sent):
                                end = min(start + chunk_size, len(sent))
                                if end < len(sent):
                                    space = sent.rfind(" ", start, end)
                                    if space > start:
                                        end = space
                                piece = sent[start:end].strip()
                                if piece:
                                    pieces.append(piece)
                                if end >= len(sent):
                                    break
                                start = max(end - overlap, start + 1)

    chunks: list[str] = []
    current_pieces: list[str] = []
    current_len = 0

    for piece in pieces:
        piece_len = len(piece) + (2 if current_pieces else 0)
        if current_pieces and (current_len + piece_len > chunk_size):
            chunk_text = "\n\n".join(current_pieces).strip()
            if chunk_text:
                chunks.append(chunk_text)
            overlap_pieces: list[str] = []
            overlap_len = 0
            for p in reversed(current_pieces):
                if overlap_len + len(p) + 2 <= overlap:
                    overlap_pieces.insert(0, p)
                    overlap_len += len(p) + 2
                else:
                    break
            current_pieces = list(overlap_pieces)
            current_len = overlap_len

        current_pieces.append(piece)
        current_len += piece_len

    if current_pieces:
        chunk_text = "\n\n".join(current_pieces).strip()
        if chunk_text:
            chunks.append(chunk_text)

    if not chunks and text.strip():
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)

    return chunks


def _split_page(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Structure-preserving page splitting separating tables and prose."""
    if not text:
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    blocks: list[tuple[str, str]] = []
    current_type: str | None = None
    current_lines: list[str] = []

    for line in lines:
        line_type = "table" if _is_table_line(line) else "prose"
        if current_type is None:
            current_type = line_type
            current_lines.append(line)
        elif current_type == line_type:
            current_lines.append(line)
        else:
            block_text = "\n".join(current_lines).strip()
            if block_text:
                blocks.append((current_type, block_text))
            current_type = line_type
            current_lines = [line]

    if current_lines:
        block_text = "\n".join(current_lines).strip()
        if block_text:
            blocks.append((current_type, block_text))

    chunks: list[str] = []
    for b_type, b_text in blocks:
        if b_type == "table":
            chunks.extend(_split_table(b_text, chunk_size, overlap))
        else:
            chunks.extend(_split_prose(b_text, chunk_size, overlap))

    merged_chunks: list[str] = []
    current_chunk = ""
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if not current_chunk:
            current_chunk = c
        elif (
            len(current_chunk) + len(c) + 2 <= chunk_size
            and not (_is_table_line(current_chunk) and _is_table_line(c))
        ):
            current_chunk += "\n\n" + c
        else:
            merged_chunks.append(current_chunk)
            current_chunk = c

    if current_chunk:
        merged_chunks.append(current_chunk)

    return merged_chunks or ([text.strip()] if text.strip() else [])


def chunk_document_pages(
    pages: Iterable[ExtractedPage | dict[str, Any]],
    chunk_size: int = DOCUMENT_CHUNK_SIZE,
    overlap: int = DOCUMENT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Split extracted pages into structure-preserving chunks with metadata."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be in [0, chunk_size).")

    chunks: list[DocumentChunk] = []
    for page in pages:
        if isinstance(page, ExtractedPage):
            page_number, text = page.page_number, page.text
            structured_data = page.structured_data
        else:
            page_number, text = page.get("page_number"), page.get("text", "")
            structured_data = tuple(page.get("structured_data", ()))

        page_chunks = _split_page(text, chunk_size, overlap)
        for chunk_text in page_chunks:
            # Associate structured_data appropriately
            matched_structures = []
            for item in structured_data:
                kind = item.get("kind")
                if kind == "table":
                    headers = item.get("headers", [])
                    if any(h and h in chunk_text for h in headers) or "|" in chunk_text:
                        matched_structures.append(item)
                elif kind == "figure":
                    if item.get("description", "") in chunk_text or "Figure" in chunk_text:
                        matched_structures.append(item)
                else:
                    matched_structures.append(item)

            if not matched_structures and (len(page_chunks) <= 1 or "|" in chunk_text):
                matched_structures = list(structured_data)

            chunks.append(
                DocumentChunk(
                    chunk_index=len(chunks),
                    page_number=page_number,
                    text=chunk_text,
                    structured_data=tuple(matched_structures),
                )
            )
    return chunks
