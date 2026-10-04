"""Format-specific parsers: PDF, DOCX, Markdown, TXT → ParsedDocument.

Every parser keeps *structure* (page numbers, heading breadcrumbs) as well as text,
because citations need pages and structure-aware chunking needs headings. Most
generic loaders discard both.
"""
from __future__ import annotations

import io
import logging
import re
from collections import Counter
from pathlib import PurePath
from typing import Callable

from app.domain import ParsedBlock, ParsedDocument
from app.ingestion.cleaning import is_noise_line, normalize_text
from app.ingestion.ocr import OcrEngine

TEXT_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}  # OCR'd
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS
OCR_MIN_CHARS = 25  # a PDF page with fewer extractable characters is treated as scanned and OCR'd
# Scans often carry a little real text ("Scanned with CamScanner", a page number, a stamp), so a
# page with a large embedded image and only sparse text is treated as scanned too.
OCR_SPARSE_CHARS = 200
OCR_MIN_IMAGE_PIXELS = 300_000  # ~ 550 x 550 px; logos and icons are far smaller


class DocumentParseError(ValueError):
    """Raised when a file cannot be turned into text (corrupt, encrypted, scanned, empty)."""


# ── Shared helpers ───────────────────────────────────────────────────────
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+(\S.{0,90})$")
_BULLET = re.compile(r"^\s*(?:[•●▪◦\-*–]|\d+[.)])\s+")
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _looks_like_heading(line: str) -> tuple[bool, int]:
    """Heuristic heading detector for layout-less text (PDF/TXT lines).

    Returns (is_heading, level). Numbered headings ("2.1 Primary Health") give a
    level from their numbering; other short, title-cased lines without terminal
    punctuation are treated as level-1 headings.
    """
    s = line.strip()
    if not s or len(s) > 90 or (_BULLET.match(s) and not _NUMBERED_HEADING.match(s)):
        return False, 0
    if s[-1] in ".,;:?!":
        return False, 0
    m = _NUMBERED_HEADING.match(s)
    if m and len(s.split()) <= 10 and m.group(2)[0].isupper():
        return True, m.group(1).count(".") + 1
    words = s.split()
    if len(words) > 8 or not s[0].isupper():
        return False, 0
    caps = sum(1 for w in words if w[0].isupper() or not w[0].isalpha())
    small = {"and", "of", "the", "for", "in", "to", "a", "an", "on", "with", "&"}
    if caps + sum(1 for w in words if w.lower() in small) >= len(words):
        return True, 1
    return False, 0


def _strip_heading_number(text: str) -> str:
    m = _NUMBERED_HEADING.match(text.strip())
    return m.group(2).strip() if m else text.strip()


def _set_heading(stack: list[str], level: int, title: str) -> list[str]:
    level = max(1, level)
    return stack[: level - 1] + [title]


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Minimal `key: value` front matter (no YAML dependency needed)."""
    m = _FRONT_MATTER.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip("\"'")
    return meta, text[m.end():]


def _lines_to_blocks(
    lines: list[str], page: int | None, stack: list[str], title_holder: list[str]
) -> tuple[list[ParsedBlock], list[str]]:
    """Group layout lines into paragraph blocks, tracking headings. Shared by PDF and TXT."""
    blocks: list[ParsedBlock] = []
    buf: list[str] = []
    buf_kind = "paragraph"

    def flush() -> None:
        nonlocal buf, buf_kind
        if buf:
            text = normalize_text(" ".join(buf))
            if text:
                blocks.append(ParsedBlock(text=text, page=page, section_path=list(stack), kind=buf_kind))
        buf, buf_kind = [], "paragraph"

    for raw in lines:
        line = raw.strip()
        if not line:
            flush()
            continue
        if is_noise_line(line):
            continue
        is_heading, level = _looks_like_heading(line)
        if is_heading:
            flush()
            heading = _strip_heading_number(line)
            if not title_holder:
                title_holder.append(heading)        # first heading = document title
            else:
                stack = _set_heading(stack, level, heading)
            continue
        if _BULLET.match(line):
            flush()
            buf_kind = "list_item"
            line = _BULLET.sub("", line)
        buf.append(line)
    flush()
    return blocks, stack


def _table_blocks(rows: list[list[str]], stack: list[str]) -> list[ParsedBlock]:
    """One block per table row, with the column headers repeated in every row.

    "Guntur | Food processing park | 6,500" means little on its own once chunked;
    "District: Guntur | Flagship project: Food processing park | Expected jobs: 6,500"
    stays understandable (and retrievable) wherever the row ends up.
    """
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return []
    header = rows[0]
    has_header = len(rows) > 1 and not any(re.search(r"\d", c) for c in header)
    body = rows[1:] if has_header else rows
    out = []
    for r in body:
        if has_header:
            text = " | ".join(f"{h}: {v}" if h else v for h, v in zip(header, r, strict=False) if v)
        else:
            text = " | ".join(c for c in r if c)
        if text:
            out.append(ParsedBlock(text=normalize_text(text), section_path=list(stack), kind="table_row",
                                   content_type="table"))
    return out


def fix_surrogates(text: str) -> str:
    """PDF text extraction can split an emoji into two UTF-16 halves, or keep only one half.
    Rejoin valid pairs and drop lone halves: tokenizers, JSON and Qdrant all reject them."""
    if not any("\ud800" <= ch <= "\udfff" for ch in text):
        return text
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace").replace("\ufffd", "")


def _finalize(title: str, file_type: str, blocks: list[ParsedBlock], **kw) -> ParsedDocument:
    title = fix_surrogates(title)
    for b in blocks:
        b.text = fix_surrogates(b.text)
        b.section_path = [fix_surrogates(p) for p in b.section_path]
    if not blocks or not any(b.text.strip() for b in blocks):
        raise DocumentParseError("No extractable text found in document.")
    return ParsedDocument(title=title, file_type=file_type, blocks=blocks, **kw)


# ── PDF ──────────────────────────────────────────────────────────────────
def _repeated_lines(pages: list[list[str]]) -> set[str]:
    """Running headers/footers: lines (digits masked) present on ≥60% of pages."""
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({re.sub(r"\d+", "#", ln.strip()) for ln in lines if ln.strip()})
    return {ln for ln, n in counts.items() if n / len(pages) >= 0.6}


def _largest_image_pixels(page) -> int:
    """Pixel count of the biggest image drawn on a PDF page (looks one level into form XObjects)."""
    best = 0

    def visit(resources, depth: int) -> None:
        nonlocal best
        try:
            xobjects = resources.get("/XObject") if resources else None
            if xobjects is None:
                return
            xobjects = xobjects.get_object()
            for name in xobjects:
                obj = xobjects[name].get_object()
                subtype = obj.get("/Subtype")
                if subtype == "/Image":
                    best = max(best, int(obj.get("/Width", 0)) * int(obj.get("/Height", 0)))
                elif subtype == "/Form" and depth < 2:
                    visit(obj.get("/Resources"), depth + 1)
        except Exception:  # odd PDFs: treat as "no image", never fail the parse
            return

    try:
        visit(page.get("/Resources"), 0)
    except Exception:
        return 0
    return best


def _looks_scanned(lines: list[str], page) -> bool:
    chars = sum(len(ln.strip()) for ln in lines)
    if chars < OCR_MIN_CHARS:
        return True
    return chars < OCR_SPARSE_CHARS and _largest_image_pixels(page) >= OCR_MIN_IMAGE_PIXELS


def parse_pdf(data: bytes, filename: str, ocr: OcrEngine | None = None) -> ParsedDocument:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover
        raise DocumentParseError("pypdf is not installed") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise DocumentParseError("PDF is encrypted.") from exc
        pages = [(p.extract_text() or "").splitlines() for p in reader.pages]
    except PdfReadError as exc:
        raise DocumentParseError(f"Malformed PDF: {exc}") from exc
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(f"Could not read PDF: {exc}") from exc

    # Pages with (almost) no text layer, or a big image plus sparse text, are scanned: OCR
    # them page by page, so a mixed document (typed pages + a scanned annex) keeps both.
    ocr_pages: list[int] = []
    ocr_errors: list[str] = []
    for i, lines in enumerate(pages):
        if not _looks_scanned(lines, reader.pages[i]):
            continue
        if ocr is None or not ocr.available or len(ocr_pages) >= ocr.max_pages:
            continue
        try:
            ocr_lines = ocr.pdf_page_lines(data, i)
        except Exception as exc:  # a broken page must not sink the whole document
            logger.warning("OCR failed on %s page %d: %s", filename, i + 1, exc)
            ocr_errors.append(f"page {i + 1}: {exc.__class__.__name__}: {exc}")
            continue
        # Keep the text layer if OCR found less (e.g. a photo with a typed caption).
        if sum(map(len, ocr_lines)) > sum(len(ln.strip()) for ln in lines):
            pages[i] = ocr_lines
            ocr_pages.append(i + 1)

    if not any(ln.strip() for lines in pages for ln in lines):
        if ocr is None or not ocr.available:
            raise DocumentParseError("PDF has no text layer (likely scanned) and OCR is not available.")
        if ocr_errors:
            raise DocumentParseError(f"OCR failed: {ocr_errors[0]}")
        raise DocumentParseError("No text found, even after OCR (blank or unreadable scan).")

    repeated = _repeated_lines(pages)
    title_holder: list[str] = []
    meta_title = ""
    try:
        if reader.metadata and reader.metadata.title:
            meta_title = str(reader.metadata.title).strip()
    except Exception:
        pass

    blocks: list[ParsedBlock] = []
    stack: list[str] = []
    for page_no, lines in enumerate(pages, start=1):
        kept = [ln for ln in lines if re.sub(r"\d+", "#", ln.strip()) not in repeated]
        page_blocks, stack = _lines_to_blocks(kept, page_no, stack, title_holder)
        if page_no in ocr_pages:
            for b in page_blocks:
                b.content_type = "ocr"
        blocks.extend(page_blocks)
    title = meta_title or (title_holder[0] if title_holder else PurePath(filename).stem)
    return _finalize(title, "pdf", blocks, page_count=len(pages), ocr_pages=ocr_pages)


# ── Images (OCR) ─────────────────────────────────────────────────────────
def parse_image(data: bytes, filename: str, ocr: OcrEngine | None = None) -> ParsedDocument:
    if ocr is None or not ocr.available:
        raise DocumentParseError("Image uploads need OCR, which is not available on this server.")
    try:
        lines = ocr.image_bytes_lines(data)
    except Exception as exc:
        raise DocumentParseError(f"Could not read image: {exc}") from exc
    title_holder: list[str] = []
    blocks, _ = _lines_to_blocks(lines, None, [], title_holder)
    for b in blocks:
        b.content_type = "ocr"
    if not blocks:
        raise DocumentParseError("No text found in the image.")
    title = title_holder[0] if title_holder else PurePath(filename).stem
    return _finalize(title, "image", blocks, ocr_pages=[0])


# ── DOCX ─────────────────────────────────────────────────────────────────
def parse_docx(data: bytes, filename: str) -> ParsedDocument:
    try:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:  # pragma: no cover
        raise DocumentParseError("python-docx is not installed") from exc
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentParseError(f"Malformed DOCX: {exc}") from exc

    title = ""
    try:
        title = (document.core_properties.title or "").strip()
    except Exception:
        pass
    stack: list[str] = []
    blocks: list[ParsedBlock] = []

    # Walk the body in order so tables stay next to the paragraphs that describe them.
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, document)
            text = normalize_text(para.text)
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            if style == "Title":
                title = title or text
                continue
            m = re.match(r"Heading (\d)", style)
            if m:
                level = int(m.group(1))
                if not title and level == 1 and not stack:
                    title = text
                    continue
                stack = _set_heading(stack, level, _strip_heading_number(text))
                continue
            kind = "list_item" if "List" in style else "paragraph"
            blocks.append(ParsedBlock(text=text, section_path=list(stack), kind=kind))
        elif tag == "tbl":
            table = Table(child, document)
            rows = []
            for row in table.rows:
                cells = [normalize_text(c.text) for c in row.cells]
                # merged cells repeat their text in python-docx; keep the first occurrence
                rows.append([c if c not in cells[:i] else "" for i, c in enumerate(cells)])
            blocks.extend(_table_blocks(rows, stack))
    return _finalize(title or PurePath(filename).stem, "docx", blocks)


# ── Markdown ─────────────────────────────────────────────────────────────
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_INLINE = [
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
    (re.compile(r"(\*\*|__)(.*?)\1"), r"\2"),
    (re.compile(r"(?<!\w)[*_](.*?)[*_](?!\w)"), r"\1"),
    (re.compile(r"`([^`]*)`"), r"\1"),
]


def _strip_md_inline(text: str) -> str:
    for pat, repl in _MD_INLINE:
        text = pat.sub(repl, text)
    return text


def parse_markdown(data: bytes, filename: str) -> ParsedDocument:
    text = _decode(data)
    front, body = _parse_front_matter(text)
    title = front.get("title", "")
    stack: list[str] = []
    blocks: list[ParsedBlock] = []
    buf: list[str] = []
    buf_kind = "paragraph"
    in_code = False

    table_rows: list[list[str]] = []

    def flush_table() -> None:
        if table_rows:
            blocks.extend(_table_blocks(table_rows, stack))
            table_rows.clear()

    def flush_par() -> None:
        nonlocal buf, buf_kind
        if buf:
            t = normalize_text(_strip_md_inline(" ".join(buf)))
            if t:
                blocks.append(ParsedBlock(text=t, section_path=list(stack), kind=buf_kind))
        buf, buf_kind = [], "paragraph"

    def flush() -> None:
        flush_table()
        flush_par()

    for raw in body.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not line.strip() or line.strip() in ("---", "***"):
            flush()
            continue
        m = _MD_HEADING.match(line) if not in_code else None
        if m:
            flush()
            level, heading = len(m.group(1)), _strip_md_inline(m.group(2)).strip()
            if level == 1 and not title:
                title = heading
                continue
            # H1 is the title, so H2 is section level 1.
            stack = _set_heading(stack, max(1, level - 1), _strip_heading_number(heading))
            continue
        stripped = line.strip()
        if stripped.startswith("|"):
            flush_par()
            cells = [_strip_md_inline(c.strip()) for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue  # table separator row
            table_rows.append(cells)
            continue
        flush_table()
        if _BULLET.match(stripped):
            flush()
            buf_kind = "list_item"
            stripped = _BULLET.sub("", stripped)
        elif stripped.startswith(">"):
            stripped = stripped.lstrip("> ")
        buf.append(stripped)
    flush()
    return _finalize(title or PurePath(filename).stem, "md", blocks, front_matter=front)


# ── TXT ──────────────────────────────────────────────────────────────────
def parse_text(data: bytes, filename: str) -> ParsedDocument:
    text = _decode(data)
    front, body = _parse_front_matter(text)
    title_holder: list[str] = [front["title"]] if front.get("title") else []
    blocks, _ = _lines_to_blocks(body.splitlines(), None, [], title_holder)
    title = title_holder[0] if title_holder else PurePath(filename).stem
    return _finalize(title, "txt", blocks, front_matter=front)


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            text = data.decode(enc)
            if enc == "utf-16" and "\x00" in text:
                continue
            return text
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# ── Registry ─────────────────────────────────────────────────────────────
PARSERS: dict[str, Callable[[bytes, str], ParsedDocument]] = {
    ".docx": parse_docx,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".txt": parse_text,
}


def parse_document(data: bytes, filename: str, ocr: OcrEngine | None = None) -> ParsedDocument:
    """Parse any supported file. `ocr` enables the scanned-page / image fallback."""
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    if not data:
        raise DocumentParseError("File is empty.")
    if ext == ".pdf":
        return parse_pdf(data, filename, ocr)
    if ext in IMAGE_EXTENSIONS:
        return parse_image(data, filename, ocr)
    return PARSERS[ext](data, filename)
