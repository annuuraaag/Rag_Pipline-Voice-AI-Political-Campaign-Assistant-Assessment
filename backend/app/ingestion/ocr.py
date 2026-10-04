"""OCR for scanned PDF pages and image uploads.

Engine: RapidOCR (PP-OCR models on ONNX Runtime). Apache-2.0, models ship inside the
pip wheel (no runtime download), and it reuses the ONNX Runtime the embedder already
needs. PDF pages are rasterised with pypdfium2.

OCR is a fallback, not the default path: a PDF page is only OCR'd when its text layer
yields almost nothing. It is slow (seconds per page on CPU), but it runs at upload
time and never on the question path.
"""
from __future__ import annotations

import io
import logging
import threading

logger = logging.getLogger(__name__)


class OcrEngine:
    def __init__(self, dpi: int = 200, det_limit_side_len: int = 1600, max_pages: int = 50,
                 max_side_px: int = 2200):
        self.dpi = dpi
        self.max_side_px = max_side_px  # cap: scans saved with odd page sizes would otherwise render huge
        self.det_limit_side_len = det_limit_side_len  # tuned: the default (736) dropped lines on A4 scans
        self.max_pages = max_pages
        self._engine = None
        self._lock = threading.Lock()
        self.status = "not loaded"

    def _get(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            # unclip 2.0 pads each detected line; tighter boxes made the recogniser drop spaces
            # between words ("TheAlliancewilllay…") on our test scans.
            self._engine = RapidOCR(det_limit_side_len=self.det_limit_side_len, det_unclip_ratio=2.0)
            self.status = "ready"
        return self._engine

    @property
    def available(self) -> bool:
        if self.status.startswith("error"):
            return False
        try:
            import pypdfium2  # noqa: F401
            import rapidocr_onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def self_test(self) -> str:
        """Load the models and run one tiny image at startup, so a broken install (missing
        OpenCV system library, bad wheel) shows up in /health instead of on the first scan."""
        if not self.available:
            self.status = "error: rapidocr-onnxruntime / pypdfium2 not installed"
            return self.status
        try:
            from PIL import Image

            self.image_lines(Image.new("RGB", (64, 32), "white"))
            self.status = "ready"
        except Exception as exc:
            self.status = f"error: {exc.__class__.__name__}: {exc}"
            logger.error("OCR self-test failed: %s", self.status)
        return self.status

    def image_lines(self, image) -> list[str]:
        """OCR a PIL image → text lines in reading order (top-to-bottom, left-to-right)."""
        import numpy as np

        with self._lock:
            result, _ = self._get()(np.asarray(image.convert("RGB")), use_cls=False)
        if not result:
            return []
        boxes = []
        for box, text, conf in result:
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            boxes.append((min(ys), max(ys), min(xs), text.strip(), float(conf)))
        boxes.sort(key=lambda b: (b[0], b[2]))
        # Merge fragments whose vertical centres are on the same line.
        lines: list[list[tuple]] = []
        for b in boxes:
            centre, height = (b[0] + b[1]) / 2, b[1] - b[0]
            if lines:
                last = lines[-1][-1]
                if abs(centre - (last[0] + last[1]) / 2) < 0.5 * max(height, 1):
                    lines[-1].append(b)
                    continue
            lines.append([b])
        return [" ".join(f[3] for f in sorted(line, key=lambda f: f[2])) for line in lines if line]

    def pdf_page_lines(self, data: bytes, page_index: int) -> list[str]:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(data)
        try:
            page = pdf[page_index]
            w, h = page.get_size()  # points (1/72 inch)
            scale = min(self.dpi / 72, self.max_side_px / max(w, h))
            image = page.render(scale=scale).to_pil()
        finally:
            pdf.close()
        return self.image_lines(image)

    def image_bytes_lines(self, data: bytes) -> list[str]:
        from PIL import Image

        return self.image_lines(Image.open(io.BytesIO(data)))
