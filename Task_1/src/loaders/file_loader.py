from __future__ import annotations

import os
import logging

logger = logging.getLogger("kb_system")


class FileLoader:
    """Reads .txt / .md / .pdf files from a directory."""

    def __init__(self, source_id: str, options: dict):
        self.source_id = source_id
        self.path = options.get("path", ".")
        self.extensions = tuple(options.get("extensions", [".txt", ".md", ".pdf"]))
        self.recursive = options.get("recursive", True)

    def _iter_files(self):
        if self.recursive:
            for root, _dirs, files in os.walk(self.path):
                for fn in files:
                    if fn.lower().endswith(self.extensions):
                        yield os.path.join(root, fn)
        else:
            if os.path.isdir(self.path):
                for fn in os.listdir(self.path):
                    full = os.path.join(self.path, fn)
                    if os.path.isfile(full) and fn.lower().endswith(self.extensions):
                        yield full

    @staticmethod
    def _read_pdf(filepath: str) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.warning("pypdf not installed; skipping %s", filepath)
            return ""
        try:
            reader = PdfReader(filepath)
            return "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to parse PDF %s: %s", filepath, exc)
            return ""

    def load(self):
        from . import RawDocument

        docs: list[RawDocument] = []
        if not os.path.exists(self.path):
            logger.warning("[%s] path does not exist: %s", self.source_id, self.path)
            return docs

        for filepath in self._iter_files():
            try:
                if filepath.lower().endswith(".pdf"):
                    text = self._read_pdf(filepath)
                else:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] failed reading %s: %s", self.source_id, filepath, exc)
                continue

            if not text.strip():
                continue

            docs.append(
                RawDocument(
                    doc_id=os.path.abspath(filepath),
                    text=text,
                    metadata={
                        "source_type": "file",
                        "source_id": self.source_id,
                        "path": filepath,
                        "filename": os.path.basename(filepath),
                    },
                )
            )
        logger.info("[%s] loaded %d file(s) from %s", self.source_id, len(docs), self.path)
        return docs
