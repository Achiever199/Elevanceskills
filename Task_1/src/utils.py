"""Shared utilities: hashing, text chunking, and logging setup."""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Iterable


def content_hash(text: str) -> str:
    """Stable hash used to detect whether a document's content changed."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """
    Simple, dependency-free recursive-ish splitter.
    Splits on paragraph boundaries first, then hard-wraps oversized paragraphs,
    and stitches chunks together up to chunk_size with the requested overlap.
    """
    text = text.strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    buffer = ""

    def flush_buffer():
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    for para in paragraphs:
        # Hard-wrap paragraphs longer than chunk_size on their own.
        while len(para) > chunk_size:
            head, para = para[:chunk_size], para[chunk_size:]
            if buffer:
                flush_buffer()
            chunks.append(head)

        if len(buffer) + len(para) + 2 <= chunk_size:
            buffer = f"{buffer}\n\n{para}" if buffer else para
        else:
            flush_buffer()
            buffer = para

    flush_buffer()

    # Apply overlap by prepending the tail of the previous chunk.
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = overlapped[i - 1][-chunk_overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        chunks = overlapped

    return chunks


def setup_logging(level: str = "INFO", log_file: str = "./logs/kb_system.log") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("kb_system")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger  # already configured (avoid duplicate handlers on reload)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


def batched(iterable: list, n: int) -> Iterable[list]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]
