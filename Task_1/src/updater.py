"""
KnowledgeBaseUpdater is the heart of the "dynamic expansion" mechanism:

For each configured source it will:
  1. Load raw documents from the source (file / web / rss).
  2. Hash each document's content.
  3. Compare against the manifest of previously-seen hashes.
       - unchanged  -> skip (no re-embedding, cheap no-op)
       - new/changed -> chunk, delete old chunks (if any), embed + upsert new chunks
  4. Detect documents that disappeared from the source and remove their
     stale chunks from the vector store.
  5. Record results in the manifest for the next run.

This module can be invoked directly (one-off `update-now`) or triggered
repeatedly by the scheduler.
"""
from __future__ import annotations

import logging
import time

from .config import AppConfig
from .loaders import get_loader
from .manifest import Manifest
from .utils import chunk_text, content_hash
from .vector_store import VectorStore

logger = logging.getLogger("kb_system")


class KnowledgeBaseUpdater:
    def __init__(self, config: AppConfig, vector_store: VectorStore, manifest: Manifest):
        self.config = config
        self.vector_store = vector_store
        self.manifest = manifest

    def update_source(self, source_cfg: dict) -> dict:
        """Runs one full update cycle for a single source. Returns a stats dict."""
        source_id = source_cfg["id"]
        stats = {"source_id": source_id, "new": 0, "updated": 0, "unchanged": 0, "removed": 0, "errors": 0}
        t0 = time.time()

        try:
            loader = get_loader(source_cfg)
            raw_docs = loader.load()
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] loader failed: %s", source_id, exc)
            stats["errors"] += 1
            return stats

        seen_doc_ids = set()
        chunk_cfg = self.config.chunking

        for raw_doc in raw_docs:
            seen_doc_ids.add(raw_doc.doc_id)
            new_hash = content_hash(raw_doc.text)
            old_hash = self.manifest.get_hash(source_id, raw_doc.doc_id)

            if old_hash == new_hash:
                stats["unchanged"] += 1
                continue

            # Content is new or changed -> remove any previous chunks for this doc first.
            old_chunk_ids = self.manifest.get_chunk_ids(source_id, raw_doc.doc_id)
            if old_chunk_ids:
                self.vector_store.delete_chunks(old_chunk_ids)

            pieces = chunk_text(
                raw_doc.text,
                chunk_size=chunk_cfg.get("chunk_size", 1000),
                chunk_overlap=chunk_cfg.get("chunk_overlap", 150),
            )
            if not pieces:
                continue

            chunk_ids = [f"{source_id}::{raw_doc.doc_id}::{i}" for i in range(len(pieces))]
            metadatas = [
                {**raw_doc.metadata, "doc_id": raw_doc.doc_id, "chunk_index": i, "content_hash": new_hash}
                for i in range(len(pieces))
            ]

            try:
                self.vector_store.upsert_chunks(chunk_ids, pieces, metadatas)
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] embedding/upsert failed for %s: %s", source_id, raw_doc.doc_id, exc)
                stats["errors"] += 1
                continue

            self.manifest.upsert(source_id, raw_doc.doc_id, new_hash, chunk_ids)
            if old_hash is None:
                stats["new"] += 1
            else:
                stats["updated"] += 1

        # Remove documents that used to exist for this source but vanished.
        previously_known = set(self.manifest.all_doc_ids_for_source(source_id))
        vanished = previously_known - seen_doc_ids
        for doc_id in vanished:
            stale_chunk_ids = self.manifest.get_chunk_ids(source_id, doc_id)
            self.vector_store.delete_chunks(stale_chunk_ids)
            self.manifest.delete(source_id, doc_id)
            stats["removed"] += 1

        stats["duration_sec"] = round(time.time() - t0, 2)
        logger.info(
            "[%s] update complete: new=%d updated=%d unchanged=%d removed=%d errors=%d (%.2fs)",
            source_id, stats["new"], stats["updated"], stats["unchanged"],
            stats["removed"], stats["errors"], stats["duration_sec"],
        )
        return stats

    def update_all(self) -> list[dict]:
        results = []
        for source_cfg in self.config.sources:
            results.append(self.update_source(source_cfg))
        return results
