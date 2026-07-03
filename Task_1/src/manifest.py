"""
Tracks a content hash + chunk-id list per (source_id, doc_id) so that repeated
updates only re-embed documents that actually changed, and so that stale
chunks can be deleted from the vector store when a document changes or
disappears from its source.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager


class Manifest:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    source_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    chunk_ids TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_id, doc_id)
                )
                """
            )

    def get_hash(self, source_id: str, doc_id: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT content_hash FROM documents WHERE source_id=? AND doc_id=?",
                (source_id, doc_id),
            ).fetchone()
            return row[0] if row else None

    def get_chunk_ids(self, source_id: str, doc_id: str) -> list[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT chunk_ids FROM documents WHERE source_id=? AND doc_id=?",
                (source_id, doc_id),
            ).fetchone()
            return json.loads(row[0]) if row else []

    def upsert(self, source_id: str, doc_id: str, content_hash: str, chunk_ids: list[str]):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO documents (source_id, doc_id, content_hash, chunk_ids, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(source_id, doc_id)
                DO UPDATE SET content_hash=excluded.content_hash,
                              chunk_ids=excluded.chunk_ids,
                              updated_at=CURRENT_TIMESTAMP
                """,
                (source_id, doc_id, content_hash, json.dumps(chunk_ids)),
            )

    def delete(self, source_id: str, doc_id: str):
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM documents WHERE source_id=? AND doc_id=?", (source_id, doc_id)
            )

    def all_doc_ids_for_source(self, source_id: str) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT doc_id FROM documents WHERE source_id=?", (source_id,)
            ).fetchall()
            return [r[0] for r in rows]
