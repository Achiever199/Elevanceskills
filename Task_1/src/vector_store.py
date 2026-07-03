from __future__ import annotations

import logging
import os

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger("kb_system")


def _build_embedding_function(provider: str, embedding_model: str):
    """
    "default"              -> Chroma's bundled ONNX MiniLM-L6-v2 (no torch, small, fast to cold-start)
    "sentence_transformers" -> full sentence-transformers model (better quality, ~2GB extra, needs torch)
    """
    if provider == "sentence_transformers":
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    return embedding_functions.DefaultEmbeddingFunction()


class VectorStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "default",
    ):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedding_fn = _build_embedding_function(embedding_provider, embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, ids: list[str], texts: list[str], metadatas: list[dict]):
        if not ids:
            return
        # Chroma upsert = insert-or-update, which is exactly what we want for
        # re-embedding a document whose content changed.
        self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    def delete_chunks(self, ids: list[str]):
        if not ids:
            return
        try:
            self.collection.delete(ids=ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed deleting chunk ids %s: %s", ids, exc)

    def query(self, query_text: str, n_results: int = 5, where: dict | None = None):
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
        )

    def count(self) -> int:
        return self.collection.count()
