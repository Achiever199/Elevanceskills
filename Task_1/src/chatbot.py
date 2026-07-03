"""
The chatbot side: retrieves relevant chunks from the (continuously updated)
vector store and, if an ANTHROPIC_API_KEY is present, asks Claude to compose
an answer grounded in that retrieved context. Without an API key it falls
back to returning the raw retrieved passages, so the system is testable
end-to-end with zero external accounts.
"""
from __future__ import annotations

import logging
import os

from .vector_store import VectorStore

logger = logging.getLogger("kb_system")

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. If the context does not contain the answer, say "
    "you don't have enough information yet rather than guessing. Cite "
    "which source each fact came from when possible."
)


class KnowledgeBaseChatbot:
    def __init__(self, vector_store: VectorStore, n_results: int = 5):
        self.vector_store = vector_store
        self.n_results = n_results
        self._anthropic_client = None

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                logger.warning("anthropic package not installed; falling back to retrieval-only mode")

    def retrieve(self, question: str) -> list[dict]:
        results = self.vector_store.query(question, n_results=self.n_results)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            {"text": d, "metadata": m, "distance": dist}
            for d, m, dist in zip(docs, metas, distances)
        ]

    def _format_context(self, passages: list[dict]) -> str:
        blocks = []
        for i, p in enumerate(passages, start=1):
            src = p["metadata"].get("url") or p["metadata"].get("path") or p["metadata"].get("source_id")
            blocks.append(f"[{i}] (source: {src})\n{p['text']}")
        return "\n\n".join(blocks)

    def ask(self, question: str) -> dict:
        passages = self.retrieve(question)

        if not passages:
            return {
                "answer": "I don't have any relevant information in the knowledge base yet.",
                "sources": [],
            }

        if self._anthropic_client is None:
            # Retrieval-only fallback: no LLM configured.
            return {
                "answer": "Here are the most relevant passages I found:\n\n" + self._format_context(passages),
                "sources": [p["metadata"] for p in passages],
            }

        context = self._format_context(passages)
        message = self._anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        answer_text = "".join(block.text for block in message.content if block.type == "text")
        return {"answer": answer_text, "sources": [p["metadata"] for p in passages]}
