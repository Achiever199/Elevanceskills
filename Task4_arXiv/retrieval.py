"""
retrieval.py
------------
Builds a searchable index over paper (title + abstract) text and retrieves
the top-k most relevant papers for a query - the retrieval half of the
Retrieval-Augmented Generation pipeline that grounds the chatbot's answers.

Primary method: sentence-transformer embeddings (semantic search) via the
'all-MiniLM-L6-v2' model - small (~80MB), fast on CPU, downloaded once from
Hugging Face on first use.

Fallback: if sentence-transformers / the model download isn't available
(no internet, offline environment, etc.), we transparently fall back to a
TF-IDF + cosine-similarity index, which is fully local and needs no
download. Either way the app keeps working.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class PaperIndex:
    """Wraps whichever retrieval backend is available behind one interface."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self._docs = (self.df["title"].fillna("") + ". " + self.df["abstract"].fillna("")).tolist()
        self.backend = None
        self._build_index()

    def _build_index(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
            self._doc_embeddings = self._model.encode(
                self._docs, show_progress_bar=False, convert_to_numpy=True
            )
            self.backend = "embeddings"
        except Exception:
            # No internet / model unavailable / package missing -> TF-IDF fallback
            self._vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
            self._doc_matrix = self._vectorizer.fit_transform(self._docs)
            self.backend = "tfidf"

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Returns [(row_index, similarity_score), ...] sorted by relevance."""
        if not query or not query.strip():
            return []

        if self.backend == "embeddings":
            query_vec = self._model.encode([query], convert_to_numpy=True)
            sims = cosine_similarity(query_vec, self._doc_embeddings)[0]
        else:
            query_vec = self._vectorizer.transform([query])
            sims = cosine_similarity(query_vec, self._doc_matrix)[0]

        order = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in order if sims[i] > 0]

    def search_papers(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Convenience wrapper returning a DataFrame of matched papers with a
        `relevance` column, in relevance order."""
        hits = self.search(query, top_k=top_k)
        if not hits:
            return self.df.iloc[0:0].copy()
        idx = [i for i, _ in hits]
        scores = [s for _, s in hits]
        result = self.df.iloc[idx].copy()
        result["relevance"] = scores
        return result
