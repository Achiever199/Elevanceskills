"""
retriever.py
TF-IDF based retrieval over MedQuAD dataset.
Uses sklearn — already installed with streamlit. Zero extra downloads.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from typing import List

INDEX_PATH = "tfidf_index.pkl"


@dataclass
class SearchResult:
    question: str
    answer: str
    focus: str
    qtype: str
    score: float


class MedicalRetriever:
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.records: List[dict] = []

    def build_index(self, csv_path: str = "medquad_data.csv") -> int:
        df = pd.read_csv(csv_path).fillna("")
        self.records = df.to_dict("records")

        # Combine question + focus for richer matching
        corpus = [
            f"{r['question']} {r['focus']} {r['type']}"
            for r in self.records
        ]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            stop_words="english",
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # Save index
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "matrix": self.tfidf_matrix,
                "records": self.records,
            }, f)

        return len(self.records)

    def load_index(self) -> bool:
        if not os.path.exists(INDEX_PATH):
            return False
        with open(INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["matrix"]
        self.records = data["records"]
        return True

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if self.vectorizer is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.01:
                r = self.records[idx]
                results.append(SearchResult(
                    question=r.get("question", ""),
                    answer=r.get("answer", ""),
                    focus=r.get("focus", ""),
                    qtype=r.get("type", ""),
                    score=round(float(scores[idx]), 4),
                ))
        return results
