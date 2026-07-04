"""
text_processing.py
-------------------
Lightweight, fully-local NLP building blocks that don't require an LLM call:
  - text cleaning
  - TF-IDF based keyword/key-phrase extraction
  - TextRank-style extractive summarization (graph centrality over sentence
    similarity), used as a fast preview summary and as a way to shrink long
    text before handing it to the LLM for abstractive summarization.

These sit alongside llm_engine.py's abstractive summarization/explanation -
the task calls for both "information extraction, summarization" (classic
NLP) and LLM-based explanation generation, so we implement both layers
rather than routing everything through the LLM.
"""

import re
from typing import List

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def extract_keywords(text: str, top_n: int = 8) -> List[str]:
    """TF-IDF keyword extraction over the sentences of a single document.
    Treating each sentence as a 'document' lets TF-IDF surface terms that
    are locally salient rather than just frequent."""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        sentences = [text, text]  # TF-IDF needs at least 2 "documents"

    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_features=2000
    )
    try:
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        return []

    scores = np.asarray(tfidf.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(scores)[::-1]
    top_terms = terms[order][:top_n]
    return list(top_terms)


def extractive_summary(text: str, num_sentences: int = 3) -> str:
    """TextRank-style summary: build a sentence-similarity graph (TF-IDF
    cosine similarity as edge weights) and rank sentences by graph
    centrality (PageRank), then return the top sentences in original order."""
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:num_sentences])

    sim_matrix = cosine_similarity(tfidf)
    np.fill_diagonal(sim_matrix, 0)

    graph = nx.from_numpy_array(sim_matrix)
    try:
        scores = nx.pagerank(graph, max_iter=200)
    except nx.PowerIterationFailedConvergence:
        scores = {i: 1.0 for i in range(len(sentences))}

    ranked_idx = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
    ranked_idx = sorted(ranked_idx)  # restore original reading order
    return " ".join(sentences[i] for i in ranked_idx)
