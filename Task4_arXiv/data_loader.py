"""
data_loader.py
---------------
Loads arXiv computer-science paper metadata for the chatbot.

Two data sources are supported:
  1. A prepared subset produced by scripts/prepare_dataset.py from the full
     Kaggle arXiv dataset (data/arxiv_cs_subset.parquet) - used automatically
     if present.
  2. A small bundled demo dataset (data/sample_arxiv_cs.json) - used as a
     fallback so the app works out of the box before anyone downloads the
     ~4GB Kaggle dataset.

Schema (both sources are normalized to this):
    id, title, abstract, authors, categories (space-separated string),
    primary_category, year (int)
"""

import json
import os
from typing import List, Optional

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPARED_PATH = os.path.join(BASE_DIR, "data", "arxiv_cs_subset.parquet")
SAMPLE_PATH = os.path.join(BASE_DIR, "data", "sample_arxiv_cs.json")

REQUIRED_COLUMNS = ["id", "title", "abstract", "authors", "categories", "primary_category", "year"]


def _load_sample() -> pd.DataFrame:
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    return df


def load_papers() -> pd.DataFrame:
    """Load the best available dataset. Returns a DataFrame with
    REQUIRED_COLUMNS. Falls back to the bundled demo sample if the full
    prepared subset hasn't been generated yet."""
    if os.path.exists(PREPARED_PATH):
        df = pd.read_parquet(PREPARED_PATH)
    else:
        df = _load_sample()

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["abstract"] = df["abstract"].fillna("").astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df = df[df["abstract"].str.len() > 0].reset_index(drop=True)
    return df


def is_using_sample_data() -> bool:
    return not os.path.exists(PREPARED_PATH)


def get_available_categories(df: pd.DataFrame) -> List[str]:
    """Flatten the space-separated categories column into a sorted unique list."""
    cats = set()
    for c in df["categories"].dropna():
        for tag in str(c).split():
            if tag.startswith("cs."):
                cats.add(tag)
    return sorted(cats)


def filter_by_categories(df: pd.DataFrame, categories: Optional[List[str]]) -> pd.DataFrame:
    if not categories:
        return df
    mask = df["categories"].apply(lambda c: any(cat in str(c).split() for cat in categories))
    return df[mask].reset_index(drop=True)


def get_paper_by_id(df: pd.DataFrame, paper_id: str) -> Optional[dict]:
    row = df[df["id"] == paper_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()
