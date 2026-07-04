"""
prepare_dataset.py
-------------------
Filters the full Kaggle arXiv metadata dataset down to a manageable
computer-science subset and saves it as a Parquet file the app loads
automatically.

--- One-time setup ---
1. Create a free Kaggle account and API token:
   https://www.kaggle.com/settings -> "Create New Token" -> downloads kaggle.json
2. Place it at ~/.kaggle/kaggle.json (chmod 600 on Linux/Mac).
3. pip install kaggle
4. Download the dataset (~4GB compressed):
     kaggle datasets download -d Cornell-University/arxiv -p ./raw_data --unzip
   This produces raw_data/arxiv-metadata-oai-snapshot.json (one JSON object
   per line, ~2.7M+ records across ALL fields, not just CS).

--- Then run this script ---
   python scripts/prepare_dataset.py --input raw_data/arxiv-metadata-oai-snapshot.json

It streams the file in chunks (it's too large to load into memory at once),
keeps only records whose categories include a cs.* tag, keeps the most
recent N per year (configurable) to bound the final size, and writes
data/arxiv_cs_subset.parquet.
"""

import argparse
import json
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "arxiv_cs_subset.parquet")

KEEP_COLUMNS = ["id", "title", "abstract", "authors", "categories", "update_date"]


def _is_cs_paper(categories: str) -> bool:
    return any(tag.startswith("cs.") for tag in (categories or "").split())


def _extract_year(update_date: str):
    try:
        return int(str(update_date)[:4])
    except (ValueError, TypeError):
        return None


def prepare(input_path: str, chunksize: int, max_per_year: int) -> None:
    records = []
    total_seen = 0
    total_kept = 0

    with open(input_path, "r", encoding="utf-8") as f:
        buffer = []
        for line in f:
            buffer.append(line)
            if len(buffer) >= chunksize:
                _process_chunk(buffer, records)
                total_seen += len(buffer)
                buffer = []
                print(f"Scanned {total_seen:,} records, kept {len(records):,} CS papers so far...")
        if buffer:
            _process_chunk(buffer, records)
            total_seen += len(buffer)

    df = pd.DataFrame(records)
    df["year"] = df["update_date"].apply(_extract_year)
    df["primary_category"] = df["categories"].apply(
        lambda c: next((t for t in str(c).split() if t.startswith("cs.")), "cs.misc")
    )
    df = df.dropna(subset=["year"])

    if max_per_year:
        df = (
            df.sort_values("year", ascending=False)
            .groupby("year", group_keys=False)
            .apply(lambda g: g.head(max_per_year))
        )

    df = df.drop(columns=["update_date"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nDone. Scanned {total_seen:,} total records.")
    print(f"Kept {len(df):,} CS papers -> {OUTPUT_PATH}")


def _process_chunk(lines, records) -> None:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not _is_cs_paper(rec.get("categories", "")):
            continue
        records.append({col: rec.get(col) for col in KEEP_COLUMNS})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=os.path.join(BASE_DIR, "raw_data", "arxiv-metadata-oai-snapshot.json"),
        help="Path to the raw Kaggle arxiv-metadata-oai-snapshot.json file",
    )
    parser.add_argument("--chunksize", type=int, default=50000, help="Lines processed per batch")
    parser.add_argument(
        "--max-per-year",
        type=int,
        default=2000,
        help="Cap papers kept per publication year to bound final dataset size (0 = no cap)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(
            f"Input file not found: {args.input}\n"
            "Download it first with the Kaggle CLI - see the docstring at the "
            "top of this script for the exact steps."
        )

    prepare(args.input, args.chunksize, args.max_per_year)
