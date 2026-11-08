"""
load_data.py  -  Phase 1, Task 1
Raw Urdu News Dataset 1M ko load karta hai aur us mein se ek BALANCED,
REPRODUCIBLE sample banata hai (default: har category se 10,000 articles).

Raw file ~2 GB hai, is liye hum usay chunks mein parhte hain taake RAM full na ho.

Run:
    python src/load_data.py
    python src/load_data.py --per-class 5000      # chhota sample
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "urdu-news-dataset-1M.csv"
OUT_FILE = ROOT / "data" / "processed" / "sample_raw.csv"

DATASET_URL = "https://data.mendeley.com/datasets/834vsxnb99"
KEEP_COLS = ["Headline", "News Text", "Category", "Source", "Date", "URL"]
SEED = 42


def balanced_sample(raw_path, per_class, seed=SEED, chunksize=100_000):
    """
    Har category se `per_class` random rows uthata hai.

    Tareeqa: har row ko ek random number dete hain (fixed seed ke saath).
    Har category mein jin rows ke number sab se chhote hain, wahi rakhte hain.
    Ye poori file par uniform random sample deta hai, aur same seed par
    har dafa bilkul same sample banta hai -> reproducible.
    """
    rng = np.random.default_rng(seed)
    kept = None
    total = 0
    for chunk in pd.read_csv(raw_path, chunksize=chunksize, usecols=KEEP_COLS,
                             encoding="utf-8"):
        total += len(chunk)
        chunk = chunk.dropna(subset=["News Text", "Category"])
        chunk["_r"] = rng.random(len(chunk))
        kept = chunk if kept is None else pd.concat([kept, chunk])
        # memory bachane ke liye har chunk ke baad sirf top `per_class` rakho
        kept = kept.sort_values("_r").groupby("Category").head(per_class)
        print(f"  rows read: {total:,}", end="\r")
    print(f"\nTotal rows in raw file: {total:,}")
    sample = kept.sort_values("_r").drop(columns="_r").reset_index(drop=True)
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=10_000)
    args = ap.parse_args()

    if not RAW_FILE.exists():
        raise SystemExit(
            f"Raw file nahi mili: {RAW_FILE}\n"
            f"1) Dataset yahan se download karo: {DATASET_URL}\n"
            f"2) Zip extract kar ke CSV ko data/raw/ mein rakho.")

    print(f"Reading {RAW_FILE.name} in chunks ...")
    sample = balanced_sample(RAW_FILE, args.per_class)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print(f"Saved {len(sample):,} rows -> {OUT_FILE.relative_to(ROOT)}")
    print(sample["Category"].value_counts().to_string())


if __name__ == "__main__":
    main()
