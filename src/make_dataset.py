"""
make_dataset.py  -  Phase 1, Task 5
sample_raw.csv (load_data.py ka output) ko clean_text() se guzaar kar
data/processed/cleaned.csv banata hai. Phase 2 isi file ko load karega.

Run (load_data.py ke baad):
    python src/make_dataset.py
"""
from collections import Counter
from pathlib import Path

import pandas as pd

from clean import clean_text

ROOT = Path(__file__).resolve().parents[1]
IN_FILE = ROOT / "data" / "processed" / "sample_raw.csv"
OUT_FILE = ROOT / "data" / "processed" / "cleaned.csv"


def vocab(texts):
    c = Counter()
    for t in texts:
        c.update(t.split())
    return c


def main():
    if not IN_FILE.exists():
        raise SystemExit("Pehle `python src/load_data.py` chalao.")
    df = pd.read_csv(IN_FILE, encoding="utf-8")
    print(f"Loaded {len(df):,} articles")

    # Headline + body dono ko jor kar ek document banate hain
    raw_docs = (df["Headline"].fillna("") + " " + df["News Text"].fillna("")).tolist()

    print("Cleaning ... (kuch minute lag sakte hain)")
    clean_docs = [clean_text(d) for d in raw_docs]

    v_before, v_after = vocab(raw_docs), vocab(clean_docs)
    tok_before = sum(v_before.values())
    tok_after = sum(v_after.values())

    out = pd.DataFrame({"clean_text": clean_docs, "label": df["Category"]})
    out = out[out["clean_text"].str.len() > 0]
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print("\n===== Vocabulary BEFORE vs AFTER cleaning =====")
    print(f"{'':12}{'Tokens':>14}{'Types (vocab)':>16}")
    print(f"{'Before':12}{tok_before:>14,}{len(v_before):>16,}")
    print(f"{'After':12}{tok_after:>14,}{len(v_after):>16,}")
    print(f"Vocab reduction: {100 * (1 - len(v_after) / len(v_before)):.1f}%")
    print(f"\nSaved {len(out):,} rows -> {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
