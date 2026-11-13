"""
count_vectors.py  -  Phase 2, Part A (BoW, TF-IDF, N-grams) + Part D1 (save)
cleaned.csv (Phase 1 ka output) par count-based representations banata hai,
un ki shape / sparsity print karta hai, aur TF-IDF vectorizer ko disk par save
karta hai taake Phase 3 dobara train kiye bina load kar sake.

Run:
    python src/count_vectors.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
CLEAN_FILE = ROOT / "data" / "processed" / "cleaned.csv"
MODELS = ROOT / "models"

# Hamara text Phase 1 mein pehle hi tokenize ho chuka hai (words space se alag).
# Is liye sklearn ko kehte hain: har space se alag cheez ek token hai
# (default pattern Urdu ke chhote words aur "NUM" ko theek se nahi pakarta).
# lowercase=False: Urdu mein case nahi hota, aur "NUM" ko "num" nahi banana.
TOKEN_PATTERN = r"\S+"
COMMON = dict(token_pattern=TOKEN_PATTERN, lowercase=False)


def load_corpus():
    df = pd.read_csv(CLEAN_FILE, encoding="utf-8").dropna(subset=["clean_text"])
    return df["clean_text"].tolist(), df["label"].tolist()


def sparsity(X):
    """Matrix mein kitne % entries zero hain."""
    total = X.shape[0] * X.shape[1]
    return 100 * (1 - X.nnz / total)


def build_bow(docs, **kw):
    vec = CountVectorizer(**COMMON, **kw)
    X = vec.fit_transform(docs)
    return vec, X


def build_tfidf(docs, **kw):
    # Default sklearn settings: smooth_idf=True, sublinear_tf=False, norm="l2"
    # idf(t) = ln((1 + N) / (1 + df(t))) + 1
    vec = TfidfVectorizer(**COMMON, **kw)
    X = vec.fit_transform(docs)
    return vec, X


def ngram_growth(docs, min_df=1):
    """n = 1, 2, 3 par vocabulary (features) kitni barhti hai."""
    rows = []
    for n, rng in [("unigram", (1, 1)), ("uni+bigram", (1, 2)), ("uni+bi+trigram", (1, 3))]:
        vec, X = build_bow(docs, ngram_range=rng, min_df=min_df)
        rows.append({"n-grams": n, "features": X.shape[1],
                     "non-zero": X.nnz, "sparsity %": round(sparsity(X), 4)})
    return pd.DataFrame(rows)


def main():
    docs, labels = load_corpus()
    print(f"Documents: {len(docs):,}")

    # A1 - Bag of Words
    bow_vec, X_bow = build_bow(docs)
    print(f"\n[A1] BoW matrix shape: {X_bow.shape}  sparsity: {sparsity(X_bow):.4f}%")

    # A2 - TF-IDF  (min_df=2: sirf ek dafa aane wale words hata kar matrix chhota)
    tfidf_vec, X_tfidf = build_tfidf(docs, min_df=2)
    print(f"[A2] TF-IDF matrix shape: {X_tfidf.shape}  sparsity: {sparsity(X_tfidf):.4f}%")

    # A3 - N-gram growth (5k docs par, warna trigrams RAM bhar dete hain)
    sub = docs[:5_000]
    print(f"\n[A3] N-gram growth on {len(sub):,} documents:")
    print(ngram_growth(sub, min_df=1).to_string(index=False))
    print("\n     same, keeping only n-grams seen in >= 5 docs (min_df=5):")
    print(ngram_growth(sub, min_df=5).to_string(index=False))

    # D1 - save
    MODELS.mkdir(exist_ok=True)
    joblib.dump(tfidf_vec, MODELS / "tfidf_vectorizer.joblib")
    joblib.dump(bow_vec, MODELS / "bow_vectorizer.joblib")
    print(f"\nSaved -> models/tfidf_vectorizer.joblib, models/bow_vectorizer.joblib")


if __name__ == "__main__":
    main()
