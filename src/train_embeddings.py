"""
train_embeddings.py  -  Phase 2, Part B (Word2Vec CBOW, Skip-gram, fastText) + D1 (save)
Teeno models SAME hyperparameters ke saath train karta hai, taake farq sirf
training ke tareeqe ka ho. Models models/ folder mein save hote hain.

Run:
    python src/train_embeddings.py
"""
import time
from pathlib import Path

import pandas as pd
from gensim.models import FastText, Word2Vec

ROOT = Path(__file__).resolve().parents[1]
CLEAN_FILE = ROOT / "data" / "processed" / "cleaned.csv"
MODELS = ROOT / "models"

# Teeno models ke liye EK jaisi settings
PARAMS = dict(
    vector_size=100,  # har word ka vector 100 numbers ka
    window=5,         # center word ke dono taraf 5 words ko context maana
    min_count=5,      # 5 se kam dafa aane wale words ignore
    epochs=5,         # poore corpus par 5 dafa training
    workers=4,        # 4 CPU threads
    seed=42,          # reproducible
)


def load_sentences():
    """Har document ko tokens ki list bana do (gensim yahi format maangta hai)."""
    df = pd.read_csv(CLEAN_FILE, encoding="utf-8").dropna(subset=["clean_text"])
    return [t.split() for t in df["clean_text"]]


def train(name, cls, sentences, **extra):
    t0 = time.time()
    model = cls(sentences=sentences, **PARAMS, **extra)
    print(f"{name:10s} vocab={len(model.wv):,}  time={time.time() - t0:.1f}s")
    return model


def main():
    sentences = load_sentences()
    n_tokens = sum(len(s) for s in sentences)
    print(f"Corpus: {len(sentences):,} documents, {n_tokens:,} tokens")
    print(f"Hyperparameters: {PARAMS}\n")

    MODELS.mkdir(exist_ok=True)
    cbow = train("CBOW", Word2Vec, sentences, sg=0)            # B1: context -> center
    skip = train("Skip-gram", Word2Vec, sentences, sg=1)       # B2: center -> context
    ft = train("fastText", FastText, sentences, sg=1,          # B4: + character n-grams
               min_n=3, max_n=6, bucket=200_000)  # bucket: n-gram table size (default 2M = 800 MB file)

    cbow.save(str(MODELS / "w2v_cbow.model"))
    skip.save(str(MODELS / "w2v_skipgram.model"))
    ft.save(str(MODELS / "fasttext.model"))
    print("\nSaved -> models/w2v_cbow.model, models/w2v_skipgram.model, models/fasttext.model")


if __name__ == "__main__":
    main()
