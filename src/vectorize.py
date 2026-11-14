"""
vectorize.py  -  Phase 2, Part D2
Phase 3 isi file ka vectorize() function call karega.

    from vectorize import vectorize
    v = vectorize("raw Urdu news text ...", method="tfidf")

Raw document -> clean_text() (Phase 1 wali SAME cleaning) -> vector.
Models sirf disk se LOAD hote hain, yahan koi training nahi hoti.
"""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from gensim.models import FastText, Word2Vec

from clean import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
METHODS = ("bow", "tfidf", "cbow", "skipgram", "fasttext")


# method -> (file, loader). Word2Vec/fastText ke liye sirf .wv (word vectors) chahiye.
_FILES = {
    "bow":      ("bow_vectorizer.joblib",   joblib.load),
    "tfidf":    ("tfidf_vectorizer.joblib", joblib.load),
    "cbow":     ("w2v_cbow.model",          lambda p: Word2Vec.load(str(p)).wv),
    "skipgram": ("w2v_skipgram.model",      lambda p: Word2Vec.load(str(p)).wv),
    "fasttext": ("fasttext.model",          lambda p: FastText.load(str(p)).wv),
}


@lru_cache(maxsize=None)          # har model sirf pehli dafa load ho, phir memory se
def _load(method):
    if method not in _FILES:
        raise ValueError(f"method must be one of {METHODS}")
    name, loader = _FILES[method]
    return loader(MODELS / name)


def vectorize(raw_doc, method="tfidf", already_clean=False):
    """
    Ek raw document ka vector.
    - bow / tfidf : sparse vector (1 x vocab size) -> numpy array mein
    - cbow / skipgram : document ke words ke vectors ka AVERAGE (100 numbers);
                        jo word vocab mein nahi, woh skip
    - fasttext : average bhi, lekin unknown words ka vector bhi character
                 n-grams se ban jata hai
    already_clean=True: text pehle se clean hai (jaise cleaned.csv), dobara
    cleaning/stemming nahi hogi.
    """
    tokens = raw_doc.split() if already_clean else clean_text(raw_doc, return_tokens=True)
    model = _load(method)

    if method in ("bow", "tfidf"):
        return model.transform([" ".join(tokens)]).toarray()[0]

    if method == "fasttext":
        vecs = [model[t] for t in tokens]
    else:
        vecs = [model[t] for t in tokens if t in model.key_to_index]
    if not vecs:                                  # koi bhi word pehchana nahi
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)


if __name__ == "__main__":
    doc = "پاکستان کرکٹ ٹیم نے سری لنکا کو 3 وکٹوں سے شکست دے دی۔"
    for m in METHODS:
        v = vectorize(doc, m)
        print(f"{m:9s} shape={v.shape}  non-zero={np.count_nonzero(v)}")
