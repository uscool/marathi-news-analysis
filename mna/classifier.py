from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from .text_utils import normalize, stem_tokens


def _word_tokens(s: str) -> list[str]:
    return stem_tokens(s)


def build_pipeline(hidden: int = 256, seed: int = 42) -> Pipeline:
    features = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1,
                                 max_features=40000, sublinear_tf=True, preprocessor=normalize)),
        ("word", TfidfVectorizer(tokenizer=_word_tokens, token_pattern=None, lowercase=False,
                                 ngram_range=(1, 2), min_df=1, max_features=20000, sublinear_tf=True)),
    ])
    mlp = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu", alpha=1e-4,
                        batch_size=64, learning_rate_init=1e-3, max_iter=60,
                        early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
                        random_state=seed)
    return Pipeline([("features", features), ("mlp", mlp)])


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "headline" not in cols or "label" not in cols:      # headerless file
        df = pd.read_csv(path, header=None, names=["headline", "label"])
    else:
        df = df.rename(columns={cols["headline"]: "headline", cols["label"]: "label"})
    df = df.dropna(subset=["headline", "label"])
    df["headline"] = df["headline"].astype(str).map(normalize)
    df["label"] = df["label"].astype(str).str.strip()
    return df[df["headline"].str.len() > 3].drop_duplicates("headline")


def train(csv_path: str | Path, out_dir: str | Path = "models/classifier",
          hidden: int = 256, test_size: float = 0.15, seed: int = 42) -> dict:
    df = load_dataset(csv_path)
    labels = sorted(df["label"].unique())
    if len(labels) != 3:
        print(f"[classifier] note: dataset has {len(labels)} classes {labels} (expected 3)")
    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(df["headline"], df["label"], test_size=test_size,
                                              random_state=seed, stratify=stratify)
    pipe = build_pipeline(hidden=hidden, seed=seed)
    if len(X_tr) < 200:                     # tiny demo sets can't spare a validation split
        pipe.set_params(mlp__early_stopping=False, mlp__max_iter=200)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    report = classification_report(y_te, pred, zero_division=0)
    macro_f1 = f1_score(y_te, pred, average="macro")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "labels": [str(c) for c in pipe.classes_], "macro_f1": macro_f1},
                out / "classifier.joblib", compress=3)
    return {"report": report, "macro_f1": macro_f1, "n_train": len(X_tr), "n_test": len(X_te),
            "labels": [str(c) for c in pipe.classes_], "path": str(out / "classifier.joblib")}


class NewsClassifier:
    def __init__(self, model_path: str | Path = "models/classifier/classifier.joblib"):
        bundle = joblib.load(model_path)
        self.pipe: Pipeline = bundle["pipeline"]
        self.labels: list[str] = bundle["labels"]
        self.macro_f1 = bundle.get("macro_f1")

    def predict_texts(self, texts: list[str]) -> np.ndarray:
        return self.pipe.predict_proba(texts)

    def classify_article(self, title: str, sentences: list[str], n_sentences: int = 8) -> dict:
        pieces, weights = [], []
        if title:
            pieces.append(title)
            weights.append(2.0)
        for s in sentences[:n_sentences]:
            pieces.append(s)
            weights.append(1.0)
        if not pieces:
            return {"label": None, "probs": {}}
        probs = np.average(self.predict_texts(pieces), axis=0, weights=weights)
        best = int(np.argmax(probs))
        return {"label": self.labels[best], "confidence": float(probs[best]),
                "probs": {l: float(p) for l, p in zip(self.labels, probs)}}
