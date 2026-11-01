"""Keyword extraction.

Step 2 (first pass): TF-IDF over sentences + KeyBERT (multilingual embeddings).
Step 4 (refinement): YAKE ranks first, then agreement with the first pass and
with NER entities lifts or confirms each keyword.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .text_utils import MARATHI_STOPWORDS, stem, stem_phrase, tokenize


def _surface_map(sentences: list[str], max_n: int = 2) -> dict[str, Counter]:
    """stemmed n-gram -> counts of the surface forms that produced it."""
    m: dict[str, Counter] = defaultdict(Counter)
    for s in sentences:
        toks = tokenize(s)
        for n in range(1, max_n + 1):
            for i in range(len(toks) - n + 1):
                gram = toks[i:i + n]
                if gram[0] in MARATHI_STOPWORDS or gram[-1] in MARATHI_STOPWORDS:
                    continue
                m[" ".join(stem(t) for t in gram)][" ".join(gram)] += 1
    return m


def tfidf_keywords(sentences: list[str], top_k: int = 15) -> list[dict]:
    """Treat each sentence as a document; words concentrated in a few sentences
    but repeated across the article score highest."""
    if len(sentences) < 2:
        return []
    vec = TfidfVectorizer(tokenizer=lambda s: [stem(t) for t in tokenize(s, remove_stopwords=True)],
                          token_pattern=None, lowercase=False, ngram_range=(1, 2),
                          min_df=1, sublinear_tf=True)
    X = vec.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=0)).ravel()
    vocab = vec.get_feature_names_out()
    surface = _surface_map(sentences)
    order = np.argsort(-scores)
    out, max_s = [], scores.max() or 1.0
    for idx in order:
        key = vocab[idx]
        if key not in surface or len(key) < 2:
            continue
        out.append({"keyword": surface[key].most_common(1)[0][0], "stem": key,
                    "score": float(scores[idx] / max_s), "source": "tfidf"})
        if len(out) >= top_k:
            break
    return out


def yake_keywords(text: str, top_k: int = 15, max_ngram: int = 3) -> list[dict]:
    import yake

    # YAKE's segmenter knows Latin punctuation, so map the danda to a full stop.
    prepared = text.replace("।", ".").replace("॥", ".")
    extractor = yake.KeywordExtractor(lan="mr", n=max_ngram, dedupLim=0.6, top=top_k * 2,
                                      stopwords=MARATHI_STOPWORDS)
    raw = extractor.extract_keywords(prepared)        # lower YAKE score = better
    out, seen = [], set()
    for kw, sc in sorted(raw, key=lambda x: x[1]):
        toks = tokenize(kw)
        if not toks or toks[0] in MARATHI_STOPWORDS or toks[-1] in MARATHI_STOPWORDS:
            continue
        key = stem_phrase(kw)
        if key in seen:
            continue
        seen.add(key)
        out.append({"keyword": " ".join(toks), "stem": key, "yake_score": float(sc), "source": "yake"})
        if len(out) >= top_k:
            break
    n = len(out)
    for r, item in enumerate(out):                    # rank-based score in [0, 1]
        item["score"] = 1.0 - r / max(n, 1)
    return out


class KeyBERTExtractor:
    """KeyBERT on a multilingual sentence-transformer. Candidates are supplied
    by us because KeyBERT's default tokeniser splits Devanagari vowel signs."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(model_name)
        self.kw_model = KeyBERT(model=self.embedder)

    def extract(self, text: str, candidates: list[str], top_k: int = 15) -> list[dict]:
        candidates = list(dict.fromkeys(c for c in candidates if c.strip()))
        if not candidates:
            return []
        res = self.kw_model.extract_keywords(text, candidates=candidates, top_n=top_k,
                                             use_mmr=True, diversity=0.4)
        return [{"keyword": kw, "stem": stem_phrase(kw), "score": float(sc), "source": "keybert"}
                for kw, sc in res]


def first_pass_keywords(sentences: list[str], text: str, entities: list[dict],
                        keybert: KeyBERTExtractor | None = None, top_k: int = 15) -> list[dict]:
    """Step 2: TF-IDF (+ KeyBERT if available), labelled with NER types."""
    tfidf = tfidf_keywords(sentences, top_k=top_k * 2)
    merged: dict[str, dict] = {}
    for item in tfidf:
        merged[item["stem"]] = {**item, "sources": ["tfidf"]}
    if keybert is not None:
        cands = [t["keyword"] for t in tfidf] + [e["text"] for e in entities]
        for item in keybert.extract(text, cands, top_k=top_k):
            cur = merged.setdefault(item["stem"], {**item, "sources": [], "score": 0.0})
            cur["score"] = max(cur["score"], 0.5 * cur["score"] + 0.5 * item["score"])
            cur["sources"].append("keybert")
    out = sorted(merged.values(), key=lambda d: -d["score"])[:top_k]
    return attach_entity_labels(out, entities)


def attach_entity_labels(keywords: list[dict], entities: list[dict]) -> list[dict]:
    ent_by_stem = {stem_phrase(e["text"]): e["label"] for e in entities}
    ent_tokens = {}
    for e in entities:
        for t in tokenize(e["text"]):
            ent_tokens.setdefault(stem(t), e["label"])
    for kw in keywords:
        label = ent_by_stem.get(kw["stem"])
        if label is None:
            hits = [ent_tokens[s] for s in kw["stem"].split() if s in ent_tokens]
            label = hits[0] if hits else None
        kw["entity"] = label or "-"
    return keywords


def refine_with_yake(text: str, first_pass: list[dict], entities: list[dict],
                     top_k: int = 12) -> list[dict]:
    """Step 4: YAKE leads; consensus with step-2 keywords and entities is a bonus."""
    yk = yake_keywords(text, top_k=top_k * 2)
    prior = {k["stem"]: k for k in first_pass}
    prior_tokens = {t for k in first_pass for t in k["stem"].split()}
    ent_stems = {stem_phrase(e["text"]) for e in entities}
    ent_tokens = {t for s in ent_stems for t in s.split()}

    refined = []
    for item in yk:
        toks = set(item["stem"].split())
        agree = 1.0 if item["stem"] in prior else (0.5 if toks & prior_tokens else 0.0)
        ent = 1.0 if item["stem"] in ent_stems else (0.5 if toks & ent_tokens else 0.0)
        score = 0.6 * item["score"] + 0.25 * agree + 0.15 * ent
        sources = ["yake"] + (prior[item["stem"]]["sources"] if item["stem"] in prior else [])
        refined.append({"keyword": item["keyword"], "stem": item["stem"], "score": round(score, 4),
                        "yake_score": item["yake_score"], "sources": sources})
    refined.sort(key=lambda d: -d["score"])
    kept: list[dict] = []                 # drop near-duplicate overlapping phrases
    for item in refined:
        toks = set(item["stem"].split())
        if any(len(toks & set(k["stem"].split())) / len(toks | set(k["stem"].split())) >= 0.5
               for k in kept):
            continue
        kept.append(item)
        if len(kept) >= top_k:
            break
    return attach_entity_labels(kept, entities)
