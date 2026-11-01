"""Runs the six steps in order. Heavy models are loaded lazily and are optional:
if a model can't be loaded the pipeline carries on and records why."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .keywords import KeyBERTExtractor, first_pass_keywords, refine_with_yake
from .ner import load_ner
from .summarizer import IndicBARTSummarizer, lexrank, score_sentences, select_summary
from .text_utils import language_check, split_sentences


@dataclass
class Options:
    classifier_path: str = "models/classifier/classifier.joblib"
    ner_model: str | None = "ai4bharat/IndicNER"
    keybert_model: str | None = "paraphrase-multilingual-MiniLM-L12-v2"
    abstractive_model: str | None = None
    sentiment_model: str | None = None
    sentiment_labels: dict | None = None
    summary_sentences: int = 3
    lexrank_threshold: float = 0.08
    top_keywords: int = 12
    semantic_graph: bool = False        # blend sentence embeddings into the graph


class ModelHub:
    """Caches loaded models; failures are remembered, not retried every call."""

    def __init__(self):
        self._cache: dict[str, Any] = {}
        self.notes: dict[str, str] = {}

    def get(self, key: str, loader):
        if key not in self._cache:
            try:
                self._cache[key] = loader()
            except Exception as exc:
                self._cache[key] = None
                self.notes[key] = f"{type(exc).__name__}: {exc}"[:300]
        return self._cache[key]


@dataclass
class Result:
    title: str
    text: str
    sentences: list[str]
    language: dict
    entities: list[dict] = field(default_factory=list)
    ner_method: str = ""
    keywords_initial: list[dict] = field(default_factory=list)
    classification: dict = field(default_factory=dict)
    sentiment: dict = field(default_factory=dict)
    keywords_refined: list[dict] = field(default_factory=list)
    key_sentences: list[dict] = field(default_factory=list)
    summary: list[dict] = field(default_factory=list)
    abstractive_summary: str = ""
    graph: Any = None
    notes: dict = field(default_factory=dict)


def analyze(text: str, title: str = "", opts: Options | None = None,
            hub: ModelHub | None = None) -> Result:
    opts = opts or Options()
    hub = hub or ModelHub()
    sentences = split_sentences(text)
    if len(sentences) < 2:
        raise ValueError("Need at least two full sentences of Marathi text.")
    res = Result(title=title, text=text, sentences=sentences, language=language_check(text))

    # Step 2: entities + first-pass important words
    ner = hub.get(f"ner:{opts.ner_model}", lambda: load_ner(opts.ner_model))
    res.entities = ner(text) if ner else []
    res.ner_method = getattr(ner, "method", "none")
    keybert = None
    if opts.keybert_model:
        keybert = hub.get(f"keybert:{opts.keybert_model}", lambda: KeyBERTExtractor(opts.keybert_model))
    res.keywords_initial = first_pass_keywords(sentences, text, res.entities, keybert, opts.top_keywords)

    # Step 3: classification (+ optional sentiment)
    if Path(opts.classifier_path).exists():
        from .classifier import NewsClassifier
        clf = hub.get(f"clf:{opts.classifier_path}", lambda: NewsClassifier(opts.classifier_path))
        if clf:
            res.classification = clf.classify_article(title, sentences)
    else:
        res.notes["classifier"] = "No trained model found. Run train_classifier.py first."
    if opts.sentiment_model:
        from .sentiment import load_sentiment
        sa = hub.get(f"sent:{opts.sentiment_model}",
                     lambda: load_sentiment(opts.sentiment_model, opts.sentiment_labels))
        if sa:
            res.sentiment = sa.analyze(sentences)

    # Step 4: YAKE-first refinement
    res.keywords_refined = refine_with_yake(text, res.keywords_initial, res.entities, opts.top_keywords)

    # Steps 5-6: sentence graph, LexRank, key sentences, summary
    embedder = keybert.embedder if (opts.semantic_graph and keybert) else None
    sg = lexrank(sentences, threshold=opts.lexrank_threshold, embedder=embedder)
    scored = score_sentences(sg, res.keywords_refined, res.entities)
    res.key_sentences = sorted(scored, key=lambda d: -d["score"])
    res.summary = select_summary(scored, sg.similarity, k=opts.summary_sentences)
    res.graph = sg
    if opts.abstractive_model:
        ab = hub.get(f"abs:{opts.abstractive_model}", lambda: IndicBARTSummarizer(opts.abstractive_model))
        if ab:
            top = sorted(res.key_sentences[:6], key=lambda d: d["index"])
            res.abstractive_summary = ab.summarize(" ".join(d["sentence"] for d in top))

    res.notes.update(hub.notes)
    return res
