"""Steps 5 and 6: key-sentence extraction and graph-based LexRank summary.

Sentence graph: nodes = sentences, edges = cosine similarity of stemmed TF-IDF
vectors (optionally blended with multilingual sentence embeddings). LexRank is
PageRank on that weighted graph. Final sentence score blends LexRank centrality,
coverage of the YAKE-refined keywords, named-entity mentions and lead position.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_utils import stem_phrase, stem_tokens


@dataclass
class SentenceGraph:
    sentences: list[str]
    similarity: np.ndarray
    graph: nx.Graph
    lexrank: np.ndarray


def similarity_matrix(sentences: list[str], embedder=None, embed_weight: float = 0.5) -> np.ndarray:
    vec = TfidfVectorizer(tokenizer=stem_tokens, token_pattern=None, lowercase=False, sublinear_tf=True)
    try:
        sim = cosine_similarity(vec.fit_transform(sentences))
    except ValueError:                           # every sentence was stopwords
        sim = np.eye(len(sentences))
    if embedder is not None:
        emb = embedder.encode(sentences, normalize_embeddings=True)
        sem = np.clip(emb @ emb.T, 0, 1)
        sim = (1 - embed_weight) * sim + embed_weight * sem
    np.fill_diagonal(sim, 0.0)
    return sim


def lexrank(sentences: list[str], threshold: float = 0.08, damping: float = 0.85,
            embedder=None) -> SentenceGraph:
    sim = similarity_matrix(sentences, embedder)
    g = nx.Graph()
    g.add_nodes_from(range(len(sentences)))
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if sim[i, j] >= threshold:
                g.add_edge(i, j, weight=float(sim[i, j]))
    if g.number_of_edges() == 0:
        scores = np.full(len(sentences), 1.0 / max(len(sentences), 1))
    else:
        pr = nx.pagerank(g, alpha=damping, weight="weight")
        scores = np.array([pr[i] for i in range(len(sentences))])
    return SentenceGraph(sentences, sim, g, scores)


def _norm(x: np.ndarray) -> np.ndarray:
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)


def score_sentences(sg: SentenceGraph, keywords: list[dict], entities: list[dict],
                    weights=(0.55, 0.25, 0.10, 0.10)) -> list[dict]:
    """Step 5: rank every sentence and say why."""
    kw_stems = [(set(k["stem"].split()), k["score"]) for k in keywords]
    ent_stems = [set(stem_phrase(e["text"]).split()) for e in entities]
    cover, ents = [], []
    for s in sg.sentences:
        toks = set(stem_tokens(s))
        hits = [sc for st, sc in kw_stems if st and st <= toks]
        cover.append(sum(hits))
        ents.append(sum(1 for st in ent_stems if st and st <= toks))
    n = len(sg.sentences)
    position = np.array([1.0 / np.sqrt(i + 1) for i in range(n)])   # news lead bias
    w_lr, w_kw, w_ent, w_pos = weights
    final = (w_lr * _norm(sg.lexrank) + w_kw * _norm(np.array(cover, float))
             + w_ent * _norm(np.array(ents, float)) + w_pos * position)
    return [{"index": i, "sentence": sg.sentences[i], "score": float(final[i]),
             "lexrank": float(sg.lexrank[i]), "keyword_hits": float(cover[i]),
             "entity_hits": int(ents[i])} for i in range(n)]


def select_summary(scored: list[dict], sim: np.ndarray, k: int = 3,
                   redundancy: float = 0.5) -> list[dict]:
    """Greedy top-k with a redundancy check (MMR-style), returned in article order."""
    chosen: list[dict] = []
    for item in sorted(scored, key=lambda d: -d["score"]):
        if all(sim[item["index"], c["index"]] < redundancy for c in chosen):
            chosen.append(item)
        if len(chosen) >= k:
            break
    return sorted(chosen, key=lambda d: d["index"])


class IndicBARTSummarizer:
    """Optional abstractive summary with ai4bharat/IndicBART-XLSum.
    Input is the extractive key sentences, which keeps it within the model's
    length limit and anchors the output to the most central content."""

    def __init__(self, model_name: str = "ai4bharat/IndicBART-XLSum", device: str = "cpu"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name, do_lower_case=False,
                                                 use_fast=False, keep_accents=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
        self.device = device
        conv = self.tok._convert_token_to_id_with_added_voc
        self.ids = dict(bos=conv("<s>"), eos=conv("</s>"), pad=conv("<pad>"), lang=conv("<2mr>"))

    def summarize(self, text: str, max_len: int = 80) -> str:
        import torch
        inp = self.tok(f"{text} </s> <2mr>", add_special_tokens=False, return_tensors="pt",
                       truncation=True, max_length=512).input_ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(inp, use_cache=True, num_beams=4, max_length=max_len,
                                      min_length=8, early_stopping=True, no_repeat_ngram_size=3,
                                      encoder_no_repeat_ngram_size=3,
                                      pad_token_id=self.ids["pad"], bos_token_id=self.ids["bos"],
                                      eos_token_id=self.ids["eos"],
                                      decoder_start_token_id=self.ids["lang"])
        return self.tok.decode(out[0], skip_special_tokens=True,
                               clean_up_tokenization_spaces=False).strip()
