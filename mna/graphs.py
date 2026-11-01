"""Interactive Plotly figures for the Streamlit app."""
from __future__ import annotations

import textwrap
from itertools import combinations

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from .summarizer import SentenceGraph
from .text_utils import stem_tokens

FONT = "Noto Sans Devanagari, Mukta, Tiro Devanagari Marathi, sans-serif"
ENTITY_COLORS = {"PER": "#2f6fb0", "LOC": "#2e8b57", "ORG": "#b5651d", "-": "#8a8f98"}
SUMMARY_COLOR, OTHER_COLOR = "#c2185b", "#9aa5b1"


def _edge_trace(g: nx.Graph, pos: dict) -> list[go.Scatter]:
    traces = []
    weights = [d["weight"] for _, _, d in g.edges(data=True)] or [1]
    wmax = max(weights)
    for u, v, d in g.edges(data=True):
        traces.append(go.Scatter(
            x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]], mode="lines", hoverinfo="skip",
            line=dict(width=0.5 + 4 * d["weight"] / wmax, color="rgba(120,130,145,0.35)"),
            showlegend=False))
    return traces


def sentence_graph_figure(sg: SentenceGraph, summary_idx: set[int]) -> go.Figure:
    g = sg.graph
    pos = nx.spring_layout(g, weight="weight", seed=7, k=1.2 / np.sqrt(max(len(g), 1)))
    lr = sg.lexrank / (sg.lexrank.max() or 1)
    xs, ys, sizes, colors, labels, hovers = [], [], [], [], [], []
    for i in g.nodes():
        xs.append(pos[i][0]); ys.append(pos[i][1])
        sizes.append(14 + 36 * lr[i])
        colors.append(SUMMARY_COLOR if i in summary_idx else OTHER_COLOR)
        labels.append(f"S{i + 1}")
        wrapped = "<br>".join(textwrap.wrap(sg.sentences[i], 60))
        hovers.append(f"<b>S{i + 1}</b> · LexRank {sg.lexrank[i]:.3f}<br>{wrapped}")
    nodes = go.Scatter(x=xs, y=ys, mode="markers+text", text=labels, textposition="middle center",
                       textfont=dict(color="white", size=10), hovertext=hovers, hoverinfo="text",
                       marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
                       showlegend=False)
    fig = go.Figure(_edge_trace(g, pos) + [nodes])
    fig.update_layout(title="Sentence similarity graph (size = LexRank, pink = in summary)",
                      font=dict(family=FONT), height=520, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def keyword_graph_figure(keywords: list[dict], sentences: list[str]) -> go.Figure:
    """Keywords linked when they occur in the same sentence; colour = NER type."""
    g = nx.Graph()
    for k in keywords:
        g.add_node(k["keyword"], score=k["score"], entity=k.get("entity", "-"), stem=set(k["stem"].split()))
    for s in sentences:
        toks = set(stem_tokens(s))
        present = [n for n, d in g.nodes(data=True) if d["stem"] and d["stem"] <= toks]
        for a, b in combinations(present, 2):
            w = g[a][b]["weight"] + 1 if g.has_edge(a, b) else 1
            g.add_edge(a, b, weight=w)
    pos = nx.spring_layout(g, weight="weight", seed=3)
    traces = _edge_trace(g, pos)
    for label, color in ENTITY_COLORS.items():
        ns = [n for n, d in g.nodes(data=True) if d["entity"] == label]
        if not ns:
            continue
        traces.append(go.Scatter(
            x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers+text",
            text=ns, textposition="top center", name={"-": "Keyword"}.get(label, label),
            hovertext=[f"{n} · score {g.nodes[n]['score']:.2f}" for n in ns], hoverinfo="text",
            marker=dict(size=[12 + 26 * g.nodes[n]["score"] for n in ns], color=color,
                        line=dict(width=1, color="white"))))
    fig = go.Figure(traces)
    fig.update_layout(title="Keyword co-occurrence graph (colour = entity type)",
                      font=dict(family=FONT), height=480, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-0.05),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def bar_figure(labels: list[str], values: list[float], title: str, color: str = "#2f6fb0") -> go.Figure:
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=color,
                           text=[f"{v:.2f}" for v in values], textposition="auto"))
    fig.update_layout(title=title, font=dict(family=FONT), height=max(220, 34 * len(labels) + 80),
                      margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(autorange="reversed"),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig
