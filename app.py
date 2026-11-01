"""Marathi news analyser — Streamlit UI.

    streamlit run app.py
"""
from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from mna.graphs import ENTITY_COLORS, bar_figure, keyword_graph_figure, sentence_graph_figure
from mna.pipeline import ModelHub, Options, analyze

st.set_page_config(page_title="मराठी बातमी विश्लेषक", page_icon="📰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Mukta:wght@400;600&family=Tiro+Devanagari+Marathi&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextArea textarea { font-family: 'Mukta', 'Noto Sans Devanagari', sans-serif; }
h1, h2, h3 { font-family: 'Tiro Devanagari Marathi', 'Mukta', serif !important; font-weight: 400 !important; }
.article { font-size: 1.08rem; line-height: 1.9; max-width: 72ch; }
.article p { margin: 0 0 .9em 0; }
mark.ent { padding: 0 .25em; border-radius: 3px; color: inherit; }
mark.ent sub { font-size: .62em; opacity: .75; margin-left: .2em; }
.summary { font-size: 1.15rem; line-height: 1.9; border-left: 4px solid #c2185b; padding: .4em 0 .4em 1em; max-width: 72ch; }
.step-note { color: #6b7280; font-size: .92rem; margin-top: -.4rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_hub() -> ModelHub:
    return ModelHub()


def highlight_entities(text: str, entities: list[dict]) -> str:
    """Wrap entity mentions in coloured <mark> tags, one paragraph per line."""
    names = sorted({e["text"] for e in entities if len(e["text"]) > 1}, key=len, reverse=True)
    label_of = {e["text"]: e["label"] for e in entities}
    paras = []
    for para in text.split("\n"):
        if not para.strip():
            continue
        esc = html.escape(para)
        if names:
            pat = re.compile("|".join(re.escape(html.escape(n)) for n in names))
            esc = pat.sub(lambda m: (
                f'<mark class="ent" style="background:{ENTITY_COLORS.get(label_of.get(html.unescape(m.group(0)), "-"))}33">'
                f'{m.group(0)}<sub>{label_of.get(html.unescape(m.group(0)), "")}</sub></mark>'), esc)
        paras.append(f"<p>{esc}</p>")
    return f'<div class="article">{"".join(paras)}</div>'


# ----------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    engine = st.radio("Browser", ["selenium", "playwright"],
                      index=0 if config.SCRAPER_ENGINE == "selenium" else 1, horizontal=True)
    use_ner = st.toggle("IndicNER model", True, help="Off = rule-based fallback tagger")
    use_keybert = st.toggle("KeyBERT", True, help="Multilingual embeddings for keyword relevance")
    use_semantic = st.toggle("Embeddings in sentence graph", False, help="Needs KeyBERT on")
    use_sent = st.toggle("Sentiment", False)
    use_abs = st.toggle("IndicBART-XLSum abstractive summary", False, help="~1 GB download, slow on CPU")
    n_sum = st.slider("Summary sentences", 1, 6, config.SUMMARY_SENTENCES)
    thr = st.slider("Graph edge threshold", 0.0, 0.4, config.LEXRANK_THRESHOLD, 0.01)
    top_k = st.slider("Keywords", 5, 25, config.TOP_KEYWORDS)

    st.divider()
    st.subheader("Classifier")
    clf_path = Path(config.CLASSIFIER_PATH)
    st.caption("Trained model found." if clf_path.exists() else "No model yet. Upload a headline,label CSV.")
    up = st.file_uploader("Training CSV (headline,label)", type="csv")
    if up and st.button("Train classifier"):
        from mna.classifier import train
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(up.getvalue())
        with st.spinner("Training…"):
            info = train(tmp.name, clf_path.parent)
        get_hub()._cache.clear()
        st.success(f"Trained on {info['n_train']} headlines. Macro-F1 {info['macro_f1']:.3f}")
        st.code(info["report"])

# -------------------------------------------------------------------------- input
st.title("मराठी बातमी विश्लेषक")
st.markdown('<p class="step-note">Paste a Marathi news link. The page is opened in a headless browser, '
            'the article is extracted, and you get entities, category, keywords, key sentences and a '
            'graph-based summary.</p>', unsafe_allow_html=True)

tab_url, tab_text = st.tabs(["From a link", "Paste text"])
with tab_url:
    url = st.text_input("News article URL", placeholder="https://…")
    go_url = st.button("Analyse article", type="primary", key="go_url")
with tab_text:
    title_in = st.text_input("Headline (optional)")
    text_in = st.text_area("Marathi article text", height=200)
    go_text = st.button("Analyse text", type="primary", key="go_text")

opts = Options(classifier_path=config.CLASSIFIER_PATH,
               ner_model=config.NER_MODEL if use_ner else None,
               keybert_model=config.KEYBERT_EMBEDDER if use_keybert else None,
               abstractive_model=config.ABSTRACTIVE_MODEL if use_abs else None,
               sentiment_model=config.SENTIMENT_MODEL if use_sent else None,
               sentiment_labels=config.SENTIMENT_LABELS,
               summary_sentences=n_sum, lexrank_threshold=thr, top_keywords=top_k,
               semantic_graph=use_semantic)

if go_url and url:
    from mna.scraper import NotMarathiError, scrape
    try:
        with st.spinner(f"Opening the page with {engine}…"):
            art = scrape(url, engine=engine)
        st.session_state["article"] = (art.title, art.text, art.method, url)
    except NotMarathiError as e:
        st.error(f"Only Marathi articles are supported. {e}")
    except Exception as e:
        st.error(f"Could not read that page ({type(e).__name__}: {e}). "
                 "Check the link, or copy the article into “Paste text”.")
if go_text and text_in.strip():
    st.session_state["article"] = (title_in, text_in, "pasted", "")

if "article" not in st.session_state:
    st.info("Add a link or paste an article to begin.")
    st.stop()

title, text, method, src = st.session_state["article"]
try:
    with st.spinner("Analysing… the first run downloads models and takes longer."):
        r = analyze(text, title, opts, get_hub())
except ValueError as e:
    st.error(str(e))
    st.stop()

if not r.language["is_marathi"]:
    st.warning(f"This text may not be Marathi: {r.language}")
for k, v in r.notes.items():
    st.caption(f"⚠️ {k}: {v}")

# ------------------------------------------------------------------- 1. article
st.header("1. Extracted article")
st.caption(f"Source: {src or 'pasted text'} · extraction: {method} · {len(r.sentences)} sentences · "
           f"{r.language['n_words']} words")
if title:
    st.subheader(title)
with st.expander("Show article with entities highlighted", expanded=False):
    st.markdown(highlight_entities(text, r.entities), unsafe_allow_html=True)

# -------------------------------------------------------- 2. important words + NER
st.header("2. Important words and named entities")
st.markdown(f'<p class="step-note">NER: {r.ner_method}. Keywords: TF-IDF'
            f'{" + KeyBERT" if use_keybert else ""}, tagged with entity types.</p>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    if r.entities:
        st.dataframe(pd.DataFrame([{"Entity": e["text"], "Type": e["label"], "Mentions": e["count"],
                                    "Confidence": round(e["score"], 2)} for e in r.entities]),
                     hide_index=True, width="stretch")
    else:
        st.write("No entities found.")
with c2:
    st.dataframe(pd.DataFrame([{"Word": k["keyword"], "Score": round(k["score"], 3), "Entity": k["entity"],
                                "Found by": ", ".join(k.get("sources", []))} for k in r.keywords_initial]),
                 hide_index=True, width="stretch")

# -------------------------------------------------------------- 3. classification
st.header("3. Category")
if r.classification:
    c1, c2 = st.columns([1, 2])
    c1.metric("Predicted category", r.classification["label"], f"{r.classification['confidence']:.0%} confidence",
              delta_color="off")
    probs = r.classification["probs"]
    c2.plotly_chart(bar_figure(list(probs), list(probs.values()), "Class probabilities"),
                    width="stretch")
else:
    st.warning("Train the classifier first (sidebar, or `python train_classifier.py --data your.csv`).")
if r.sentiment:
    st.caption(f"Sentiment: **{r.sentiment['overall']}** · distribution {r.sentiment['distribution']}")

# ------------------------------------------------------------ 4. YAKE refinement
st.header("4. Refined keywords (YAKE first)")
st.markdown('<p class="step-note">YAKE ranks phrases; agreement with step 2 and with named entities '
            'raises the score.</p>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
kw = r.keywords_refined
c1.plotly_chart(bar_figure([k["keyword"] for k in kw], [k["score"] for k in kw], "Keyword score",
                           color="#2e8b57"), width="stretch")
c2.plotly_chart(keyword_graph_figure(kw, r.sentences), width="stretch")

# ------------------------------------------------------------- 5. key sentences
st.header("5. Key sentences")
st.dataframe(pd.DataFrame([{"#": d["index"] + 1, "Score": round(d["score"], 3),
                            "LexRank": round(d["lexrank"], 3), "Keyword hits": round(d["keyword_hits"], 2),
                            "Entities": d["entity_hits"], "Sentence": d["sentence"]}
                           for d in r.key_sentences[:8]]),
             hide_index=True, width="stretch")

# ---------------------------------------------------------- 6. graph + summary
st.header("6. Summary")
summary_text = " ".join(d["sentence"] for d in r.summary)
st.markdown(f'<div class="summary">{html.escape(summary_text)}</div>', unsafe_allow_html=True)
st.caption("Extractive: sentences chosen by LexRank centrality, keyword coverage, entities and position "
           f"(sentences {', '.join('S' + str(d['index'] + 1) for d in r.summary)}).")
if r.abstractive_summary:
    st.markdown("**IndicBART-XLSum (abstractive):** " + html.escape(r.abstractive_summary))
st.plotly_chart(sentence_graph_figure(r.graph, {d["index"] for d in r.summary}), width="stretch")

st.download_button("Download results (JSON)", data=pd.Series({
    "title": title, "category": r.classification, "entities": r.entities,
    "keywords": r.keywords_refined, "summary": summary_text,
    "abstractive_summary": r.abstractive_summary}).to_json(force_ascii=False, indent=2),
    file_name="marathi_analysis.json", mime="application/json")
