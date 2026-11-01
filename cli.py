"""Run the pipeline from the terminal.

    python cli.py --url https://marathi.example.com/news/...
    python cli.py --file data/sample_article.txt --no-keybert --no-ner-model
"""
import argparse
import json

import config
from mna.pipeline import Options, analyze

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url")
    src.add_argument("--file")
    ap.add_argument("--engine", default=config.SCRAPER_ENGINE, choices=["selenium", "playwright"])
    ap.add_argument("--no-keybert", action="store_true")
    ap.add_argument("--no-ner-model", action="store_true", help="use the rule-based NER fallback")
    ap.add_argument("--abstractive", action="store_true", help="add IndicBART-XLSum summary")
    a = ap.parse_args()

    title = ""
    if a.url:
        from mna.scraper import scrape
        art = scrape(a.url, engine=a.engine)
        title, text = art.title, art.text
        print(f"[scraped via {art.method}] {title}\n")
    else:
        text = open(a.file, encoding="utf-8").read()

    opts = Options(classifier_path=config.CLASSIFIER_PATH,
                   ner_model=None if a.no_ner_model else config.NER_MODEL,
                   keybert_model=None if a.no_keybert else config.KEYBERT_EMBEDDER,
                   abstractive_model=config.ABSTRACTIVE_MODEL if a.abstractive else None,
                   summary_sentences=config.SUMMARY_SENTENCES,
                   lexrank_threshold=config.LEXRANK_THRESHOLD, top_keywords=config.TOP_KEYWORDS)
    r = analyze(text, title, opts)
    out = {
        "language": r.language,
        "ner_method": r.ner_method,
        "entities": [(e["text"], e["label"], e["count"]) for e in r.entities[:15]],
        "keywords_step2": [(k["keyword"], round(k["score"], 3), k["entity"]) for k in r.keywords_initial],
        "classification": r.classification,
        "keywords_step4_yake": [(k["keyword"], k["score"], k["entity"]) for k in r.keywords_refined],
        "key_sentences": [(d["index"] + 1, round(d["score"], 3), d["sentence"]) for d in r.key_sentences[:5]],
        "summary": " ".join(d["sentence"] for d in r.summary),
        "abstractive_summary": r.abstractive_summary,
        "notes": r.notes,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
