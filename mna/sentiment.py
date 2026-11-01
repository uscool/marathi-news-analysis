from __future__ import annotations

from collections import Counter


class SentimentAnalyzer:
    def __init__(self, model_name: str, label_map: dict | None = None, device: int = -1):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=model_name, tokenizer=model_name,
                             device=device, truncation=True, max_length=256)
        self.label_map = label_map or {}

    def analyze(self, sentences: list[str]) -> dict:
        if not sentences:
            return {}
        preds = self.pipe(sentences, batch_size=16)
        rows, totals = [], Counter()
        for s, p in zip(sentences, preds):
            label = self.label_map.get(p["label"], p["label"])
            rows.append({"sentence": s, "label": label, "score": float(p["score"])})
            totals[label] += p["score"]
        overall = max(totals, key=totals.get)
        share = {k: round(v / sum(totals.values()), 3) for k, v in totals.items()}
        return {"overall": overall, "distribution": share, "sentences": rows}


def load_sentiment(model_name: str | None, label_map: dict | None = None):
    if not model_name:
        return None
    try:
        return SentimentAnalyzer(model_name, label_map)
    except Exception as exc:
        print(f"[sentiment] unavailable ({exc}); skipping")
        return None
