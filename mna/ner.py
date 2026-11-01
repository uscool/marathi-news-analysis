"""Named-entity recognition for Marathi.

Primary: ai4bharat/IndicNER (PER / LOC / ORG), run sentence by sentence so long
articles stay under the 512-token limit. Fallback: a small rule-based tagger,
so the pipeline still works offline (clearly marked as such in the output).
"""
from __future__ import annotations

import re

from .text_utils import WORD_RE, split_sentences

LABEL_NAMES = {"PER": "Person", "LOC": "Location", "ORG": "Organisation"}


class TransformerNER:
    method = "transformer"

    def __init__(self, model_name: str = "ai4bharat/IndicNER", device: int = -1):
        from transformers import pipeline
        self.model_name = model_name
        self.pipe = pipeline("token-classification", model=model_name, tokenizer=model_name,
                             aggregation_strategy="simple", device=device)

    def __call__(self, text: str, min_score: float = 0.6) -> list[dict]:
        ents, cursor = [], 0
        for sent in split_sentences(text, min_words=1):
            offset = text.find(sent, cursor)
            if offset < 0:
                offset = text.find(sent)
            cursor = max(cursor, offset + len(sent)) if offset >= 0 else cursor
            for e in self.pipe(sent):
                label = e["entity_group"].split("-")[-1]
                if e["score"] < min_score or label not in LABEL_NAMES:
                    continue
                start, end = e["start"], e["end"]
                ents.append({"text": sent[start:end].strip(), "label": label,
                             "score": float(e["score"]),
                             "start": (offset + start) if offset >= 0 else None})
        return _merge(ents)


class RuleNER:
    """Offline fallback: honorifics/titles -> PER, place suffixes -> LOC,
    organisation words -> ORG. Low recall; replace with IndicNER when possible."""
    method = "rule-based fallback"

    PER_TRIGGERS = r"(?:श्री\.?|श्रीमती|सौ\.|डॉ\.|प्रा\.|मंत्री|मुख्यमंत्री|उपमुख्यमंत्री|खासदार|आमदार|जिल्हाधिकारी|आयुक्त|अध्यक्ष|कर्णधार|अभिनेता|अभिनेत्री)"
    LOC_SUFFIX = re.compile(r".+(पूर|पुर|नगर|बाद|गाव|वाडी|वाडा|गड|घर|ाबाद)$")
    LOC_WORDS = {"महाराष्ट्र", "मुंबई", "पुणे", "नागपूर", "नाशिक", "ठाणे", "कोल्हापूर", "सातारा", "सांगली",
                 "सोलापूर", "औरंगाबाद", "छत्रपती संभाजीनगर", "कोकण", "मराठवाडा", "विदर्भ", "दिल्ली",
                 "भारत", "गोवा", "कर्नाटक", "गुजरात", "रत्नागिरी", "रायगड", "अमरावती", "लातूर", "जळगाव"}
    ORG_WORDS = re.compile(r"(पक्ष|महामंडळ|मंडळ|संघटना|समिती|विद्यापीठ|महापालिका|महानगरपालिका|"
                           r"परिषद|बँक|कंपनी|विभाग|आयोग|न्यायालय|दल|संस्था)")

    def __call__(self, text: str) -> list[dict]:
        from .text_utils import stem
        ents = []
        from .text_utils import MARATHI_STOPWORDS
        for m in re.finditer(r"(?:" + self.PER_TRIGGERS + r"\s*)+((?:[\u0900-\u097F]+\s?){1,3})", text):
            words = WORD_RE.findall(m.group(1))
            name = []
            for w in words:                       # stop at a stopword or case marker
                if w in MARATHI_STOPWORDS:
                    break
                name.append(stem(w))
                if stem(w) != w:
                    break
            if name:
                ents.append({"text": " ".join(name[:3]), "label": "PER", "score": 0.5, "start": m.start(1)})
        for m in WORD_RE.finditer(text):
            w, s = m.group(0), stem(m.group(0))
            if s in self.LOC_WORDS or (len(s) > 4 and self.LOC_SUFFIX.match(s)):
                ents.append({"text": s, "label": "LOC", "score": 0.5, "start": m.start()})
        for m in re.finditer(r"((?:[\u0900-\u097F]+\s){0,2}[\u0900-\u097F]*" + self.ORG_WORDS.pattern + r")", text):
            ents.append({"text": m.group(1).strip(), "label": "ORG", "score": 0.4, "start": m.start()})
        return _merge(ents)


def _merge(ents: list[dict]) -> list[dict]:
    """Deduplicate by (text, label), keep best score, count mentions."""
    out: dict[tuple, dict] = {}
    for e in ents:
        if len(e["text"]) < 2:
            continue
        key = (e["text"], e["label"])
        if key in out:
            out[key]["count"] += 1
            out[key]["score"] = max(out[key]["score"], e["score"])
        else:
            out[key] = {**e, "count": 1}
    return sorted(out.values(), key=lambda d: (-d["count"], -d["score"]))


def load_ner(model_name: str | None = "ai4bharat/IndicNER"):
    if model_name:
        try:
            return TransformerNER(model_name)
        except Exception as exc:  # no network / no torch -> fallback
            print(f"[ner] transformer NER unavailable ({exc}); using rule-based fallback")
    return RuleNER()
