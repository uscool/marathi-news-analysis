"""Marathi text utilities: normalisation, Marathi-vs-Hindi check, sentence
splitting, tokenisation, stopwords and a light suffix stemmer."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Devanagari letters + vowel signs (excludes danda U+0964/0965 and digits U+0966-096F)
_LETTERS = "\u0900-\u0963\u0971-\u097F"
WORD_RE = re.compile(rf"[{_LETTERS}\u200c\u200d]+")
DEV_CHAR_RE = re.compile(rf"[{_LETTERS}]")
ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)

MARATHI_STOPWORDS = set("""
आणि व किंवा पण परंतु तर की कि हे ही हा ह्या या ते ती तो त्या त्यांना त्यांनी त्याचा त्याची त्याचे
त्यांचा त्यांची त्यांचे त्याला तिला त्याने तिने आहे आहेत होते होता होती होत्या होतं आहोत आहात असे
असा अशी असून असल्याचे असल्याने असते असतो असतात असेल असलेल्या असलेले असलेला असलेली नाही नाहीत
नसून मी आम्ही आपण तुम्ही तू मला आम्हाला आपल्या आपला आपली आपले माझा माझी माझे आमचा आमची आमचे
तुमचा तुमची तुमचे एक दोन तीन काही सर्व अनेक प्रत्येक इतर अन्य अधिक कमी खूप फार जास्त
येथे तेथे इथे तिथे जेथे कुठे आता तेव्हा जेव्हा कधी नंतर आधी पूर्वी मात्र देखील सुद्धा फक्त केवळ
म्हणजे म्हणून म्हणाले म्हणाल्या म्हटले सांगितले केले केली केला केल्या करून करणे करण्यात करण्याचे
करण्याची करण्यासाठी करत करते करतो करतात झाले झाली झाला झाल्या झाल्याने झाल्याचे होऊन होणार
होईल गेले गेली गेला गेल्या दिले दिली दिला आला आली आले आल्या येणार येत यांनी यांचा यांची यांचे
यांना यांच्या याच्या याचा याची याचे यात यावर यामुळे त्यामुळे यासाठी त्यासाठी मध्ये वर खाली
साठी पासून पर्यंत कडे कडून सोबत बरोबर विरुद्ध नुसार दरम्यान असं तसं जसे तसे जशी तशी कसे
काय कोण कोणी कोणते कोणता कोणती का कारण जर तरी तेही हेही ना नये नको नवा नवी नवे दिवशी
सुरू निर्माण जारी घेतली घेतला घेत घेण्यात देण्यात आलेल्या आलेले गेल्या पुढील मोठ्या मोठे मोठा मोठी प्रमाणात
रोजी सध्या आज काल उद्या वेळी वेळा सुमारे जवळ जवळपास बाबत संदर्भात तसेच तसंच शिवाय यांच्याकडून
""".split())

# Hindi function words that are rare in Marathi; used to reject Hindi pages.
_HINDI_MARKERS = {"है", "हैं", "और", "के", "की", "में", "था", "थे", "थी", "से", "को", "नहीं", "यह", "वह", "लिए", "गया", "किया"}
_MARATHI_MARKERS = {"आहे", "आहेत", "आणि", "होते", "होता", "नाही", "मध्ये", "झाले", "केले", "असून", "यांनी", "त्यामुळे", "करण्यात", "असे", "आला"}

_ABBREVIATIONS = ["डॉ", "श्री", "सौ", "प्रा", "कु", "मा", "स्व", "ता", "जि", "क्र", "रु", "इ", "स", "वि", "पं", "ले"]

# Inflectional suffixes, longest first (light, conservative stemmer).
_SUFFIXES = sorted(set("""
ांच्याकडून ांच्यासाठी ांच्यामध्ये च्याकडून च्यासाठी च्यामध्ये ांच्या ाच्या च्या ांमध्ये ामध्ये मध्ये
ांसाठी ासाठी साठी ांपासून पासून पर्यंत ांकडून कडून ांकडे कडे बरोबर सोबत ांना ांनी ाला ाचा ाची ाचे
ांचा ांची ांचे चा ची चे ला ना नी ने ही ावर ांवर वर ात ांत त
""".split()), key=len, reverse=True)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[“”„]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def devanagari_ratio(text: str) -> float:
    letters = ALPHA_RE.findall(text) + re.findall(r"[\u093e-\u094d\u0901-\u0903]", text)
    if not letters:
        return 0.0
    return len(DEV_CHAR_RE.findall(text)) / max(len(letters), 1)


def language_check(text: str) -> dict:
    """Heuristic: Devanagari share + Marathi vs Hindi function-word counts."""
    words = WORD_RE.findall(text)
    counts = Counter(words)
    mr = sum(counts[w] for w in _MARATHI_MARKERS)
    hi = sum(counts[w] for w in _HINDI_MARKERS)
    ratio = devanagari_ratio(text)
    is_mr = ratio >= 0.6 and len(words) >= 15 and mr >= hi
    return {"is_marathi": is_mr, "devanagari_ratio": round(ratio, 3),
            "marathi_markers": mr, "hindi_markers": hi, "n_words": len(words)}


def split_sentences(text: str, min_words: int = 4) -> list[str]:
    text = normalize(text)
    abbr = "|".join(sorted(map(re.escape, _ABBREVIATIONS), key=len, reverse=True))
    text = re.sub(rf"(?<![{_LETTERS}])({abbr})\.", r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?।॥])\s+|\n+", text)
    out = []
    for p in parts:
        p = p.replace("<DOT>", ".").strip(" -–—•\t")
        if len(WORD_RE.findall(p)) >= min_words:
            out.append(p)
    return out


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    toks = WORD_RE.findall(text)
    toks = [t.strip("\u200c\u200d") for t in toks]
    if remove_stopwords:
        toks = [t for t in toks if t not in MARATHI_STOPWORDS and len(t) > 1]
    return toks


def stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf):
            base = word[: -len(suf)]
            min_len = 4 if suf in ("त", "ने", "ही", "ना", "ला") else 2
            if len(base) >= min_len:
                return base
    return word


def stem_tokens(text: str) -> list[str]:
    """Tokenise, drop stopwords, stem. Used for TF-IDF / similarity."""
    return [stem(t) for t in tokenize(text, remove_stopwords=True)]


def stem_phrase(phrase: str) -> str:
    return " ".join(stem(t) for t in tokenize(phrase))
