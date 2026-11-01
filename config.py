"""Model names and defaults. Set any optional model to None to disable it."""

CLASSIFIER_PATH = "models/classifier/classifier.joblib"

NER_MODEL = "ai4bharat/IndicNER"                                  # PER / LOC / ORG
KEYBERT_EMBEDDER = "paraphrase-multilingual-MiniLM-L12-v2"         # supports Marathi
ABSTRACTIVE_MODEL = "ai4bharat/IndicBART-XLSum"                    # optional, ~1 GB

# Any Marathi or multilingual sentiment checkpoint from the Hugging Face Hub.
# Check its model card and map raw labels to readable names below.
SENTIMENT_MODEL = "l3cube-pune/MarathiSentiment"
SENTIMENT_LABELS = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

SCRAPER_ENGINE = "selenium"        # or "playwright"
SUMMARY_SENTENCES = 3
LEXRANK_THRESHOLD = 0.08
TOP_KEYWORDS = 12
