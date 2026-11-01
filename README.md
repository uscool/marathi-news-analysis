<div align="center">

# 📰 Marathi News Analyzer

### मराठी बातमी विश्लेषक

**An NLP-powered application for extracting, classifying, and summarizing Marathi news articles.**

Built with **Python · Streamlit · Scikit-learn · Transformers · Plotly**

</div>

---

## 📌 Overview

**Marathi News Analyzer** is a Python application that processes Marathi news from a webpage or pasted text. It extracts article content, identifies important keywords and named entities, classifies the news, ranks key sentences, and generates a concise summary. A Streamlit dashboard presents the results, including an interactive sentence-similarity graph, and lets you export the analysis as JSON.

The pipeline supports lightweight methods and optional pretrained transformer models. If certain optional models cannot load, the application can continue using simpler methods and report the limitation.

## ✨ Features

- **Article extraction:** Open a Marathi news URL with Selenium (or optional Playwright), extract readable article text using Trafilatura, and apply fallback content filtering.
- **Language validation:** Check whether extracted text is Marathi, including attempts to distinguish it from Hindi written in the same script.
- **Keyword extraction:** Combine sentence-level TF-IDF with optional multilingual KeyBERT embeddings, then refine using YAKE and entity agreement.
- **Named entity recognition:** Identify people, locations, and organizations with `ai4bharat/IndicNER`, with a rule-based fallback.
- **News classification:** Categorize content using an MLP trained on character n-grams and stemmed word TF-IDF features.
- **Key sentence ranking:** Prioritize informative sentences using LexRank centrality, keywords, entity mentions, and lead position.
- **Text summarization:** Generate extractive summaries with LexRank; optionally generate an abstractive summary using IndicBART-XLSum.
- **Interactive visualizations:** Explore relationships among sentences with Plotly-based graphs.
- **Optional sentiment analysis:** Run a configured Marathi sentiment model when enabled.
- **Two interfaces:** Analyze articles in the Streamlit web app or run the pipeline through the CLI.
- **JSON export:** Download structured analysis results from the dashboard.

## 🧠 How It Works

```text
Marathi News URL / Pasted Text / Local Text File
                      |
                      v
        Article Extraction & Cleaning
                      |
                      v
           Marathi Language Check
                      |
                      v
       Keywords + Named Entities (NER)
                      |
                      v
        News Category Classification
                      |
                      v
           YAKE Keyword Refinement
                      |
                      v
         Key Sentence Ranking (LexRank)
                      |
                      v
       Summary + Interactive Graph + JSON
```

### Core techniques

| Task | Method / Technology |
| --- | --- |
| Web scraping | Selenium / optional Playwright, BeautifulSoup, Trafilatura |
| Text normalization | Marathi-oriented preprocessing and lightweight suffix stemming |
| Keywords | TF-IDF, optional KeyBERT, YAKE |
| Named entities | IndicNER (`PER`, `LOC`, `ORG`), rule-based fallback |
| News category | Character 2–4-grams + word TF-IDF + one-hidden-layer MLP |
| Sentence graph | Sentence similarity and LexRank / PageRank |
| Summarization | Extractive LexRank; optional IndicBART-XLSum |
| Visualization | Streamlit and Plotly |

## 🛠️ Tech Stack

**Language:** Python  
**Web UI:** Streamlit  
**Machine Learning:** Scikit-learn, NumPy, Pandas, Joblib  
**NLP:** Transformers, PyTorch, Sentence Transformers, KeyBERT, YAKE  
**Scraping:** Selenium, BeautifulSoup, Trafilatura  
**Graphs:** NetworkX, Plotly

## 📁 Project Structure

```text
marathi_news_analyzer/
├── app.py                     # Streamlit dashboard
├── cli.py                     # Command-line analysis, JSON output
├── config.py                  # Model names and pipeline defaults
├── train_classifier.py        # Train the news-category classifier
├── requirements.txt           # Python dependencies
├── README.md
├── data/
│   ├── sample_article.txt     # Example Marathi article
│   └── sample_headlines.csv   # Small demonstration training set
├── models/
│   └── classifier/
│       └── classifier.joblib  # Bundled trained classifier
└── mna/
    ├── __init__.py
    ├── pipeline.py            # Six-step processing pipeline
    ├── scraper.py             # Article extraction
    ├── text_utils.py          # Preprocessing and language checks
    ├── ner.py                 # Named entity recognition
    ├── keywords.py            # Keyword generation and refinement
    ├── classifier.py          # Category classification
    ├── sentiment.py           # Optional sentiment analysis
    ├── summarizer.py          # LexRank and abstractive summaries
    └── graphs.py              # Plotly graph creation
```

## 🚀 Getting Started

### Prerequisites

- Python with support for the versions specified in `requirements.txt` (a recent Python 3 installation is recommended)
- `pip`
- **Google Chrome** for Selenium-based URL extraction
- Internet access for first-time downloads of optional pretrained models

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

Alternatively, extract the project ZIP and open a terminal inside `marathi_news_analyzer/`.

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The dependency list includes transformer-related packages. Initial model downloads and inference may require substantial memory and disk space. Playwright is an optional alternative to Selenium.

### 4. Launch the web application

```bash
streamlit run app.py
```

Open the local URL shown in your terminal (typically `http://localhost:8501`).

## 📖 Usage

### Streamlit dashboard

1. Open the **From a link** tab to enter a Marathi news URL, or use **Paste text** to supply article text directly.
2. Select **Analyse article** or **Analyse text**.
3. Explore extracted content, important keywords, named entities, predicted category, refined keywords, key sentences, and summary.
4. Inspect the interactive visualizations and use **Download results (JSON)** to export the output.

### Command-line interface

Analyze the included sample article:

```bash
python cli.py --file data/sample_article.txt
```

Analyze an online article:

```bash
python cli.py --url "https://example.com/marathi-news-article"
```

Run without transformer-based NER and KeyBERT:

```bash
python cli.py --file data/sample_article.txt --no-ner-model --no-keybert
```

Request an optional abstractive summary:

```bash
python cli.py --file data/sample_article.txt --abstractive
```

Use Playwright as an alternative scraping engine (after installing it):

```bash
pip install playwright
playwright install chromium
python cli.py --url "https://example.com/marathi-news-article" --engine playwright
```

The CLI prints structured JSON containing the language check, extracted entities, keywords, classification, key sentences, summary, optional abstractive summary, and processing notes.

## 🎓 Training the News Classifier

The project includes `data/sample_headlines.csv`, a **45-row demonstration dataset** covering three example categories: **sports, entertainment, and state**. The sample dataset is for demonstrating the workflow, **not for claiming production-level classification accuracy**.

To train the model:

```bash
python train_classifier.py --data data/sample_headlines.csv
```

For your own dataset, prepare a CSV in this format:

```csv
headline,label
"नमुना मराठी क्रीडा बातमी",sports
"नमुना मराठी चित्रपट बातमी",entertainment
"नमुना मराठी राज्य बातमी",state
```

Then run:

```bash
python train_classifier.py --data data/your_headlines.csv
```

By default, the trained model is saved under `models/classifier/`. The training script reports a classification report and macro-F1 on its held-out split. Use a substantially larger, representative dataset and independent evaluation before relying on category predictions.

## ⚙️ Configuration

Edit `config.py` to adjust model selection and analysis parameters:

```python
NER_MODEL = "ai4bharat/IndicNER"
KEYBERT_EMBEDDER = "paraphrase-multilingual-MiniLM-L12-v2"
ABSTRACTIVE_MODEL = "ai4bharat/IndicBART-XLSum"
SENTIMENT_MODEL = "l3cube-pune/MarathiSentiment"

SCRAPER_ENGINE = "selenium"     # or "playwright"
SUMMARY_SENTENCES = 3
LEXRANK_THRESHOLD = 0.08
TOP_KEYWORDS = 12
```

Some models are optional and can be disabled with `None`. Verify the sentiment checkpoint's actual label mapping before interpreting sentiment results; default mappings in `config.py` are configuration assumptions, not guarantees about every model.

## ⚠️ Limitations

- **Demo dataset:** The provided headlines CSV is small; classification quality may be unreliable on unseen articles or different categories.
- **Scraping:** Some websites block automation, use unusual page structures, or require additional access steps.
- **Language detection:** Marathi/Hindi separation is heuristic and can make mistakes on short or mixed-language passages.
- **Optional models:** Initial downloads can be large; resource constraints or unavailable models may cause fallbacks.
- **Summary quality:** Extractive summarization selects original sentences and may not capture all nuance. Abstractive summaries should be checked against the source for factual consistency.
- **Input requirement:** Analysis expects at least two complete Marathi sentences.
- **Research/demo usage:** Outputs should be reviewed before use in sensitive, editorial, or automated publishing workflows.

## 🔮 Future Improvements

- Train and evaluate on a larger, balanced Marathi news dataset.
- Add stronger multilingual and Marathi-specific language validation.
- Benchmark against transformer-based classifiers such as IndicBERT or MuRIL.
- Add reproducible evaluation metrics and automated tests for the full pipeline.
- Improve cross-site scraping reliability and input validation.
- Add richer entity linking, source citations, and summary quality checks.
- Provide reproducible deployment instructions and optional containerization.

