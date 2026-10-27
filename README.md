# Digital Decade country reports: TF-IDF keyword analysis

Finds the keywords that are distinctive to each of the 27 European Commission
"Digital Decade 2026 country reports" (one per EU Member State, same template),
using TF-IDF. The 2025 edition can be processed later as a separate corpus.

- Plan and cleaning decisions: [docs/PLAN.md](docs/PLAN.md)
- Progress log, step by step: [docs/LOG.md](docs/LOG.md)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python setup_nltk.py        # downloads NLTK tokenizer, stopwords, tagger, WordNet
```

## Data

Download the English PDFs yourself from the
[landing page](https://digital-strategy.ec.europa.eu/en/library/digital-decade-2026-country-reports)
and save them as `data/raw/2026/<country>.pdf`, using these exact file names:

```
austria   belgium  bulgaria  croatia     cyprus      czechia  denmark
estonia   finland  france    germany     greece      hungary  ireland
italy     latvia   lithuania luxembourg  malta       netherlands
poland    portugal romania   slovakia    slovenia    spain    sweden
```

The 2025 edition goes in `data/raw/2025/` with the same names.

## Pipeline

| Step | Script | Input | Output |
|---|---|---|---|
| 2. Extract | `extract.py` | `data/raw/<year>/*.pdf` | `data/text/<year>/*.txt`, `*.spans.csv` (text + font info) |
| 3. Clean | `clean.py` | `data/text/<year>/*.spans.csv` | `data/clean/<year>/*.txt` (lemma chunks), `*.body.txt` (readable) |
| 4–6. TF-IDF | `tfidf.py` | `data/clean/<year>/*.txt` | `output/<year>/` |
| 7. Keyness | `keyness.py` | `data/clean/<year>/*.txt`, `top_terms.csv` | `output/<year>/keyness_*` |

Run each script with `.venv/bin/python <script>.py`. The year is set in `config.py`.

## Project structure

```
config.py         paths, country list, names removed during cleaning
setup_nltk.py     one-time NLTK data download
extract.py        PDF -> text (PyMuPDF)
clean.py          text cleaning, NLTK tokenizing / stopwords / lemmatizing
tfidf.py          TF-IDF, top terms, sanity check, charts
keyness.py        log-likelihood (G²) keyness, comparison with TF-IDF
docs/             plan and progress log
data/             raw PDFs, extracted and cleaned text (not versioned)
output/           results per year
```

## Libraries

| Library | Used for |
|---|---|
| PyMuPDF | text extraction from PDF |
| NLTK | tokenizing, English stopwords, POS tagging, lemmatizing (WordNet), word counts (`FreqDist`) |
| scikit-learn | `TfidfVectorizer`, `CountVectorizer` |
| SciPy (installed with scikit-learn) | `xlogy` for G², Spearman correlation |
| pandas | tables and CSV output |
| matplotlib | charts |
