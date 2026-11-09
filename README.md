# Digital Decade country reports: TF-IDF keyword analysis

A course project for *Introduction to Scripting, Data Mining and Machine Learning*
in the MSc in Techno-Anthropology at Aalborg University. It uses NLP to support
qualitative analysis: instead of reading every report closely, the researcher gets
the keywords that are distinctive to each one as a starting point.

The corpus is the European Commission's "Digital Decade country reports" (one per
EU Member State, same template). TF-IDF finds the distinctive keywords in each report.
The 2025 and 2026 editions are analysed together as 54 reports, so a term's scores
can be compared across countries and years.

- Plan and cleaning decisions: [docs/PLAN.md](docs/PLAN.md)
- Progress log, step by step: [docs/LOG.md](docs/LOG.md)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python setup_nltk.py        # downloads NLTK tokenizer, stopwords, tagger, WordNet
```

## Data

All data is committed, so the analysis can be rerun straight after cloning:

- `data/raw/<year>/` holds the original English PDFs, one per Member State, from the
  European Commission's [2025](https://digital-strategy.ec.europa.eu/en/library/digital-decade-2025-country-reports)
  and [2026](https://digital-strategy.ec.europa.eu/en/library/digital-decade-2026-country-reports)
  country reports (© European Union, reused under the Commission's reuse policy).
- `data/text/<year>/` and `data/clean/<year>/` are generated from the PDFs by `extract.py`
  and `clean.py`. Rerun those scripts after changing them, so the committed files stay in sync.

To start from a fresh download, save the PDFs as `data/raw/<year>/<country>.pdf`,
using these exact file names:

```
austria   belgium  bulgaria  croatia     cyprus      czechia  denmark
estonia   finland  france    germany     greece      hungary  ireland
italy     latvia   lithuania luxembourg  malta       netherlands
poland    portugal romania   slovakia    slovenia    spain    sweden
```

## Pipeline

| Step | Script | Input | Output |
|---|---|---|---|
| 2. Extract | `extract.py` | `data/raw/<year>/*.pdf` | `data/text/<year>/*.txt`, `*.spans.csv` (text + font info) |
| 3. Clean | `clean.py` | `data/text/<year>/*.spans.csv` | `data/clean/<year>/*.txt` (lemma chunks), `*.body.txt` (readable) |
| 4–6. TF-IDF | `tfidf.py` | `data/clean/<year>/*.txt` | `output/combined/` |
| 8. Charts and word clouds | `visualization.py` | `top_terms.csv`, `sanity_doc_freq.csv`, `data/clean/<year>/*.txt` | `output/combined/charts/`, `output/combined/visualization.html` |

Run each script with `.venv/bin/python <script>.py`. In `config.py`, `YEAR` is the year
`extract.py` and `clean.py` process (run them once per year), and `YEARS` lists the years
`tfidf.py` and `visualization.py` analyse together.

## Project structure

```
config.py         paths, country list, names removed during cleaning
setup_nltk.py     one-time NLTK data download
extract.py        PDF -> text (PyMuPDF)
clean.py          text cleaning, NLTK tokenizing / stopwords / lemmatizing
tfidf.py          TF-IDF, top terms, sanity check
visualization.py  charts, interactive word clouds (HTML, from visualization_template.html)
docs/             plan and progress log
data/             raw PDFs, extracted and cleaned text (not versioned)
output/combined/  results for both years together
```

## Libraries

| Library | Used for |
|---|---|
| PyMuPDF | text extraction from PDF |
| NLTK | tokenizing, English stopwords, POS tagging, lemmatizing (WordNet), word counts (`FreqDist`) |
| scikit-learn | `TfidfVectorizer`, `CountVectorizer` |
| pandas | tables and CSV output |
| matplotlib | charts |
