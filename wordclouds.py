"""Step 8: interactive word clouds in one HTML page.

Input:  output/combined/top_terms.csv        top 20 terms per report (from tfidf.py)
        output/combined/sanity_doc_freq.csv  every term with its document frequency
        data/clean/<year>/<country>.txt      cleaned text, to count each term per report
Output: output/combined/wordclouds.html      open in a browser; no server or internet
                                             needed. Shows the PNGs from
                                             output/combined/charts/ next to the clouds,
                                             so keep the two together.

One cloud per report (word size = TF-IDF score), grouped by country, and one cloud of
the most common terms overall (word size = total count). Clicking a word lists every
country that uses it, with its count and TF-IDF score in each year.

Usage: .venv/bin/python wordclouds.py
"""

import json

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config import COUNTRIES, RESULTS_DIR, ROOT, YEARS
from tfidf import MAX_DF, MIN_DF, SUBLINEAR_TF, load_corpus, ngrams_in_chunks

N_COMMON = 50  # terms in the "most common overall" cloud


def main():
    top = pd.read_csv(RESULTS_DIR / "top_terms.csv")
    common = pd.read_csv(RESULTS_DIR / "sanity_doc_freq.csv").head(N_COMMON)

    # Count and score every displayed term in every report, so a click can show where
    # it occurs. TF-IDF uses the same settings as tfidf.py; terms max_df drops get None.
    terms = sorted(set(top["term"]) | set(common["term"]))
    reports, texts = load_corpus()
    C = CountVectorizer(analyzer=ngrams_in_chunks, vocabulary=terms).transform(texts).toarray()
    tfidf = TfidfVectorizer(analyzer=ngrams_in_chunks, sublinear_tf=SUBLINEAR_TF,
                            min_df=MIN_DF, max_df=MAX_DF)
    X = tfidf.fit_transform(texts)
    column = tfidf.vocabulary_
    counts = {  # term -> {report: [count, score]}
        term: {
            r: [int(C[i, j]), round(float(X[i, column[term]]), 4) if term in column else None]
            for i, r in enumerate(reports) if C[i, j] > 0
        }
        for j, term in enumerate(terms)
    }

    data = {
        "years": YEARS,
        "countries": list(COUNTRIES),
        "reports": {
            report: [[r.term, r.score, r.count] for r in top[top["report"] == report].itertuples()]
            for report in reports
        },
        "common": [[r.term, r.total_count, r.doc_freq] for r in common.itertuples()],
        "counts": counts,
    }
    template = (ROOT / "wordclouds_template.html").read_text(encoding="utf-8")
    html = template.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    path = RESULTS_DIR / "wordclouds.html"
    path.write_text(html, encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
