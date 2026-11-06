"""Steps 4-6: TF-IDF, top terms per country, sanity check.

Input:  data/clean/<year>/<country>.txt   lemmatized chunks from clean.py, one per line
Output: output/combined/top_terms.csv               main result: top 20 per report
        output/combined/top_terms_by_ngram.csv      top 10 per report per n-gram length
        output/combined/sanity_doc_freq.csv         every term with its document frequency
        output/combined/summary.md                  readable summary

Each report is ONE document, named <country>_<year>. All reports of all YEARS share one
vocabulary and one IDF, so a term's scores can be compared across countries and years.
N-grams (1-3 words) are built inside chunks only, so they never cross a removed
stopword, number or sentence end (docs/PLAN.md).

Usage: .venv/bin/python tfidf.py
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config import COUNTRIES, RESULTS_DIR, ROOT, YEARS

# --- Parameters (documented in docs/PLAN.md) ---------------------------------------
NGRAM_RANGE = (1, 3)    # applied in ngrams_in_chunks(), not by sklearn
SUBLINEAR_TF = True     # tf -> 1 + log(tf)
MIN_DF = 1              # keep terms found in only one report
MAX_DF = 0.4            # drop terms found in more than 40% of the reports
SUBSUME_SHARE = 0.8     # drop a term if a longer ranked term containing it occurs
                        # at least 80% as often in that report
TOP_N = 20              # terms per report in the main ranking
TOP_N_PER_LENGTH = 10   # terms per report per n-gram length
SHOW_REPORTS = None     # reports to print and chart; None = all available

def ngrams_in_chunks(text):
    """All 1-, 2- and 3-grams inside each line (chunk) of a cleaned document."""
    terms = []
    for line in text.splitlines():
        tokens = line.split()
        for n in range(NGRAM_RANGE[0], NGRAM_RANGE[1] + 1):
            terms += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return terms


def load_corpus():
    """One label ('austria_2025') and one text per report, for every year in YEARS."""
    paths = {
        f"{country}_{year}": ROOT / "data" / "clean" / year / f"{country}.txt"
        for year in YEARS for country in COUNTRIES
    }
    reports = [r for r, path in paths.items() if path.exists()]
    texts = [paths[r].read_text(encoding="utf-8") for r in reports]
    return reports, texts


def title(report):
    """'austria_2025' -> 'Austria 2025'."""
    return report.replace("_", " ").capitalize()


def n_words(term):
    return term.count(" ") + 1


# ===================================================================================
# Ranking with the overlap filter
# ===================================================================================

def rank_terms(scores, counts):
    """Rank one report's terms by TF-IDF after the overlap filter: a term is dropped
    when a longer term containing it occurs at least SUBSUME_SHARE times as often,
    i.e. the shorter term hardly appears on its own ('broadband router' inside
    'mobile broadband router').
    """
    longest_container = defaultdict(int)  # term -> highest count of a longer term containing it
    for term in scores:
        tokens = term.split()
        for n in range(1, len(tokens)):
            for i in range(len(tokens) - n + 1):
                part = " ".join(tokens[i:i + n])
                longest_container[part] = max(longest_container[part], counts[term])
    kept = {
        t: s for t, s in scores.items()
        if longest_container[t] < SUBSUME_SHARE * counts[t]
    }
    return sorted(kept.items(), key=lambda item: -item[1])


def per_document(matrix, vocabulary, row):
    """{term: value} for the non-zero entries of one row of a sparse matrix."""
    r = matrix.getrow(row)
    return {vocabulary[j]: v for j, v in zip(r.indices, r.data)}


def build_rankings(reports, X, tfidf_terms, C, count_terms, df_by_term):
    main, by_length = [], []
    for i, report in enumerate(reports):
        scores = per_document(X, tfidf_terms, i)
        counts = defaultdict(int, per_document(C, count_terms, i))

        def rows(ranked, limit):
            return [{
                "report": report, "rank": r, "term": t, "n": n_words(t),
                "score": round(s, 4), "count": counts[t], "doc_freq": df_by_term[t],
            } for r, (t, s) in enumerate(ranked[:limit], start=1)]

        ranked = rank_terms(scores, counts)
        main += rows(ranked, TOP_N)
        for n in range(NGRAM_RANGE[0], NGRAM_RANGE[1] + 1):
            by_length += rows([(t, s) for t, s in ranked if n_words(t) == n], TOP_N_PER_LENGTH)
    return pd.DataFrame(main), pd.DataFrame(by_length)


# ===================================================================================
# Sanity check: does template language get low weight?
# ===================================================================================

def sanity_table(texts, C, count_terms, max_doc_count):
    """Every term with document frequency, total count, IDF, the mean TF-IDF it would
    get *without* the max_df cut (averaged over the reports that contain it), and
    whether the cut removes it.

    sklearn's smoothed IDF is ln((1 + N) / (1 + df)) + 1, so it never reaches 0: a
    term in every report keeps IDF = 1. This table shows how much of the template
    removal is done by IDF alone and how much by max_df.
    """
    no_cut = TfidfVectorizer(analyzer=ngrams_in_chunks, sublinear_tf=SUBLINEAR_TF, min_df=MIN_DF)
    X_all = no_cut.fit_transform(texts)
    terms_all = no_cut.get_feature_names_out()
    present = np.asarray((X_all > 0).sum(axis=0)).ravel()
    mean_where_present = np.asarray(X_all.sum(axis=0)).ravel() / present
    mean_score = dict(zip(terms_all, mean_where_present))
    idf = dict(zip(terms_all, no_cut.idf_))

    df = np.asarray((C > 0).sum(axis=0)).ravel()
    total = np.asarray(C.sum(axis=0)).ravel()
    table = pd.DataFrame({
        "term": count_terms, "n": [n_words(t) for t in count_terms],
        "doc_freq": df, "total_count": total,
    })
    table["idf"] = table["term"].map(idf).round(3)
    table["mean_tfidf_without_cut"] = table["term"].map(mean_score).round(5)
    table["dropped_by_max_df"] = table["doc_freq"] > max_doc_count
    return table.sort_values(["doc_freq", "total_count"], ascending=False)


# ===================================================================================
# Summary
# ===================================================================================

def write_summary(path, reports, top, sanity, max_doc_count):
    lines = [
        f"# Distinctive terms per report: Digital Decade {' + '.join(YEARS)} country reports",
        "",
        f"Corpus: {len(reports)} of {len(COUNTRIES) * len(YEARS)} reports. Exploratory: with "
        "this few documents IDF takes only a handful of values.",
        "",
        "| Parameter | Value |", "|---|---|",
        f"| n-grams | {NGRAM_RANGE[0]}-{NGRAM_RANGE[1]} words, built inside chunks |",
        f"| sublinear_tf | {SUBLINEAR_TF} |",
        f"| min_df | {MIN_DF} |",
        f"| max_df | {MAX_DF} (terms in more than {max_doc_count} reports are dropped) |",
        f"| overlap filter | shorter term dropped if a longer one containing it occurs "
        f"{SUBSUME_SHARE:.0%}+ as often |",
        "",
        "Scores are TF-IDF (L2-normalized per report); `(n)` after a term is its count in "
        "the report.",
        "",
    ]
    for report in reports:
        rows = top[top["report"] == report]
        terms = ", ".join(f"{t} ({c})" for t, c in zip(rows["term"], rows["count"]))
        lines += [f"## {title(report)}", "", terms, ""]

    common = sanity.head(15)
    lines += [
        "## Sanity check: most widespread terms", "",
        "These terms appear in the most reports. `idf` is sklearn's smoothed IDF (1.0 = in "
        "every report). `mean TF-IDF without cut` is the score the term would get, averaged "
        "over the reports containing it, if max_df were not applied. `dropped` shows "
        "whether max_df removes it.", "",
        "| term | reports | total count | idf | mean TF-IDF without cut | dropped |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r.term} | {r.doc_freq} | {r.total_count} | {r.idf:.2f} | {r.mean_tfidf_without_cut:.4f} | "
        f"{'yes' if r.dropped_by_max_df else 'no'} |"
        for r in common.itertuples()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ===================================================================================

def main():
    reports, texts = load_corpus()
    if not reports:
        raise SystemExit("No cleaned documents; run clean.py first.")
    expected = len(COUNTRIES) * len(YEARS)
    if len(reports) < expected:
        print(f"WARNING: only {len(reports)} of {expected} reports. Terms are "
              "distinctive relative to these reports only; rankings will change with "
              "the full corpus.\n")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    counter = CountVectorizer(analyzer=ngrams_in_chunks)
    C = counter.fit_transform(texts)
    count_terms = counter.get_feature_names_out()
    df_by_term = dict(zip(count_terms, np.asarray((C > 0).sum(axis=0)).ravel()))

    vectorizer = TfidfVectorizer(
        analyzer=ngrams_in_chunks, sublinear_tf=SUBLINEAR_TF, min_df=MIN_DF, max_df=MAX_DF,
    )
    X = vectorizer.fit_transform(texts)
    tfidf_terms = vectorizer.get_feature_names_out()
    max_doc_count = int(MAX_DF * len(reports))

    print(f"Documents: {len(reports)}; vocabulary: {len(count_terms)} terms; "
          f"after max_df (df <= {max_doc_count}): {len(tfidf_terms)} terms")

    top, by_length = build_rankings(reports, X, tfidf_terms, C, count_terms, df_by_term)
    top.to_csv(RESULTS_DIR / "top_terms.csv", index=False)
    by_length.to_csv(RESULTS_DIR / "top_terms_by_ngram.csv", index=False)

    sanity = sanity_table(texts, C, count_terms, max_doc_count)
    sanity.to_csv(RESULTS_DIR / "sanity_doc_freq.csv", index=False)
    print("\nSanity check: terms with the highest document frequency")
    print(sanity.head(25).to_string(index=False))
    kept = sanity[~sanity["dropped_by_max_df"]]
    dropped = sanity[sanity["dropped_by_max_df"]]
    print(f"\nMean TF-IDF without the cut (where present): terms max_df drops "
          f"{dropped['mean_tfidf_without_cut'].mean():.5f}, terms it keeps "
          f"{kept['mean_tfidf_without_cut'].mean():.5f}")
    print(f"Terms max_df drops: {len(dropped)} of {len(sanity)}")

    for report in [r for r in (SHOW_REPORTS or reports) if r in reports]:
        print(f"\n== {report} ==")
        print(top[top["report"] == report][["rank", "term", "score", "count", "doc_freq"]]
              .to_string(index=False))

    write_summary(RESULTS_DIR / "summary.md", reports, top, sanity, max_doc_count)
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
