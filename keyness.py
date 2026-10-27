"""Step 7: log-likelihood (G²) keyness, compared with the TF-IDF results.

Each report is compared with the other 26 combined (the reference corpus). For each
term, G² measures how far its count in the report is from the count expected if it
were spread evenly over all reports (Dunning 1993). Positive keywords are used more
than expected, negative keywords less.

Uses the same terms (1-3-grams inside chunks), phrase minimum and overlap filter as
tfidf.py, so differences come from the measure only. No max_df: G² can show that a
widespread word is still overused in one report.

Input:  data/clean/<year>/<country>.txt, output/<year>/top_terms.csv (from tfidf.py)
Output: output/<year>/keyness_g2.csv            top 20 positive keywords per country
        output/<year>/keyness_negative.csv      top 10 negative keywords per country
        output/<year>/keyness_comparison.csv    overlap with TF-IDF per country
        output/<year>/keyness_comparison.md     readable comparison
        output/<year>/charts/keyness_overlap.png

Usage: .venv/bin/python keyness.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config import OUTPUT_DIR, YEAR
from tfidf import (BAR, INK, INK_MUTED, MAX_DF, MIN_DF, MIN_PHRASE_COUNT, SUBLINEAR_TF,
                   SURFACE, TOP_N, load_corpus, n_words, ngrams_in_chunks, rank_terms,
                   style_axes)

G2_CRITICAL = 10.83     # p < 0.001 with 1 degree of freedom
TOP_N_NEGATIVE = 10


def g2_matrix(C, term_lengths):
    """G² and log ratio for every (report, term), report vs the rest of the corpus.

    Corpus sizes are counted per n-gram length: a bigram is compared with the number
    of bigrams in the report and in the reference corpus.
    Returns (g2, log_ratio, overused), each of shape (reports, terms).
    """
    a = C.toarray().astype(float)                  # count in the report
    b = a.sum(axis=0) - a                          # count in the other reports

    size = np.zeros_like(a)                        # report size, per term's n-gram length
    for n in np.unique(term_lengths):
        cols = term_lengths == n
        size[:, cols] = a[:, cols].sum(axis=1, keepdims=True)
    ref_size = size.sum(axis=0) - size             # reference corpus size
    total = size + ref_size

    expected_a = size * (a + b) / total
    expected_b = ref_size * (a + b) / total
    g2 = 2 * (xlogy(a, a / expected_a) + xlogy(b, b / expected_b))

    # Log ratio (effect size, Hardie 2014): log2 of the relative frequency ratio,
    # +0.5 so terms absent from the reference corpus don't divide by zero.
    log_ratio = np.log2(((a + 0.5) / size) / ((b + 0.5) / ref_size))
    overused = a / size > b / ref_size
    return g2, log_ratio, overused


def tfidf_scores(texts, terms):
    """TF-IDF with the same settings as tfidf.py, aligned to `terms` (NaN where max_df
    removed the term)."""
    vectorizer = TfidfVectorizer(analyzer=ngrams_in_chunks, sublinear_tf=SUBLINEAR_TF,
                                 min_df=MIN_DF, max_df=MAX_DF)
    X = vectorizer.fit_transform(texts).toarray()
    column = {t: j for j, t in enumerate(vectorizer.get_feature_names_out())}
    aligned = np.full((X.shape[0], len(terms)), np.nan)
    for j, t in enumerate(terms):
        if t in column:
            aligned[:, j] = X[:, column[t]]
    return aligned


def chart_overlap(comp, path):
    data = comp.sort_values("overlap")
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(data) + 1.2), facecolor=SURFACE)
    style_axes(ax)
    ax.barh(data["country"].str.capitalize(), data["overlap"], color=BAR, height=0.7)
    ax.tick_params(axis="y", labelcolor=INK, labelsize=9)
    ax.set_xlim(0, TOP_N)
    ax.set_xticks(range(0, TOP_N + 1, 5))
    ax.set_xlabel(f"Terms in both top-{TOP_N} lists (TF-IDF and G²)", color=INK_MUTED, fontsize=9)
    ax.set_title(f"How much TF-IDF and log-likelihood agree ({YEAR})",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_markdown(path, comp, pos, neg, tfidf_top):
    lines = [
        f"# TF-IDF vs log-likelihood (G²): Digital Decade {YEAR}", "",
        "Each report is compared with the other 26. G² ranks terms that a report uses "
        f"significantly more often than the rest (G² ≥ {G2_CRITICAL}, p < 0.001); "
        "same n-grams, phrase minimum and overlap filter as TF-IDF, but no max_df.", "",
        "| | TF-IDF | G² |", "|---|---|---|",
        f"| mean count in report (top {TOP_N}) | {tfidf_top['count'].mean():.1f} | {pos['count'].mean():.1f} |",
        f"| mean number of reports containing the term | {tfidf_top['doc_freq'].mean():.1f} | {pos['doc_freq'].mean():.1f} |",
        f"| terms found in only one report | {(tfidf_top['doc_freq'] == 1).mean():.0%} | {(pos['doc_freq'] == 1).mean():.0%} |",
        f"| terms found in more than half the reports | {(tfidf_top['doc_freq'] > 13).mean():.0%} | {(pos['doc_freq'] > 13).mean():.0%} |",
        f"| single words / 2-grams / 3-grams | "
        f"{' / '.join(str((tfidf_top['n'] == n).sum()) for n in (1, 2, 3))} | "
        f"{' / '.join(str((pos['n'] == n).sum()) for n in (1, 2, 3))} |",
        "",
        f"Mean overlap of the top-{TOP_N} lists: **{comp['overlap'].mean():.1f}**; "
        f"mean Spearman correlation of all scores within a report: "
        f"**{comp['spearman_all_terms'].mean():.2f}**.", "",
        "## Per country", "",
    ]
    for r in comp.itertuples():
        negatives = ", ".join(neg[neg["country"] == r.country]["term"].head(5))
        lines += [
            f"### {r.country.capitalize()} (overlap {r.overlap}/{TOP_N})", "",
            f"- **both:** {r.shared or '-'}",
            f"- **only TF-IDF:** {r.only_tfidf or '-'}",
            f"- **only G²:** {r.only_g2 or '-'}",
            f"- **underused (negative G²):** {negatives or '-'}", "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    countries, texts = load_corpus()
    counter = CountVectorizer(analyzer=ngrams_in_chunks)
    C = counter.fit_transform(texts)
    terms = counter.get_feature_names_out()
    index = {t: j for j, t in enumerate(terms)}
    lengths = np.array([n_words(t) for t in terms])
    doc_freq = np.asarray((C > 0).sum(axis=0)).ravel()
    counts = C.toarray()
    totals = counts.sum(axis=0)

    g2, log_ratio, overused = g2_matrix(C, lengths)
    signed_g2 = np.where(overused, g2, -g2)
    tfidf = tfidf_scores(texts, terms)
    tfidf_top = pd.read_csv(OUTPUT_DIR / "top_terms.csv")

    positive, negative, comparison = [], [], []
    for i, country in enumerate(countries):
        def row(j, rank):
            return {"country": country, "rank": rank, "term": terms[j], "n": int(lengths[j]),
                    "g2": round(g2[i, j], 1), "log_ratio": round(log_ratio[i, j], 2),
                    "count": int(counts[i, j]), "ref_count": int(totals[j] - counts[i, j]),
                    "doc_freq": int(doc_freq[j])}

        # Positive keywords: overused and significant, then the same filters as TF-IDF
        key = np.where(overused[i] & (g2[i] >= G2_CRITICAL))[0]
        ranked = rank_terms({terms[j]: g2[i, j] for j in key},
                            {terms[j]: int(counts[i, j]) for j in key}, MIN_PHRASE_COUNT)[:TOP_N]
        positive += [row(index[t], r) for r, (t, _) in enumerate(ranked, start=1)]

        # Negative keywords: used significantly less than in the other reports
        under = np.where(~overused[i] & (g2[i] >= G2_CRITICAL))[0]
        under = under[np.argsort(-g2[i, under])][:TOP_N_NEGATIVE]
        negative += [row(j, r) for r, j in enumerate(under, start=1)]

        # Agreement with TF-IDF: top-list overlap, and rank correlation over all
        # terms in the report that TF-IDF kept
        t_terms = tfidf_top[tfidf_top["country"] == country]["term"].tolist()
        g_terms = [t for t, _ in ranked]
        both = (counts[i] > 0) & ~np.isnan(tfidf[i])
        comparison.append({
            "country": country,
            "overlap": sum(t in t_terms for t in g_terms),
            "spearman_all_terms": round(spearmanr(tfidf[i, both], signed_g2[i, both])[0], 2),
            "significant_keywords": int(len(key)),
            "shared": " | ".join(t for t in g_terms if t in t_terms),
            "only_tfidf": " | ".join(t for t in t_terms if t not in g_terms),
            "only_g2": " | ".join(t for t in g_terms if t not in t_terms),
        })

    pos, neg, comp = pd.DataFrame(positive), pd.DataFrame(negative), pd.DataFrame(comparison)
    pos.to_csv(OUTPUT_DIR / "keyness_g2.csv", index=False)
    neg.to_csv(OUTPUT_DIR / "keyness_negative.csv", index=False)
    comp.to_csv(OUTPUT_DIR / "keyness_comparison.csv", index=False)
    write_markdown(OUTPUT_DIR / "keyness_comparison.md", comp, pos, neg, tfidf_top)
    chart_overlap(comp, OUTPUT_DIR / "charts" / "keyness_overlap.png")

    print(comp[["country", "overlap", "spearman_all_terms", "significant_keywords"]]
          .to_string(index=False))
    print(f"\nMean overlap: {comp['overlap'].mean():.1f} of {TOP_N}; "
          f"mean Spearman: {comp['spearman_all_terms'].mean():.2f}")
    print(f"Wrote results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
