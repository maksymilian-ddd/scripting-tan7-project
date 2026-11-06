"""Charts and interactive word clouds, built from the results of tfidf.py.

Input:  output/combined/top_terms.csv        top 20 terms per report (from tfidf.py)
        output/combined/sanity_doc_freq.csv  every term with its document frequency
        data/clean/<year>/<country>.txt      cleaned text, to count each term per report
Output: output/combined/charts/*.png         bar chart per report, sanity chart
        output/combined/visualization.html      open in a browser; no server or internet
                                             needed. Shows the PNGs next to the clouds,
                                             so keep the two together.

One cloud per report (word size = TF-IDF score), grouped by country, and one cloud of
the most common terms overall (word size = total count). Clicking a word lists every
country that uses it, with its count and TF-IDF score in each year.

Usage: .venv/bin/python visualization.py
"""

import json

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config import COUNTRIES, RESULTS_DIR, ROOT, YEARS
from tfidf import (MAX_DF, MIN_DF, SHOW_REPORTS, SUBLINEAR_TF, load_corpus, ngrams_in_chunks,
                   title)

N_COMMON = 50  # terms in the "most common overall" cloud

# Chart style: one series, so one hue; text in neutral inks.
SURFACE, INK, INK_MUTED, GRID, BAR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


# ===================================================================================
# Charts
# ===================================================================================

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_top_terms(top, report, path):
    data = top[top["report"] == report].sort_values("score")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(data) + 1.2), facecolor=SURFACE)
    style_axes(ax)
    ax.barh(data["term"], data["score"], color=BAR, height=0.7)
    ax.tick_params(axis="y", labelcolor=INK, labelsize=9)
    ax.xaxis.set_major_locator(MultipleLocator(0.01))  # same two-decimal ticks on every chart
    ax.set_xlabel("TF-IDF score", color=INK_MUTED, fontsize=9)
    ax.set_title(f"{title(report)}: top {len(data)} distinctive terms",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_doc_freq(table, max_doc_count, n_docs, path):
    """TF-IDF without the cut (mean over reports containing the term) against
    document frequency, one dot per term."""
    rng = np.random.default_rng(0)
    x = table["doc_freq"] + rng.uniform(-0.25, 0.25, len(table))
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.scatter(x, table["mean_tfidf_without_cut"], s=6, color=BAR, alpha=0.35, linewidths=0)
    ax.axvline(max_doc_count + 0.5, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(max_doc_count + 0.6, ax.get_ylim()[1] * 0.95, f"max_df cut: terms right of\n"
            f"this line are dropped", color=INK_MUTED, fontsize=8, va="top")
    ax.set_xticks(range(0, n_docs + 1, 6))
    ax.set_xlabel("Document frequency (number of reports containing the term)",
                  color=INK_MUTED, fontsize=9)
    ax.set_ylabel("Mean TF-IDF where present, no max_df", color=INK_MUTED, fontsize=9)
    ax.set_title("TF-IDF weight by document frequency, before the max_df cut",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===================================================================================
# Word clouds
# ===================================================================================

def write_wordclouds(top, common, reports, texts):

    # Count and score every displayed term in every report, so a click can show where
    # it occurs. TF-IDF uses the same settings as tfidf.py; terms max_df drops get None.
    terms = sorted(set(top["term"]) | set(common["term"]))
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
    template = (ROOT / "visualization_template.html").read_text(encoding="utf-8")
    html = template.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    path = RESULTS_DIR / "visualization.html"
    path.write_text(html, encoding="utf-8")
    print(f"Wrote {path}")


# ===================================================================================

def main():
    top = pd.read_csv(RESULTS_DIR / "top_terms.csv")
    sanity = pd.read_csv(RESULTS_DIR / "sanity_doc_freq.csv")
    reports, texts = load_corpus()

    charts_dir = RESULTS_DIR / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    for report in [r for r in (SHOW_REPORTS or reports) if r in reports]:
        chart_top_terms(top, report, charts_dir / f"top_terms_{report}.png")
    chart_doc_freq(sanity, int(MAX_DF * len(reports)), len(reports),
                   charts_dir / "doc_freq_vs_score.png")

    write_wordclouds(top, sanity.head(N_COMMON), reports, texts)


if __name__ == "__main__":
    main()
