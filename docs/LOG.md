# Progress log

## Step 1 — Setup (2026-10-27)

- Created a virtual environment `.venv` (Python 3.14.5) and `requirements.txt`.
- Libraries: PyMuPDF, NLTK, pandas, scikit-learn, matplotlib.
  - First tried spaCy for lemmatizing; switched to NLTK on request (tokenizing,
    stopwords, word counts). NLTK lemmatizing uses POS tags + WordNet.
  - pdfplumber not needed; PyMuPDF covers extraction.
- Tested NLTK: `word_tokenize` keeps `high-performance` as one token;
  `WordNetLemmatizer` gives `facilities → facility`, `deploying → deploy` when given the
  POS tag.
- NLTK data is installed inside the venv (`.venv/nltk_data`) by `setup_nltk.py`.
- Folder layout created; each year has its own folders (`2026`, `2025`).
- Wrote `config.py` (paths, 27 countries + name forms to remove) and `extract.py`.
- Wrote the plan with cleaning decisions: [PLAN.md](PLAN.md).
- Download is manual (the user puts the PDFs in `data/raw/2026/`), so there is no
  `download.py`.

**Next:** PDFs into `data/raw/2026/`, then run `extract.py` and inspect the structure.

## Step 2 — Extraction and structure inspection (2026-10-27)

- The user put 4 of the 27 PDFs in `data/raw/2026/`: Austria, Belgium, Czechia, Denmark.
  The other 23 are still missing.
- `extract.py` ran: 30–40 pages per report, 82–122k characters, no scanned pages.
- Inspecting the text showed the plain-text approach has problems: footnote markers glued
  to words, KPI table rows mixed into text, footnote sizes that differ per report.
- Inspected fonts with PyMuPDF and found that every document part has its own font
  size/style/position (table in [PLAN.md](PLAN.md#document-structure-found-4-reports-inspected-at-be-cz-dk)).
- Changed `extract.py` to also save all text spans with font info (`<country>.spans.csv`),
  so cleaning can filter by font.

## Step 3 — Cleaning (2026-10-27)

- Wrote `clean.py`: font-based body-text filter → paragraphs → normalize → country-name
  removal → NLTK tokenize / POS-tag / lemmatize / stopwords → chunks.
- Results on 4 reports:

  | country | raw chars | body chars | names removed | tokens kept | unique lemmas |
  |---|---|---|---|---|---|
  | austria | 82,216 | 74,916 | 245 | 6,062 | 1,265 |
  | belgium | 116,541 | 108,229 | 232 | 9,112 | 1,786 |
  | czechia | 77,642 | 70,336 | 196 | 5,681 | 1,198 |
  | denmark | 78,582 | 70,729 | 190 | 5,796 | 1,426 |

- Checks done:
  - KPI table, footnotes, TOC and headers are gone; the Recommendations box is kept.
  - No glued words. Long unknown tokens are real terms (often national-language
    names, which are useful).
  - Multi-word terms survive (`high-performance computing`, `digital public service`).
- Fixes after the first run: split `a/b` words; noun fallback in the lemmatizer for
  mis-tagged plurals (`technologies`).
- Removed Flanders/Wallonia/Flemish/Walloon from the country-name list, pending the
  decision on regions.
- NLTK `FreqDist` top words are template words for all 4 (*eu, digital, average,
  support*), which confirms `max_df` is needed.

**Next:** the remaining 23 PDFs → run `extract.py` and `clean.py` on all 27 (check the
new reports follow the same layout) → write and run `tfidf.py`.

## Step 4–6 — TF-IDF script (2026-10-27)

- Explained to the user why chunks: TF-IDF still runs on one document per country;
  chunks only stop false n-grams (Austria: 3,218 of 6,061 flat bigrams would be false).
- Agreed ranking design: one combined top 20 + top 10 per n-gram length;
  **phrase minimum** (2-/3-grams need 2+ occurrences, a deviation from the brief, also
  output without it); **overlap filter** (drop a term mostly occurring inside a longer
  ranked term, 80% threshold).
- Wrote `tfidf.py`; smoke-tested on the 4 available reports (not meaningful yet:
  `max_df` 0.5 of 4 = drop anything in 3+ reports).
- Smoke test exposed cleaning bugs, now fixed in `clean.py`:
  - `-speaking community` left over from "German-speaking" → name removal now takes the
    attached hyphen compound
  - `i.e` (tokenizer drops the final dot) → added to stopwords
  - discourse words (`moreover`, `furthermore`, …) → added to stopwords
- **Sanity check fixed.** The first version averaged scores over all reports (0 where
  absent), which made rare terms look low and the check meaningless. It now reports IDF
  and the mean score where present. Finding: smoothed IDF never reaches 0 (min 1.0), so
  frequent template words still score high without `max_df`, and **`max_df` is what
  removes boilerplate**. Re-check on 27 reports.
- Early observations (4 reports, to be re-checked on 27):
  - Belgium's list is dominated by regions/federal structure (`wallonia`, `flanders`,
    `brussels-capital region`, `governance level`): this depends on the open decision
    about regions.
  - Currency codes `czk`, `dkk` rank #1 for Czechia/Denmark. Like country names, they
    identify the country without saying anything about content; possibly remove.
  - Generic report phrasing (`figure remain`, `coverage stand`, `see new measure`)
    should mostly disappear with 27 reports (higher df).

**Next:** remaining 23 PDFs → `extract.py`, `clean.py`, `tfidf.py` on all 27 → inspect.

## First results on 7 reports (2026-10-27)

- Reports: Austria, Belgium, Bulgaria, Croatia, Cyprus, Czechia, Denmark.
- Layout check of the 3 new ones: same structure. Croatia has many more footnotes
  (9 pt, correctly dropped); the 11 pt "text" above the header line in Croatia/Cyprus is
  only whitespace. Body text kept: 86–92%.
- With 7 reports `max_df` 0.5 drops terms in 4+ reports (1,685 of 20,708 terms).
- Results look meaningful for most countries: agencies, programmes and national
  platforms rank high (Austria `bba2030`, `federal chancellery`; Bulgaria `pridst`;
  Croatia `e-citizen`, `carnet`, `hakom`; Cyprus `dmrid`, `gov.cy`, `submarine cable`;
  Czechia `op tac`, `twist programme`; Denmark `technology comprehension`, `eifo`).
- Country identifiers that leak through name removal: currency codes (`czk`, `dkk`),
  ISO country code (`cy`, `ncc-cy`), capitals (`sofia`, `brussels`), regions (Belgium).
- Generic phrasing still present (`still point`, `refer`, `see new measure`, `sense`,
  `mislead`); expected to drop with more reports.

## Codes removed; run on 12 reports (2026-10-27)

- User decisions: **remove** currency codes and ISO country codes; **keep** capitals and
  regions.
- Added `COUNTRY_CODES` to `config.py`; `clean.py` strips them at token level, also in
  compounds/domains (`NCC-CY` → `ncc`, `gov.cy` → `gov`). `IT` excluded (information
  technology). Codes were removed almost only from their own report (DKK 18× in
  Denmark, CY 13× in Cyprus, CZK 10× in Czechia).
- 5 new reports (Estonia, Finland, France, Germany, Greece): same layout; all checks pass.
- Smoke-run fixes: added `pps`, `km` and French/German function words (`la`, `de`, …)
  to stopwords (they ranked high for France from names like "Osez l'IA").
- 12 reports: `max_df` 0.5 → terms in 7+ reports dropped (1,458 of 30,808).
  The 25 most widespread terms (in all 12) are all template words and all dropped.
- Observations:
  - Top terms are mostly national institutions/programmes (`lumi`, `vtt`, `traficom`,
    `sitra` for Finland; `mittelstand-digital`, `bsi`, `hightech agenda` for Germany;
    `eett`, `hellasqci` for Greece; `ria`, `eesti.ai`, `ai leap` for Estonia).
  - **Writing style shows through**: each report has its own authors, so phrases like
    `still point`, `refer` (Croatia), `annual rate`, `year-on-year` (Estonia), `might`
    (France) rank high. This is real distinctiveness, but stylistic, not topical.
  - Leftover after name removal: `num` (from "France Num"), `gov` (from gov.cy/gov.gr).

**Next:** remaining 15 PDFs (Hungary … Sweden) → re-run all three scripts.

## Full corpus: all 27 reports (2026-10-27)

- All 27 PDFs extracted and cleaned (25–40 pages each; body text kept 83–93%).
- Automatic layout check on all 27. 3 flags, all false alarms: Czechia/Portugal have a body
  sentence containing the KPI label text; Latvia has a stray "s" in the header zone
  (dropped by the position filter). Codes removed from other countries' reports are
  real mentions (`BE` in Luxembourg = "Arlon (BE)").
- Fixes from inspecting the 27 lists:
  - all-caps acronyms are no longer lemmatized (`PTS` → was `pt`, now `pts`)
  - hyphen variants normalized (`scale‑up` with U+2011 was a separate term)
  - more foreign function words from programme names (`di`, `dla`, …)
- TF-IDF: 59,168 terms; `max_df` 0.5 (> 13 reports) drops 1,557. Top 20 mix:
  222 unigrams, 215 bigrams, 103 trigrams; 381 of 540 top terms occur in only one report.
- **Sanity check passes** (see PLAN step 6): with 27 reports IDF already gives template
  terms low weight; `max_df` removes the remaining frequent ones.
- **Phrase minimum has no effect** on the top 20 (identical with/without); documented.
- Console output of the last run: `output/2026/_run_log.txt`.

**Possible next steps:** inspect the lists by eye; 2025 corpus (step 7).
