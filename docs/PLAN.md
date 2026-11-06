# Plan and cleaning decisions

Status: **full pipeline run on all 54 reports (2025 + 2026)**. Results in `output/combined/`.

## Goal

For each of the 27 Digital Decade 2026 country reports, find the 15–20 terms that are
most distinctive to that report compared to the other 26. All reports follow the same
template, so shared template language should get low weight and country-specific
content (projects, policies, problems) should rise to the top.

## Steps

| # | Step | Script | Status |
|---|---|---|---|
| 1 | Setup: venv, requirements, docs | — | done |
| 2 | Extract text from PDFs, inspect structure | `extract.py` | done (27/27) |
| 3 | Clean text | `clean.py` | done (27/27) |
| 4 | TF-IDF | `tfidf.py` | done (27/27) |
| 5 | Top 15–20 terms per country: CSV + Markdown summary | `tfidf.py` | done (27/27) |
| 6 | Sanity check: document frequency vs. weight | `tfidf.py` | done (27/27) |
| 7 | 2025 corpus, year comparison in one 54-report analysis | `extract.py` + `clean.py` per year, then `tfidf.py` | done |
| 8 | Interactive word clouds per country and year + most common terms | `wordclouds.py` | done |

## Step 2: extraction

- **PyMuPDF**. For each country `extract.py` writes:
  - `<country>.txt`: plain text, pages separated by a form feed (`\f`), for reading.
  - `<country>.spans.csv`: every text span with page, block, line, font, size and
    position. **Cleaning works on this file**, because font and position identify
    the document parts much more reliably than text patterns.
  - `_structure.csv` (pages, characters, words, near-empty pages).

### Document structure found (4 reports inspected: AT, BE, CZ, DK)

All four follow the same layout, with the same fonts (Calibri) and positions:

| Part | Where | Font | Action |
|---|---|---|---|
| Cover page (SWD number, "Accompanying the document…") | p. 1 | Times New Roman 12 | drop |
| Title page ("DIGITAL DECADE COUNTRY REPORT 2026") | p. 2 | large; text over an image | drop |
| Table of contents (dot leaders) | p. 3 | 11 | drop |
| Running header = country name | top of every page, y < 60 | 20 pt | drop |
| Page number | bottom, y ≈ 782 | 11 pt | drop |
| Section headings | — | 15 / 26 pt, Calibri-Light | drop |
| Subheadings ("Performance assessment", "Policy context…") | — | 11–12 pt Calibri-Light | drop |
| **Body text, incl. bold sentences and the Recommendations box** | — | **11 pt Calibri / Bold / Italic** | **keep** |
| KPI table in the executive summary + its notes | p. 5–6 | 8 pt (title 14 pt) | drop |
| Footnotes | page bottom | 9 or 10 pt (varies by report) | drop |
| Superscript footnote markers, glued to words (`roadmap1`) | inline | 6.5–7 pt | drop |
| Chart captions ("RRF spillover effects to Austria") | annex | 9 pt italic | drop |
| Charts, regional coverage tables, maps | — | **images** | no text, nothing to do |

**Unexpected / worth knowing:**

- **Charts and the NUTS-2 regional tables are images**, so their content isn't in the
  text at all. The body text often describes them, though.
- **Footnote markers are glued to words** in plain-text extraction (`roadmap1`,
  `billion)2`). The font-based filter removes them cleanly.
- **Footnote size differs between reports** (9 pt in some, 10 pt in others). A fixed
  "drop 10 pt" rule would have missed some, so the rule is "keep only 11 pt".
- **Annex II is almost word-for-word template** with only numbers and the
  country name changed, and it names other countries ("spillover effects from RRPs of
  Italy, Germany, Spain"). Country-name removal and `max_df` handle this.
- **The Eurobarometer paragraph and the "Funding" paragraph** are also template text
  with country-specific numbers.
- **National-language terms appear** (`Unternehmensserviceportal`, `Datenzugangsgesetz`,
  `MinVirksomhed`, `IT4Innovations`, `Quantum4Belgium`). They are kept on purpose:
  they're specific to each country and are likely to become top TF-IDF terms.
- **Belgium mentions "German" and "French"** (8× and 1×) as its language communities.
  These are removed as country adjectives, which is acceptable.
- The near-empty page flagged by `extract.py` in CZ and DK is the title page (text over
  an image), not a scanned page. No OCR is needed.

## Step 3: cleaning decisions (implemented in `clean.py`)

1. **Keep body text only, chosen by font**: spans that are 11 pt, not Calibri-Light,
   between y = 60 and y = 770, on pages after the one containing "Contents". This
   removes the cover, title page, TOC, header, page numbers, headings, KPI table,
   footnotes, superscript markers and chart captions in one rule (see table above).
   Safety net: any paragraph with dot leaders (`.....`) is dropped too.
   Result: about 90% of the characters are kept (e.g. AT 82k → 75k).
   - Headings are dropped deliberately. They're identical in every report, so they'd
     only add template words, and they would raise the document frequency of topic
     words like *semiconductor* even in reports that barely discuss them.
2. **Rebuild paragraphs**: lines of one PDF text block are joined. A line ending in `-`
   is joined without a space (`cross-` + `border` → `cross-border`). In these
   Word-generated PDFs a line-end hyphen is a real compound hyphen, not word-splitting.
3. **Normalize**: curly quotes → straight, dashes → spaced hyphen, hyphen variants
   (U+2010, U+2011, soft hyphen) → `-`, narrow/non-breaking spaces → space, bullets removed, `a/b` → `a / b`.
4. **Country names** are replaced by a marker, case-sensitively, including possessive
   `'s` and an attached hyphen compound (`German-speaking`, `Austria-wide`, otherwise
   `-speaking` was left behind), longest form first (*Czech Republic* before *Czech*). Forms are in `config.py`.
   All removals → `data/clean/<year>/_removed_names.csv` (document, form, count,
   own_country). The 4 reports had 190–245 removals each, almost all of them the
   report's own country.
   - Capitals, regions and agency names are **not** removed (open question for the
     user). Flanders/Wallonia/Flemish/Walloon were taken out of the removal list so
     that all regions are handled the same way.
4b. **Country codes and currencies** (decided with the user): ISO codes and national
   currency codes in `config.COUNTRY_CODES` (`AT`, `CY`, `EL`/`GR`, `CZK`, `DKK`,
   `PLN`, …) are removed **at token level**, uppercase only, and only when they stand
   alone. Tokens that contain a code stay whole (`gov.cy`, `Red.es`, `NCC-CY`), because
   cutting them lost information (`Red.es` became `red`). The cost: `ncc-cy` and `ncc-hr`
   are separate terms although each is a National Competence Centre. `IT` (Italy) is left out on
   purpose: in these reports it means information technology. `HR` (Croatia) could in
   theory mean human resources; in the 12 reports it was only removed from the Croatian
   report. Removals are counted in `_removed_names.csv`.
   Capitals and regions are **kept** (user decision): Belgium's regions and Germany's
   `länder` are part of their reports' content.
5. **NLTK tokenizing**: `sent_tokenize` → `word_tokenize`. Hyphenated compounds
   stay one token.
6. **NLTK POS tagging + lemmatizing**: `pos_tag_sents`, Penn tag mapped to
   WordNet (adj/verb/adv/noun), `WordNetLemmatizer`. Tagging uses the original case;
   lemmas are lowercase. Extra rules:
   - plural acronyms lose the `s` (`SMEs` → `sme`, `EDICs` → `edic`); all-caps
     acronyms are never lemmatized (`PTS` stays `pts`, WordNet would make it `pt`)
   - in hyphenated words only the last part is lemmatized (`start-ups` → `start-up`)
   - if the tagged POS gives no change, retry as a noun (fixes mis-tagged plurals
     like `Technologies`). There's no verb retry, so `computing` stays `computing`
     (needed for "high-performance computing").
7. **Stopwords and filtering**: NLTK English stopwords (198) + a short custom list
   (discourse words `also, however, moreover, furthermore, nevertheless, therefore,
   thus, meanwhile, additionally, e.g., i.e., etc, via, within, well, yet` (both
   `i.e.` and `i.e`, since the tokenizer drops the final dot), units/currency
   `eur, million, billion, per, cent, percent, pp, pps, kg, km`, month names, roman-numeral
   list markers `ii, iii, iv, …` (560× in the corpus, from "(ii)", "(iii)"), and French/German
   function words that occur inside programme names: `la, le, les, de, des, du, et, der,
   das, und, für` and Italian/Polish/Spanish/Swedish `di, del, della, delle, dla, na, el,
   los, las, para, och, av, för`; not NLTK's full French/German lists, which contain English words
   like `war`, `will`). Tokens without a
   letter (numbers, percentages, punctuation) and 1-character tokens are dropped.
   `5g` and `4g` stay (they contain a letter).
   Template words (*digital, eu, member state, commission*) are **not** stopwords;
   `max_df` should remove them.
8. **Word counts with NLTK** `FreqDist` → `data/clean/<year>/_word_counts.csv`
   (top 50 per document). The raw top words are template words (*eu, digital, average,
   support, service*) for all four reports, as expected. They have to be dropped by
   `max_df` for TF-IDF to work.

### N-grams (1–3 words)

If stopwords are deleted first and n-grams built afterwards, false phrases appear:
*"use of cloud by enterprises"* → *"cloud enterprise"*. To avoid this:

- Stopwords and removed country names are replaced by a **boundary marker**, and
  sentence ends are boundaries too.
- N-grams (1, 2, 3 words) are built only **inside** stretches without a boundary.
  So *"digital public services"* → `digital`, `public`, `service`, `digital public`,
  `public service`, `digital public service`.
- The cleaned text is saved one chunk per line in `data/clean/<year>/<country>.txt`,
  readable by eye.

## Step 4: TF-IDF (implemented in `tfidf.py`)

Each report is **one document**, labelled `<country>_<year>`. The chunks only decide which n-grams exist:
all 1-, 2- and 3-grams inside each chunk are pooled into one bag of terms per report.
A flat token list would create false n-grams across removed words: in the Austrian
report 3,218 of 6,061 flat bigrams were false (`eu growth`, `cloud data` from
"cloud *and* data").

`TfidfVectorizer(analyzer=ngrams_in_chunks, ...)`: a custom analyzer, so sklearn
doesn't re-tokenize. sklearn's own `stop_words` + `ngram_range` would create the same
false n-grams.

| Parameter | Value | Why |
|---|---|---|
| n-grams | 1–3, inside chunks | multi-word terms without false phrases |
| `sublinear_tf` | True | tf → 1 + ln(tf), so a word used 40 times isn't 40× as important |
| `min_df` | 1 | keep terms in only one report: the most distinctive |
| `max_df` | 0.4 | drop terms in more than 40% of the reports (> 21 of 54) |
| `smooth_idf` | True (default) | idf = ln((1+N)/(1+df)) + 1 |
| `norm` | `"l2"` (default) | long and short reports comparable |

### Ranking filter (added on top of the brief, agreed with the user)

**Overlap filter:** a term is dropped when a longer ranked term containing it occurs
at least 80% as often in the same report (`SUBSUME_SHARE = 0.8`). So
`broadband router` goes if it nearly always appears inside `mobile broadband
router`, but `router` stays if it is also used on its own. This keeps one idea from
filling several places in the top 20.

The filter affects only the ranking. The TF-IDF scores themselves are computed on
the full vocabulary.

## Step 5: outputs (`output/combined/`)

| File | Content |
|---|---|
| `top_terms.csv` | **main result**: report, rank, term, n, score, count, doc_freq (top 20) |
| `top_terms_by_ngram.csv` | top 10 per country for each n-gram length (1, 2, 3) |
| `sanity_doc_freq.csv` | every term: n, doc_freq, total count, idf, score without cut, dropped |
| `summary.md` | parameters, top terms per country, sanity table |
| `charts/top_terms_<country>_<year>.png` | bar chart for the reports in `SHOW_REPORTS` |
| `charts/doc_freq_vs_score.png` | sanity chart |
| `wordclouds.html` | word clouds (from `wordclouds.py`): click a word to see every report that uses it |

## Step 6: sanity check

- Prints the terms with the highest document frequency, with their IDF, the score they
  *would* get without `max_df` (mean over the reports containing them), and whether
  `max_df` drops them.
- **Important finding (smoke test, 4 reports):** sklearn's smoothed IDF never reaches 0.
  A term in every report keeps IDF = 1.0; the maximum (df = 1) is ln(28/2)+1 ≈ 3.6 with
  27 reports. Very frequent template words (`digital`, 762 times) can therefore still
  score as high as distinctive terms. **IDF alone does not remove boilerplate; `max_df`
  does.** The check has to be repeated on all 27 reports.
- **Full corpus (27 reports):** the check passes. Terms in 1–3 reports reach mean
  scores up to 0.09; terms in 14+ reports stay around 0.01–0.03 even without the cut
  (`charts/doc_freq_vs_score.png`). With 27 documents IDF does most of the work;
  `max_df` removes the remaining very frequent template words (1,557 of 59,168 terms,
  among them all 25 most widespread terms, each present in all 27 reports).
- A first version averaged scores over *all* reports (0 where absent), which made rare
  terms look low-weighted and the check meaningless. That was fixed.

## Step 7: 2025 corpus and year comparison

Set `YEAR` in `config.py` and run `extract.py` + `clean.py` once per year. `tfidf.py` and
`wordclouds.py` then read every year in `YEARS` as one corpus of 54 reports.

- **Why one corpus:** with one vocabulary and one IDF, a term's 2025 and 2026 scores can
  be compared directly. Separate runs give each year its own IDF.
- **Why `max_df` is 0.4:** at 0.5 a term must be in more than 27 of 54 reports to be
  dropped, so wording used in all 27 reports of one year (`df` = 27) would stay. 190 of
  the 570 terms with `df` 22-27 are concentrated (85%+) in one year. In practice the
  change is small: 0.4 and 0.5 differ in 1 of the 1,080 top-20 places.
- **Known bias:** a term in both of a country's reports has `df` 2 instead of 1, so IDF
  4.31 instead of 4.72, which favours terms from one year. The effect on the comparison
  is small: top-20 lists share 2.5 of 20 terms per country here, 2.7 in separate runs.
- 2025 layout: same structure as 2026; 89.8% of characters kept (2026: 91.2%).
