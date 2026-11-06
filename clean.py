"""Step 3: clean the extracted text and turn it into lemmatized chunks for TF-IDF.

Input:  data/text/<year>/<country>.spans.csv   (from extract.py)
Output: data/clean/<year>/<country>.body.txt   body text after structural cleaning and
                                               country-name removal (readable check)
        data/clean/<year>/<country>.txt        lemmatized chunks, one per line; a chunk
                                               is a run of words with no stopword,
                                               punctuation, number or country name
                                               in between, so n-grams never cross them
        data/clean/<year>/_word_counts.csv     NLTK FreqDist: top words per document
        output/<year>/removed_names.csv        country-name removals per document

See docs/PLAN.md for the reasoning behind each rule.

Usage: .venv/bin/python clean.py
"""

import re
from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer

from config import CLEAN_DIR, COUNTRIES, COUNTRY_CODES, OTHER_COUNTRIES, OUTPUT_DIR, TEXT_DIR

# --- Structural cleaning (font-based, see docs/PLAN.md) ---------------------------
BODY_SIZE = 11      # body text is 11 pt; footnotes 9-10, KPI table 8, superscripts ~7
HEADER_Y = 60       # running header (country name, 20 pt) sits above this
FOOTER_Y = 770      # page number sits below this

# --- Token filtering ---------------------------------------------------------------
BOUNDARY = "|"      # marks a break between chunks
COUNTRY_MARK = "XXCOUNTRYXX"
EXTRA_STOPWORDS = {
    # discourse words and abbreviations
    "also", "however", "moreover", "furthermore", "nevertheless", "therefore", "thus",
    "meanwhile", "additionally", "e.g.", "e.g", "i.e.", "i.e", "etc", "via", "within",
    "well", "yet",
    # roman numerals used as list markers: "(ii)", "(iii)"
    "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii",
    # units and currency (the numbers themselves are dropped anyway)
    "eur", "euro", "million", "billion", "per", "cent", "percent", "pp", "pps", "kg", "km",
    # function words inside national programme names ("Osez l'IA", "Politecnico di
    # Milano", "Fundusze Europejskie dla ..."): French, German, Italian, Polish,
    # Spanish, Swedish
    "la", "le", "les", "de", "des", "du", "et", "der", "das", "und", "für",
    "di", "del", "della", "delle", "dla", "na", "el", "los", "las", "para", "och", "av", "för",
    # months
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}
STOPWORDS = set(stopwords.words("english")) | EXTRA_STOPWORDS

lemmatizer = WordNetLemmatizer()


# ===================================================================================
# 1. Structure: keep body text only
# ===================================================================================

def first_content_page(spans):
    """First page after the table of contents (the page with a 'Contents' line)."""
    contents = spans[spans["text"].str.strip() == "Contents"]
    return int(contents["page"].min()) + 1 if len(contents) else 1


def body_paragraphs(spans):
    """Keep 11 pt regular/bold/italic text between header and footer, after the TOC.

    Drops: cover, title page and TOC; running header and page numbers; headings
    (Calibri-Light, 15+ pt); footnotes and their superscript markers; the KPI
    table (8 pt); chart captions (9 pt). Returns one string per PDF text block.
    """
    keep = (
        (spans["page"] >= first_content_page(spans))
        & (spans["size"].round() == BODY_SIZE)
        & ~spans["font"].str.contains("Light")
        & (spans["y0"] > HEADER_Y)
        & (spans["y1"] < FOOTER_Y)
    )
    body = spans[keep]

    paragraphs = []
    for _, block in body.groupby(["page", "block"], sort=True):
        lines = [
            "".join(line["text"].astype(str)).strip()
            for _, line in block.groupby("line", sort=True)
        ]
        paragraphs.append(join_lines([l for l in lines if l]))
    return [p for p in paragraphs if p and not re.search(r"\.{5,}", p)]


def join_lines(lines):
    """Join the lines of one block. A line ending in '-' is a compound word broken
    over two lines ('cross-' + 'border'), so it is joined without a space."""
    text = ""
    for line in lines:
        if text.endswith("-"):
            text += line
        else:
            text += " " + line if text else line
    return text


def normalize(text):
    """Unify quotes, dashes and spaces."""
    text = text.replace("’", "'").replace("‘", "'")      # normalizing quotes
    text = text.replace("“", '"').replace("”", '"')      # normalizing double quotes
    text = text.replace("–", " - ").replace("—", " - ")  # splitting words so they become two different tokens
    text = re.sub(r"[   ]", " ", text)                   # normalizing spaces
    text = text.replace("•", " ")                        # removing bullet points
    text = re.sub(r"[‐‑­]", "-", text)                    # normalizing hyphens
    text = re.sub(r"(?<=[A-Za-z])/(?=[A-Za-z])", " / ", text)  # splitting words so they become two different tokens
    return re.sub(r"\s+", " ", text).strip()             # normalizing multiple whitespace characters into a single space


# ===================================================================================
# 2. Country names
# ===================================================================================

ALL_NAMES = sorted(
    {form for forms in COUNTRIES.values() for form in forms} | set(OTHER_COUNTRIES),
    key=len, reverse=True,  # longest first: "Czech Republic" before "Czech"
)
NAME_PATTERN = re.compile(
    # a hyphenated compound goes with the name: "German-speaking", "Austria-wide"
    r"\b(" + "|".join(re.escape(n) for n in ALL_NAMES) + r")(?:'s|-\w+)?\b"
)


def remove_country_names(text, country, counter):
    """Replace country names (case-sensitive) with a marker; count what was removed."""
    def replace(match):
        counter[match.group(1)] += 1
        return f" {COUNTRY_MARK} "
    return NAME_PATTERN.sub(replace, text)


CODES = {code for codes in COUNTRY_CODES.values() for code in codes}


# ===================================================================================
# 3. NLTK: tokenize, POS-tag, lemmatize, remove stopwords
# ===================================================================================

def wordnet_pos(tag):
    """Map a Penn Treebank tag to the WordNet POS the lemmatizer expects."""
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag[0], wordnet.NOUN)


def lemmatize(word, tag):
    """Lemmatize one token. Plural acronyms ('SMEs') lose the 's'; in hyphenated
    compounds only the last part is lemmatized ('start-ups' -> 'start-up')."""
    if re.fullmatch(r"[A-Z]{2,}s", word):
        return word[:-1].lower()
    if re.fullmatch(r"[A-Z0-9]{2,}", word):  # acronym: never lemmatize ("PTS" is no plural)
        return word.lower()
    word = word.lower()
    if "-" in word:
        head, _, tail = word.rpartition("-")
        return f"{head}-{lemmatize_word(tail, tag)}" if tail else word
    return lemmatize_word(word, tag)


def lemmatize_word(word, tag):
    """Lemmatize with the tagged POS; if that changes nothing, try it as a noun
    (catches plurals the tagger got wrong, e.g. 'Technologies' tagged as a verb)."""
    lemma = lemmatizer.lemmatize(word, wordnet_pos(tag))
    return lemmatizer.lemmatize(word, wordnet.NOUN) if lemma == word else lemma


def is_content(lemma):
    """A token is kept if it has a letter, is 2+ characters and is not a stopword."""
    return (
        len(lemma) >= 2
        and re.search(r"[a-z]", lemma) is not None
        and lemma not in STOPWORDS
        and lemma != COUNTRY_MARK.lower()
    )


def to_chunks(paragraphs, removed):
    """Turn paragraphs into chunks of consecutive content lemmas.

    Sentence ends, stopwords, punctuation, numbers and removed country names and
    codes all end a chunk. N-grams are later built inside chunks only.
    """
    sentences = [nltk.word_tokenize(s) for p in paragraphs for s in nltk.sent_tokenize(p)]
    chunks = []
    for tagged in nltk.pos_tag_sents(sentences):
        current = []
        for word, tag in tagged:
            word = word.strip("'")
            if word in CODES:  # standalone only: "gov.cy" and "NCC-CY" stay whole
                removed[word] += 1
                word = ""
            lemma = lemmatize(word, tag) if word else ""
            if is_content(lemma):
                current.append(lemma)
            elif current:
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)
    return chunks


# ===================================================================================

def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    removed_rows, count_rows, stats = [], [], []
    for country in COUNTRIES:
        spans_path = TEXT_DIR / f"{country}.spans.csv"
        if not spans_path.exists():
            continue
        spans = pd.read_csv(spans_path, keep_default_na=False)

        paragraphs = [normalize(p) for p in body_paragraphs(spans)]
        removed = Counter()
        paragraphs = [remove_country_names(p, country, removed) for p in paragraphs]
        (CLEAN_DIR / f"{country}.body.txt").write_text("\n\n".join(paragraphs), encoding="utf-8")

        chunks = to_chunks(paragraphs, removed)
        (CLEAN_DIR / f"{country}.txt").write_text(
            "\n".join(" ".join(c) for c in chunks), encoding="utf-8"
        )

        own_forms = set(COUNTRIES[country]) | set(COUNTRY_CODES[country])
        for form, n in removed.most_common():
            removed_rows.append({"document": country, "removed": form, "count": n,
                                 "own_country": form in own_forms})

        freq = FreqDist(lemma for chunk in chunks for lemma in chunk)
        for rank, (term, n) in enumerate(freq.most_common(50), start=1):
            count_rows.append({"document": country, "rank": rank, "term": term, "count": n})

        stats.append({
            "country": country,
            "raw_chars": int(spans["text"].astype(str).str.len().sum()),
            "body_chars": sum(len(p) for p in paragraphs),
            "names_removed": sum(removed.values()),
            "tokens_kept": freq.N(),
            "unique_lemmas": freq.B(),
            "chunks": len(chunks),
        })

    print(pd.DataFrame(stats).to_string(index=False))
    pd.DataFrame(removed_rows).to_csv(OUTPUT_DIR / "removed_names.csv", index=False)
    pd.DataFrame(count_rows).to_csv(CLEAN_DIR / "_word_counts.csv", index=False)
    


if __name__ == "__main__":
    main()
