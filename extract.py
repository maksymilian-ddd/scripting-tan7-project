"""Step 2: extract plain text from each country PDF with PyMuPDF.

Input:  data/raw/<year>/<country>.pdf
Output: data/text/<year>/<country>.txt   pages separated by a form feed (\\f), so
                                          clean.py can still work page by page
        data/text/<year>/<country>.spans.csv  every text span with page, position,
                                              font and size; clean.py uses this to
                                              keep only body text
        data/text/<year>/_structure.csv   per-document page, character and word
                                          counts, with a TOTAL row

Usage: .venv/bin/python extract.py
"""

import pandas as pd
import pymupdf

from config import COUNTRIES, RAW_DIR, TEXT_DIR


def extract_pages(pdf_path):
    """Return a list with the text of each page, in reading order."""
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text("text", sort=True) for page in doc]


def extract_spans(pdf_path):
    """Return one row per text span: page, block, line, font, size, position, text."""
    rows = []
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            for block_no, block in enumerate(page.get_text("dict", sort=True)["blocks"]):
                for line_no, line in enumerate(block.get("lines", [])):
                    for span in line["spans"]:
                        x0, y0, x1, y1 = span["bbox"]
                        rows.append({
                            "page": page_no, "block": block_no, "line": line_no,
                            "font": span["font"], "size": round(span["size"], 1),
                            "x0": round(x0, 1), "y0": round(y0, 1),
                            "x1": round(x1, 1), "y1": round(y1, 1),
                            "text": span["text"],
                        })
    return pd.DataFrame(rows)


def main():
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [c for c in COUNTRIES if not (RAW_DIR / f"{c}.pdf").exists()]
    unexpected = [p.name for p in RAW_DIR.glob("*.pdf") if p.stem not in COUNTRIES]
    if missing:
        print(f"Missing PDFs ({len(missing)}): {', '.join(missing)}")
    if unexpected:
        print(f"Unexpected file names (not processed): {', '.join(unexpected)}")

    rows = []
    for country in COUNTRIES:
        pdf_path = RAW_DIR / f"{country}.pdf"
        if not pdf_path.exists():
            continue
        pages = extract_pages(pdf_path)
        (TEXT_DIR / f"{country}.txt").write_text("\f".join(pages), encoding="utf-8")
        extract_spans(pdf_path).to_csv(TEXT_DIR / f"{country}.spans.csv", index=False)

        chars = [len(p) for p in pages]
        rows.append({
            "country": country,
            "pages": len(pages),
            "chars": sum(chars),
            "words": sum(len(p.split()) for p in pages),
            "empty_pages": sum(c < 50 for c in chars),  # possibly scanned or image-only
        })
        print(f"{country:12s} {len(pages):3d} pages {sum(chars):8d} chars")

    if rows:
        structure = pd.DataFrame(rows)
        total = structure.drop(columns="country").sum().to_dict()
        structure.loc[len(structure)] = {"country": "TOTAL", **total}
        structure.to_csv(TEXT_DIR / "_structure.csv", index=False)
        print(f"\nExtracted {len(rows)} of {len(COUNTRIES)} documents into {TEXT_DIR}")


if __name__ == "__main__":
    main()
