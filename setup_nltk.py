"""Download the NLTK data packages this project uses into the virtual environment.

Run once after `pip install -r requirements.txt`:
    .venv/bin/python setup_nltk.py
"""

import sys
from pathlib import Path

import nltk

PACKAGES = [
    "punkt_tab",                       # word_tokenize / sent_tokenize
    "stopwords",                       # English stopword list
    "averaged_perceptron_tagger_eng",  # POS tags, so the lemmatizer knows noun vs verb
    "wordnet", "omw-1.4",              # WordNetLemmatizer
]

# <venv>/nltk_data is on NLTK's default search path, so nothing else needs configuring.
target = Path(sys.prefix) / "nltk_data"
for package in PACKAGES:
    nltk.download(package, download_dir=str(target), quiet=True)
print(f"NLTK data installed in {target}")
