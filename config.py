"""Shared settings: paths, the list of countries, and the names removed during cleaning."""

from pathlib import Path

ROOT = Path(__file__).parent
YEAR = "2026"             # the year extract.py and clean.py process
YEARS = ["2025", "2026"]  # the years tfidf.py and wordclouds.py analyse together

RAW_DIR = ROOT / "data" / "raw" / YEAR      # PDFs, one per country: <slug>.pdf
TEXT_DIR = ROOT / "data" / "text" / YEAR    # raw extracted text
CLEAN_DIR = ROOT / "data" / "clean" / YEAR  # cleaned text
OUTPUT_DIR = ROOT / "output" / YEAR         # removed_names.csv from clean.py
# TF-IDF results, CSV, Markdown, charts
RESULTS_DIR = ROOT / "output" / ("combined" if len(YEARS) > 1 else YEARS[0])

# slug (file name) -> forms of the country name that get removed from that report
# and from all others. Matching is case-sensitive on the original text, so
# "Polish" is removed but the verb "polish" is not.
COUNTRIES = {
    "austria":     ["Austria", "Austrian", "Austrians"],
    "belgium":     ["Belgium", "Belgian", "Belgians"],
    "bulgaria":    ["Bulgaria", "Bulgarian", "Bulgarians"],
    "croatia":     ["Croatia", "Croatian", "Croatians"],
    "cyprus":      ["Cyprus", "Cypriot", "Cypriots"],
    "czechia":     ["Czechia", "Czech Republic", "Czech", "Czechs"],
    "denmark":     ["Denmark", "Danish", "Danes"],
    "estonia":     ["Estonia", "Estonian", "Estonians"],
    "finland":     ["Finland", "Finnish", "Finns"],
    "france":      ["France", "French"],
    "germany":     ["Germany", "German", "Germans"],
    "greece":      ["Greece", "Greek", "Greeks", "Hellenic"],
    "hungary":     ["Hungary", "Hungarian", "Hungarians"],
    "ireland":     ["Ireland", "Irish"],
    "italy":       ["Italy", "Italian", "Italians"],
    "latvia":      ["Latvia", "Latvian", "Latvians"],
    "lithuania":   ["Lithuania", "Lithuanian", "Lithuanians"],
    "luxembourg":  ["Luxembourg", "Luxembourgish", "Luxembourger", "Luxembourgers"],
    "malta":       ["Malta", "Maltese"],
    "netherlands": ["the Netherlands", "Netherlands", "Dutch"],
    "poland":      ["Poland", "Polish", "Poles"],
    "portugal":    ["Portugal", "Portuguese"],
    "romania":     ["Romania", "Romanian", "Romanians"],
    "slovakia":    ["Slovakia", "Slovak", "Slovaks", "Slovakian"],
    "slovenia":    ["Slovenia", "Slovenian", "Slovenians", "Slovene"],
    "spain":       ["Spain", "Spanish", "Spaniards"],
    "sweden":      ["Sweden", "Swedish", "Swedes"],
}

# Non-EU countries likely to be mentioned (partnerships, comparisons). Also removed.
OTHER_COUNTRIES = [
    "United States", "US", "USA", "American", "China", "Chinese", "Ukraine", "Ukrainian",
    "United Kingdom", "UK", "British", "Switzerland", "Swiss", "Norway", "Norwegian",
    "Iceland", "Japan", "Japanese", "South Korea", "Korea", "Korean", "India", "Indian",
    "Moldova", "Western Balkans",
]

# Country and currency codes removed at token level (clean.py), when they stand alone.
# Tokens that contain a code stay whole: "gov.cy", "Red.es", "NCC-CY".
# "IT" (Italy) is deliberately left out: in these reports it means information technology.
COUNTRY_CODES = {
    "austria": ["AT"], "belgium": ["BE"], "bulgaria": ["BG", "BGN"], "croatia": ["HR"],
    "cyprus": ["CY"], "czechia": ["CZ", "CZK"], "denmark": ["DK", "DKK"],
    "estonia": ["EE"], "finland": ["FI"], "france": ["FR"], "germany": ["DE"],
    "greece": ["EL", "GR"], "hungary": ["HU", "HUF"], "ireland": ["IE"], "italy": [],
    "latvia": ["LV"], "lithuania": ["LT"], "luxembourg": ["LU"], "malta": ["MT"],
    "netherlands": ["NL"], "poland": ["PL", "PLN"], "portugal": ["PT"],
    "romania": ["RO", "RON"], "slovakia": ["SK"], "slovenia": ["SI"], "spain": ["ES"],
    "sweden": ["SE", "SEK"],
}
