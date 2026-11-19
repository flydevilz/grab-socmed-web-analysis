"""
Custom Manglish / Malay lexicon for VADER (Step 4 of the Manglish pivot, 2026-11-18).

Scores use VADER's scale (-4 most negative ... +4 most positive). Applied with
`analyzer.lexicon.update(MANGLISH_LEXICON)`. Multi-word entries use "_" and only match after
`manglish_normalise.normalise_manglish` has joined the phrase (e.g. "puas hati" -> "puas_hati").

SEED words were given by the team. Further words are added only after the team reviews the
top unscored tokens in the dataset (see notebook 04), so every score can be explained.
"""
from __future__ import annotations

SEED = {
    "teruk": -2.5, "bagus": 2.5, "mantap": 2.5, "scam": -3.0, "penipu": -3.0, "lembap": -2.0,
    "lambat": -1.5, "cepat": 2.0, "murah": 1.5, "mahal": -1.5, "sabar": 0.5, "bodoh": -2.5,
    "kecewa": -2.3, "puas_hati": 2.3, "memang_best": 2.5, "terbaik": 2.5, "ok": 0.8,
    "power": 1.5, "padu": 2.0, "sedap": 2.0, "tipu": -3.0, "bising": -1.5, "kurang_ajar": -3.0,
    "cancel": -1.5, "surge": -1.5, "mesra": 2.0, "senang": 1.5, "susah": -1.5, "rugi": -2.0,
    "jimat": 1.8,
}

# Added after the team reviewed the top unscored tokens (approved 2026-11-18)
REVIEWED: dict[str, float] = {
    "baik": 1.8, "ramah": 2.0, "sopan": 2.0, "laju": 1.5, "cun": 2.0, "syukur": 2.0, "lega": 1.5,
    "terima_kasih": 1.8,
    "kena": -0.8, "geram": -2.0, "marah": -2.3, "menyampah": -2.5, "hampeh": -2.5, "kasar": -2.0,
    "lembab": -2.0, "lmbt": -1.5, "sakit_hati": -2.5, "sial": -2.5, "babi": -2.5,
}

# Malay intensifiers, treated like VADER's boosters ("very", "really") instead of scored words
BOOSTERS = {"memang": 0.293, "mmg": 0.293}

MANGLISH_LEXICON = {**SEED, **REVIEWED}


def vader_with_manglish(extra: dict | None = None):
    """A fresh VADER analyzer with the Manglish lexicon added (plain VADER stays untouched)."""
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(MANGLISH_LEXICON | (extra or {}))
    # VADER keeps boosters in a module-level dict; copying it keeps plain VADER unchanged.
    sia.constants.BOOSTER_DICT = {**sia.constants.BOOSTER_DICT, **BOOSTERS}
    return sia
