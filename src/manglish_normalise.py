"""
Manglish normalisation for VADER scoring (Step 3 of the Manglish pivot, 2026-11-18).

Produces `text_manglish_norm`, used ONLY for sentiment scoring. `text` and `text_clean`
are never changed. Steps, in order:

1. Lower-case.
2. Collapse letters repeated 3+ times to one ("cepatttt" -> "cepat", "sooo" -> "so").
3. Join multi-word Malay expressions into one token so the lexicon can score them
   (PHRASES, e.g. "puas hati" -> "puas_hati", "kurang ajar" -> "kurang_ajar").
4. Replace Malay / Manglish short forms with the English word VADER understands (WORD_MAP).
   Mapping "tak" / "x" to "not" lets VADER's own negation rule flip the next word
   ("tak best" -> "not best").
5. "gila" ("crazy") -> "very" ONLY when the next word is positive in the lexicon
   ("gila sedap" -> "very sedap"); otherwise it is left for the lexicon to judge.
6. Remove discourse particles (PARTICLES) that carry no sentiment ("lah", "lor", "meh" ...).
"""
from __future__ import annotations

import re

# Short forms -> English (one token in, one or two tokens out)
WORD_MAP = {
    # negation
    "x": "not", "tak": "not", "tk": "not", "xde": "not", "takde": "not", "tiada": "not",
    "tidak": "not", "bukan": "not",
    "xboleh": "cannot", "takboleh": "cannot", "tak_boleh": "cannot",
    "xnak": "dont want", "taknak": "dont want", "tak_nak": "dont want",
    "jgn": "dont", "jangan": "dont",
    # intensifiers / quantity
    "sgt": "very", "sangat": "very", "byk": "many", "banyak": "many",
    # time / aspect
    "dh": "already", "dah": "already", "da": "already", "sudah": "already",
    # demonstratives
    "ni": "this", "tu": "that",
    # modal
    "boleh": "can", "blh": "can",
}

# Multi-word expressions joined into one token (scored in manglish_lexicon.py).
# "tak boleh" / "tak nak" are joined first so they map to "cannot" / "dont want".
PHRASES = {
    "tak boleh": "tak_boleh", "tak nak": "tak_nak",
    "puas hati": "puas_hati", "tak puas hati": "tak puas_hati",
    "memang best": "memang_best", "kurang ajar": "kurang_ajar",
    "makan gaji": "makan_gaji", "buang masa": "buang_masa",
    "terima kasih": "terima_kasih", "sakit hati": "sakit_hati",
}

# Discourse particles: removed from the scoring text only
PARTICLES = {"lah", "la", "lor", "leh", "meh", "kan", "ah", "sia", "wei", "weh", "hor", "mah", "liao"}

REPEAT_RE = re.compile(r"([a-z])\1{2,}")
TOKEN_RE = re.compile(r"[a-z_']+|[^\sa-z_']+")


def normalise_manglish(text: str, positive_words: set[str] | frozenset = frozenset()) -> str:
    """Return the scoring text for VADER. `positive_words` = lexicon words with score > 0,
    used for the 'gila' rule (pass the merged VADER + Manglish lexicon)."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = REPEAT_RE.sub(r"\1", t)
    for phrase, joined in sorted(PHRASES.items(), key=lambda kv: -len(kv[0])):
        t = re.sub(rf"\b{re.escape(phrase)}\b", joined, t)
    toks = t.split()
    out: list[str] = []
    for i, tok in enumerate(toks):
        core = tok.strip(".,!?;:\"'()[]")
        if core in PARTICLES:
            continue
        if core == "gila":
            nxt = toks[i + 1].strip(".,!?;:\"'()[]") if i + 1 < len(toks) else ""
            if nxt in positive_words:
                out.append(tok.replace(core, "very"))
                continue
        if core in WORD_MAP:
            out.append(tok.replace(core, WORD_MAP[core]))
            continue
        out.append(tok)
    return " ".join(out)
