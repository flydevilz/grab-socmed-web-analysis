"""
Build data/processed/labelled_dataset_v3.csv (Manglish pivot, 2026-11-18).

v2 (1,341 rows, English-only) + posts restored by the Manglish language rule + X round 3,
each cleaned with src/clean.py and labelled by Claude against data/labelling/LABELLING_GUIDE.md.
Adds two columns to the 27-column v2 schema: `lang_group` and `text_manglish_norm`.
v2 is never modified; it is backed up before the merge.
"""
from __future__ import annotations

import shutil

import pandas as pd

from src.clean import detect_english, lang_group
from src.collect import PROJECT_ROOT
from src.label import add_new_labelled
from src.manglish_lexicon import vader_with_manglish
from src.manglish_normalise import normalise_manglish

P = PROJECT_ROOT / "data" / "processed"
L = PROJECT_ROOT / "data" / "labelling"

NEW_SOURCES = [  # (cleaned file, label file)
    ("manglish_restored_clean.csv", "claude_labels_manglish_restored.csv"),
    ("twitter_round3_clean.csv", "claude_labels_twitter_round3.csv"),
]
# Existing v2 rows re-read with the Manglish rules (Step 5): post_id -> (label, reason)
RECHECK_CHANGES = {"1992165330396037136": ("positive", "praises the Grab app tracking; re-checked 2026-11-18")}


def build_v3(recheck_changes: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    recheck_changes = recheck_changes if recheck_changes is not None else RECHECK_CHANGES
    v2_path = P / "labelled_dataset_v2.csv"
    backup = P / "labelled_dataset_v2_backup_20261118.csv"
    if not backup.exists():
        shutil.copy2(v2_path, backup)
    v2 = pd.read_csv(v2_path, dtype={"post_id": str})
    v2["lang_group"] = [lang_group(t, detect_english(t)) for t in v2["text"].fillna("")]
    for pid, (label, reason) in recheck_changes.items():
        m = v2["post_id"] == pid
        v2.loc[m, ["claude_label", "unsure", "reason"]] = [label, True, reason]
        # the teammate's correction (if any) still wins
        v2.loc[m & v2["corrected_label"].isna(), "sentiment"] = label
    merged, irrelevant = v2, []
    for clean_f, label_f in NEW_SOURCES:
        nc = pd.read_csv(P / clean_f, dtype={"post_id": str})
        nl = pd.read_csv(L / label_f, dtype={"post_id": str})
        merged, irr = add_new_labelled(merged, nc, nl)
        irrelevant.append(irr.assign(source=clean_f.replace("_clean.csv", "")))
    sia = vader_with_manglish()
    pos = frozenset(w for w, s in sia.lexicon.items() if s > 0)
    merged["text_manglish_norm"] = merged["text_clean"].fillna("").map(lambda t: normalise_manglish(t, pos))
    return merged, pd.concat(irrelevant, ignore_index=True)
