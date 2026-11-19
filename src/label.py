"""
Helpers for notebook 03_manual_labelling.

Part 1: draw a stratified random sample and build the review file from the hand labels
        (data/labelling/claude_labels.csv, written by reading each post against the guide).
Part 2: merge the second checker's corrections, drop "irrelevant" posts and save the
        labelled dataset.
"""
from __future__ import annotations

import pandas as pd

LABELS = ["positive", "negative", "neutral", "irrelevant"]


def stratified_sample(df: pd.DataFrame, n: int, by: str = "platform", seed: int = 42) -> pd.DataFrame:
    """Sample n rows with each `by` group in proportion to its size (largest-remainder rounding)."""
    sizes = df[by].value_counts()
    exact = sizes / sizes.sum() * n
    alloc = exact.astype(int)
    for g in (exact - alloc).sort_values(ascending=False).index[: n - alloc.sum()]:
        alloc[g] += 1
    parts = [df[df[by] == g].sample(int(k), random_state=seed) for g, k in alloc.items()]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def thread_titles(df: pd.DataFrame) -> pd.Series:
    """For Reddit comments: the first line of the parent post (the thread title), for context."""
    posts = df[(df["platform"] == "reddit") & (~df["is_comment"])]
    titles = posts.set_index("post_id")["text_clean"].str.split(r"\s{2,}|\n").str[0].str.slice(0, 150)
    return df["parent_post_id"].map(titles)


def build_review_file(sample: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join hand labels to the sample; unsure posts first; empty corrected_label column."""
    out = sample[["post_id", "platform", "url", "thread_title", "text_clean"]].rename(columns={"text_clean": "text"})
    out = out.merge(labels[["post_id", "label", "unsure", "reason"]], on="post_id", how="left", validate="1:1")
    missing = out["label"].isna().sum()
    if missing:
        raise ValueError(f"{missing} sampled posts have no label yet")
    bad = set(out["label"]) - set(LABELS)
    if bad:
        raise ValueError(f"Unknown labels: {bad}")
    out["unsure"] = out["unsure"].fillna(False).astype(bool)
    out["reason"] = out["reason"].fillna("")
    out["corrected_label"] = ""
    return (out.sort_values(["unsure", "platform"], ascending=[False, True])
               [["post_id", "platform", "url", "unsure", "label", "reason", "corrected_label", "thread_title", "text"]]
               .reset_index(drop=True))


def apply_corrections(review: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """final_label = corrected_label where filled in, else Claude's label. Returns (df, n_changed)."""
    r = review.copy()
    corr = r["corrected_label"].fillna("").astype(str).str.strip().str.lower()
    bad = set(corr[corr != ""]) - set(LABELS)
    if bad:
        raise ValueError(f"corrected_label contains unknown values: {bad}. Use one of {LABELS}.")
    r["final_label"] = corr.where(corr != "", r["label"])
    n_changed = int((r["final_label"] != r["label"]).sum())
    return r, n_changed


def label_distribution(labels: pd.Series) -> pd.DataFrame:
    counts = labels.value_counts().reindex([l for l in LABELS if l in set(labels)])
    return pd.DataFrame({"count": counts, "percent": (counts / counts.sum() * 100).round(1)})


TYPO_FIXES = {"irrlevant": "irrelevant", "irelevant": "irrelevant", "irrelavant": "irrelevant",
              "postive": "positive", "possitive": "positive", "negtive": "negative", "nuetral": "neutral",
              "netural": "neutral"}


def merge_review(full_review: pd.DataFrame, partial: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Copy a checker's corrected_label values (from a partial/older review file) into the full
    review file, matched on post_id. Fixes common spelling slips and reports what it fixed."""
    corr = partial.set_index("post_id")["corrected_label"].fillna("").astype(str).str.strip().str.lower()
    notes = [f"{pid}: '{v}' read as '{TYPO_FIXES[v]}'" for pid, v in corr.items() if v in TYPO_FIXES]
    corr = corr.replace(TYPO_FIXES)
    corr = corr[corr != ""]
    out = full_review.copy()
    out["corrected_label"] = out["post_id"].map(corr).fillna(out["corrected_label"].fillna(""))
    return out, notes


def add_new_labelled(old: pd.DataFrame, new_clean: pd.DataFrame, new_labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Append newly cleaned + labelled posts to an existing labelled dataset, keeping its exact
    column order. New rows have no second-checker correction yet, so sentiment = claude_label.
    Returns (merged relevant rows, the new rows labelled irrelevant).
    """
    lab = new_labels.copy()
    lab["post_id"] = lab["post_id"].astype(str)
    new = new_clean.copy()
    new["post_id"] = new["post_id"].astype(str)
    new = new.merge(lab[["post_id", "label", "unsure", "reason"]], on="post_id", how="left", validate="1:1")
    if new["label"].isna().any():
        raise ValueError(f"{new['label'].isna().sum()} new posts have no label")
    bad = set(new["label"]) - set(LABELS)
    if bad:
        raise ValueError(f"Unknown labels: {bad}")
    new["claude_label"] = new["label"]
    new["sentiment"] = new["label"]
    new["corrected_label"] = pd.NA
    new["unsure"] = new["unsure"].fillna(False).astype(bool)
    new["reason"] = new["reason"].fillna("").where(new["unsure"], pd.NA)
    if "manually_added" in old.columns and "manually_added" not in new.columns:
        new["manually_added"] = False
    irrelevant = new[new["sentiment"] == "irrelevant"]
    new = new[new["sentiment"] != "irrelevant"]
    old_ids = set(old["post_id"].astype(str))
    dupes = new["post_id"].isin(old_ids)
    if dupes.any():
        raise ValueError(f"{dupes.sum()} new posts are already in the labelled dataset")
    merged = pd.concat([old, new.reindex(columns=old.columns)], ignore_index=True)
    return merged, irrelevant
