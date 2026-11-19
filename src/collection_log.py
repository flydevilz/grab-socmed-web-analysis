"""
Documentation of the EXTRA social-media collections (after notebook 01's first rounds), for the report's
Data Preparation section. Read-only: it reads the saved raw Apify files and the final dataset, never calls Apify.

Collections documented
  * X round 3 (Malay keywords)                   xquik/x-tweet-scraper
  * X/Threads/TikTok/Facebook/Reddit posts restored by the Manglish rule (no new collection, re-cleaned raw files)
  * Threads round 2 (brand search + replies to @grabmy)  futurizerush/meta-threads-scraper
  * Facebook (comments under facebook.com/GrabMY posts)  apify/facebook-posts-scraper + apify/facebook-comments-scraper
  * TikTok (comments under @grabmy videos)               clockworks/tiktok-comments-scraper
Only aggregate counts and search settings are reported: no usernames leave data/raw/.
Costs are not reported here: the raw files store only start fees, not per-item charges (see CLAUDE.md budget notes).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

FINAL = "data/grab_sentiment_clean.csv"

# raw-file pattern, platform, plain-English description, clean file that links raw items to the final dataset
COLLECTIONS = [
    {"collection": "X round 3 - Malay keywords", "platform": "twitter",
     "raw": "data/raw/twitter_round3_raw_*.json", "clean": "data/processed/twitter_round3_clean.csv",
     "what": "Tweets matching 14 Grab terms combined with Malaysian-Malay marker words"},
    {"collection": "Threads round 2", "platform": "threads",
     "raw": "data/raw/threads_round2_raw_*.json", "clean": "data/processed/threads_round2_clean.csv",
     "what": "Threads search for brand words/phrases + customer posts that @grabmy replied to"},
    {"collection": "Facebook - GrabMY page", "platform": "facebook",
     "raw": "data/raw/facebook_raw_*.json", "clean": None,
     "what": "Public comments under posts on facebook.com/GrabMY"},
    {"collection": "TikTok - @grabmy videos", "platform": "tiktok",
     "raw": "data/raw/tiktok_raw_*.json", "clean": None,
     "what": "Public comments under @grabmy TikTok videos posted in 2026"},
    {"collection": "Restored Malay/Manglish posts (no new scraping)", "platform": "several",
     "raw": None, "clean": "data/processed/manglish_restored_clean.csv",
     "what": "Posts from earlier rounds' saved raw files that were dropped only for being Malay; re-cleaned "
             "after the team switched to Manglish mode (18 Nov 2026)"},
]


def _input_summary(inp: dict) -> str:
    """Short, readable version of an actor's input (search terms, page, profile, limits, dates)."""
    keep = ["searchTerms", "keywords", "usernames", "mode", "max_posts", "maxItems", "startUrls", "resultsLimit",
            "profiles", "commentsPerPost", "oldestPostDateUnified", "start_date", "end_date", "includeNestedComments"]
    parts = []
    for k in keep:
        if k in inp:
            v = inp[k]
            if k == "startUrls":
                v = [u.get("url") if isinstance(u, dict) else u for u in v][:3]
            parts.append(f"{k}={v}")
    return "; ".join(parts) if parts else json.dumps(inp)[:200]


def runs_table(root: Path = Path(".")) -> pd.DataFrame:
    """One row per actor run in the extra collections (test runs marked)."""
    rows = []
    for c in COLLECTIONS:
        if not c["raw"]:
            continue
        for p in sorted(glob.glob(str(root / c["raw"]))):
            d = json.loads(Path(p).read_text())
            for r in d.get("runs", []):
                rows.append({"collection": c["collection"], "raw_file": Path(p).name,
                             "run_date": str(d.get("collected_at", ""))[:10], "actor": r.get("actor_id"),
                             "run_label": r.get("label"), "status": r.get("status"),
                             "items_returned": len(r.get("items", [])),
                             "input": _input_summary(r.get("input") or {})})
    t = pd.DataFrame(rows)
    # the first two 10-item TikTok runs were tests of the Free-plan cap before the full run
    if not t.empty:
        t["note"] = ""
        tk = t["collection"].str.startswith("TikTok") & (t["items_returned"] <= 10)
        t.loc[tk, "note"] = "test run (Free-plan item cap check)"
    return t


def summary_table(root: Path = Path(".")) -> pd.DataFrame:
    """One row per collection: actor(s), search settings, content dates and items in the final dataset."""
    final = pd.read_csv(root / FINAL, dtype={"post_id": str})
    runs = runs_table(root)
    restored = set(pd.read_csv(root / "data/processed/manglish_restored_clean.csv", dtype={"post_id": str}).post_id)
    rows = []
    for c in COLLECTIONS:
        if c["clean"]:
            ids = set(pd.read_csv(root / c["clean"], dtype={"post_id": str}).post_id)
            in_final = final[final.post_id.isin(ids)]
        else:
            in_final = final[final.platform == c["platform"]]
        r = runs[runs.collection == c["collection"]] if not runs.empty else runs
        real = r[r.get("note", "") == ""] if len(r) else r
        d = pd.to_datetime(in_final["created_at"])
        rows.append({
            "collection": c["collection"], "what": c["what"],
            "apify_actor(s)": "; ".join(sorted(set(real["actor"]))) if len(real) else "none (re-cleaning only)",
            "collected_on": "; ".join(sorted(set(real["run_date"]))) if len(real) else "2026-11-18",
            "actor_runs": len(real), "items_returned_by_actor": int(real["items_returned"].sum()) if len(real) else None,
            "posts_in_final_dataset": len(in_final),
            "of_which_malay_or_mixed": int(in_final.lang_group.isin(["ms", "mixed"]).sum()),
            "of_which_restored_by_manglish_rule": int(in_final.post_id.isin(restored).sum())
            if c["clean"] != "data/processed/manglish_restored_clean.csv" else len(in_final),
            "content_from": str(d.min().date()) if len(d) else "", "content_to": str(d.max().date()) if len(d) else "",
            "by_platform": in_final.platform.value_counts().to_dict(),
        })
    return pd.DataFrame(rows)


def round3_terms(cfg: dict) -> pd.DataFrame:
    """The X round-3 search settings, exactly as configured."""
    r3 = cfg["social"]["platforms"]["twitter"]["round3"]
    return pd.DataFrame([{"term": t, "query_sent_to_X": f"{t} {r3['markers']} {r3['exclude']} "
                          f"since:{r3['since']} until:{r3['until']}", "cap_per_term": r3["cap_per_term"]}
                         for t in r3["terms"]])
