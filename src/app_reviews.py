"""
Grab app reviews from Google Play and the Apple App Store (Part 1: website & app analysis).

Collection (free, no Apify)
  * Google Play  com.grabtaxi.passenger  -> google-play-scraper: newest reviews, lang "en" and "ms", country "my"
                                             plus the public app details page (lifetime rating, histogram).
  * App Store    id647268330             -> Apple's public customer-review RSS feed (Malaysia store),
                                             up to 10 pages x 50 reviews, plus the iTunes lookup API (lifetime rating).

robots.txt: Play's review endpoint (/_/...) and Apple's RSS feed (/*/rss/*) are disallowed for crawlers. The user
approved a one-off exception for this small academic sample (2026-11-18): 3 requests to Google and 11 to Apple,
spaced out, no login, usernames hashed. This is recorded in the README and the limitations.

Privacy: raw JSON (contains reviewer names) is saved only under data/raw/app_reviews/ (git-ignored).
The clean table keeps author_hash (SHA-256 of the lower-cased name), never the name.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from src import clean
from src.collect import hash_author as author_hash
from src.findings import INK, INK2, SERIES, TOPICS, _note, _save, set_style, tag_topics

PLAY_ID = "com.grabtaxi.passenger"
APPLE_ID = "647268330"
COUNTRY = "my"
WA = Path("website and app analysis")
RAW_DIR = Path("data/raw/app_reviews")
UA = "Mozilla/5.0 (academic research; BAA5113 student project)"
STORE_NAME = {"google_play": "Google Play", "app_store": "App Store"}
# Google Play's review filter works by LANGUAGE, not by the reviewer's country: English reviews come from every Grab
# market (Indonesia, Philippines, Vietnam, Thailand ...), Malay reviews are mostly Malaysian. Apple's feed is per
# country store. So results are reported per source, not per store.
SOURCES = ["App Store (Malaysia store)", "Google Play - Malay reviews", "Google Play - English reviews (all markets)"]

# Reviews that mention support / interactive features (user's keyword list, 2026-11-18)
SUPPORT_RE = r"chat ?bot|live chat|support|help cent|agent|AI assistant|recommend|ticket"
SUPPORT_CORE_RE = r"chat ?bot|live chat|support|help cent|agent|AI assistant|ticket"   # without 'recommend'
SUPPORT_PARTS = {
    "chatbot": r"chat ?bots?", "live chat": r"live chat", "support": r"support", "help centre": r"help cent(?:re|er)",
    "agent": r"agents?", "AI assistant": r"ai assistant", "recommend": r"recommend\w*", "ticket": r"tickets?",
}


# ---------------------------------------------------------------- collection
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _save_raw(obj, name: str, root: Path) -> Path:
    d = root / RAW_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}_{_now()}.json"
    p.write_text(json.dumps(obj, default=str, ensure_ascii=False, indent=1))
    return p


def collect_play(root: Path = Path("."), langs=("en", "ms"), count: int = 1000, pause_s: float = 5) -> dict:
    from google_play_scraper import Sort, app, reviews
    info = app(PLAY_ID, lang="en", country=COUNTRY)          # public details page (robots-allowed)
    info = {k: info.get(k) for k in ["title", "score", "ratings", "reviews", "histogram", "installs",
                                     "realInstalls", "version", "updated", "released", "developer"]}
    rows = []
    for lang in langs:
        time.sleep(pause_s)
        got, _ = reviews(PLAY_ID, lang=lang, country=COUNTRY, sort=Sort.NEWEST, count=count)
        for r in got:
            r["lang_requested"] = lang
        rows += got
    raw = {"app_info": info, "reviews": rows, "collected_at": datetime.now(timezone.utc).isoformat()}
    return {"raw": raw, "path": _save_raw(raw, "play_reviews", root)}


def collect_appstore(root: Path = Path("."), pages: int = 10, pause_s: float = 3) -> dict:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    look = s.get("https://itunes.apple.com/lookup", params={"id": APPLE_ID, "country": COUNTRY}, timeout=30).json()
    res = (look.get("results") or [{}])[0]
    info = {k: res.get(k) for k in ["trackName", "averageUserRating", "userRatingCount", "version",
                                    "currentVersionReleaseDate", "averageUserRatingForCurrentVersion",
                                    "userRatingCountForCurrentVersion"]}
    entries = []
    for page in range(1, pages + 1):
        time.sleep(pause_s)
        url = (f"https://itunes.apple.com/{COUNTRY}/rss/customerreviews/page={page}/id={APPLE_ID}"
               f"/sortby=mostrecent/json")
        r = s.get(url, timeout=30)
        if r.status_code != 200:
            break
        e = r.json().get("feed", {}).get("entry", [])
        e = [e] if isinstance(e, dict) else e
        e = [x for x in e if "im:rating" in x]           # skip the app-info entry some pages carry
        if not e:
            break
        entries += e
    raw = {"app_info": info, "reviews": entries, "collected_at": datetime.now(timezone.utc).isoformat()}
    return {"raw": raw, "path": _save_raw(raw, "appstore_reviews", root)}


# ---------------------------------------------------------------- normalise + clean
def _iso(x) -> str | None:
    t = pd.to_datetime(x, utc=True, errors="coerce")
    return None if pd.isna(t) else t.isoformat()


def normalise_play(raw: dict) -> pd.DataFrame:
    rows = []
    for r in raw["reviews"]:
        rows.append({
            "review_id": f"gp_{r.get('reviewId')}", "store": "google_play", "lang_requested": r.get("lang_requested"),
            "created_at": _iso(r.get("at")), "rating": r.get("score"), "title": "", "text": r.get("content") or "",
            "thumbs_up": r.get("thumbsUpCount"), "app_version": r.get("reviewCreatedVersion") or r.get("appVersion"),
            "developer_replied": bool(r.get("replyContent")), "reply_at": _iso(r.get("repliedAt")),
            "author_hash": author_hash(r.get("userName")), "collected_at": raw["collected_at"],
        })
    return pd.DataFrame(rows)


def normalise_appstore(raw: dict) -> pd.DataFrame:
    g = lambda e, k: (e.get(k) or {}).get("label")
    rows = []
    for e in raw["reviews"]:
        rows.append({
            "review_id": f"as_{g(e, 'id')}", "store": "app_store", "lang_requested": "any",
            "created_at": _iso(g(e, "updated")), "rating": int(g(e, "im:rating")), "title": g(e, "title") or "",
            "text": g(e, "content") or "", "thumbs_up": int(g(e, "im:voteSum") or 0),
            "app_version": g(e, "im:version"), "developer_replied": np.nan, "reply_at": None,   # not in the feed
            "author_hash": author_hash((e.get("author") or {}).get("name", {}).get("label")),
            "collected_at": raw["collected_at"],
        })
    return pd.DataFrame(rows)


def clean_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dedupe, drop empty reviews, anonymise text, add language group, star-based sentiment, topics."""
    log = [("collected", len(df))]
    df = df.drop_duplicates("review_id")
    log.append(("after removing duplicates (same review returned for en and ms)", len(df)))
    df = df[df["text"].str.strip().str.len() > 0].copy()
    log.append(("after removing empty reviews", len(df)))
    full = (df["title"].fillna("") + ". " + df["text"]).where(df["title"].fillna("") != "", df["text"])
    df["text_clean"] = full.map(clean.anonymise).map(clean.clean_text)
    df["lang_group"] = df["text_clean"].map(clean.lang_group)
    df["n_words"] = df["text_clean"].str.split().str.len()
    df["source"] = np.select([df["store"].eq("app_store"), df["lang_requested"].eq("ms")], SOURCES[:2], SOURCES[2])
    df["star_sentiment"] = pd.cut(df["rating"], [0, 2, 3, 5], labels=["negative", "neutral", "positive"]).astype(str)
    df["mentions_support"] = df["text_clean"].str.contains(SUPPORT_RE, case=False, regex=True)
    df = tag_topics(df)
    df["reply_hours"] = ((pd.to_datetime(df["reply_at"], utc=True) - pd.to_datetime(df["created_at"], utc=True))
                         .dt.total_seconds() / 3600).round(1)
    # Play shows the date of the latest EDIT: a review edited after Grab replied looks older than the reply
    df.loc[df["reply_hours"] < 0, "reply_hours"] = np.nan
    df = df.drop(columns=["title", "text"])     # text_clean is the anonymised version
    return df.reset_index(drop=True), pd.DataFrame(log, columns=["step", "reviews_remaining"])


# ---------------------------------------------------------------- tables
def store_overview(df: pd.DataFrame, play_info: dict, apple_info: dict) -> pd.DataFrame:
    rows = []
    life = {"google_play": (play_info.get("score"), play_info.get("ratings")),
            "app_store": (apple_info.get("averageUserRating"), apple_info.get("userRatingCount"))}
    for src in SOURCES:
        s = df[df["source"] == src]
        avg, n = life[s["store"].iloc[0]]
        d = pd.to_datetime(s["created_at"])
        rows.append({
            "source": src, "store_lifetime_avg_rating (all countries)": round(avg, 2) if avg else None,
            "store_lifetime_n_ratings": n, "sample_n_reviews": len(s),
            "sample_avg_rating": round(s["rating"].mean(), 2),
            "sample_pct_1_2_star": round(100 * (s["rating"] <= 2).mean(), 1),
            "sample_pct_4_5_star": round(100 * (s["rating"] >= 4).mean(), 1),
            "sample_from": str(d.min().date()), "sample_to": str(d.max().date()),
            "pct_with_developer_reply": round(100 * s["developer_replied"].mean(), 1)
            if s["developer_replied"].notna().any() else None,
        })
    return pd.DataFrame(rows)


def rating_distribution(df: pd.DataFrame, play_info: dict) -> pd.DataFrame:
    rows = []
    for src in SOURCES:
        s = df[df["source"] == src]
        c = s["rating"].value_counts().reindex(range(1, 6), fill_value=0)
        rows.append({"group": f"{src} - recent", "n": int(c.sum()),
                     **{f"pct_{k}_star": round(100 * v / c.sum(), 1) for k, v in c.items()}})
    h = play_info.get("histogram")
    if h:
        tot = sum(h)
        rows.append({"group": "Google Play - all-time star ratings (all countries)", "n": int(tot),
                     **{f"pct_{k}_star": round(100 * v / tot, 1) for k, v in zip(range(1, 6), h)}})
    return pd.DataFrame(rows)


def by_topic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in TOPICS:
        s = df[df[f"topic::{t}"]]
        if len(s) == 0:
            continue
        rows.append({"topic": t, "n_reviews": len(s), "pct_of_reviews": round(100 * len(s) / len(df), 1),
                     "avg_rating": round(s["rating"].mean(), 2), "pct_1_2_star": round(100 * (s.rating <= 2).mean(), 1),
                     **{f"avg_rating {src}": round(s.loc[s.source == src, "rating"].mean(), 2) for src in SOURCES}})
    rest = df[df["n_topics"] == 0]
    rows.append({"topic": "No topic keyword (e.g. 'good app')", "n_reviews": len(rest),
                 "pct_of_reviews": round(100 * len(rest) / len(df), 1), "avg_rating": round(rest.rating.mean(), 2),
                 "pct_1_2_star": round(100 * (rest.rating <= 2).mean(), 1),
                 **{f"avg_rating {src}": round(rest.loc[rest.source == src, "rating"].mean(), 2) for src in SOURCES}})
    return pd.DataFrame(rows).sort_values("pct_1_2_star", ascending=False).reset_index(drop=True)


def support_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rec = df["text_clean"].str.contains(SUPPORT_PARTS["recommend"], case=False, regex=True)
    sup = df["text_clean"].str.contains(SUPPORT_CORE_RE, case=False, regex=True)
    for name, m in [("Mentions support / interactive features (any keyword)", df["mentions_support"]),
                    ("Mentions support, chatbot, agent, help centre or ticket", sup),
                    ("Mentions 'recommend' only (mostly 'I recommend this app')", rec & ~sup),
                    ("Mentions none of these", ~df["mentions_support"])]:
        s = df[m]
        rows.append({"group": name, "n_reviews": len(s), "pct_of_reviews": round(100 * len(s) / len(df), 1),
                     "avg_rating": round(s.rating.mean(), 2),
                     **{f"n_{k}_star": int((s.rating == k).sum()) for k in range(1, 6)},
                     "pct_1_2_star": round(100 * (s.rating <= 2).mean(), 1)})
    for name, rx in SUPPORT_PARTS.items():
        s = df[df["text_clean"].str.contains(rx, case=False, regex=True)]
        rows.append({"group": f"  keyword: {name}", "n_reviews": len(s),
                     "pct_of_reviews": round(100 * len(s) / len(df), 1),
                     "avg_rating": round(s.rating.mean(), 2) if len(s) else None,
                     **{f"n_{k}_star": int((s.rating == k).sum()) for k in range(1, 6)},
                     "pct_1_2_star": round(100 * (s.rating <= 2).mean(), 1) if len(s) else None})
    return pd.DataFrame(rows)


def developer_replies(df: pd.DataFrame) -> pd.DataFrame:
    s = df[df["store"] == "google_play"]          # the App Store feed has no developer replies
    t = s.groupby(["source", "rating"]).agg(n_reviews=("review_id", "size"),
                                pct_replied=("developer_replied", lambda x: round(100 * x.mean(), 1)),
                                median_reply_hours=("reply_hours", "median")).reset_index()
    return t


def by_language(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["source", "lang_group"]).agg(n_reviews=("review_id", "size"), avg_rating=("rating", "mean"),
                                                 pct_1_2_star=("rating", lambda x: 100 * (x <= 2).mean()))
    return t.round(2).reset_index()


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["created_at"]).dt.tz_convert(None).dt.to_period("W").dt.start_time
    return df.assign(week=d).groupby(["source", "week"]).agg(n_reviews=("review_id", "size"),
                                                             avg_rating=("rating", "mean")).round(2).reset_index()


def examples(df: pd.DataFrame, n: int = 3, seed: int = 3) -> pd.DataFrame:
    """A few short, readable example reviews per star band and for support mentions (for quoting)."""
    out = []
    ok = df[df["n_words"].between(10, 45)]
    for label, m in [("1-2 star", ok.rating <= 2), ("3 star", ok.rating == 3), ("4-5 star", ok.rating >= 4),
                     ("mentions support / chatbot", ok.mentions_support)]:
        s = ok[m]
        for _, r in s.sample(min(n, len(s)), random_state=seed).iterrows():
            out.append({"group": label, "source": r.source, "rating": r.rating,
                        "date": str(r.created_at)[:10], "text": r.text_clean})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- charts
STAR_COLORS = ["#c43b3b", "#ef8a85", "#c9c8c1", "#86b6ef", "#2a78d6"]     # diverging: 1-2 red, 3 grey, 4-5 blue
STAR_TEXT = ["white", INK, INK, INK, "white"]


def fig_rating_dist(rd: pd.DataFrame, out: Path) -> Path:
    t = rd.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.6, 0.6 * len(t) + 1.3))
    y = np.arange(len(t))
    left = np.zeros(len(t))
    for k in range(1, 6):
        v = t[f"pct_{k}_star"].to_numpy(dtype=float)
        ax.barh(y, v, left=left, height=0.6, color=STAR_COLORS[k - 1], edgecolor="white", linewidth=1.2,
                label=f"{k}★")
        for yi, l, w in zip(y, left, v):
            if w >= 5:
                ax.text(l + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=8, color=STAR_TEXT[k - 1])
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}\n(n = {n:,})" for g, n in zip(t["group"], t["n"])], fontsize=8)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, handlelength=1)
    ax.set_title("Grab app star ratings: recent reviews vs all-time ratings", pad=22)
    _note(fig, "Recent = newest written reviews, collected 18 Nov 2026. App Store = Malaysia store; Google Play filters "
               "by language, not country. All-time = every Google Play star rating, with or without text.",
          y=-0.3 / fig.get_figheight())
    return _save(fig, out, "app_fig01_rating_distribution.png")


def fig_topics(bt: pd.DataFrame, out: Path, min_n: int = 20) -> Path:
    t = bt[bt["n_reviews"] >= min_n].iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.barh(y, t["pct_1_2_star"], height=0.6, color=STAR_COLORS[0])
    for yi, v, n, a in zip(y, t["pct_1_2_star"], t["n_reviews"], t["avg_rating"]):
        ax.text(v + 1, yi, f"{v:.0f}%   (n={n:,}, avg {a:.1f}★)", va="center", fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(t["topic"], fontsize=8); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.set_xlim(0, 115); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of reviews on the topic that give 1 or 2 stars"); ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("Which topics come with low app ratings?")
    _note(fig, "Topics from the same English + Malay keyword dictionary used for the social-media posts; "
               "one review can touch several topics. "
               f"Topics with fewer than {min_n} reviews are left out (see table).", y=-0.45 / fig.get_figheight())
    return _save(fig, out, "app_fig02_low_ratings_by_topic.png")


def fig_support(sf: pd.DataFrame, out: Path) -> Path:
    t = sf[sf["group"].isin(["Mentions support, chatbot, agent, help centre or ticket",
                             "Mentions 'recommend' only (mostly 'I recommend this app')",
                             "Mentions none of these"])].copy()
    tot = t[[f"n_{k}_star" for k in range(1, 6)]].sum(axis=1)
    for k in range(1, 6):
        t[f"pct_{k}_star"] = 100 * t[f"n_{k}_star"] / tot
    t = t.rename(columns={"n_reviews": "n"})
    fig, ax = plt.subplots(figsize=(7.8, 2.8))
    y = np.arange(len(t))[::-1]
    left = np.zeros(len(t))
    for k in range(1, 6):
        v = t[f"pct_{k}_star"].to_numpy(dtype=float)
        ax.barh(y, v, left=left, height=0.55, color=STAR_COLORS[k - 1], edgecolor="white", linewidth=1.2, label=f"{k}★")
        for yi, l, w in zip(y, left, v):
            if w >= 5:
                ax.text(l + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=8, color=STAR_TEXT[k - 1])
        left += v
    ax.set_yticks(y); ax.set_yticklabels([f"{g}\n(n = {n:,}, avg {a:.1f}★)" for g, n, a in
                                          zip(t["group"], t["n"], t["avg_rating"])], fontsize=8)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, handlelength=1)
    ax.set_title("Reviews mentioning support, chatbots or agents", pad=22)
    _note(fig, "Keywords: chatbot, live chat, support, help centre, agent, AI assistant, ticket; 'recommend' shown "
               "separately because it mostly means 'I recommend this app'.",
          y=-0.3 / fig.get_figheight())
    return _save(fig, out, "app_fig03_support_mentions.png")


def fig_weekly(wk: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    for i, src in enumerate(SOURCES):
        s = wk[(wk["source"] == src) & (wk["n_reviews"] >= 5)]
        ax.plot(s["week"], s["avg_rating"], color=SERIES[i], lw=2, marker="o", ms=4, label=src)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_ylim(1, 5); ax.set_ylabel("Average star rating"); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("Average rating of new reviews, by week")
    fig.autofmt_xdate()
    _note(fig, "Weeks with fewer than 5 reviews are left out. Each source covers a different period because the newest "
               "1,000 (Play) or 500 (App Store) reviews reach back different distances.", y=-0.55 / fig.get_figheight())
    return _save(fig, out, "app_fig04_weekly_rating.png")


# ---------------------------------------------------------------- run
def latest_raw(root: Path, prefix: str) -> Path:
    files = sorted((root / RAW_DIR).glob(f"{prefix}_*.json"))
    if not files:
        raise FileNotFoundError(f"No {prefix} file in {RAW_DIR}; run the collection first.")
    return files[-1]


def run(root: Path = Path("."), collect: bool = False, rebuild_from_raw: bool = False) -> dict:
    """Default: work from the saved, anonymised review table + public store info (both in the repo).
    rebuild_from_raw=True re-cleans the raw JSON in data/raw/app_reviews/ (local only, git-ignored);
    collect=True first fetches fresh reviews (free, 14 requests, robots.txt exception - ask the team first)."""
    root = Path(root)
    wa = root / WA
    for sub in ["data", "tables", "figures"]:
        (wa / sub).mkdir(parents=True, exist_ok=True)
    clean_path, info_path = wa / "data/app_reviews_clean.csv", wa / "data/app_store_info.json"
    if collect or rebuild_from_raw:
        if collect:
            collect_play(root); collect_appstore(root)
        play = json.loads(latest_raw(root, "play_reviews").read_text())
        apple = json.loads(latest_raw(root, "appstore_reviews").read_text())
        df, funnel = clean_reviews(pd.concat([normalise_play(play), normalise_appstore(apple)], ignore_index=True))
        keep = [c for c in df.columns if not c.startswith("topic::")]
        df[keep].to_csv(clean_path, index=False)
        info = {"google_play": play["app_info"], "app_store": apple["app_info"],
                "collected_at": {"google_play": play["collected_at"], "app_store": apple["collected_at"]},
                "cleaning_funnel": funnel.to_dict(orient="records")}
        info_path.write_text(json.dumps(info, indent=1, default=str))
    else:
        df = tag_topics(pd.read_csv(clean_path))
        info = json.loads(info_path.read_text())
        funnel = pd.DataFrame(info["cleaning_funnel"])
    play, apple = {"app_info": info["google_play"]}, {"app_info": info["app_store"]}

    rd = rating_distribution(df, play["app_info"])
    T = {
        "app_tab01_store_overview": store_overview(df, play["app_info"], apple["app_info"]),
        "app_tab02_cleaning_funnel": funnel,
        "app_tab03_rating_distribution": rd,
        "app_tab04_topics": by_topic(df),
        "app_tab05_support_features": support_features(df),
        "app_tab06_developer_replies_google_play": developer_replies(df),
        "app_tab07_by_language": by_language(df),
        "app_tab08_weekly_rating": weekly(df),
        "app_tab09_example_reviews": examples(df),
    }
    for k, t in T.items():
        t.to_csv(wa / "tables" / f"{k}.csv", index=False)
    set_style()
    F = [fig_rating_dist(rd, wa / "figures"), fig_topics(T["app_tab04_topics"], wa / "figures"),
         fig_support(T["app_tab05_support_features"], wa / "figures"),
         fig_weekly(T["app_tab08_weekly_rating"], wa / "figures")]
    return {"df": df, "tables": T, "figures": F, "play_info": play["app_info"], "apple_info": apple["app_info"]}
