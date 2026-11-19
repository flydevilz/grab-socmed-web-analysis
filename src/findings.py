"""
Report-ready charts, topic summaries, comparison tables (notebook 05).

Inputs
  data/grab_sentiment_clean.csv    final clean, labelled posts (2,015)
  result/01_model_comparison.csv             from notebook 06 (model comparison)
  website and app analysis/tables/pagespeed_median.csv, onpage_seo.csv   (run_web only; notebook 01 Section A)

Outputs
  outputs/figures/figNN_*.png  (300 dpi, white background, ready to paste into Word)
  outputs/tables/tabNN_*.csv

Topics are found with a transparent keyword dictionary (TOPICS below, English + Malay), so every
topic tag can be explained in the report. A post can belong to several topics. An unsupervised
NMF topic model is run as a cross-check (tab07).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

# ---------------------------------------------------------------- style
SENT = ["negative", "neutral", "positive"]
SENT_COLOR = {"negative": "#e34948", "neutral": "#b5b4ad", "positive": "#2a78d6"}   # diverging: red / grey / blue
SENT_TEXT = {"negative": "white", "neutral": "#0b0b0b", "positive": "white"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PLATFORM_NAME = {"twitter": "X (Twitter)", "reddit": "Reddit", "threads": "Threads",
                 "tiktok": "TikTok", "facebook": "Facebook"}
LANG_NAME = {"en": "English", "ms": "Malay", "mixed": "Mixed (Manglish)"}
SITE_NAME = {"https://www.grab.com/my/": "grab.com/my (main)", "https://food.grab.com/my/": "GrabFood",
             "https://mart.grab.com/my": "GrabMart", "https://express.grab.com/ms": "GrabExpress",
             "https://dineout.grab.com/my": "Grab Dine Out"}


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": GRID,
        "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
    })


def _note(fig, text, y=None):
    """Source/caption line below the plot; offset fixed in inches so it clears the x-axis label."""
    if y is None:
        y = -0.12 / fig.get_figheight()
    fig.text(0.0, y, text, fontsize=7, color=INK2, ha="left", va="top", transform=fig.transFigure)


def _save(fig, out: Path, name: str) -> Path:
    p = out / name
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- topics
# Keyword dictionary (lower-case, matched on whole words in text_clean). English + Malay.
TOPICS: dict[str, list[str]] = {
    "Price, fares & fees": [r"prices?", r"pricing", r"fares?", r"mahal", r"expensive", r"overpriced", r"pricey",
                            r"surge\w*", r"cheap(?:er|est)?", r"murah", r"fees?", r"charge[sd]?", r"costs?", r"costly",
                            r"tolls?", r"harga", r"caj", r"tambang", r"ringgit", r"rm ?\d+", r"afford\w*", r"mark-?up"],
    "Drivers & rides (GrabCar)": [r"drivers?", r"rides?", r"grab ?car", r"car", r"pick ?up", r"drop ?off", r"route",
                                  r"trips?", r"airport", r"klia", r"kereta", r"pemandu", r"e-?hailing", r"passengers?",
                                  r"penumpang"],
    "Food & delivery (GrabFood/Mart/Express)": [r"grab ?food", r"food", r"deliver(?:y|ies|ed)?", r"orders?", r"ordered",
                                                r"riders?", r"makanan", r"restaurants?", r"merchants?", r"grab ?mart",
                                                r"mart", r"parcel", r"grab ?express", r"seller", r"kedai", r"makan"],
    "Cancellations & waiting time": [r"cancel\w*", r"wait(?:ing|ed)?", r"tunggu", r"lambat", r"late", r"slow",
                                     r"lama", r"delay(?:ed)?", r"no drivers?", r"lembab", r"lembap", r"took forever"],
    "Customer service & refunds": [r"customer service", r"customer support", r"support", r"cs", r"refunds?",
                                   r"refunded", r"complain\w*", r"report(?:ed)?", r"tickets?", r"respon\w*",
                                   r"repl(?:y|ied|ies)", r"agent", r"hotline", r"help ?cent(?:re|er)", r"aduan"],
    "App & technology (incl. AI)": [r"apps?", r"update[sd]?", r"bugs?", r"glitch\w*", r"crash\w*", r"error",
                                    r"maps?", r"gps", r"log ?in", r"accounts?", r"ai", r"features?", r"notifications?",
                                    r"ads", r"interface", r"otp", r"system", r"sistem", r"aplikasi", r"robotaxis?"],
    "Payments & GrabPay": [r"grab ?pay", r"e-?wallet", r"wallet", r"payments?", r"paylater", r"pay ?later",
                           r"cards?", r"cash", r"touch ?n ?go", r"tng", r"bank", r"duit"],
    "Promos, vouchers & rewards": [r"promos?", r"vouchers?", r"discounts?", r"promo ?codes?", r"deals?", r"rewards?",
                                   r"points", r"cashback", r"subscription", r"grab ?unlimited", r"offers?", r"diskaun",
                                   r"baucar", r"free delivery"],
    "Safety & trust": [r"safe", r"safety", r"unsafe", r"scam\w*", r"harass\w*", r"police", r"accidents?",
                       r"dangerous", r"danger", r"scary", r"scared", r"tipu", r"penipu", r"fraud\w*", r"assault\w*",
                       r"reckless", r"threat\w*", r"takut", r"bahaya", r"polis"],
    "Driver welfare & regulation": [r"gig", r"socso", r"perkeso", r"earnings?", r"income", r"commissions?",
                                    r"incentives?", r"apad", r"jpj", r"government", r"law", r"bill", r"regulat\w*",
                                    r"minister", r"loke", r"pendapatan", r"gaji", r"union", r"strike", r"protest",
                                    r"mogok", r"psv", r"evp", r"kerajaan"],
    "Competitors & alternatives": [r"airasia", r"maxim", r"indrive\w*", r"foodpanda", r"panda", r"shopee ?food",
                                   r"bolt", r"lalamove", r"uber", r"taxi", r"teksi", r"mycar", r"competitor\w*",
                                   r"alternatives?"],
}
OTHER = "Other / general"
# Whole-word matching: \b on both sides, so "car" never matches "card" or "scar".
# Whole-word matching (\b on both sides): "car" never matches "card" or "scar".
_TOPIC_RE = {t: re.compile(r"\b(?:" + "|".join(ws) + r")\b", re.I) for t, ws in TOPICS.items()}


def tag_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Add one boolean column per topic (`topic::<name>`) and `topics` (semicolon list)."""
    out = df.copy()
    text = out["text_clean"].fillna("").str.lower()
    for t, rx in _TOPIC_RE.items():
        out[f"topic::{t}"] = text.str.contains(rx)
    cols = [f"topic::{t}" for t in TOPICS]
    out["topics"] = out[cols].apply(lambda r: "; ".join(t for t, v in zip(TOPICS, r) if v) or OTHER, axis=1)
    out["n_topics"] = out[cols].sum(axis=1)
    return out


# ---------------------------------------------------------------- word statistics
MALAY_STOP = set("""yang dan di ke dari ini itu tu ni dengan untuk pada ada tak tidak x je jer la lah pun dah sudah
nak mau akan saya aku kau kamu awak dia kita kami mereka korang diorang dorang org orang ada kalau kat dekat
sebab pasal macam mcm camtu camni lagi juga jugak boleh blh bole buat bila bukan apa mana siapa kenapa sbb utk
dgn dlm dalam atau tapi tp pastu lepas sama tau tahu ke ka kan pon kot ja ah eh oh weh wei je jgn jangan
dh yg sy tk ni nk kt""".split())
EXTRA_STOP = {"grab", "grabmy", "user", "url", "im", "dont", "just", "like", "got", "get", "amp", "lol", "u", "s",
              "t", "ve", "ll", "don", "didn", "doesn", "isn", "wasn", "really", "know", "think", "people"}
STOP = set(ENGLISH_STOP_WORDS) | MALAY_STOP | EXTRA_STOP


def pretty_term(t: str) -> str:
    return f"[{t[6:].replace('_', ' ')} emoji]" if t.startswith("emoji_") else t


def distinctive_words(df: pd.DataFrame, n: int = 12, min_count: int = 5) -> pd.DataFrame:
    """Words most typical of each sentiment vs the other two: weighted log-odds with an informative
    Dirichlet prior (Monroe, Colaresi & Quinn, 2008). z > 1.96 ~ significant at 5%."""
    vec = CountVectorizer(token_pattern=r"(?u)\b[a-z_][a-z_]+\b", stop_words=list(STOP), min_df=min_count)
    X = vec.fit_transform(df["text_norm"].fillna(""))
    vocab = vec.get_feature_names_out()
    prior = np.asarray(X.sum(axis=0)).ravel() + 0.01
    a0 = prior.sum()
    rows = []
    for s in SENT:
        m = (df["sentiment"] == s).to_numpy()
        yi, yj = np.asarray(X[m].sum(axis=0)).ravel(), np.asarray(X[~m].sum(axis=0)).ravel()
        ni, nj = yi.sum(), yj.sum()
        delta = (np.log((yi + prior) / (ni + a0 - yi - prior)) - np.log((yj + prior) / (nj + a0 - yj - prior)))
        z = delta / np.sqrt(1 / (yi + prior) + 1 / (yj + prior))
        for r, j in enumerate(np.argsort(-z)[:n]):
            rows.append({"sentiment": s, "rank": r + 1, "term": pretty_term(vocab[j]), "z_score": round(z[j], 2),
                         "count_in_class": int(yi[j]), "count_elsewhere": int(yj[j])})
    return pd.DataFrame(rows)


def nmf_topics(df: pd.DataFrame, k: int = 8, n_words: int = 10, seed: int = 42) -> pd.DataFrame:
    """Data-driven topics (NMF on TF-IDF) as a cross-check of the keyword topics."""
    vec = TfidfVectorizer(token_pattern=r"(?u)\b[a-z][a-z]+\b", stop_words=list(STOP), min_df=5, max_df=0.5)
    X = vec.fit_transform(df["text_norm"].fillna(""))
    nmf = NMF(n_components=k, random_state=seed, init="nndsvd", max_iter=600)
    W = nmf.fit_transform(X)
    vocab = vec.get_feature_names_out()
    dom = W.argmax(axis=1)
    has = W.max(axis=1) > 0
    rows = []
    for i in range(k):
        m = (dom == i) & has
        mix = df.loc[m, "sentiment"].value_counts(normalize=True).reindex(SENT, fill_value=0)
        rows.append({"nmf_topic": i + 1, "top_words": ", ".join(vocab[np.argsort(-nmf.components_[i])[:n_words]]),
                     "n_posts": int(m.sum()), **{f"pct_{s}": round(100 * mix[s], 1) for s in SENT}})
    return pd.DataFrame(rows).sort_values("n_posts", ascending=False)


# ---------------------------------------------------------------- tables
def sentiment_table(df: pd.DataFrame, by: str, names: dict | None = None) -> pd.DataFrame:
    ct = pd.crosstab(df[by], df["sentiment"]).reindex(columns=SENT, fill_value=0)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    t = pd.DataFrame({"n_posts": ct.sum(axis=1)})
    for s in SENT:
        t[f"n_{s}"] = ct[s]
    for s in SENT:
        t[f"pct_{s}"] = pct[s].round(1)
    t["net_sentiment"] = (pct["positive"] - pct["negative"]).round(1)
    t = t.sort_values("n_posts", ascending=False)
    total = pd.DataFrame({"n_posts": [len(df)], **{f"n_{s}": [(df.sentiment == s).sum()] for s in SENT},
                          **{f"pct_{s}": [round(100 * (df.sentiment == s).mean(), 1)] for s in SENT}},
                         index=["All posts"])
    total["net_sentiment"] = total["pct_positive"] - total["pct_negative"]
    t = pd.concat([t, total])
    if names:
        t.index = [names.get(i, i) for i in t.index]
    return t.rename_axis(by).reset_index()


def _example(sub: pd.DataFrame, sentiment: str, seed: int = 7, max_words: int = 35) -> str:
    base = sub[(sub["sentiment"] == sentiment) & sub["n_words"].between(8, max_words) & sub["mentions_grab"]]
    c = base[base["n_topics"] <= 1]          # prefer posts that are clearly about this one topic
    if c.empty:
        c = base
    if c.empty:
        c = sub[(sub["sentiment"] == sentiment) & sub["n_words"].between(5, 60)]
    if c.empty:
        return ""
    r = c.sample(1, random_state=seed).iloc[0]
    return f"[{PLATFORM_NAME.get(r.platform, r.platform)}] {r.text_clean.strip()}"


def topic_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in list(TOPICS) + [OTHER]:
        m = df[f"topic::{t}"] if t != OTHER else df["n_topics"].eq(0)
        sub = df[m]
        mix = sub["sentiment"].value_counts(normalize=True).reindex(SENT, fill_value=0) * 100
        rows.append({"topic": t, "n_posts": len(sub), "pct_of_all_posts": round(100 * len(sub) / len(df), 1),
                     **{f"pct_{s}": round(mix[s], 1) for s in SENT},
                     "net_sentiment": round(mix["positive"] - mix["negative"], 1),
                     "pct_malay_or_mixed": round(100 * sub["lang_group"].isin(["ms", "mixed"]).mean(), 1),
                     "keywords_used": ", ".join(w.replace("\\", "").replace("(?:", "(") for w in TOPICS.get(t, ["(none of the above)"])[:8]),
                     "example_negative": _example(sub, "negative"), "example_positive": _example(sub, "positive")})
    return pd.DataFrame(rows)


def topic_by_platform(df: pd.DataFrame) -> pd.DataFrame:
    """% of each platform's posts that touch each topic (columns = platforms)."""
    t = pd.DataFrame({PLATFORM_NAME[p]: [round(100 * g[f"topic::{tp}"].mean(), 1) for tp in TOPICS]
                      for p, g in df.groupby("platform")}, index=list(TOPICS))
    return t[[PLATFORM_NAME[p] for p in df["platform"].value_counts().index]].rename_axis("topic").reset_index()


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    m = pd.to_datetime(df["created_at"]).dt.to_period("M")
    t = sentiment_table(df.assign(month=m.astype(str)), "month").query("month != 'All posts'").sort_values("month")
    first, last = m.min(), m.max()
    t["note"] = np.where(t["month"].isin([str(first), str(last)]), "partial month", "")
    return t.reset_index(drop=True)


def engagement(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["platform", "sentiment"])
    t = g.agg(n_posts=("post_id", "size"), median_likes=("likes", "median"), mean_likes=("likes", "mean"),
              median_replies=("replies", "median")).round(1).reset_index()
    t["platform"] = t["platform"].map(PLATFORM_NAME)
    return t


def addressed_to_grab(df: pd.DataFrame) -> pd.DataFrame:
    """Posts that talk TO Grab (tag @GrabMY etc., or reply/comment under Grab's own posts) vs ABOUT Grab."""
    to = (df["text_clean"].str.contains(r"@grab", case=False, na=False)
          | df["search_query"].str.contains(r"answered by @grabmy|comment on @grabmy|comment on GrabMY", case=False, na=False))
    return sentiment_table(df.assign(audience=np.where(to, "Talking TO Grab (tag / reply / comment)",
                                                        "Talking ABOUT Grab")), "audience")


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    ct = pd.crosstab(df["platform"].map(PLATFORM_NAME), df["lang_group"].map(LANG_NAME), margins=True,
                     margins_name="Total")
    d = pd.to_datetime(df["created_at"])
    first = df.assign(d=d).groupby(df["platform"].map(PLATFORM_NAME))["d"].agg(["min", "max"])
    ct["earliest_post"] = first["min"].dt.date.astype(str).reindex(ct.index).fillna(str(d.min().date()))
    ct["latest_post"] = first["max"].dt.date.astype(str).reindex(ct.index).fillna(str(d.max().date()))
    ct["pct_comments"] = (df.groupby(df["platform"].map(PLATFORM_NAME))["is_comment"].mean() * 100).round(1) \
        .reindex(ct.index).fillna(round(100 * df["is_comment"].mean(), 1))
    return ct.sort_values("Total", ascending=False).rename_axis("platform").reset_index()


def web_summary(psi: pd.DataFrame) -> pd.DataFrame:
    t = psi.copy()
    t.insert(0, "site", t["url"].map(SITE_NAME))
    keep = {"site": "site", "strategy": "device", "performance_score": "performance", "accessibility_score":
            "accessibility", "best_practices_score": "best_practices", "seo_score": "seo",
            "lab_lcp_ms": "lab_LCP_s", "lab_tbt_ms": "lab_TBT_ms", "lab_cls": "lab_CLS",
            "field_lcp_category": "real_users_LCP", "field_inp_category": "real_users_INP",
            "field_cls_category": "real_users_CLS", "field_overall_category": "real_users_overall"}
    t = t[list(keep)].rename(columns=keep)
    t["lab_LCP_s"] = (t["lab_LCP_s"] / 1000).round(1)
    t["lab_TBT_ms"] = t["lab_TBT_ms"].round(0).astype(int)
    t["performance_rating"] = pd.cut(t["performance"], [-1, 49, 89, 100], labels=["poor (0-49)",
                                     "needs improvement (50-89)", "good (90-100)"]).astype(str)
    return t.sort_values(["site", "device"]).reset_index(drop=True)


def seo_summary(seo: pd.DataFrame) -> pd.DataFrame:
    t = seo.copy()
    t.insert(0, "site", t["url"].map(SITE_NAME))
    t = t[["site", "http_status", "robots_allowed", "title_length", "meta_description_length", "h1_count",
           "h2_count", "hreflang_count", "img_total", "img_missing_alt", "has_viewport"]]
    issues = []
    for _, r in t.iterrows():
        i = []
        if r.http_status != 200:
            i.append(f"blocked to crawlers (HTTP {r.http_status})")
        else:
            if not r.title_length or r.title_length == 0:
                i.append("no <title>")
            elif r.title_length > 60:
                i.append("title > 60 chars")
            if not r.meta_description_length:
                i.append("no meta description")
            elif r.meta_description_length > 160:
                i.append("meta description > 160 chars")
            if r.h1_count != 1:
                i.append(f"{int(r.h1_count) if pd.notna(r.h1_count) else 0} H1 tags (1 expected)")
            if r.img_missing_alt and r.img_missing_alt > 0:
                i.append(f"{int(r.img_missing_alt)} image(s) without alt text")
        issues.append("; ".join(i) or "no basic issues")
    t["issues_found"] = issues
    return t


# ---------------------------------------------------------------- charts
def _stacked_share(ax, t: pd.DataFrame, label_col: str, min_label: float = 4.0):
    """100% stacked horizontal bars (negative | neutral | positive), first row at the top."""
    y = np.arange(len(t))[::-1]
    left = np.zeros(len(t))
    for s in SENT:
        v = t[f"pct_{s}"].to_numpy(dtype=float)
        ax.barh(y, v, left=left, height=0.62, color=SENT_COLOR[s], edgecolor="white", linewidth=1.2, label=s.capitalize())
        for yi, l, w in zip(y, left, v):
            if w >= min_label:
                ax.text(l + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=8, color=SENT_TEXT[s])
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a}  (n={n:,})" for a, n in zip(t[label_col], t["n_posts"])])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, handlelength=1, borderaxespad=0.2)


def fig_overall(df, out):
    c = df["sentiment"].value_counts().reindex(SENT)
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    y = np.arange(3)[::-1]
    ax.barh(y, c.values, height=0.55, color=[SENT_COLOR[s] for s in SENT])
    for yi, v in zip(y, c.values):
        ax.text(v + 12, yi, f"{v:,}  ({100 * v / c.sum():.1f}%)", va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([s.capitalize() for s in SENT]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, c.max() * 1.3); ax.set_xlabel("Number of posts"); ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title(f"Overall sentiment towards Grab Malaysia (n = {len(df):,} posts)")
    _note(fig, "Source: labelled posts (Claude + team review) from X, Reddit, Threads, TikTok and Facebook, Nov 2025 - Nov 2026.",
          y=-0.3 / fig.get_figheight())
    return _save(fig, out, "fig01_sentiment_overall.png")


def fig_share(t, label_col, title, note, out, name, height=None):
    t = t[t[label_col] != "All posts"]
    fig, ax = plt.subplots(figsize=(7.2, height or 0.45 * len(t) + 1.2))
    _stacked_share(ax, t, label_col)
    ax.set_title(title, pad=22)
    _note(fig, note)
    return _save(fig, out, name)


ANNOTATE_MONTHS = {
    "2026-04": "Apr 2026: complaints about Grab moving\ncustomer service to chatbots + long food waits",
    "2026-11": "Nov 2026: 68% of posts are replies to @GrabMY\n(Threads collection round) - not a like-for-like month",
}


def fig_monthly(mt, out):
    mt = mt.copy()
    x = np.arange(len(mt))
    labels = [pd.Period(m).strftime("%b\n%Y") if (i == 0 or m.endswith("-01")) else pd.Period(m).strftime("%b")
              for i, m in enumerate(mt["month"])]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 4.8), sharex=True, gridspec_kw={"height_ratios": [1, 1.5]})
    partial = mt["note"].eq("partial month").to_numpy()
    a1.bar(x, mt["n_posts"], width=0.6, color=np.where(partial, "#b7d3f6", SERIES[0]))
    a1.set_ylabel("Posts"); a1.yaxis.grid(True); a1.set_axisbelow(True)
    a1.set_title("Post volume and sentiment by month")
    for s in ["negative", "positive"]:
        a2.plot(x, mt[f"pct_{s}"], color=SENT_COLOR[s], lw=2, marker="o", ms=4)
        a2.text(x[-1] + 0.3, mt[f"pct_{s}"].iloc[-1], f"% {s}", color=INK, va="center", fontsize=8)
    a2.set_ylabel("Share of the month's posts (%)"); a2.set_ylim(0, 80)
    for m, txt in ANNOTATE_MONTHS.items():
        if m in set(mt["month"]):
            i = int(np.flatnonzero(mt["month"].to_numpy() == m)[0])
            a2.annotate(txt, xy=(i, mt["pct_negative"].iloc[i]), xytext=(i - 0.4, 77), fontsize=7, color=INK2,
                        ha="right", va="top", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    a2.yaxis.grid(True); a2.set_axisbelow(True)
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=8)
    a2.set_xlim(-0.6, len(x) + 0.6)
    _note(fig, "Lighter bars = partial months (collection window 7 Nov 2025 - 17 Nov 2026). Monthly shares rest on 84-327 "
               "posts each.\nNovember 2026 includes the extra Threads collection of customer posts answered by @GrabMY "
               "(mostly complaints), so it is more negative and not comparable with other months.")
    fig.tight_layout()
    return _save(fig, out, "fig04_sentiment_by_month.png")


def fig_topics(ts, out):
    t = ts[ts["topic"] != OTHER].sort_values("net_sentiment", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.8, 0.42 * len(t) + 1.3))
    _stacked_share(ax, t, "topic")
    ax.set_title("Sentiment by topic (sorted from most positive to most negative)", pad=22)
    _note(fig, "Topics found with a keyword dictionary (English + Malay); one post can touch several topics. "
               "n = posts mentioning the topic.")
    return _save(fig, out, "fig05_sentiment_by_topic.png")


def fig_topic_volume(ts, out):
    t = ts[ts["topic"] != OTHER].sort_values("n_posts").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(t) + 1.0))
    y = np.arange(len(t))
    ax.barh(y, t["pct_of_all_posts"], height=0.6, color=SERIES[0])
    for yi, v, n in zip(y, t["pct_of_all_posts"], t["n_posts"]):
        ax.text(v + 0.6, yi, f"{v:.0f}%  ({n:,})", va="center", fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(t["topic"]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.set_xlim(0, t["pct_of_all_posts"].max() * 1.25)
    ax.set_xlabel("% of all posts mentioning the topic"); ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("What people talk about when they talk about Grab")
    _note(fig, "One post can touch several topics, so percentages add up to more than 100%.")
    return _save(fig, out, "fig06_topic_volume.png")


def fig_words(dw, out, n=10):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    for ax, s in zip(axes, SENT):
        d = dw[dw["sentiment"] == s].head(n).iloc[::-1]
        ax.barh(np.arange(len(d)), d["z_score"], height=0.6, color=SENT_COLOR[s])
        ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d["term"], fontsize=8); ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False); ax.xaxis.grid(True); ax.set_axisbelow(True)
        ax.set_title(f"{s.capitalize()} posts", fontsize=10, color=INK)
        ax.set_xlabel("z-score (log-odds)")
    fig.suptitle("Words most typical of each sentiment", x=0.0, ha="left", fontweight="bold", fontsize=11)
    _note(fig, "Weighted log-odds with an informative Dirichlet prior (Monroe et al., 2008), each class vs the other two; "
               "z > 1.96 = significant at 5%. Malay and English stop-words removed.")
    fig.tight_layout()
    return _save(fig, out, "fig07_distinctive_words.png")


def fig_models(mc, out):
    t = mc.copy()
    t["method"] = t["method"].str.replace(r"^\(\w\) ", "", regex=True)
    t = t.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    y = np.arange(len(t))
    for off, col, colr, lab in [(0.18, "accuracy", SERIES[0], "Accuracy"), (-0.18, "macro_f1", SERIES[1], "Macro-F1")]:
        ax.barh(y + off, t[col], height=0.32, color=colr, label=lab)
        for yi, v in zip(y + off, t[col]):
            ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(t["method"]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.set_xlim(0, 0.8); ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, handlelength=1)
    ax.set_title(f"Sentiment methods compared on the same test posts (n = {int(t['n'].iloc[0])})", pad=20)
    _note(fig, "Stratified 80/20 train/test split; Logistic Regression trained on the 80% only. "
               "Macro-F1 weights negative, neutral and positive equally.")
    return _save(fig, out, "fig08_model_comparison.png")


def fig_web_perf(ws, out):
    t = ws.pivot(index="site", columns="device", values="performance").sort_values("mobile")
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    y = np.arange(len(t))
    for lo, hi, c in [(0, 50, "#fbe9e9"), (50, 90, "#fdf3dc"), (90, 100, "#e3f3e3")]:
        ax.axvspan(lo, hi, color=c, zorder=0)
    ax.hlines(y, t["mobile"], t["desktop"], color="#c3c2b7", lw=2, zorder=1)
    ax.scatter(t["mobile"], y, s=60, color=SERIES[1], zorder=2, label="Mobile", edgecolor="white", linewidth=1.5)
    ax.scatter(t["desktop"], y, s=60, color=SERIES[0], zorder=2, label="Desktop", edgecolor="white", linewidth=1.5)
    for yi, m, d in zip(y, t["mobile"], t["desktop"]):
        ax.text(m - 2, yi + 0.28, f"{m:.0f}", ha="center", fontsize=8, color=INK)
        ax.text(d + 2, yi + 0.28, f"{d:.0f}", ha="center", fontsize=8, color=INK)
    for x, lab in [(25, "Poor (0-49)"), (70, "Needs improvement (50-89)"), (95, "Good")]:
        ax.text(x, len(t) - 0.35, lab, ha="center", fontsize=7, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(t.index); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(t) - 0.1)
    ax.set_xlabel("Lighthouse performance score (median of 3 runs)")
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, handlelength=1)
    ax.set_title("Grab Malaysia websites: mobile vs desktop speed", pad=20)
    _note(fig, "Source: Google PageSpeed Insights API (Lighthouse 13.6), 4 Nov 2026. Bands are Google's thresholds.", y=-0.3 / fig.get_figheight())
    return _save(fig, out, "web_fig01_performance_mobile_vs_desktop.png")


def fig_web_scores(ws, out):
    cats = ["performance", "accessibility", "best_practices", "seo"]
    t = ws.set_index(ws["site"] + " - " + ws["device"])[cats]
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(t) + 1.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#e8f1fc", "#86b6ef", "#2a78d6", "#104281"])
    ax.imshow(t.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v >= 60 else INK)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(["Performance", "Accessibility", "Best practices", "SEO"])
    ax.xaxis.tick_top(); ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index, fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cats)), minor=True); ax.set_yticks(np.arange(-0.5, len(t)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
    ax.set_title("Lighthouse category scores (0-100)", pad=24)
    _note(fig, "Source: Google PageSpeed Insights API, median of 3 runs per site and device, 4 Nov 2026.")
    return _save(fig, out, "web_fig02_lighthouse_scores.png")


# ---------------------------------------------------------------- run everything
def run(root: Path = Path(".")) -> dict:
    set_style()
    root = Path(root)
    fig_dir, tab_dir = root / "outputs/figures", root / "outputs/tables"
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)

    df = tag_topics(pd.read_csv(root / "data/grab_sentiment_clean.csv"))
    mc = pd.read_csv(root / "result/01_model_comparison.csv")

    T = {
        "tab01_dataset_overview": dataset_overview(df),
        "tab02_sentiment_by_platform": sentiment_table(df, "platform", PLATFORM_NAME),
        "tab03_sentiment_by_language": sentiment_table(df, "lang_group", LANG_NAME),
        "tab04_sentiment_by_month": monthly(df),
        "tab05_topic_summary": topic_summary(df),
        "tab06_topic_by_platform": topic_by_platform(df),
        "tab07_nmf_topics_crosscheck": nmf_topics(df),
        "tab08_distinctive_words": distinctive_words(df),
        "tab09_model_comparison": mc.round(3),
        "tab10_engagement_by_sentiment": engagement(df),
        "tab11_talking_to_vs_about_grab": addressed_to_grab(df),
    }
    for name, t in T.items():
        t.to_csv(tab_dir / f"{name}.csv", index=False)

    F = [
        fig_overall(df, fig_dir),
        fig_share(T["tab02_sentiment_by_platform"], "platform", "Sentiment by platform",
                  "Facebook and TikTok = comments under Grab Malaysia's own posts/videos (small samples).",
                  fig_dir, "fig02_sentiment_by_platform.png"),
        fig_share(T["tab03_sentiment_by_language"], "lang_group", "Sentiment by language of the post",
                  "Language group from the share of Malay function words in each post.",
                  fig_dir, "fig03_sentiment_by_language.png"),
        fig_monthly(T["tab04_sentiment_by_month"], fig_dir),
        fig_topics(T["tab05_topic_summary"], fig_dir),
        fig_topic_volume(T["tab05_topic_summary"], fig_dir),
        fig_words(T["tab08_distinctive_words"], fig_dir),
        fig_models(mc, fig_dir),
    ]
    dist, per_post = distribution_by_method(root)
    T["tab14_distribution_by_method"] = dist
    T["tab15_vader_positive_but_label_negative"] = vader_false_positives(per_post)
    for name in ["tab14_distribution_by_method", "tab15_vader_positive_but_label_negative"]:
        T[name].to_csv(tab_dir / f"{name}.csv", index=False)
    F.append(fig_distribution_by_method(dist, fig_dir))
    appendix = export_appendix(root)
    return {"df": df, "tables": T, "figures": F, "per_post": per_post, "appendix": appendix}


# ---------------------------------------------------------------- method comparison on ALL posts (tab14, fig11)
METHOD_LABELS = {
    "true_label": "Labels (reference)",
    "a_vader_plain": "(a) VADER",
    "c_vader_lexicon_norm": "(c) VADER + lexicon + normalisation",
    "d_tfidf_logreg_oof": "(d) TF-IDF + LogReg (5-fold out-of-sample)",
}


def logreg_out_of_fold(df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """A prediction for EVERY post from a model that never saw it: StratifiedKFold(5, shuffle, 42) with the same
    TF-IDF + Logistic Regression pipeline as notebook 06 (src.evaluate.tfidf_logreg)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from src.evaluate import tfidf_logreg
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_predict(tfidf_logreg(), df["text_norm"].fillna(""), df["sentiment"], cv=cv)


def distribution_by_method(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Share of negative / neutral / positive predicted by each method on all posts (+ the per-post table)."""
    allp = pd.read_csv(root / "result/10_all_posts_with_split.csv", dtype={"post_id": str})
    final = pd.read_csv(root / "data/grab_sentiment_clean.csv", dtype={"post_id": str})
    final["d_tfidf_logreg_oof"] = logreg_out_of_fold(final)
    per_post = allp.merge(final[["post_id", "n_words", "d_tfidf_logreg_oof"]], on="post_id", how="left", validate="1:1")
    assert per_post["d_tfidf_logreg_oof"].notna().all() and len(per_post) == len(final)
    rows = []
    for col, name in METHOD_LABELS.items():
        v = per_post[col].value_counts().reindex(SENT, fill_value=0)
        rows.append({"method": name, "n_posts": int(v.sum()), **{f"n_{s}": int(v[s]) for s in SENT},
                     **{f"pct_{s}": round(100 * v[s] / v.sum(), 1) for s in SENT},
                     "agreement_with_labels_pct": round(100 * (per_post[col] == per_post["true_label"]).mean(), 1)})
    return pd.DataFrame(rows), per_post


def vader_false_positives(per_post: pd.DataFrame, n: int = 5, seed: int = 42) -> pd.DataFrame:
    """Posts where plain VADER says positive but the label is negative (readable length), for the discussion."""
    m = (per_post["a_vader_plain"] == "positive") & (per_post["true_label"] == "negative")
    pool = per_post[m & per_post["n_words"].between(8, 40)]
    out = pool.sample(min(n, len(pool)), random_state=seed)
    out = out[["post_id", "platform", "lang_group", "true_label", "a_vader_plain", "vader_plain_compound",
               "c_vader_lexicon_norm", "vader_lexnorm_compound", "text_clean"]].copy()
    out.insert(0, "n_such_posts_in_dataset", int(m.sum()))
    return out.reset_index(drop=True)


def fig_distribution_by_method(t: pd.DataFrame, out: Path) -> Path:
    colors = ["#6e6d68", SERIES[0], SERIES[1], SERIES[2]]      # grey = reference labels, then categorical slots 1-3
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    x = np.arange(len(SENT)); w = 0.2
    for i, (_, r) in enumerate(t.iterrows()):
        vals = [r[f"pct_{s}"] for s in SENT]
        bars = ax.bar(x + (i - 1.5) * w, vals, w * 0.92, color=colors[i], label=r["method"])
        ax.bar_label(bars, fmt="%.0f%%", fontsize=7, padding=1.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in SENT], fontsize=10)
    ax.set_ylim(0, 60); ax.set_ylabel("Share of all posts (%)"); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(ncol=2, loc="upper right", fontsize=8, handlelength=1)
    ax.set_title(f"Predicted sentiment distribution by method (all {int(t['n_posts'].iloc[0]):,} posts)")
    _note(fig, "VADER methods need no training. Logistic Regression: 5-fold stratified cross-validation, so every post "
               "is predicted by a model that did not see it.", y=-0.3 / fig.get_figheight())
    return _save(fig, out, "fig11_distribution_by_method.png")


def export_appendix(root: Path) -> Path:
    """All posts for the report appendix: id, platform, date, language group, anonymised text, label. No usernames."""
    f = pd.read_csv(root / "data/grab_sentiment_clean.csv", dtype={"post_id": str})
    a = f[["post_id", "platform", "created_at", "lang_group", "text_clean", "sentiment"]].rename(
        columns={"text_clean": "text_anonymised"})
    handles = a["text_anonymised"].str.findall(r"(?<![\w@])@(\w+)").explode().dropna().str.lower()
    bad = handles[~handles.isin(["user"]) & ~handles.str.startswith("grab")]
    assert bad.empty, f"personal @handles left in text: {sorted(set(bad))[:10]}"
    assert not a["text_anonymised"].str.contains(r"(?<![\w/])/?u/(?!user\b)[A-Za-z0-9_-]{3,20}\b", regex=True).any()
    p = root / "outputs/appendix_grab_posts_2015.csv"
    a.to_csv(p, index=False)
    return p


WEB_APP_DIR = Path("website and app analysis")


def run_web(root: Path = Path(".")) -> dict:
    """Part 1 (website analysis): PSI + on-page SEO summary tables and charts, in 'website and app analysis/'."""
    set_style()
    wa = Path(root) / WEB_APP_DIR
    (wa / "figures").mkdir(parents=True, exist_ok=True)
    psi = pd.read_csv(wa / "tables/pagespeed_median.csv")
    seo = pd.read_csv(wa / "tables/onpage_seo.csv")
    T = {"web_tab01_performance_summary": web_summary(psi), "web_tab02_onpage_seo_summary": seo_summary(seo)}
    for name, t in T.items():
        t.to_csv(wa / "tables" / f"{name}.csv", index=False)
    F = [fig_web_perf(T["web_tab01_performance_summary"], wa / "figures"),
         fig_web_scores(T["web_tab01_performance_summary"], wa / "figures")]
    return {"tables": T, "figures": F}
