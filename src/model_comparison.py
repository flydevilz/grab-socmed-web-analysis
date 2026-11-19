"""
Final model comparison on the very clean dataset (data/grab_sentiment_clean.csv).

Methods (all judged against the labels in `sentiment`):
  (a) VADER (plain)                          lexicon-based, no training
  (b) VADER + Manglish lexicon               on text_clean
  (c) VADER + Manglish lexicon + normalising on text_manglish_norm
  (d) TF-IDF + Logistic Regression           supervised, trained on the 80% TRAIN split only

The data is split ONCE (stratified 80/20, random_state=42). The supervised model never sees the
test rows. VADER needs no training, so it is scored on the same 20% test rows for a fair comparison
(and on the full dataset as extra information).

Everything is written to `result/` (see write_all()).
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.evaluate import CLASSES, METHODS, tfidf_logreg, vader_label
from src.manglish_lexicon import vader_with_manglish

SEED = 42
TEST_SIZE = 0.2


def add_vader(df: pd.DataFrame) -> pd.DataFrame:
    """VADER compound scores + labels for (a)-(c). Needs no training, so it runs on every row."""
    out = df.copy()
    plain, mang = SentimentIntensityAnalyzer(), vader_with_manglish()
    out["vader_plain_compound"] = out["text_clean"].fillna("").map(lambda t: plain.polarity_scores(t)["compound"])
    out["vader_lex_compound"] = out["text_clean"].fillna("").map(lambda t: mang.polarity_scores(t)["compound"])
    out["vader_lexnorm_compound"] = out["text_manglish_norm"].fillna("").map(lambda t: mang.polarity_scores(t)["compound"])
    out["a_vader_plain"] = out["vader_plain_compound"].map(vader_label)
    out["b_vader_lexicon"] = out["vader_lex_compound"].map(vader_label)
    out["c_vader_lexicon_norm"] = out["vader_lexnorm_compound"].map(vader_label)
    return out


def split(df: pd.DataFrame, truth: str = "sentiment"):
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df[truth], random_state=SEED)
    return train.copy(), test.copy()


def train_logreg(train: pd.DataFrame, truth: str = "sentiment"):
    """Fit TF-IDF + LR on the train split; also 5-fold CV on the train split (model selection check)."""
    X, y = train["text_norm"].fillna(""), train[truth]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_f1 = cross_val_score(tfidf_logreg(), X, y, cv=cv, scoring="f1_macro")
    model = tfidf_logreg().fit(X, y)
    return model, cv_f1


def metrics_row(name: str, y, p, split_name: str) -> dict:
    f1s = f1_score(y, p, labels=CLASSES, average=None, zero_division=0)
    return {"method": name, "evaluated_on": split_name, "n": len(y),
            "accuracy": accuracy_score(y, p),
            "macro_f1": f1_score(y, p, labels=CLASSES, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, p, labels=CLASSES, average="weighted", zero_division=0),
            **{f"f1_{c}": f for c, f in zip(CLASSES, f1s)}}


def top_lr_words(model, n: int = 15) -> pd.DataFrame:
    """The n words/phrases that push the LR model most strongly towards each class."""
    vec, lr = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
    vocab = vec.get_feature_names_out()
    rows = []
    for i, cls in enumerate(lr.classes_):
        order = lr.coef_[i].argsort()[::-1][:n]
        rows += [{"class": cls, "rank": r + 1, "term": vocab[j], "weight": round(lr.coef_[i][j], 3)}
                 for r, j in enumerate(order)]
    return pd.DataFrame(rows)


def run(df: pd.DataFrame, truth: str = "sentiment") -> dict:
    df = add_vader(df[df[truth].isin(CLASSES)])
    train, test = split(df, truth)
    model, cv_f1 = train_logreg(train, truth)
    test["d_tfidf_logreg"] = model.predict(test["text_norm"].fillna(""))
    proba = model.predict_proba(test["text_norm"].fillna(""))
    test["d_tfidf_logreg_confidence"] = proba.max(axis=1).round(3)
    train_pred = model.predict(train["text_norm"].fillna(""))
    baseline = DummyClassifier(strategy="most_frequent").fit(train[["text_norm"]], train[truth])
    base_pred = baseline.predict(test[["text_norm"]])

    # 1) main comparison: all methods on the SAME test rows
    rows = [metrics_row(name, test[truth], test[col], "test (20%)") for col, name in METHODS.items()]
    rows.append(metrics_row("Baseline: always '%s'" % base_pred[0], test[truth], base_pred, "test (20%)"))
    comparison = pd.DataFrame(rows)

    # 2) supervised-model evaluation: train vs CV vs test
    lr_eval = pd.DataFrame([
        metrics_row(METHODS["d_tfidf_logreg"], train[truth], train_pred, "train (80%) - seen data"),
        {"method": METHODS["d_tfidf_logreg"], "evaluated_on": "5-fold CV within train",
         "n": len(train), "macro_f1": cv_f1.mean(), "macro_f1_sd": cv_f1.std()},
        metrics_row(METHODS["d_tfidf_logreg"], test[truth], test["d_tfidf_logreg"], "test (20%) - unseen data"),
    ])

    # 3) VADER on the full dataset (no training needed, extra information only)
    vader_full = pd.DataFrame([metrics_row(name, df[truth], df[col], "full dataset")
                               for col, name in METHODS.items() if col != "d_tfidf_logreg"])

    # 4) classification reports + confusion matrices
    reports = {col: classification_report(test[truth], test[col], labels=CLASSES, digits=3, zero_division=0)
               for col in METHODS}
    cms = {col: confusion_matrix(test[truth], test[col], labels=CLASSES) for col in METHODS}

    # 5) how often the methods agree with each other on the test rows
    cols = list(METHODS)
    agree = pd.DataFrame(index=[METHODS[c] for c in cols], columns=[METHODS[c] for c in cols], dtype=float)
    for a, b in [(a, b) for a in cols for b in cols]:
        agree.loc[METHODS[a], METHODS[b]] = (test[a] == test[b]).mean()
    pair_rows = []
    for a, b in combinations(cols, 2):
        both_right = ((test[a] == test[truth]) & (test[b] == test[truth])).sum()
        only_a = ((test[a] == test[truth]) & (test[b] != test[truth])).sum()
        only_b = ((test[a] != test[truth]) & (test[b] == test[truth])).sum()
        pair_rows.append({"method_1": METHODS[a], "method_2": METHODS[b], "agreement": (test[a] == test[b]).mean(),
                          "both_correct": both_right, "only_method_1_correct": only_a,
                          "only_method_2_correct": only_b,
                          "both_wrong": len(test) - both_right - only_a - only_b})
    pairs = pd.DataFrame(pair_rows)

    # 6) results by language group and platform (test rows)
    def by_group(g):
        return pd.DataFrame([{g: k, "n": len(s), **{f"{METHODS[c]} macro_f1":
                              f1_score(s[truth], s[c], labels=CLASSES, average="macro", zero_division=0)
                              for c in cols}} for k, s in test.groupby(g)])
    by_lang, by_platform = by_group("lang_group"), by_group("platform")

    # 7) per-post predictions
    test["split"] = "test"
    test["vader_correct"] = test["c_vader_lexicon_norm"] == test[truth]
    test["logreg_correct"] = test["d_tfidf_logreg"] == test[truth]
    test["vader_vs_logreg"] = (test["c_vader_lexicon_norm"] == test["d_tfidf_logreg"]).map({True: "agree", False: "disagree"})
    keep = ["post_id", "platform", "lang_group", "url", "text_clean", truth,
            "vader_plain_compound", "a_vader_plain", "vader_lexnorm_compound", "c_vader_lexicon_norm",
            "b_vader_lexicon", "d_tfidf_logreg", "d_tfidf_logreg_confidence",
            "vader_correct", "logreg_correct", "vader_vs_logreg"]
    test_pred = test[keep].rename(columns={truth: "true_label"})
    disagree = test_pred[test_pred["vader_vs_logreg"] == "disagree"].sort_values(["logreg_correct", "vader_correct"], ascending=False)

    train_ids = set(train["post_id"])
    all_pred = df.assign(split=df["post_id"].map(lambda i: "train" if i in train_ids else "test"))
    all_pred = all_pred.merge(test[["post_id", "d_tfidf_logreg"]], on="post_id", how="left")
    all_pred = all_pred[["post_id", "platform", "lang_group", "split", "text_clean", truth,
                         "vader_plain_compound", "a_vader_plain", "vader_lexnorm_compound",
                         "c_vader_lexicon_norm", "d_tfidf_logreg"]].rename(columns={truth: "true_label"})

    return dict(df=df, train=train, test=test, model=model, comparison=comparison, lr_eval=lr_eval,
                vader_full=vader_full, reports=reports, cms=cms, agreement=agree, pairs=pairs,
                by_lang=by_lang, by_platform=by_platform, test_pred=test_pred, disagree=disagree,
                all_pred=all_pred, top_words=top_lr_words(model))


def _round(d: pd.DataFrame) -> pd.DataFrame:
    return d.round(3)


def write_all(r: dict, out: Path, source: str) -> list[Path]:
    """Save every table, report and figure into `out` (the result/ folder)."""
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    written = []

    def save(d: pd.DataFrame, name: str, index=False):
        p = out / name
        _round(d).to_csv(p, index=index)
        written.append(p)

    save(r["comparison"], "01_model_comparison.csv")
    save(r["lr_eval"], "02_logreg_evaluation.csv")
    save(r["vader_full"], "03_vader_full_dataset.csv")
    save(r["agreement"], "04_agreement_matrix.csv", index=True)
    save(r["pairs"], "05_agreement_pairs.csv")
    save(r["by_lang"], "06_macro_f1_by_language.csv")
    save(r["by_platform"], "07_macro_f1_by_platform.csv")
    save(r["test_pred"], "08_test_predictions.csv")
    save(r["disagree"], "09_vader_vs_logreg_disagreements.csv")
    save(r["all_pred"], "10_all_posts_with_split.csv")
    save(r["top_words"], "11_logreg_top_words.csv")

    p = out / "classification_reports.txt"
    p.write_text("\n\n".join(f"=== {METHODS[c]} (test set, n={len(r['test'])}) ===\n{rep}"
                             for c, rep in r["reports"].items()))
    written.append(p)

    # confusion matrices: one per method + one combined figure
    fig_all, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax_all, (col, cm) in zip(axes, r["cms"].items()):
        for ax in (ax_all, plt.subplots(figsize=(4.4, 4))[1]):
            ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(METHODS[col], fontsize=10)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True label")
        f = ax.figure; f.tight_layout()
        p = out / "figures" / f"confusion_{col}.png"; f.savefig(p, dpi=150); plt.close(f); written.append(p)
    fig_all.tight_layout()
    p = out / "figures" / "confusion_all_methods.png"; fig_all.savefig(p, dpi=150); plt.close(fig_all); written.append(p)

    # bar chart: accuracy and macro-F1 per method
    c = r["comparison"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = range(len(c)); w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], c["accuracy"], w, label="Accuracy", color="#4C72B0")
    b2 = ax.bar([i + w / 2 for i in x], c["macro_f1"], w, label="Macro-F1", color="#DD8452")
    ax.bar_label(b1, fmt="%.2f", fontsize=8); ax.bar_label(b2, fmt="%.2f", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels([m.replace(" + ", "\n+ ") for m in c["method"]], fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("Score (test set)"); ax.legend(frameon=False)
    ax.set_title(f"Model comparison on the same test set (n={int(c['n'].iloc[0])})", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); p = out / "figures" / "model_comparison_bar.png"; fig.savefig(p, dpi=150); plt.close(fig)
    written.append(p)

    # README
    p = out / "README.md"
    p.write_text(_readme(r, source)); written.append(p)
    return written


def _md(d: pd.DataFrame) -> str:
    """Small markdown table writer (avoids the optional `tabulate` dependency)."""
    fmt = lambda v: "" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def _readme(r: dict, source: str) -> str:
    c, ev, df, tr, te = r["comparison"], r["lr_eval"], r["df"], r["train"], r["test"]
    best = c.iloc[c["macro_f1"].idxmax()]
    lab = df["sentiment"].value_counts().reindex(CLASSES)
    pairs = r["pairs"]
    vl = pairs[(pairs.method_1 == METHODS["c_vader_lexicon_norm"]) & (pairs.method_2 == METHODS["d_tfidf_logreg"])].iloc[0]
    tbl = c[["method", "accuracy", "macro_f1", "f1_negative", "f1_neutral", "f1_positive"]].pipe(_md)
    evt = _md(ev[["evaluated_on", "n", "accuracy", "macro_f1"]])
    return f"""# Model comparison results

Input: `{source}` ({len(df):,} posts; negative {lab['negative']}, neutral {lab['neutral']}, positive {lab['positive']}).
Produced by `notebooks/06_model_comparison.ipynb` (logic in `src/model_comparison.py`).

## How the comparison was done
- **Split:** the data was split once into **train 80% ({len(tr):,} posts)** and **test 20% ({len(te):,} posts)**,
  stratified so each split has the same mix of labels (random_state = 42).
- **Logistic Regression** (TF-IDF words + 2-word phrases, balanced class weights) was trained on the train split only.
  The test posts were never seen during training.
- **VADER** needs no training. It was scored on the **same test posts**, so every method is compared on the same posts.
- **Baseline:** always predict the most common label. A useful model must beat this.

## Main result (test set, n = {len(te)})
{tbl}

**Best method: {best['method']}** (macro-F1 {best['macro_f1']:.3f}, accuracy {best['accuracy']:.3f}).
Macro-F1 is the main metric. It averages the three classes equally, so the small positive class counts as much as the others.

## Is the Logistic Regression over-fitting?
{evt}

On the posts it was trained on, the model scores macro-F1 {ev.macro_f1.iloc[0]:.3f}, but on unseen test posts it
scores {ev.macro_f1.iloc[2]:.3f}. That gap means the model partly memorises its training posts, which is common for
word-based text models trained on about 1,600 posts. Cross-validation inside the train split gave macro-F1
{ev.macro_f1.iloc[1]:.3f} ± {ev.macro_f1_sd.iloc[1]:.3f}. The test score is the honest figure to report.

## Weak spot: positive posts
Only {lab['positive']} of {len(df):,} posts are positive. The F1 for the positive class is LogReg {c.f1_positive.iloc[3]:.3f}
and VADER (+lexicon+norm) {c.f1_positive.iloc[2]:.3f}. Every method struggles to find positive posts, and VADER does
slightly better on this class. LogReg's higher overall score comes from the negative and neutral classes.

## VADER vs Logistic Regression on the test posts
- They give the same label on **{vl['agreement']:.0%}** of test posts.
- Both correct: {vl['both_correct']} · only VADER (+lexicon+norm) correct: {vl['only_method_1_correct']} ·
  only LogReg correct: {vl['only_method_2_correct']} · both wrong: {vl['both_wrong']}.
- Examples where they disagree are in `09_vader_vs_logreg_disagreements.csv`.

## Files
| File | What it shows |
|---|---|
| `01_model_comparison.csv` | Accuracy, macro-F1 and F1 per class for every method on the test set |
| `02_logreg_evaluation.csv` | Logistic Regression on train vs 5-fold CV vs test (over-fitting check) |
| `03_vader_full_dataset.csv` | VADER scored on all {len(df):,} posts (no training needed) |
| `04_agreement_matrix.csv` | % of test posts on which each pair of methods gives the same label |
| `05_agreement_pairs.csv` | For each pair: both correct / only one correct / both wrong |
| `06_macro_f1_by_language.csv` | Macro-F1 for English, Malay and mixed posts |
| `07_macro_f1_by_platform.csv` | Macro-F1 for each platform |
| `08_test_predictions.csv` | Every test post with its true label and each method's prediction |
| `09_vader_vs_logreg_disagreements.csv` | Test posts where VADER and LogReg disagree |
| `10_all_posts_with_split.csv` | All posts, whether each was train or test, and their VADER scores |
| `11_logreg_top_words.csv` | Words that push LogReg most strongly towards each label |
| `classification_reports.txt` | Precision / recall / F1 per class for each method |
| `figures/model_comparison_bar.png` | Accuracy and macro-F1 bar chart |
| `figures/confusion_*.png` | Confusion matrices (one per method, plus all four together) |
"""


def main(source: str = "data/grab_sentiment_clean.csv", out: str = "result") -> dict:
    df = pd.read_csv(source)
    r = run(df)
    write_all(r, Path(out), source)
    return r
