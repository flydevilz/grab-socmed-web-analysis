"""
Sentiment models and their comparison (notebook 04, Step 6 of the Manglish pivot).

(a) plain VADER on text_clean
(b) VADER + Manglish lexicon on text_clean
(c) VADER + Manglish lexicon on text_manglish_norm (normalised Manglish)
(d) TF-IDF + Logistic Regression (class_weight="balanced") trained on the labels

All four are scored on the SAME held-out test set (stratified 20%), so the numbers are comparable.
The reference labels are the `sentiment` column (Claude's labels + teammate corrections) unless a
hand-checked `my_label` column is supplied.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline

from src.manglish_lexicon import vader_with_manglish

CLASSES = ["negative", "neutral", "positive"]
METHODS = {
    "a_vader_plain": "(a) VADER",
    "b_vader_lexicon": "(b) VADER + Manglish lexicon",
    "c_vader_lexicon_norm": "(c) VADER + lexicon + normalisation",
    "d_tfidf_logreg": "(d) TF-IDF + Logistic Regression",
}


def vader_label(compound: float) -> str:
    """Standard VADER thresholds (Hutto & Gilbert, 2014)."""
    return "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"


def tfidf_logreg():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    )


def score_all(df: pd.DataFrame) -> pd.DataFrame:
    """Add VADER compound scores + labels for (a)-(c) and out-of-fold TF-IDF+LR predictions (d)."""
    out = df.copy()
    plain, mang = SentimentIntensityAnalyzer(), vader_with_manglish()
    out["vader_plain_compound"] = out["text_clean"].fillna("").map(lambda t: plain.polarity_scores(t)["compound"])
    out["vader_lex_compound"] = out["text_clean"].fillna("").map(lambda t: mang.polarity_scores(t)["compound"])
    out["vader_lexnorm_compound"] = out["text_manglish_norm"].fillna("").map(lambda t: mang.polarity_scores(t)["compound"])
    out["a_vader_plain"] = out["vader_plain_compound"].map(vader_label)
    out["b_vader_lexicon"] = out["vader_lex_compound"].map(vader_label)
    out["c_vader_lexicon_norm"] = out["vader_lexnorm_compound"].map(vader_label)
    # 5-fold out-of-fold predictions, so every row gets a prediction from a model that never saw it
    out["d_tfidf_logreg"] = cross_val_predict(tfidf_logreg(), out["text_norm"].fillna(""), out["sentiment"], cv=5)
    return out


def compare(df: pd.DataFrame, truth_col: str = "sentiment", test_size: float = 0.2,
            fig_dir: Path | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Train (d) on a stratified 80% split; evaluate (a)-(d) on the same 20% test split.
    Returns (results table, confusion matrices, test rows with predictions)."""
    train, test = train_test_split(df, test_size=test_size, stratify=df[truth_col], random_state=42)
    model = tfidf_logreg().fit(train["text_norm"].fillna(""), train[truth_col])
    test = test.copy()
    test["d_tfidf_logreg"] = model.predict(test["text_norm"].fillna(""))
    rows, cms = [], {}
    for col, name in METHODS.items():
        y, p = test[truth_col], test[col]
        f1s = f1_score(y, p, labels=CLASSES, average=None, zero_division=0)
        rows.append({"method": name, "accuracy": accuracy_score(y, p),
                     "macro_f1": f1_score(y, p, labels=CLASSES, average="macro", zero_division=0),
                     **{f"f1_{c}": f for c, f in zip(CLASSES, f1s)}, "n_test": len(test)})
        cms[col] = confusion_matrix(y, p, labels=CLASSES)
        if fig_dir is not None:
            fig, ax = plt.subplots(figsize=(4.2, 3.8))
            ConfusionMatrixDisplay(cms[col], display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Label")
            fig.tight_layout()
            fig.savefig(Path(fig_dir) / f"confusion_{col}.png", dpi=150)
            plt.close(fig)
    res = pd.DataFrame(rows)
    res[res.columns[1:-1]] = res[res.columns[1:-1]].round(3)
    return res, cms, test


def fixed_by_manglish(test: pd.DataFrame, truth_col: str = "sentiment", n: int = 8) -> pd.DataFrame:
    """Rows where plain VADER (a) was wrong and lexicon + normalisation (c) was right."""
    m = (test["a_vader_plain"] != test[truth_col]) & (test["c_vader_lexicon_norm"] == test[truth_col])
    cols = ["post_id", "platform", "lang_group", truth_col, "a_vader_plain", "vader_plain_compound",
            "c_vader_lexicon_norm", "vader_lexnorm_compound", "text_clean"]
    return test.loc[m, cols].sort_values("lang_group").head(n)


def hand_check_sample(df: pd.DataFrame, n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Random sample stratified by label x platform (proportional, at least 1 per non-empty cell)."""
    groups = df.groupby(["sentiment", "platform"])
    sizes = groups.size()
    alloc = (sizes / sizes.sum() * n).round().clip(lower=1).astype(int)
    while alloc.sum() > n:
        alloc[alloc.idxmax()] -= 1
    while alloc.sum() < n:
        alloc[(sizes - alloc).idxmax()] += 1
    parts = [g.sample(min(int(alloc[k]), len(g)), random_state=seed) for k, g in groups]
    s = pd.concat(parts).sample(frac=1, random_state=seed)
    return (s.assign(my_label="")[["post_id", "platform", "lang_group", "url", "sentiment", "my_label", "text_clean"]]
             .rename(columns={"sentiment": "claude_label", "text_clean": "text"}).reset_index(drop=True))
