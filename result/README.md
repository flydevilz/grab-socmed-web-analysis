# Model comparison results

Input: `data/grab_sentiment_clean.csv` (2,015 posts; negative 899, neutral 898, positive 218).
Produced by `notebooks/06_model_comparison.ipynb` (logic in `src/model_comparison.py`).

## How the comparison was done
- **Split:** the data was split once into **train 80% (1,612 posts)** and **test 20% (403 posts)**,
  stratified so each split has the same mix of labels (random_state = 42).
- **Logistic Regression** (TF-IDF words + 2-word phrases, balanced class weights) was trained on the train split only.
  The test posts were never seen during training.
- **VADER** needs no training. It was scored on the **same test posts**, so every method is compared on the same posts.
- **Baseline:** always predict the most common label. A useful model must beat this.

## Main result (test set, n = 403)
| method | accuracy | macro_f1 | f1_negative | f1_neutral | f1_positive |
|---|---|---|---|---|---|
| (a) VADER | 0.390 | 0.374 | 0.516 | 0.336 | 0.272 |
| (b) VADER + Manglish lexicon | 0.437 | 0.406 | 0.614 | 0.331 | 0.275 |
| (c) VADER + lexicon + normalisation | 0.439 | 0.408 | 0.616 | 0.331 | 0.277 |
| (d) TF-IDF + Logistic Regression | 0.618 | 0.509 | 0.690 | 0.640 | 0.195 |
| Baseline: always 'negative' | 0.447 | 0.206 | 0.617 | 0.000 | 0.000 |

**Best method: (d) TF-IDF + Logistic Regression** (macro-F1 0.509, accuracy 0.618).
Macro-F1 is the main metric. It averages the three classes equally, so the small positive class counts as much as the others.

## Is the Logistic Regression over-fitting?
| evaluated_on | n | accuracy | macro_f1 |
|---|---|---|---|
| train (80%) - seen data | 1612 | 0.951 | 0.954 |
| 5-fold CV within train | 1612 |  | 0.567 |
| test (20%) - unseen data | 403 | 0.618 | 0.509 |

On the posts it was trained on, the model scores macro-F1 0.954, but on unseen test posts it
scores 0.509. That gap means the model partly memorises its training posts, which is common for
word-based text models trained on about 1,600 posts. Cross-validation inside the train split gave macro-F1
0.567 ± 0.027. The test score is the honest figure to report.

## Weak spot: positive posts
Only 218 of 2,015 posts are positive. The F1 for the positive class is LogReg 0.195
and VADER (+lexicon+norm) 0.277. Every method struggles to find positive posts, and VADER does
slightly better on this class. LogReg's higher overall score comes from the negative and neutral classes.

## VADER vs Logistic Regression on the test posts
- They give the same label on **41%** of test posts.
- Both correct: 116 · only VADER (+lexicon+norm) correct: 61 ·
  only LogReg correct: 133 · both wrong: 93.
- Examples where they disagree are in `09_vader_vs_logreg_disagreements.csv`.

## Files
| File | What it shows |
|---|---|
| `01_model_comparison.csv` | Accuracy, macro-F1 and F1 per class for every method on the test set |
| `02_logreg_evaluation.csv` | Logistic Regression on train vs 5-fold CV vs test (over-fitting check) |
| `03_vader_full_dataset.csv` | VADER scored on all 2,015 posts (no training needed) |
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
