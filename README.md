# Grab Malaysia: Web & Social Media Sentiment Analysis

Group project for **BAA5113 Web & Social Media Analytics**. We look at Grab Malaysia (GrabCar, GrabFood, GrabPay …) from two sides:

- **Part 1 – Grab's own channels:** speed, SEO and accessibility of five Grab Malaysia websites, plus Google Play and App Store reviews of the Grab app.
- **Part 2 – Public sentiment:** 2,015 posts and comments about Grab Malaysia from X (Twitter), Reddit, Threads, TikTok and Facebook (Nov 2025 – Nov 2026). They are labelled negative, neutral or positive, and used to compare VADER with a supervised Logistic Regression model.

## Repository layout

```
data/
  grab_sentiment_clean.csv       the final dataset (2,015 labelled posts, anonymised)
  LABELLING_GUIDE.md             rules used to label each post
notebooks/
  06_model_comparison.ipynb      VADER vs TF-IDF + Logistic Regression  -> result/
  05_visualisation_and_findings.ipynb   charts and tables for Part 2      -> outputs/
  history/                       how the dataset was collected, cleaned and labelled (record only)
result/                          model-comparison results (see result/README.md)
outputs/
  FINDINGS.md                    written findings for Part 2
  figures/, tables/              report-ready charts (300 dpi) and tables
  tables/data_preparation/       collection and cleaning tables for the Data Preparation section
  appendix_grab_posts_2015.csv   all posts for the report appendix
website and app analysis/        Part 1: notebook, README (findings), figures, tables, data
src/                             all the Python code the notebooks call
config.yaml                      settings (search terms, URLs, limits)
```

## The dataset: `data/grab_sentiment_clean.csv`

| | |
|---|---|
| Posts | 2,015 (X 1,033 · Reddit 676 · Threads 224 · TikTok 51 · Facebook 31) |
| Period | 7 Nov 2025 – 17 Nov 2026 |
| Language | English 1,348 · Malay 530 · mixed Malay-English (Manglish) 137 |
| Labels | negative 899 (44.6%) · neutral 898 (44.6%) · positive 218 (10.8%) |

**How the data was prepared:**
- **Collection:** public posts were collected with Apify actors. Search terms and actors are listed in `notebooks/history/`.
- **Cleaning:**
  - Duplicates, bots and official Grab accounts, off-topic posts and other languages were removed.
  - Personal @usernames, links, emails and phone numbers were replaced by placeholders.
- **Labelling:**
  - Each post was labelled by Claude (an AI assistant), reading it against `data/LABELLING_GUIDE.md`.
  - A team member then reviewed the posts flagged as unsure and corrected 12 labels.

Main columns:
- `post_id`, `platform`, `created_at`, `url` (username-free)
- `likes`, `replies`
- `text_clean` (anonymised text), `text_norm` (normalised text for the ML model), `text_manglish_norm` (for VADER)
- `lang_group` (en / ms / mixed)
- `sentiment` (the label)

## Main results

| Method (same 403 held-out test posts) | Accuracy | Macro-F1 |
|---|---|---|
| VADER | 0.390 | 0.374 |
| VADER + Manglish lexicon + normalisation | 0.439 | 0.408 |
| **TF-IDF + Logistic Regression** (trained on 1,612 posts) | **0.618** | **0.509** |

- **Overall mood:** sentiment towards Grab is mainly negative or neutral. The most negative topics are cancellations, customer service and price.
- **VADER over-predicts positive:** it labels about 39% of posts positive, against 10.8% in the labels.
- **Full write-ups:** [`outputs/FINDINGS.md`](outputs/FINDINGS.md) (Part 2) and [`website and app analysis/README.md`](website%20and%20app%20analysis/README.md) (Part 1).

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.13
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon wordnet omw-1.4

jupyter nbconvert --to notebook --execute --inplace notebooks/06_model_comparison.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_visualisation_and_findings.ipynb
jupyter nbconvert --to notebook --execute --inplace "website and app analysis/website_and_app_analysis.ipynb"
```

- Run notebook 06 before 05, because 05 reads 06's results.
- All three notebooks work from the files in this repository. They need no API keys and make no web requests.
- New data collection needs Apify and PageSpeed keys in a `.env` file (template: `.env.example`), and is not needed to reproduce the results.

## Ethics and privacy
- **Public data only.** Only public posts, comments and app reviews were used.
- **No usernames.** Usernames never appear in this repository; reviewer names were replaced by an irreversible SHA-256 hash. The raw scraper output, which contains usernames, stays in `data/raw/`, which is git-ignored.
- **robots.txt.** The website SEO scrape respects robots.txt. The app-store review feeds are disallowed for crawlers; the group made a one-off exception for a small (14-request) academic sample, as documented in `website and app analysis/README.md`.
- **Secrets.** API keys live only in `.env`, which is git-ignored.
