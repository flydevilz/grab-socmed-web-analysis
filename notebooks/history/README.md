# How the dataset was built (record only)

These notebooks show how `data/grab_sentiment_clean.csv` was produced. They are kept **with their saved outputs**,
so the steps and numbers can be read on GitHub and described in the report's Data Preparation section.

| Notebook | What it records |
|---|---|
| `01_data_collection.ipynb` | Website tests (PageSpeed Insights, SEO) and the first social-media collection (Reddit, X, Threads), 4–6 Nov 2026 |
| `01b_extra_data_collection.ipynb` | Extra collections on 17–18 Nov 2026: X with Malay keywords, Threads round 2, Facebook and TikTok comments; Apify actors, search terms, dates, item counts |
| `02_data_cleaning.ipynb` | Cleaning rules and the first (English-only) cleaning round |
| `03_manual_labelling.ipynb` | First labelling round and the team review of unsure posts |
| `04_sentiment_analysis.ipynb` | First model comparison on an earlier version of the data (superseded by `../06_model_comparison.ipynb`) |

**They are not meant to be re-run.** They need the raw scraper files and intermediate datasets, which are not in this
repository: the raw files contain usernames, and the intermediate files were superseded by the final dataset.
Report the numbers from the final dataset and from notebooks 05 and 06, not the intermediate numbers shown here.

Tables from these steps that the report can use are in `outputs/tables/data_preparation/`.
