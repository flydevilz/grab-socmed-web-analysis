# CLAUDE.md – Grab Malaysia Sentiment Analysis

University group project for **BAA5113 Web & Social Media Analytics**. We analyse public
sentiment towards **Grab Malaysia** (GrabCar, GrabFood, GrabMart, GrabPay, etc.) by combining
web analytics on Grab's Malaysian sites with social-media posts (Reddit, X/Twitter, Threads).
Deadline: **11 Dec 2026**.

## Working agreement
- Build incrementally; **stop for the user's review after each step**.
- **Never run a paid Apify collection without explicit confirmation.** Full runs were approved and done on
  2026-11-05 (`pilot_mode: false`); any further paid run needs the user's OK. Notebook 01 defaults to load-only.
- Groupmates who don't code will read the notebooks: keep them thin, with plenty of
  plain-English markdown. Put logic in `src/`.

## Folder conventions (tidied for GitHub 2026-11-18)
The ONLY dataset is `data/grab_sentiment_clean.csv` (2,015 posts, user decision 2026-11-18). Backups, intermediate datasets,
old label files, superseded outputs and logs were moved OUT of the repo to `../sentiment_analysis_archive_20261118/`
(same folder structure; nothing deleted). Don't recreate backups or intermediate copies inside the repo.

| Path | Contents |
|---|---|
| `config.yaml` | All tunable settings (URLs, queries, actors, caps, prices). Edit here, not in notebooks. |
| `.env` | Secrets only (`APIFY_TOKEN*`, `PAGESPEED_API_KEY`). Git-ignored. Template: `.env.example`. |
| `data/grab_sentiment_clean.csv` | **The final dataset.** `data/LABELLING_GUIDE.md` = labelling rules. |
| `data/raw/` | Raw Apify / app-review JSON. **Git-ignored** (usernames). Local only. |
| `notebooks/06_model_comparison.ipynb` → `result/` | VADER (a–c) vs TF-IDF+LogReg, stratified 80/20 split (`src/model_comparison.py`). Run first. |
| `notebooks/05_visualisation_and_findings.ipynb` → `outputs/` | Part 2 charts `fig01-08, fig11`, tables `tab01-11, tab14-15`, `appendix_grab_posts_2015.csv`, brief `FINDINGS.md` (`src/findings.py`). Reads `result/10_all_posts_with_split.csv`. |
| `notebooks/history/` | 01, 01b, 02, 03, 04: record of collection/cleaning/labelling with saved outputs. NOT re-runnable (inputs archived). |
| `outputs/tables/data_preparation/` | Collection and cleaning funnels, extra-collection tables, lexicon-review token lists. |
| `website and app analysis/` | **Part 1**: notebook, `README.md` (findings), `tables/`, `figures/`, `data/pagespeed_raw/` (final PSI runs), `data/app_reviews_clean.csv` + `data/app_store_info.json` (the notebook runs from these; `rebuild_from_raw=True` needs `data/raw/app_reviews/`). |
| `src/` | All logic. `collect.py`, `clean.py`, `label.py`, `merge_v3.py`, `collection_log.py` are used by history notebooks / app_reviews. |

## Normalised post schema
`post_id, platform, query, text, created_at (ISO UTC), author_hash (SHA-256 of lower-cased
username), url (username-free form), likes, replies, is_comment, collected_at`.
Raw usernames never leave `data/raw/`.

## App reviews (Part 1, 2026-11-18)
- Both review endpoints are robots.txt-disallowed (Play `/_/`, Apple `/*/rss/*`); the user approved a **one-off exception**
  (14 requests, free). Don't re-collect without asking. The notebook defaults to `RUN_APP_COLLECTION = False`.
- Google Play filters by **language, not country**: English reviews are SEA-wide, Malay mostly Malaysian. Results are reported
  per `source` (App Store MY / Play Malay / Play English all markets).

## Actor notes (input schemas checked on Apify, Nov 2026)
- Apify account is on the **Free plan: US$5 credit/month shared by all platforms**. **User's hard limit: US$5/month,
  never more** (`social.monthly_budget_usd`; `run_actor` checks live usage and caps/skips runs). Per-platform
  run caps: `platforms.<p>.max_charge_per_run_usd`. Platform-specific `queries` override `social.queries`.
- `fatihtahta/reddit-scraper-search-fast`: `subredditName` (single) + `subredditKeywords`, `dateFrom/dateTo`,
  `maxPosts`, `scrapeComments`, `maxComments`, `strictSearch`, `sentiment_analysis` (AFINN, English only).
  One run per subreddit. Strict search + 12 months found ~91 posts; caps were never hit.
- `xquik/x-tweet-scraper` (X): `searchTerms` (with `since:`/`until:`), `maxItems` (run total), `queryType`,
  `includeSearchTerms`. Free-plan OK, US$0.00015/row, no start fee. One run per query.
  (`apidojo/tweet-scraper` was dropped: no API access on the Free plan, 50-tweet minimum.)
- `futurizerush/meta-threads-scraper`: `mode: "search"`, `keywords`, `max_posts` (per keyword, min 10),
  `search_filter`, `start_date/end_date`. US$0.0025/post + US$0.02 start: the priciest platform.
- Each platform writes `data/interim/[pilot_]posts_<platform>.csv`; `posts_raw_combined.csv` is rebuilt
  from all of them by `combine_platform_files()`.
- X (`xquik`) returns newest first, so a busy query hits its cap within weeks. A twitter query in config may be a dict
  `{q, since, until, split_into, cap}` to split the window and reach older tweets. English Malaysian content on X is
  largely exhausted (rounds 2–5 all came back under their caps).
- **Budget status (cycle 4 Nov – 3 Dec 2026): US$4.963 of US$5 used.** No further paid runs possible this cycle.
- **Second Apify account (token `APIFY_TOKEN_REDDIT` in `.env`, cycle 17 Nov – 16 Dec 2026): ~US$4.66 of US$5 used** (2026-11-18;
  Manglish round: X round 3 US$0.12, 813 tweets; 784 posts restored free from saved raw files)
  (Threads round 2 US$3.28, Facebook ~US$0.48, TikTok ~US$0.44, tests). Same US$5 hard limit and guard.
- Round 2 (2026-11-17): Threads search (`threads.round2` in config; multi-word queries return mostly junk, brand words
  like `GrabMY` work), Threads `user_replies` on @grabmy keeps the *customer* post answered. Reddit was blocked
  ("temporary Reddit access issue", 0 items, not charged). Facebook = comments on facebook.com/GrabMY posts
  (`apify/facebook-comments-scraper`; Facebook shows only a few top comments per post to scrapers). TikTok = comments on
  @grabmy videos (`clockworks/tiktok-comments-scraper`; `apidojo/...` caps the Free plan at 10 items). Facebook/TikTok
  comments under Grab's own posts count as on-topic in cleaning (like Reddit comments in a Grab thread).
- New-data files: `data/processed/<tag>_clean.csv` + `<tag>_removed_audit.csv`, labels in
  `data/labelling/claude_labels_<tag>.csv` (hand-labelled by Claude against the guide, not the API; user decision
  2026-11-17). Merged result: `data/processed/labelled_dataset_v2.csv` (1,341 rows, same 27 columns); original backed up as
  `labelled_dataset_backup_20261117.csv`. Hand-check sample: `data/labelling/review_unsure.csv`.
- 7 Threads posts were added by hand from user-supplied links (`data/raw/manual_threads_posts_20261106.json`,
  `collect.add_manual_posts`, flag `manually_added`). Cleaning skips only the automatic relevance and Malaysia-context
  rules for them. Cleaned dataset: exactly 2,000 rows; 1,222 relevant after labelling (provisional).
- PageSpeed: PSI caches recent results, so repeat runs go in waves at least 60 s apart, and a test that
  repeats an earlier `fetchTime` is rejected. food.grab.com sometimes returns NOT_HTML to Lighthouse.
  Keep `food.grab.com/my/` as the URL (user's choice: Malaysia region entry point), even though it redirects to `/my/en/`.

## Environment
Python 3.13 virtualenv in `.venv/` (`.venv/bin/python`). Pinned deps in `requirements.txt`.
Execute notebooks with `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/<nb>.ipynb`.
