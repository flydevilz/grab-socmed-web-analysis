# Part 1 – Website and app analysis (Grab Malaysia)

This folder holds everything about Grab Malaysia's **own digital channels**: its websites and its mobile app.
Part 2 (social-media sentiment) is in `outputs/FINDINGS.md`.

| What | Where |
|---|---|
| Notebook (run top to bottom) | `website_and_app_analysis.ipynb` |
| Charts (PNG, 300 dpi, ready for Word) | `figures/` |
| Tables (CSV, open in Excel) | `tables/` |
| Clean, anonymised app reviews (2,500) | `data/app_reviews_clean.csv` |
| Raw PageSpeed test results (JSON) | `data/pagespeed_raw/` |
| Code | `src/findings.py` (`run_web`), `src/app_reviews.py`; PageSpeed tests in `notebooks/history/01_data_collection.ipynb` |

---

## A. Websites

**What we measured.**
- Google PageSpeed Insights (Lighthouse 13.6) tested five Grab Malaysia sites on a simulated phone (mobile) and a laptop (desktop), 3 runs each, on 4 Nov 2026. We report the median.
- We also checked each page's HTML for basic on-page SEO: title, meta description, H1 heading and image alt text.
- The bands below are Google's: 90–100 good, 50–89 needs improvement, 0–49 poor.

**W1. All Grab sites are much slower on mobile than on desktop, and GrabFood is the worst** (web_fig01, web_tab01).

| Site | Mobile | Desktop |
|---|---|---|
| Grab Dine Out | 70 | 94 |
| GrabMart | 54 | 74 |
| grab.com/my (main) | 46 | 65 |
| GrabExpress | 43 | 59 |
| **GrabFood** | **15** | 76 |

- On a mobile phone, GrabFood's main content takes about **25 seconds** to appear in the lab test.
- Data from real Chrome users also rates GrabFood's mobile experience **"slow"**, mainly because the layout jumps around while loading (CLS).
- Most Malaysians browse on their phones, so mobile is the score that matters.

**W2. The other quality scores are generally good** (web_fig02).
- Accessibility scores 75–95, best practices 77–96 and SEO 85–100.
- Speed on mobile is the one clear weak spot.

**W3. On-page SEO has a few clear gaps** (web_tab02).
- **GrabExpress:** the HTML has **no title, no meta description and no H1 heading**. The page is probably built in the browser with JavaScript, so search engines that don't run JavaScript see an empty page.
- **GrabFood:** no H1 heading.
- **Main site:** 4 images without alt text.
- **GrabMart:** refused our crawler (HTTP 403), so it could not be audited.
- **Grab Dine Out:** no basic issues.

---

## B. App reviews (Google Play and App Store)

**What we collected (18 Nov 2026, free, no Apify credit).**

| Source | Reviews | Period covered | Avg ★ in sample | Store's public score |
|---|---|---|---|---|
| App Store – **Malaysia store** | 500 (newest) | 12 Aug – 16 Nov 2026 | **2.76** | 4.73 (905,812 ratings) |
| Google Play – **Malay** reviews | 1,000 (newest) | 4 May – 16 Nov 2026 | **4.41** | 4.81 (16.86M ratings*) |
| Google Play – **English** reviews (all markets) | 1,000 (newest) | 31 Oct – 17 Nov 2026 | **3.24** | (same) |

\* The public Play score is shared by all countries. On the collection day it was 4.81★ from 16,863,093 ratings.
That is slightly more than the "4.8 from 16.2M" figure quoted earlier, because the count keeps growing.

> **Google Play filters reviews by language, not by country.**
> - The English reviews come from every Grab market. We found clues from Indonesia, the Philippines, Vietnam and Thailand.
> - The Malay reviews are mostly Malaysian.
> - Apple's feed is the Malaysia store only.
>
> So results are reported **per source**, and the English Play reviews should be described as "Grab users in Southeast Asia", not "Malaysian users".

**A1. Recent written reviews are far less positive than the 4.7–4.8★ shown on the store pages** (app_fig01, app_tab03).
- 91% of all Google Play star ratings are 5★.
- Among recent written reviews:
  - App Store Malaysia: **55% are 1–2★** and 39% are 5★.
  - English Play reviews: 43% are 1–2★.
  - Malay Play reviews: 13% are 1–2★.
- **Why the gap:**
  - The headline score includes millions of quick star taps, often prompted inside the app right after a good trip.
  - People who write a review are more often people with a problem.
- So the store score shows broad satisfaction, while written reviews show the pain points.

**A2. App Store (Malaysia) ratings have fallen over the last three months** (app_fig04).
- The monthly average rating dropped from **3.3★ in September to 2.5★ in October and 2.1★ in November 2026**.
- Malay Play reviews stayed steady at about 4.3–4.6★ from May to November.
- The App Store sample is small (86–177 reviews per month). Treat this as a warning sign, not proof of a trend.

**A3. Customer service and cancellations bring the lowest ratings, the same as on social media** (app_fig02, app_tab04).

| Topic in the review | % of reviews | % giving 1–2★ | Avg ★ |
|---|---|---|---|
| Customer service & refunds | 7% | **91%** | 1.3 |
| Cancellations & waiting time | 10% | **88%** | 1.4 |
| Price, fares & fees | 9% | 72% | 2.0 |
| App & technology | 22% | 53% | 2.8 |
| Food & delivery | 22% | 51% | 2.9 |
| Drivers & rides | 18% | 51% | 2.9 |
| No topic word (e.g. "good app", "bagus") | 50% | 12% | 4.5 |

Half of all reviews are short praise without a topic word, and these carry the high ratings. Reviews that describe a specific experience are mostly negative.

**A4. Grab replies to English reviews but almost never to Malay ones** (app_tab06, Google Play only; Apple's feed has no replies).
- Grab replied to **44% of English** Play reviews, more often to 1–2★ reviews (55–58%) than to 5★ reviews (34%). The median reply came 18–36 hours after the review.
- Grab replied to **0.4% of Malay** reviews (4 of 1,000), even though 132 of them gave 1–2★.
- Malay-speaking customers who complain on Google Play get almost no public response.

---

## C. Support and interactive features (chatbot, live chat, agents, help centre)

We searched reviews for the group's keyword list: chatbot, live chat, support, help centre, agent, AI assistant, recommend, ticket (app_fig03, app_tab05).

**S1. Reviews that mention support channels are overwhelmingly negative.**
- 76 reviews (3%) mention support, a chatbot, an agent, the help centre or a ticket. **84% of them give 1–2★**, with an average of **1.6★**.
- Reviews that mention none of these average 3.7★.
- **Every review mentioning the chatbot (15) or an agent (15) gave 1–2★.**
- A typical complaint: *"chat bot is not helping, cannot even connect me to a live agent."*
- This matches the April 2026 complaints on social media about Grab moving customer service to chatbots/AI (Part 2, F5).

**S2. "Recommend" is not about a feature.** 24 reviews contain only "recommend" from the list (avg 3.5★). They mostly say "I recommend this app", so they are shown separately and are not counted as support mentions.

**S3. Malay reviewers almost never mention support channels** (0.2% of Malay Play reviews, against about 6–7% elsewhere). Malay reviews on Play tend to be short ratings of the service itself.

---

## Suggested figure captions

| File | Caption |
|---|---|
| figures/web_fig01_performance_mobile_vs_desktop.png | Figure W1. Lighthouse performance score of Grab Malaysia websites, mobile vs desktop (median of 3 runs, 4 Nov 2026). |
| figures/web_fig02_lighthouse_scores.png | Figure W2. Lighthouse scores for performance, accessibility, best practices and SEO, by site and device. |
| figures/app_fig01_rating_distribution.png | Figure A1. Star ratings of recent written reviews vs all-time Google Play ratings. |
| figures/app_fig02_low_ratings_by_topic.png | Figure A2. Share of 1–2★ reviews by topic (Google Play + App Store, n = 2,500). |
| figures/app_fig03_support_mentions.png | Figure A3. Star ratings of reviews mentioning support, chatbots or agents. |
| figures/app_fig04_weekly_rating.png | Figure A4. Weekly average rating of new reviews by source. |

## Tables

| File | What it shows |
|---|---|
| `pagespeed_runs.csv`, `pagespeed_median.csv`, `onpage_seo.csv` | Raw PageSpeed runs, medians, and the SEO scrape (from notebook 01) |
| `web_tab01_performance_summary.csv` | Scores, lab speed and real-user ratings per site and device |
| `web_tab02_onpage_seo_summary.csv` | SEO checks with a plain-English list of issues per site |
| `app_tab01_store_overview.csv` | Public store score vs our sample, per source |
| `app_tab02_cleaning_funnel.csv` | Reviews kept at each cleaning step |
| `app_tab03_rating_distribution.csv` | % of 1–5★ per source, plus the all-time Play histogram |
| `app_tab04_topics.csv` | Topics: share of reviews, avg ★, % 1–2★, avg ★ per source |
| `app_tab05_support_features.csv` | Support / interactive-feature keywords and their ratings |
| `app_tab06_developer_replies_google_play.csv` | Grab's reply rate and reply time by star rating |
| `app_tab07_by_language.csv` | Ratings by detected language of the review |
| `app_tab08_weekly_rating.csv` | Weekly review count and average rating |
| `app_tab09_example_reviews.csv` | Anonymised example reviews for quoting (picked automatically; check before use) |

## Method and ethics notes

- **Collection:**
  - Google Play via `google-play-scraper` 1.2.7: newest reviews, `lang` = en and ms, `country` = my, plus the public app details page.
  - App Store via Apple's public customer-review RSS feed (Malaysia store, 10 pages × 50) and the iTunes lookup API.
  - In total: 3 requests to Google (2 for reviews, 1 for the details page) and 11 to Apple, spaced 3–5 seconds apart.
- **robots.txt exception:**
  - Both review endpoints are disallowed for crawlers: Google Play `/_/` and Apple `/*/rss/*`.
  - The group decided on 18 Nov 2026 to make a one-off exception for this small, non-commercial academic sample. Neither store offers an official review API to outside researchers.
  - The website SEO scrape fully respected robots.txt.
- **Privacy:**
  - Reviewer names were replaced by a SHA-256 hash.
  - Personal @mentions, links, emails and phone numbers in the text were replaced by placeholders.
  - Raw files with names stay in `data/raw/app_reviews/`, which is git-ignored and not shared.
- **Star sentiment:** 1–2★ = negative, 3★ = neutral, 4–5★ = positive (the `star_sentiment` column).
- **Topics:** the same English + Malay keyword dictionary as Part 2 (`src/findings.py`, `TOPICS`).

## Limitations

1. **Newest reviews only.**
   - The samples are the most recent reviews, not a random sample of all reviews.
   - Each source covers a different period: 2.5 weeks for English Play, 3 months for the App Store and 6 months for Malay Play.
2. **English Play reviews are not Malaysia-only** (see the note in Section B).
3. **Written reviews are skewed towards problems**, so they should not be read as the average customer's view. The store score tells that story.
4. **Keyword topics and support keywords** can miss or mis-tag reviews. For example, "support" can mean "I support Grab".
5. **The web tests are a snapshot** from one day, and lab speed varies between runs. GrabMart's HTML could not be audited (HTTP 403).
