# Grab Malaysia – Findings for the report

All charts are in `outputs/figures/` (PNG, 300 dpi, ready to insert into Word).
All tables are in `outputs/tables/` (CSV files that open in Excel).
Numbers come from `notebooks/05_visualisation_and_findings.ipynb` and `notebooks/06_model_comparison.ipynb`.
Part 1 (websites + app reviews) is in the folder `website and app analysis/`.

---

## 1. Data at a glance (Table: `tab01_dataset_overview.csv`)

- **2,015 posts and comments** about Grab Malaysia, posted between **7 Nov 2025 and 17 Nov 2026**.
- They come from five platforms: X/Twitter 1,033, Reddit 676, Threads 224, TikTok 51 and Facebook 31.
- **Language:** English 1,348 (67%), Malay 530 (26%) and mixed Malay-English ("Manglish") 137 (7%).
- **Labels:** Claude (an AI assistant) labelled each post negative, neutral or positive using the labelling guide (`data/LABELLING_GUIDE.md`). A team member reviewed the uncertain posts; only 12 labels were corrected.
- **Privacy:** usernames were replaced by `@user` before analysis. No personal data is reported.

## 2. Key findings

**F1. Public sentiment towards Grab Malaysia is mainly negative or neutral. Praise is rare.**
- Negative 44.6%, neutral 44.6%, positive 10.8% (Fig. 1). That gives a net sentiment of −34 points (% positive − % negative).
- When people bother to post an opinion about Grab, it is about four times more likely to be a complaint than praise.

**F2. People use Grab's social accounts as a complaints desk.**
- Posts that talk **to** Grab are **70% negative**. These are posts that tag @GrabMY, reply to it, or comment under Grab's own posts.
- Posts that only talk **about** Grab are 35% negative (Table 11).
- This explains why Facebook (87% negative), Threads (73%) and TikTok (59%) look much harsher than Reddit (36%) and X (42%) in Fig. 2.
- The difference comes from **where** the posts were collected, not from the platform's users being angrier.

**F3. Service failures drive the negativity, not the product idea.** Topics sorted by how negative they are (Fig. 5, Table 5):

| Topic | % of posts | % negative | % positive |
|---|---|---|---|
| Cancellations & waiting time | 15% | **62%** | 9% |
| Customer service & refunds | 15% | **58%** | 10% |
| Price, fares & fees | 26% | **52%** | 11% |
| App & technology (incl. AI) | 17% | 48% | 12% |
| Food & delivery | 28% | 46% | 9% |
| Drivers & rides | 45% | 42% | 13% |
| Promos, vouchers & rewards | 5% | 36% | **21%** |
| Competitors & alternatives | 9% | 36% | 16% |
| Safety & trust | 6% | 43% | 18% |

- Drivers/rides is the topic people mention most (45% of posts). Price, waiting time and customer service are the most negative.
- **"cancel"** and **"mahal"** (Malay for "expensive") are the words most typical of negative posts (Fig. 7).
- The most positive topics are promotions and vouchers, and comparisons with taxis or competitors. In those comparisons, Grab's fixed, in-app prices are seen as safer and fairer than street taxis.
- Visitors from abroad (e.g. India) often praise Grab as easy and reliable compared with options at home.

**F4. Malay and Manglish posts are more negative than English posts.**
- Malay posts are 53% negative and mixed posts 56%, against 40% for English posts (Fig. 3).
- Complaints about price ("mahal", "harga"), being let down ("kena", "cancel") and waiting ("tunggu", "lambat") are mostly written in Malay.
- An English-only analysis would have **under-stated** the negativity. That is the reason Malay and Manglish posts were kept.

**F5. Sentiment is fairly stable over the year, with one real negative spike.**
- Most months sit at 30–48% negative (Fig. 4).
- **April 2026 (62% negative)** matches complaints that Grab was moving customer service to chatbots/AI, plus long GrabFood delays.
- The **November 2026** figure (69%) **should not be read as a trend**. Most posts that month came from a separate collection round of replies to @GrabMY on Threads (see F2).

**F6. Engagement does not depend on sentiment.**
- On X, the median post gets about 5 likes whether it is negative, neutral or positive (Table 10).
- Negative posts are not rewarded with more likes. Their visibility comes from their volume.

**F7. Machine learning beats a ready-made sentiment dictionary, but neither is reliable enough on its own** (Fig. 8, Table 9). All methods were tested on the same 403 test posts, which were held out from training:

| Method | Accuracy | Macro-F1 |
|---|---|---|
| VADER (dictionary, English) | 0.39 | 0.37 |
| VADER + our Manglish word list + normalisation | 0.44 | 0.41 |
| **TF-IDF + Logistic Regression (trained on 80%)** | **0.62** | **0.51** |
| Baseline: always guess "negative" | 0.45 | 0.21 |

- **VADER does worse than simply guessing "negative" every time** (0.39 accuracy against 0.45), because it labels many neutral posts as positive.
- Adding our Malay/Manglish word list improves VADER, mainly on Malay posts: macro-F1 rises from 0.37 to 0.47 on those.
- Logistic Regression is the best method. Its weak point is positive posts: it finds only 8 of the 43 positive test posts.
- Logistic Regression scores 0.95 on posts it was trained on but 0.51 on new posts, so it partly memorises its training data. 0.51 is the honest figure to report.
- **The methods predict very different sentiment mixes** (Fig. 11, Table 14, all 2,015 posts):

| Method | Negative | Neutral | Positive |
|---|---|---|---|
| Labels (reference) | 44.6% | 44.6% | 10.8% |
| (a) VADER | 32.3% | 28.7% | 39.0% |
| (c) VADER + lexicon + normalisation | 39.2% | 21.9% | 38.9% |
| (d) TF-IDF + LogReg (5-fold out-of-sample) | 46.4% | 43.5% | 10.1% |

- Both VADER versions call almost **4 times as many posts positive** as the labels do.
- VADER scores friendly single words ("legit", "help", "please", "thank you") and emojis, and misses complaints that are written politely or sarcastically.
- Plain VADER labels 252 negative posts as positive. Five examples are in Table 15.
- Logistic Regression's out-of-sample predictions (made for every post by 5-fold cross-validation) reproduce the label mix closely.

**F8–F9. Websites and app reviews (Part 1)** are written up separately in
**`website and app analysis/README.md`**, with their own charts and tables in that folder.

## 3. Putting web, app and social together (for the discussion section)
Details of the web and app numbers are in `website and app analysis/README.md`.

- **Speed is a problem in both data sources.** Social complaints focus on waiting (cancellations, late food, slow customer service), and the GrabFood website is the slowest of Grab's sites on mobile.
- **App reviews tell the same story as social media.** In app-store reviews, too, customer service (91% give 1–2★) and cancellations/waiting time (88%) are the worst topics. Three independent sources (social posts, app reviews and web tests) point to speed and support as Grab's weak spots.
- Most orders are placed in the app, not on the website, so the slow site does not directly cause the app complaints. It does, however, weaken the **first impression** for new customers who arrive from Google.
- **Customer service is moving onto social media.** One in four posts (27%) is addressed to Grab directly, and 70% of those are complaints.
- Faster, human responses on social channels, together with clearer pricing and cancellation rules, would address the three most negative topics.

## 4. Suggested figure captions

| File | Caption |
|---|---|
| fig01_sentiment_overall.png | Figure 1. Overall sentiment towards Grab Malaysia (n = 2,015 posts, Nov 2025 – Nov 2026). |
| fig02_sentiment_by_platform.png | Figure 2. Sentiment by platform. Facebook and TikTok are comments under Grab's own posts. |
| fig03_sentiment_by_language.png | Figure 3. Sentiment by language of the post (English, Malay, mixed). |
| fig04_sentiment_by_month.png | Figure 4. Monthly post volume and share of negative and positive posts. |
| fig05_sentiment_by_topic.png | Figure 5. Sentiment by topic, from most positive to most negative. |
| fig06_topic_volume.png | Figure 6. Share of posts mentioning each topic. |
| fig07_distinctive_words.png | Figure 7. Words most typical of negative, neutral and positive posts (weighted log-odds). |
| fig08_model_comparison.png | Figure 8. Accuracy and macro-F1 of four sentiment methods on the same held-out test set. |
| fig11_distribution_by_method.png | Figure 11. Share of negative, neutral and positive posts predicted by each method vs the labels (all 2,015 posts). |

## 5. Method notes (for the methodology section)

- **Topics:**
  - A keyword dictionary in English and Malay (11 topics; lists in `src/findings.py`). One post can have several topics.
  - Cross-checked with an unsupervised NMF topic model (Table 7), which found the same main themes.
- **Distinctive words:** weighted log-odds with an informative Dirichlet prior (Monroe, Colaresi & Quinn, 2008).
- **Model comparison:**
  - Stratified 80/20 train/test split (random_state 42). Logistic Regression was trained on the 80% only.
  - VADER needs no training and was scored on the same 20% of posts.
  - Macro-F1 is the main metric because positive posts are rare.
- **Web analytics and app reviews:** see `website and app analysis/README.md`.

## 6. Limitations (be honest about these in the report)

1. **Collection method shapes the results.** Posts addressed to @GrabMY were collected on purpose in later rounds, so they make Threads, Facebook, TikTok and November 2026 look more negative (see F2).
2. **Small groups.** Facebook (31) and TikTok (51) are too small for firm conclusions about those platforms.
3. **Keyword topics** can miss posts that use other words, and can occasionally tag a post wrongly. Example quotes in Table 5 were picked automatically; check them before quoting.
4. **Labels** were made by Claude (an AI assistant) following the labelling guide, and a team member reviewed only the uncertain posts (12 corrected). The other labels were not double-checked by a person, and some posts (sarcasm, jokes) are genuinely ambiguous.
5. **Reach of the data.** Social posts are not a random sample of Grab customers. Unhappy customers post more often than happy ones.

### Model limitations

6. **Over-fitting.**
   - Logistic Regression reaches macro-F1 **0.954 on its training posts**, but only **0.567** in 5-fold cross-validation and **0.509 on the unseen test posts**.
   - It partly memorises its 1,612 training posts, so report the test score.
7. **Spurious words from a small training set.**
   - Some of the words the model links most strongly to *positive* have no real sentiment: **"semalam"** (yesterday), **"kl"**, **"one"**, **"our"** (`result/11_logreg_top_words.csv`).
   - With only 218 positive posts, a word that happens to appear in a few positive posts gets a large weight.
   - Treat the model's word list as a description of this dataset, not as general "positive words".
8. **"grabmy" is the strongest negative word.**
   - That reflects **where** posts were collected, not the word itself.
   - Posts that talk *to* Grab (tagging or replying to @GrabMY) are **70.3% negative**, against **35.0%** for posts *about* Grab (F2, Table 11).
   - The model has learned "addressed to Grab = complaint", which may not hold in other data.
9. **The positive class is weak for every method.**
   - F1 for *positive* is only **0.195** for Logistic Regression and **0.277** for VADER + lexicon + normalisation (test set).
   - Neither method can reliably find the rare positive posts (10.8% of the data). LogReg misses most of them, while VADER finds more but also wrongly calls many neutral and negative posts positive.
