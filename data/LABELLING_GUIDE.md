# Labelling guide: Grab Malaysia sentiment

Each post gets **one** label. Read the whole post. For a Reddit comment, read the thread title
too, because "same, happened to me" only makes sense next to it.

| Label | Use it when the post… | Examples |
|---|---|---|
| **positive** | expresses a favourable feeling about Grab Malaysia: its service, app, drivers/riders, prices, promotions, support, or the company | "Grab driver returned my lost wallet, so kind 🥰"; "GrabFood is a lifesaver at 2am" |
| **negative** | expresses an unfavourable feeling: complaints, frustration, anger, distrust, criticism of prices/fees/drivers/support/app | "Charged RM40 for 5km, ridiculous"; "@GrabMY 3 days and still no refund" |
| **neutral** | is about Grab Malaysia but expresses no clear feeling: questions, factual statements, news, neutral advice | "Is GrabPay accepted at KLIA?"; "Grab rides to JB can now be booked 6 hours ahead" |
| **irrelevant** | is **not about Grab Malaysia** at all | "grab" used as a verb; another country's Grab; spam; a comment in a Grab thread that talks about something unrelated |

## Rules for difficult cases
1. **Mixed posts → the main feeling.** Pick whichever feeling dominates or ends the post.
   "Driver was nice but the fare was insane and I'm deleting the app" → **negative**.
2. **Sarcasm → the intended meaning.** "Wow great job Grab, 40 minutes late again 👏" → **negative**.
3. **Malay / Manglish → label normally.** "Grab mahal gila now lah" → **negative**.
4. **News → neutral, unless the poster adds an opinion.** A shared headline is **neutral**.
   A headline plus "about time!" or "this is why I stopped using Grab" takes the poster's feeling.
5. **Irrelevant = not about Grab Malaysia.** Use it when the post isn't about Grab, or is clearly about Grab
   in another country. Comments in a Grab thread count as relevant if they discuss the thread's
   topic (e-hailing, the delivery, the driver, alternatives to Grab), even without the word "Grab".
6. **Feelings about the topic, not the poster's day.** Judge the feeling towards Grab or the Grab-related
   situation, not unrelated parts of the post.
7. **Comparisons.** "foodpanda is way better than Grab" is **negative** for Grab. "Just use Grab, much easier
   than a taxi" is **positive**.
8. **Advice / recommendations.** "Take Grab from KLIA, it's easy" → **positive**. "Avoid Grab at peak hours, surge is crazy"
   → **negative**. "You can take Grab or the train" → **neutral**.
9. **Questions** are **neutral**, unless they're rhetorical complaints ("why is Grab so expensive now??" → **negative**).

## Manglish / Malay posts (added 2026-11-18)
Posts may be in Malaysian English, Manglish (mixed) or Malay. Read slang for its meaning: "teruk", "lembap",
"hampeh", "kena tipu", "mahal gila" are complaints; "mantap", "padu", "terbaik", "laju" are praise. Short forms:
x / tak / takde = not, sgt = very, dah = already. Particles (lah, kan, sia, weh) carry no feeling on their own.
"Abang grab" means the driver/rider. Sarcasm is common ("terbaik la grab, sejam tunggu" → **negative**).
Indonesian posts are **irrelevant** when they are clearly about Grab in Indonesia (rupiah / "rb", Gojek, Jakarta,
@GrabID, Indonesian slang such as "gak", "banget", "aja", "gw"); Malay posts about Malaysia are labelled normally.

## Unsure?
If a post could reasonably take two labels, choose the best one, mark it **unsure**, and write a short
reason (e.g. "sarcasm? could be positive"). Unsure posts are listed first in the review file so
the second checker looks at them first.

## Second checker
Open `labels_for_review.csv`, read each post (click `url` to see the original post with its images, replies and thread), and type a label in **`corrected_label`** only where you
**disagree**. Leave it empty if you agree. Use exactly: `positive`, `negative`, `neutral`, `irrelevant`.
Save the file as **`labels_reviewed.csv`** in the same folder.
