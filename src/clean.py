"""
Text-cleaning helpers for notebook 02_data_cleaning.

Every filter adds a plain-English `drop_reason` instead of silently deleting rows, so the
notebook can show a "cleaning funnel" and the team can audit what was removed and why.

Main entry point: `clean_posts(raw_df, cfg)` returns (kept_df, dropped_df, funnel_df).
"""
from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache

import emoji
import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect_langs
from nltk.corpus import stopwords, wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

from src.collect import hash_author

DetectorFactory.seed = 0  # langdetect is random by default; fix it so results are repeatable

# ---------------------------------------------------------------------------
# Word lists used by the language and relevance rules
# ---------------------------------------------------------------------------
EN_FUNCTION_WORDS = set(stopwords.words("english")) | {
    "im", "dont", "cant", "wont", "didnt", "doesnt", "isnt", "ive", "youre", "thats", "u", "ur",
}
# Malay + Indonesian markers. langdetect has no Malay model and labels Malay as "id", and
# Manglish mixes both, so we count these words directly. Words that are also common
# English words (e.g. "di" in names, "lah" is kept: it signals Malay/Manglish) are left out.
MS_ID_WORDS = set(stopwords.words("indonesian")) | {
    "tak", "nak", "dah", "lah", "la", "kat", "je", "ni", "tu", "mcm", "macam", "sbb", "sebab",
    "org", "orang", "mmg", "memang", "kena", "buat", "pun", "pon", "jgk", "jugak", "juga",
    "dgn", "dengan", "utk", "untuk", "yg", "yang", "tapi", "tp", "kalau", "klau", "kalo", "bila",
    "aku", "saya", "kau", "korang", "kita", "kami", "dia", "diorang", "depa", "awak", "hang",
    "ada", "takde", "tiada", "tidak", "boleh", "blh", "sudah", "sangat", "sgt", "lagi", "lg",
    "baru", "sikit", "banyak", "byk", "semua", "mana", "sini", "situ", "apa", "camne",
    "macamana", "kenapa", "pastu", "lepas", "dekat", "dkt", "naik", "jalan", "duit", "mahal",
    "murah", "lambat", "cepat", "betul", "ke", "ini", "itu", "dan", "atau", "nya", "gila",
    "gak", "nggak", "ga", "banget", "udah", "aja", "gw", "gue", "lu", "temen", "gimana", "wkwk",
    "wkwkwk", "bgt", "sih", "deh", "dong", "kok", "nih", "tuh", "abang", "abg", "kak", "akak",
    "bang", "tolong", "terima", "kasih", "pakai", "guna", "beli", "harga", "bayar", "sampai",
    "rumah", "kerja", "makan", "nasi", "hari", "esok", "semalam", "tadi", "nanti", "pergi",
}
# Tagalog / Filipino markers ("mahal" is also Tagalog, which is why it was a noisy keyword)
TL_WORDS = {
    "ang", "ng", "mga", "sa", "naman", "po", "ko", "yung", "lang", "talaga", "kasi", "hindi",
    "ako", "ikaw", "siya", "natin", "namin", "kayo", "sila", "pa", "na", "ba", "din", "rin",
    "nga", "kaya", "pero", "wala", "meron", "sana", "grabe", "ito", "iyan", "yan", "dito",
}

# Mentions of Grab the company / its services (not "grab a coffee").
BRAND_PATTERNS = [
    r"\bgrab ?(?:car|pay|mart|express|bike|taxi|unlimited|rewards|my|malaysia|share|hitch|"
    r"dine ?out|driver|drivers|rider|riders|delivery|deliveries|app|ride|rides|order|orders|"
    r"account|support|reward|card|cc|scam|scams|fare|fares|fee|fees|price|prices|pricing|promo|promos|voucher|vouchers|"
    r"biz|kiosk|premium|family|ads|financial|wallet|points|cs|customer|surge|car's)\b",
    r"[@#]grab\w*",
    # "GrabFood" is always the service; lower-case "grab food" only with service context
    # (otherwise "grab food yourself" / "grab food and drinks" would count)
    r"\bgrabfood\b",
    r"\b(?:order|ordered|ordering|via|on|use|used|using|from|through|by|the|my|our|a|delete|"
    r"deleted|open|opened|kat|naik|vs) grab ?food\b",
    r"\bgrab ?food (?:rider|riders|delivery|deliveries|driver|drivers|order|orders|app|promo|"
    r"promos|voucher|vouchers|guy|guys|man|uncle|fee|fees|price|prices|unlimited|account)\b",
    r"\b(?:take|took|taking|takes|book|booked|booking|order|ordered|ordering|call|called|use|"
    r"used|using|via|on|through|from|than|vs|naik|delete|deleted|uninstall|uninstalled|"
    r"open|opened|like|unlike|both|either|neither) grab\b"
    r"(?! (?:a|an|the|some|your|yours|my|his|her|their|our|it|them|this|that|me|him|us)\b)",
    r"(?<!of )(?<!what )(?<!nice )(?<!great )(?<!good )"
    r"\b(?:the|my|a|our|your|their|his|her|no|any|every|cheap|cheaper|expensive) grabs?\b"
    r"(?! (?:a|an|the|some|your|my|his|her|their|our|it|them|this|that|me|him|us|bag|bags|"
    r"bar|bars)\b)",
    # Malay / Manglish (added 2026-11-18): "naik grab", "abang grab", "guna grab", "grab lambat"
    r"\b(?:guna|gune|pakai|tempah|tunggu|panggil|abang|abg|bang|pakcik|makcik|naik|kat|dkt|dekat|"
    r"dgn|dengan|ambil|amik|order|oder|driver|rider|delete|bukak|buka) grab\b",
    r"\bgrab (?:lambat|lmbt|mahal|lembap|teruk|cancel|cancelled|tak|tk|x|xde|takde|tipu|penipu|"
    r"sampai|smpai|dah|dh|harini|hari ni|skrg|sekarang|kat|dkt|ni|tu|je|pun|pon|mmg|memang|kena)\b",
    r"\bgrab (?:and|or|/) (?:gojek|foodpanda|panda|shopee|indrive|maxim|airasia|lalamove|"
    r"bolt|uber|taxi|e-?hailing)\b",
    r"\b(?:gojek|foodpanda|panda|shopee ?food|indrive|maxim|airasia ride|lalamove|bolt|uber|"
    r"taxi|e-?hailing)(?: and| or| vs|,|/) ?grab\b",
]
BRAND_CASE_SENSITIVE = [
    # Capital-G "Grab" right after a lower-case word ("daily Grab surges", "for Grab"), which
    # excludes Title Case headlines ("Political Money Grab") and promos ("... ⚡ Grab both now!")
    r"(?<![A-Za-z'])[a-z][a-z']*,? Grab\b(?! (?:a|an|the|some|your|my|it|them|this|that|both|one|yours)\b)",
    r"\bGrab Food\b",
    r"\bGrab (?:is|was|has|have|had|should|always|never|will|would|can|cannot|can't|won't|"
    r"charges?|charged|raised|increased?|keeps?|just|really|still|now|also|made|makes|said|"
    r"says|needs?|sucks?|lah|pls|please|support|team|Malaysia)\b",
]
BRAND_RE = re.compile("|".join(BRAND_PATTERNS), re.IGNORECASE)
BRAND_CS_RE = re.compile("|".join(BRAND_CASE_SENSITIVE), re.MULTILINE)

MY_CONTEXT_RE = re.compile(
    r"\b(?:malaysia|malaysian|malaysians|m'sia|msia|kl|kuala lumpur|selangor|penang|johor|jb|"
    r"johor bahru|klia|klia2|putrajaya|cyberjaya|petaling|pj|subang|shah alam|klang|ipoh|"
    r"melaka|malacca|sabah|sarawak|kuching|kota kinabalu|seremban|kedah|kelantan|terengganu|"
    r"pahang|perak|bangsar|bukit bintang|mont kiara|puchong|cheras|ringgit|jpj|apad|mydin|"
    r"grabmy|bolehland|rm ?\d[\d,.]*k?)\b|\brm\d",
    re.IGNORECASE,
)
FOREIGN_RE = re.compile(
    r"\$grab\b|\bnasdaq\b|\b(?:stock|stocks|shares|earnings|investors?|eps|revenue|"
    r"q[1-4] ?20\d\d)\b|\b(?:philippines|filipino|pinoy|manila|cebu|quezon|makati|pasig|"
    r"taguig|indonesia|indonesian|jakarta|bali|bandung|surabaya|rupiah|singapore|singaporean|"
    r"thailand|thai|bangkok|vietnam|vietnamese|hanoi|saigon|ho chi minh|cambodia|phnom penh|"
    r"myanmar|yangon)\b|₱|\bphp ?\d|\brp ?\d|฿|\bsgd\b|\bs\$",
    re.IGNORECASE,
)
NON_LATIN_SCRIPT_RE = re.compile(r"[฀-๿぀-ヿ一-鿿가-힯]")

# Business adverts and fan-club campaigns mention GrabFood but aren't customer opinions
PROMO_RE = re.compile(
    r"#grab\w*x\w+|#\w+xgrab\w*|\bgrab ?food (?:x|with) (?-i:[A-Z]{4,})|"
    r"\b(?:order now|now available|available (?:now )?(?:on|at|via) (?:grab|foodpanda|shopee)|"
    r"order (?:on|via|through|from) (?:grab|foodpanda|shopee)\w* (?:or|and|now|today)|"
    r"while stocks? last|limited time only|link in (?:bio|our bio)|dm (?:us )?to order|"
    r"we deliver|open daily|opening hours|visit us at|follow us|order here|"
    r"dine-?in,? (?:pick-?up|take-?out|takeaway)|(?:pick-?up|take-?out),? or delivery)\b",
    re.IGNORECASE,
)

BOT_TEXT_RE = re.compile(
    r"i am a bot|this action was performed automatically|^\[(?:deleted|removed)\]$|"
    r"contact the moderators of this subreddit|your (?:post|submission) has been removed",
    re.IGNORECASE,
)
# Replies that carry no opinion on their own, even though they contain words
FILLER_PHRASES = {
    "thanks", "thank you", "thank u", "thx", "ty", "tq", "tqvm", "lol", "lmao", "lmfao", "haha",
    "same", "this", "true", "yes", "no", "ok", "okay", "agreed", "agree", "noted", "nice",
    "wow", "damn", "bro", "sis", "wtf", "omg", "yep", "yup", "nope", "indeed", "exactly",
    "second this", "this one", "me too", "same here", "so true", "well said", "good one",
    "thanks bro", "thank you so much", "thanks for sharing", "following", "bump", "up",
}

# ---------------------------------------------------------------------------
# Anonymisation: usernames, URLs, emails, phone numbers
# ---------------------------------------------------------------------------
URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|my|net|org|io|ly|gg|co|me|app|link)"
    r"(?:/\S*)?\b",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Any phone-like number: optional +, starts with a digit, 10-16 digits/spaces/dashes in total
# (catches Malaysian 01x-xxx xxxx, +60..., and foreign numbers such as 0923xxxxxxx)
PHONE_RE = re.compile(r"(?<![\w.])\+?\d[\d -]{8,14}\d(?![\w.])")
MENTION_RE = re.compile(r"(?<![\w@])@([A-Za-z0-9_]{1,30})")
REDDIT_USER_RE = re.compile(r"(?<![\w/])/?u/[A-Za-z0-9_-]{3,20}\b")


def anonymise(text: str) -> str:
    """Replace URLs, emails, phone numbers and personal @usernames with placeholder tokens.

    Brand accounts (handles starting with "grab", e.g. @GrabMY) are kept: they are a
    company, not a private person, and show who the post is talking to.
    """
    t = EMAIL_RE.sub("EMAIL", text)
    t = URL_RE.sub("URL", t)
    t = PHONE_RE.sub("PHONE", t)
    t = REDDIT_USER_RE.sub("u/user", t)
    t = MENTION_RE.sub(lambda m: m.group(0) if m.group(1).lower().startswith("grab") else "@user", t)
    return t


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "‛": "'", "“": '"', "”": '"', "„": '"',
                           "–": "-", "—": "-", "‐": "-", "…": "...", " ": " "})


def _emoji_to_words(text: str, style: str) -> str:
    """😂 -> ' [face with tears of joy] ' (readable) or ' emoji_face_with_tears_of_joy ' (token)."""
    def repl(chars, data):
        name = data["en"].strip(":").lower()
        if style == "words":
            return f" [{name.replace('_', ' ')}] "
        return " emoji_" + re.sub(r"[^a-z0-9]+", "_", name).strip("_") + " "
    return emoji.replace_emoji(text, replace=repl)


def reduce_repeats(text: str) -> str:
    """'sooooo goooood!!!!!!' -> 'soo good!!!', 'hahahaha' -> 'haha', '.....' -> '...'."""
    t = re.sub(r"\b(?:[hw]a){2,}h?\b|\b(?:he){3,}\b|\b(?:hi){3,}\b", "haha", text, flags=re.IGNORECASE)
    t = re.sub(r"([A-Za-z])\1{2,}", r"\1\1", t)            # 3+ same letters -> 2
    t = re.sub(r"([!?])\1{3,}", r"\1\1\1", t)               # max 3 ! or ? (VADER uses these)
    t = re.sub(r"\.{4,}", "...", t)
    t = re.sub(r"([^\w\s!?.'\"])\1{2,}", r"\1", t)          # ~~~~ ---- **** -> single
    return t


def clean_text(text: str) -> str:
    """Readable cleaned text for VADER and for people: casing, ! and ? kept, emojis as words."""
    t = html.unescape(str(text))
    t = unicodedata.normalize("NFKC", t).translate(QUOTE_MAP)
    t = anonymise(t)
    t = _emoji_to_words(t, "words")
    t = reduce_repeats(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_lemmatizer = WordNetLemmatizer()


@lru_cache(maxsize=50_000)
def _lemma(word: str) -> str:
    # Short words and stopwords are left alone: WordNet would turn "has" into "ha", "was" into "wa"
    if len(word) <= 3 or word in EN_FUNCTION_WORDS or word.startswith("emoji_"):
        return word
    return _lemmatizer.lemmatize(word)


def normalise_text(text: str) -> str:
    """Fully normalised text for the ML model: lower-case, no punctuation, emoji tokens, lemmas."""
    t = html.unescape(str(text))
    t = unicodedata.normalize("NFKC", t).translate(QUOTE_MAP)
    t = anonymise(t)
    t = _emoji_to_words(t, "token")
    t = reduce_repeats(t).lower()
    t = t.replace("&", " and ").replace("@user", " user ").replace("u/user", " user ")
    t = re.sub(r"'", "", t)                     # don't -> dont
    t = re.sub(r"[^\w\s]|_(?!\w)", " ", t)      # strip punctuation, keep emoji_tokens intact
    tokens = [_lemma(w) for w in t.split()]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Language, relevance and information checks
# ---------------------------------------------------------------------------
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _words(text: str) -> list[str]:
    return [w.replace("'", "") for w in WORD_RE.findall(text.lower())]


@lru_cache(maxsize=100_000)
def _is_english_word(w: str) -> bool:
    """Dictionary word (WordNet), English function word, or a Grab brand word."""
    return w in EN_FUNCTION_WORDS or w.startswith("grab") or bool(wordnet.synsets(w))


def detect_english(text: str) -> dict:
    """Decide whether a post is written in English.

    Uses langdetect PLUS counts of English vs Malay/Indonesian/Tagalog function words,
    because langdetect alone is unreliable on short or mixed (Manglish) text.
    A post counts as English if langdetect says English (or it has 3+ English function
    words, or - for short replies langdetect can't judge - at least 70% of its words are
    English dictionary words) AND non-English marker words make up at most 25% of the
    English ones.
    """
    probe = anonymise(text)
    probe = emoji.replace_emoji(probe, " ")
    probe = re.sub(r"URL|EMAIL|PHONE|@\w+|u/user|#", " ", probe)
    words = _words(probe)
    en = sum(w in EN_FUNCTION_WORDS for w in words)
    ms = sum(w in MS_ID_WORDS for w in words)
    tl = sum(w in TL_WORDS for w in words)
    try:
        langs = detect_langs(probe) if words else []
        top, prob = (langs[0].lang, langs[0].prob) if langs else ("unknown", 0.0)
    except LangDetectException:
        top, prob = "unknown", 0.0
    if NON_LATIN_SCRIPT_RE.search(text):
        is_en = False
    else:
        dict_share = sum(map(_is_english_word, words)) / len(words) if words else 0.0
        short_dict_en = len(words) <= 8 and dict_share >= 0.7
        is_en = (top == "en" or en >= 3 or short_dict_en) and (ms + tl) <= 0.25 * en
    return {"lang_detected": top, "lang_prob": round(prob, 3), "n_en_words": en,
            "n_ms_id_words": ms, "n_tl_words": tl, "is_english": is_en}


def lang_group(text: str, det: dict | None = None) -> str:
    """Rough language group for the Manglish pipeline (user decision 2026-11-18):
    'en' (mostly English), 'ms' (mostly Malay/Indonesian), 'mixed' (code-mixed Manglish),
    'other' (another script or language, e.g. Chinese, Thai, Tagalog, Spanish).

    Heuristic: share of known Malay/Indonesian marker words among Malay + English function
    words. Under 20% Malay -> en, over 80% -> ms, in between -> mixed. Posts with neither kind
    of word fall back to the English-dictionary share and langdetect.
    """
    det = det or detect_english(text)
    letters = [ch for ch in text if ch.isalpha()]
    if letters and sum(bool(NON_LATIN_SCRIPT_RE.match(ch)) for ch in letters) / len(letters) > 0.3:
        return "other"
    en, ms, tl = det["n_en_words"], det["n_ms_id_words"], det["n_tl_words"]
    if tl >= 2 and tl > ms and tl > en:
        return "other"
    if en + ms == 0:
        probe = re.sub(r"URL|EMAIL|PHONE|@\w+|u/user|#", " ", emoji.replace_emoji(anonymise(text), " "))
        words = _words(probe)
        if words and sum(map(_is_english_word, words)) / len(words) >= 0.6:
            return "en"
        if det["lang_detected"] in ("id", "ms"):
            return "ms"
        return "en" if det["lang_detected"] == "en" else "other"
    share = ms / (en + ms)
    return "en" if share <= 0.2 else "ms" if share >= 0.8 else "mixed"


NOT_BRAND_RE = re.compile(
    r"\b(?:power|money|cash|land|smash[- ]and|easy|quick|talent|data|attention|grab[- ]and)[- ]grabs?\b|"
    r"\bgrab[- ](?:bag|bags|bar|bars|handle|handles|and[- ]go)\b",
    re.IGNORECASE,
)


def mentions_grab(text: str) -> bool:
    """True if the text refers to Grab the company/service (not the verb 'grab')."""
    text = NOT_BRAND_RE.sub(" ", text)
    return bool(BRAND_RE.search(text) or BRAND_CS_RE.search(text))


@lru_cache(maxsize=1)
def _vader_words() -> frozenset:
    from src.manglish_lexicon import MANGLISH_LEXICON  # Malay opinion words count too (2026-11-18)
    lex = {**SentimentIntensityAnalyzer().lexicon, **MANGLISH_LEXICON}
    return frozenset(w for w, s in lex.items() if abs(s) >= 1.0)


PLACEHOLDER_RE = re.compile(r"URL|EMAIL|PHONE|@user|u/user|\[[a-z0-9 :'&.-]+\]")


def content_words(text_clean: str) -> list[str]:
    """Words in the cleaned text, ignoring placeholders (URL, @user) and emoji names."""
    return _words(PLACEHOLDER_RE.sub(" ", text_clean))


def information_check(text_clean: str, min_words: int) -> str | None:
    """Return a drop reason if the text has too little content, else None.

    Short posts are KEPT if they contain an opinion word (from the VADER sentiment lexicon)
    or an emoji, e.g. "Grab so expensive", "worst service 😡". Short posts with no
    opinion ("At subang jaya", "Try indrive") and pure filler ("thanks!", "same") are dropped.
    """
    words = content_words(text_clean)
    if not words:
        return "no words (link / emoji / image only)"
    phrase = " ".join(words)
    if phrase in FILLER_PHRASES:
        return "filler reply with no opinion"
    if len(words) < min_words:
        has_emoji = "[" in text_clean
        has_opinion = any(w in _vader_words() for w in words)
        if not (has_opinion or has_emoji):
            return f"under {min_words} words and no opinion word"
    return None


# ---------------------------------------------------------------------------
# The full cleaning pipeline
# ---------------------------------------------------------------------------
def clean_posts(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply every cleaning step in order. Each row is dropped by the FIRST rule it fails."""
    ccfg = cfg["cleaning"]
    df = raw.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["drop_reason"] = pd.NA
    funnel = [("Raw items collected", len(df))]

    def drop(mask: pd.Series, reason) -> None:
        mask = mask & df["drop_reason"].isna()
        if callable(reason):
            df.loc[mask, "drop_reason"] = df.loc[mask].apply(reason, axis=1)
        else:
            df.loc[mask, "drop_reason"] = reason
        funnel.append((reason if isinstance(reason, str) else "low-information content",
                       int((df["drop_reason"].isna()).sum())))

    # 1. Empty, deleted or removed
    drop(df["text"].str.strip().isin(["", "[deleted]", "[removed]", "nan"]), "empty / deleted / removed")

    # 2. Exact duplicates: same post collected by several search queries
    df["_query_all"] = df.groupby(["platform", "post_id"])["query"].transform(
        lambda s: "; ".join(dict.fromkeys(s.dropna().astype(str))))
    drop(df.duplicated(["platform", "post_id"], keep="first"), "duplicate (same post, several queries)")

    # 3. Official Grab accounts and bots: not public opinion
    official = {hash_author(u) for u in ccfg["official_accounts"]}
    bots = {hash_author(u) for u in ccfg["bot_accounts"]}
    drop(df["author_hash"].isin(official), "official Grab account")
    drop(df["author_hash"].isin(bots) | df["text"].str.contains(BOT_TEXT_RE), "bot / moderator message")

    # 4. English only
    lang = pd.DataFrame([detect_english(t) for t in df["text"]], index=df.index)
    df = df.join(lang)
    df["lang_group"] = [lang_group(t, d) for t, d in zip(df["text"], lang.to_dict("records"))]
    if set(ccfg.get("languages", ["en"])) == {"en"}:
        # Original English-only rule (rounds 1-2, user decision 2026-11-06)
        drop(~df["is_english"] & df["lang_detected"].eq("id") | ~df["is_english"] & (df["n_ms_id_words"] > 0),
             "not English (Malay / Indonesian)")
        drop(~df["is_english"], "not English (other language)")
    else:
        # Manglish mode (user decision 2026-11-18): keep English, Malay and code-mixed text
        drop(~df["lang_group"].isin(ccfg["languages"]), "not English / Malay / Manglish (other language or script)")

    # 5. Relevance: must be about Grab (the company), in a Malaysian setting
    df["mentions_grab"] = df["text"].map(mentions_grab)
    is_reddit = df["platform"].eq("reddit")
    reddit_posts = df[is_reddit & ~df["is_comment"]]
    relevant_threads = set(reddit_posts.loc[reddit_posts["mentions_grab"], "post_id"])
    # Reddit comments inherit relevance from their thread (a reply in a Grab thread is about
    # Grab even if it doesn't repeat the word). Everything else must mention Grab itself.
    # Posts the team added by hand (notebook 01, `manually_added`) were checked by a person for
    # being about Grab Malaysia, so they skip the two automatic relevance rules below.
    manual = df["manually_added"].fillna(False).astype(bool) if "manually_added" in df else pd.Series(False, index=df.index)
    # Facebook / TikTok rows are comments under Grab Malaysia's own posts and videos, so (like a
    # Reddit comment in a Grab thread) they are about Grab even without the word itself.
    is_fb_comment = df["platform"].isin(["facebook", "tiktok"]) & df["is_comment"].fillna(False).astype(bool)
    on_topic = (df["mentions_grab"] | manual | is_fb_comment
                | (is_reddit & df["is_comment"] & df["parent_post_id"].isin(relevant_threads)))
    drop(is_reddit & df["is_comment"] & ~on_topic, "Reddit comment in a thread not about Grab")
    drop(~on_topic, "does not mention Grab (or uses 'grab' as a verb)")

    has_my = df["text"].str.contains(MY_CONTEXT_RE)
    has_foreign = df["text"].str.contains(FOREIGN_RE)
    # Reddit data comes from Malaysian subreddits only, so the setting is already Malaysian
    drop(~is_reddit & has_foreign & ~has_my, "about another country or Grab's stock")
    if ccfg.get("threads_require_malaysia_context", True):
        drop(df["platform"].eq("threads") & ~has_my & ~manual, "Threads post with no Malaysian context")
    n_tags = df["text"].str.count(r"#\w+")
    drop(n_tags >= ccfg["max_hashtags"], f"promo / fan spam ({ccfg['max_hashtags']}+ hashtags)")
    drop(~is_reddit & df["text"].str.contains(PROMO_RE), "business advert / fan-club promotion")

    # 6. Anonymise + normalise (done before the information check, which reads clean text)
    df["text_clean"] = df["text"].map(clean_text)
    df["text_norm"] = df["text"].map(normalise_text)
    df["n_words"] = df["text_clean"].map(lambda t: len(content_words(t)))

    # 7. Low-information content (meaningful short opinions are kept)
    min_words = ccfg["min_words"]
    info = df["text_clean"].map(lambda t: information_check(t, min_words))
    drop(info.notna(), lambda r: info[r.name])

    # 8. Same text posted more than once (copy-paste spam, cross-posts)
    #    (checked among the rows still kept, so a dropped copy doesn't knock out a good one)
    still_kept = df["drop_reason"].isna()
    repeated = pd.Series(False, index=df.index)
    repeated[still_kept] = df.loc[still_kept].duplicated("text_norm", keep="first")
    drop(repeated, "repeated text (copy-paste / cross-post)")

    # Final tidy-up. `text` is replaced by its anonymised form: no usernames / links leave here.
    df["query"] = df["_query_all"]
    df["text"] = df["text"].map(anonymise)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

    keep_cols = ["post_id", "platform", "is_comment", "parent_post_id", "subreddit", "query",
                 "created_at", "author_hash", "url", "likes", "replies", "text", "text_clean",
                 "text_norm", "n_words", "lang_detected", "lang_prob", "lang_group", "mentions_grab",
                 "actor_sentiment_score", "actor_sentiment_label", "manually_added", "collected_at"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    kept = df.loc[df["drop_reason"].isna(), keep_cols].sort_values(["platform", "created_at"]).reset_index(drop=True)
    dropped = df.loc[df["drop_reason"].notna(), ["post_id", "platform", "is_comment", "drop_reason", "text"]]

    funnel_df = pd.DataFrame(funnel, columns=["step", "items_remaining"])
    funnel_df["removed_at_this_step"] = (-funnel_df["items_remaining"].diff()).fillna(0).astype(int)
    funnel_df = funnel_df.groupby("step", sort=False, as_index=False).agg(
        items_remaining=("items_remaining", "last"), removed_at_this_step=("removed_at_this_step", "sum"))
    return kept, dropped.reset_index(drop=True), funnel_df


def platform_breakdown(kept: pd.DataFrame) -> pd.DataFrame:
    """Final counts per platform, posts vs comments/replies."""
    t = pd.crosstab(kept["platform"], kept["is_comment"].map({False: "posts", True: "comments/replies"}),
                    margins=True, margins_name="TOTAL")
    return t
