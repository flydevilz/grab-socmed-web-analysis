"""
Data-collection helpers for notebook 01_data_collection.

Section A – web performance:
    * Google PageSpeed Insights (PSI) API v5  -> run_pagespeed_batch(), pagespeed_median_table()
    * On-page SEO scrape (requests + BS4)     -> scrape_onpage_seo_many()

Section B – social media via Apify:
    * collect_social()  runs Reddit / X (Twitter) / Threads actors, saves raw JSON,
      and returns one normalised DataFrame with the project schema (POST_SCHEMA).

Secrets (APIFY_TOKEN, PAGESPEED_API_KEY) are read from .env only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable
from urllib import robotparser
from urllib.parse import urlparse

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Final column order for every social-media post, whatever the platform.
POST_SCHEMA = [
    "post_id", "platform", "query", "text", "created_at", "author_hash",
    "url", "likes", "replies", "is_comment", "collected_at",
]


# ===========================================================================
# General utilities
# ===========================================================================
def load_config(path: str | Path | None = None) -> dict:
    """Read config.yaml (project root by default)."""
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(rel: str) -> Path:
    """Absolute path for a project-relative folder, created if missing."""
    p = PROJECT_ROOT / rel
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_secret(name: str) -> str:
    """Read a secret from .env. Raises a friendly error if it's missing."""
    load_dotenv(PROJECT_ROOT / ".env")
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and add your key."
        )
    return value


def setup_logging(name: str = "collection") -> logging.Logger:
    """Log INFO+ to logs/<name>_<timestamp>.log and to the notebook output."""
    logger = logging.getLogger(f"grab.{name}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(project_path("logs") / f"{name}_{ts_slug()}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ts_slug(dt: datetime | None = None) -> str:
    """Timestamp safe for filenames, e.g. 20261105T031500Z."""
    return (dt or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def url_slug(url: str) -> str:
    """https://food.grab.com/my/ -> food.grab.com_my"""
    p = urlparse(url)
    return re.sub(r"[^A-Za-z0-9.]+", "_", (p.netloc + p.path)).strip("_")


def with_retries(fn: Callable[[], Any], *, attempts: int = 3, base_delay: float = 5.0,
                 logger: logging.Logger | None = None, label: str = "") -> Any:
    """Call fn(); on error wait (exponential backoff + jitter) and try again."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise on the last attempt
            if i == attempts:
                if logger:
                    logger.error("%s failed after %d attempts: %s", label, attempts, exc)
                raise
            delay = base_delay * 2 ** (i - 1) + random.uniform(0, 1)
            if logger:
                logger.warning("%s attempt %d/%d failed (%s); retrying in %.0fs",
                               label, i, attempts, exc, delay)
            time.sleep(delay)


def hash_author(username: Any) -> str | None:
    """SHA-256 of the lower-cased username. Raw usernames are never kept in our tables."""
    if username is None or (isinstance(username, float) and pd.isna(username)):
        return None
    name = str(username).strip().lstrip("@").lower()
    if not name or name in {"[deleted]", "[removed]"}:
        return None
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def to_iso_utc(value: Any) -> str | None:
    """Convert unix seconds / ISO strings / Twitter-style dates to ISO-8601 UTC."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            ts = pd.to_datetime(value, unit="s", utc=True)
        else:
            ts = pd.to_datetime(value, utc=True)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return None


# ===========================================================================
# Section A1 – PageSpeed Insights
# ===========================================================================
class PageSpeedError(RuntimeError):
    pass


def fetch_pagespeed(url: str, strategy: str, api_key: str, endpoint: str,
                    categories: list[str], timeout: int = 180) -> dict:
    """One PSI API call. The API key is never included in error messages."""
    params = [("url", url), ("strategy", strategy), ("key", api_key)]
    params += [("category", c) for c in categories]
    try:
        r = requests.get(endpoint, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise PageSpeedError(f"network error: {type(exc).__name__}") from None
    if r.status_code != 200:
        try:
            msg = r.json().get("error", {}).get("message", "")
        except ValueError:
            msg = r.text[:200]
        raise PageSpeedError(f"HTTP {r.status_code}: {msg[:300]}")
    return r.json()


def extract_pagespeed_metrics(data: dict) -> dict:
    """Flatten one PSI response into a single tidy row."""
    lh = data.get("lighthouseResult", {}) or {}
    audits = lh.get("audits", {}) or {}
    cats = lh.get("categories", {}) or {}

    def cat_score(key):
        s = (cats.get(key) or {}).get("score")
        return None if s is None else round(s * 100)

    def audit_value(key):
        return (audits.get(key) or {}).get("numericValue")

    row = {
        "performance_score": cat_score("performance"),
        "accessibility_score": cat_score("accessibility"),
        "best_practices_score": cat_score("best-practices"),
        "seo_score": cat_score("seo"),
        # Lab data (Lighthouse, simulated device)
        "lab_fcp_ms": audit_value("first-contentful-paint"),
        "lab_lcp_ms": audit_value("largest-contentful-paint"),
        "lab_tbt_ms": audit_value("total-blocking-time"),
        "lab_cls": audit_value("cumulative-layout-shift"),
        "lab_speed_index_ms": audit_value("speed-index"),
    }

    # Field data (Chrome UX Report, real users, last 28 days).
    # Prefer page-level data; fall back to whole-origin data if the page has none.
    field, source = data.get("loadingExperience") or {}, "page"
    if not field.get("metrics"):
        field, source = data.get("originLoadingExperience") or {}, "origin"
    m = field.get("metrics") or {}

    def pct(k):
        return (m.get(k) or {}).get("percentile")

    def cat(k):
        return (m.get(k) or {}).get("category")

    cls = pct("CUMULATIVE_LAYOUT_SHIFT_SCORE")
    row.update({
        "field_data_source": source if m else None,
        "field_lcp_ms": pct("LARGEST_CONTENTFUL_PAINT_MS"),
        "field_lcp_category": cat("LARGEST_CONTENTFUL_PAINT_MS"),
        "field_inp_ms": pct("INTERACTION_TO_NEXT_PAINT"),
        "field_inp_category": cat("INTERACTION_TO_NEXT_PAINT"),
        "field_cls": None if cls is None else cls / 100,  # CrUX reports CLS x100
        "field_cls_category": cat("CUMULATIVE_LAYOUT_SHIFT_SCORE"),
        "field_overall_category": field.get("overall_category") if m else None,
        "final_url": lh.get("finalDisplayedUrl") or lh.get("finalUrl"),
        "lighthouse_version": lh.get("lighthouseVersion"),
        "runtime_error": (lh.get("runtimeError") or {}).get("code"),
    })
    return row


def run_pagespeed_batch(cfg: dict, logger: logging.Logger,
                        only: list[tuple[str, str, int]] | None = None,
                        max_workers: int | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """
    Run every URL x strategy x run, save each raw JSON, return a tidy DataFrame.
    `only` = list of (url, strategy, run) to re-run just those (e.g. earlier failures).
    """
    ps = cfg["pagespeed"]
    api_key = get_secret("PAGESPEED_API_KEY")
    out_dir = project_path(cfg["paths"]["web"])
    retry = cfg["social"]["retries"]

    jobs = only or [(u, s, r) for r in range(1, ps["runs_per_combo"] + 1)
                    for u in ps["urls"] for s in ps["strategies"]]
    logger.info("PageSpeed: %d API calls queued", len(jobs))
    min_gap = ps.get("min_gap_between_runs_s", 60)
    # Lighthouse fetchTime already seen per (url, strategy). PSI caches recent results,
    # so a repeat test of the same page can silently return the previous one.
    seen: dict[tuple[str, str], set[str]] = {}

    def fetch_fresh(url, strategy):
        data = fetch_pagespeed(url, strategy, api_key, ps["endpoint"],
                               ps["categories"], ps["request_timeout_s"])
        ft = (data.get("lighthouseResult") or {}).get("fetchTime")
        if ft and ft in seen.get((url, strategy), set()):
            raise PageSpeedError("PSI returned a cached copy of an earlier run")
        seen.setdefault((url, strategy), set()).add(ft)
        return data

    def job(url, strategy, run):
        label = f"PSI {url} [{strategy} run {run}]"
        data = with_retries(lambda: fetch_fresh(url, strategy),
                            attempts=retry["attempts"], base_delay=max(30, retry["base_delay_s"]),
                            logger=logger, label=label)
        test_dt = data.get("analysisUTCTimestamp") or utc_now().isoformat()
        fname = out_dir / f"psi_{url_slug(url)}_{strategy}_run{run}_{ts_slug()}.json"
        fname.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        row = {"url": url, "strategy": strategy, "run": run, "test_datetime": test_dt}
        row.update(extract_pagespeed_metrics(data))
        row["lighthouse_fetch_time"] = (data.get("lighthouseResult") or {}).get("fetchTime")
        row["raw_file"] = fname.name
        logger.info("%s -> performance %s", label, row["performance_score"])
        return row

    # Run in "waves": every URL x strategy for run 1, then run 2, ... so the same page is
    # never tested twice at once, with at least `min_gap` seconds between waves.
    rows, failures = [], []
    for run_no in sorted({j[2] for j in jobs}):
        wave = [j for j in jobs if j[2] == run_no]
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=max_workers or ps["max_workers"]) as pool:
            futures = {pool.submit(job, *j): j for j in wave}
            for fut in as_completed(futures):
                url, strategy, run = futures[fut]
                try:
                    rows.append(fut.result())
                except Exception as exc:  # noqa: BLE001 - log & continue
                    failures.append({"url": url, "strategy": strategy, "run": run, "error": str(exc)})
        wait = min_gap - (time.monotonic() - started)
        if wait > 0 and run_no != max(j[2] for j in jobs):
            logger.info("Waiting %.0fs before next wave (avoids PSI cache)", wait)
            time.sleep(wait)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["url", "strategy", "run"]).reset_index(drop=True)
    logger.info("PageSpeed finished: %d ok, %d failed", len(rows), len(failures))
    return df, failures


def pagespeed_median_table(df: pd.DataFrame) -> pd.DataFrame:
    """Median of each numeric metric per URL x strategy (the number we report)."""
    numeric = [c for c in df.columns
               if c.endswith(("_score", "_ms", "_cls")) or c in ("lab_cls", "field_cls")]
    med = df.groupby(["url", "strategy"])[numeric].median().round(3)
    # Categorical CrUX fields are the same across runs, so take the first.
    cat_cols = ["field_data_source", "field_lcp_category", "field_inp_category",
                "field_cls_category", "field_overall_category"]
    cats = df.groupby(["url", "strategy"])[cat_cols].first()
    out = med.join(cats)
    out.insert(0, "n_runs", df.groupby(["url", "strategy"]).size())
    return out.reset_index()


# ===========================================================================
# Section A2 – On-page SEO scrape
# ===========================================================================
def robots_allowed(url: str, user_agent: str, timeout: int = 15) -> tuple[bool, str]:
    """Check robots.txt before scraping. 4xx = no robots file = allowed; 5xx/error = not allowed."""
    p = urlparse(url)
    robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
    try:
        r = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=timeout)
    except requests.RequestException:
        return False, robots_url
    if 400 <= r.status_code < 500:
        return True, robots_url
    if r.status_code >= 500:
        return False, robots_url
    rp = robotparser.RobotFileParser()
    rp.parse(r.text.splitlines())
    return rp.can_fetch(user_agent, url), robots_url


def scrape_onpage_seo(url: str, user_agent: str, timeout: int = 30) -> dict:
    """Pull basic on-page SEO elements from the server-rendered HTML of one page."""
    allowed, robots_url = robots_allowed(url, user_agent)
    row: dict[str, Any] = {"url": url, "robots_txt": robots_url,
                           "robots_allowed": allowed, "scraped_at": utc_now().isoformat()}
    if not allowed:
        row["status"] = "skipped (disallowed by robots.txt)"
        return row

    r = requests.get(url, headers={"User-Agent": user_agent,
                                   "Accept-Language": "en-MY,en;q=0.9,ms;q=0.8"},
                     timeout=timeout)
    row.update({"http_status": r.status_code, "final_url": r.url, "html_bytes": len(r.content)})
    if r.status_code != 200:
        # e.g. a CDN/WAF block page. Don't parse it as if it were the real site.
        row["status"] = f"not parsed (HTTP {r.status_code})"
        return row
    soup = BeautifulSoup(r.text, "lxml")

    title = soup.title.get_text(strip=True) if soup.title else None
    desc_tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    desc = (desc_tag.get("content") or "").strip() if desc_tag else None
    h1 = [h.get_text(" ", strip=True) for h in soup.find_all("h1")]
    h2 = [h.get_text(" ", strip=True) for h in soup.find_all("h2")]
    canonical = soup.find("link", rel="canonical")
    hreflang = [f'{t.get("hreflang")}={t.get("href")}'
                for t in soup.find_all("link", hreflang=True)]
    imgs = soup.find_all("img")
    viewport = soup.find("meta", attrs={"name": re.compile(r"^viewport$", re.I)})

    row.update({
        "status": "ok",
        "title": title,
        "title_length": len(title) if title else 0,
        "meta_description": desc,
        "meta_description_length": len(desc) if desc else 0,
        "h1_count": len(h1),
        "h1_text": " | ".join(h1),
        "h2_count": len(h2),
        "h2_text": " | ".join(h2),
        "canonical": canonical.get("href") if canonical else None,
        "hreflang_count": len(hreflang),
        "hreflang": " | ".join(hreflang),
        "img_total": len(imgs),
        "img_missing_alt": sum(1 for i in imgs if not i.has_attr("alt")),
        "img_empty_alt": sum(1 for i in imgs if i.has_attr("alt") and not i["alt"].strip()),
        "has_viewport": viewport is not None,
        "viewport_content": viewport.get("content") if viewport else None,
    })
    return row


def scrape_onpage_seo_many(cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    ua = cfg["project"]["user_agent"]
    seo = cfg["onpage_seo"]
    rows = []
    for i, url in enumerate(cfg["pagespeed"]["urls"]):
        if i:
            time.sleep(seo["delay_between_requests_s"])  # be polite
        try:
            rows.append(scrape_onpage_seo(url, ua, seo["request_timeout_s"]))
            logger.info("SEO scraped %s", url)
        except Exception as exc:  # noqa: BLE001
            logger.error("SEO scrape failed for %s: %s", url, exc)
            rows.append({"url": url, "status": f"error: {exc}"})
    return pd.DataFrame(rows)


# ===========================================================================
# Section B – Social media via Apify
# ===========================================================================
def date_window(cfg: dict, platform: str | None = None) -> tuple[str, str]:
    """(date_from, date_to) as YYYY-MM-DD. A platform's own lookback_days overrides the global one."""
    days = cfg["social"]["lookback_days"]
    if platform:
        days = cfg["social"]["platforms"][platform].get("lookback_days", days)
    today = utc_now().date()
    return (today - timedelta(days=days)).isoformat(), today.isoformat()


def platform_queries(cfg: dict, platform: str) -> list[str]:
    """A platform's own `queries` list overrides the shared social.queries."""
    return cfg["social"]["platforms"][platform].get("queries") or cfg["social"]["queries"]


def cap_per_query(pcfg: dict, pilot: bool) -> int:
    return pcfg["pilot_cap_per_query"] if pilot else pcfg["full_cap_per_query"]


def build_run_plan(cfg: dict, pilot: bool, platforms: list[str] | None = None) -> list[dict]:
    """
    List of actor runs to make. Each entry: platform, actor_id, label, input, max_items.
    `platforms` (e.g. ["reddit"]) restricts the plan; default = all enabled platforms.
    Input field names were checked against each actor's published input schema (Nov 2026).
    """
    soc = cfg["social"]

    def wanted(p):
        return soc["platforms"][p]["enabled"] and (platforms is None or p in platforms)

    plan: list[dict] = []

    # --- Reddit: one run per subreddit, all queries as subredditKeywords ---
    rc = soc["platforms"]["reddit"]
    if wanted("reddit"):
        queries = platform_queries(cfg, "reddit")
        d_from, d_to = date_window(cfg, "reddit")
        cap = cap_per_query(rc, pilot)
        for sub in rc["subreddits"]:
            plan.append({
                "platform": "reddit", "actor_id": rc["actor_id"], "label": f"r/{sub}",
                "queries": queries,
                "max_items": cap * len(queries) * (1 + (rc["max_comments_per_post"] if rc["scrape_comments"] else 0)),
                "input": {
                    "subredditName": sub,
                    "subredditKeywords": queries,
                    "subredditSort": rc["sort"],
                    "subredditTimeframe": rc["timeframe"],
                    "dateFrom": d_from,
                    "dateTo": d_to,
                    "maxPosts": cap,
                    "scrapeComments": rc["scrape_comments"],
                    "maxComments": rc["max_comments_per_post"],
                    "sentiment_analysis": rc.get("sentiment_analysis", False),
                    "strictSearch": rc.get("strict_search", False),
                    "includeNsfw": False,
                },
            })

    # --- X / Twitter (xquik/x-tweet-scraper): one run per query, because the actor's
    # maxItems is a total across the run. No start fee, so extra runs cost nothing. ---
    tc = soc["platforms"]["twitter"]
    if wanted("twitter"):
        queries = platform_queries(cfg, "twitter")
        d_from, d_to = date_window(cfg, "twitter")
        default_cap = cap_per_query(tc, pilot)
        for entry in queries:
            # A query is either a plain string, or a dict {q, since, until, split_into, cap}.
            # X returns newest tweets first, so a busy query hits its cap within weeks;
            # split_into cuts the date range into N equal windows to reach older tweets too.
            if isinstance(entry, str):
                q, a, b, n, cap = entry, d_from, d_to, 1, default_cap
            else:
                q = entry["q"]
                a, b = entry.get("since", d_from), entry.get("until", d_to)
                n, cap = entry.get("split_into", 1), entry.get("cap", default_cap)
                if pilot:
                    cap = default_cap
            start, end = date.fromisoformat(str(a)), date.fromisoformat(str(b))
            step = (end - start) / n
            for i in range(n):
                w_from = (start + step * i).isoformat()
                w_to = end.isoformat() if i == n - 1 else (start + step * (i + 1)).isoformat()
                plan.append({
                    "platform": "twitter", "actor_id": tc["actor_id"],
                    "label": q if n == 1 else f"{q} [{w_from}..{w_to}]",
                    "queries": [q], "max_items": cap,
                    "input": {
                        # since:/until: is standard X advanced-search syntax; the actor checks every row
                        "searchTerms": [f"{q} since:{w_from} until:{w_to}"],
                        "maxItems": cap,
                        "queryType": tc["sort"],
                        "includeSearchTerms": True,
                    },
                })

    # --- Threads: one run, all queries as keywords (max_posts is per keyword) ---
    hc = soc["platforms"]["threads"]
    if wanted("threads"):
        queries = platform_queries(cfg, "threads")
        d_from, d_to = date_window(cfg, "threads")
        cap = max(10, cap_per_query(hc, pilot))  # actor minimum = 10
        plan.append({
            "platform": "threads", "actor_id": hc["actor_id"], "label": "all queries",
            "queries": queries, "max_items": cap * len(queries),
            "input": {
                "mode": "search",
                "keywords": [q.replace('"', "") for q in queries],
                "max_posts": cap,
                "search_filter": hc["search_filter"],
                "start_date": d_from,
                "end_date": d_to,
            },
        })
    return plan


def estimate_cost(cfg: dict, plan: list[dict]) -> pd.DataFrame:
    """Upper-bound cost if every run returns its maximum number of items."""
    rows = []
    for p in plan:
        pc = cfg["social"]["platforms"][p["platform"]]
        rows.append({"platform": p["platform"], "run": p["label"], "max_items": p["max_items"],
                     "max_cost_usd": p["max_items"] * pc["price_per_item_usd"] + pc["price_per_start_usd"]})
    return pd.DataFrame(rows)


def _status(run) -> str:
    s = getattr(run, "status", None)
    return str(getattr(s, "value", s))


def monthly_usage_usd(client) -> float:
    """Apify account's month-to-date usage (USD), as Apify reports it."""
    limits = client.user().limits()
    data = limits.model_dump() if hasattr(limits, "model_dump") else limits
    return float(data["current"]["monthly_usage_usd"])


class BudgetExceeded(RuntimeError):
    pass


def run_actor(client, entry: dict, cfg: dict, pilot: bool, logger: logging.Logger) -> dict:
    """Run one actor with retries and a hard spend cap; return run metadata + items."""
    retry = cfg["social"]["retries"]
    mode = "pilot" if pilot else "full"
    # Platform-specific cap (platforms.<p>.max_charge_per_run_usd.<mode>) overrides the global one.
    cap_usd = ((cfg["social"]["platforms"][entry["platform"]].get("max_charge_per_run_usd") or {}).get(mode)
               or cfg["social"]["max_charge_per_run_usd"][mode])
    label = f"{entry['platform']} | {entry['label']}"

    # Hard monthly budget: never let a run push month-to-date usage past monthly_budget_usd.
    budget = cfg["social"].get("monthly_budget_usd")
    if budget is not None:
        margin = cfg["social"].get("budget_safety_margin_usd", 0.10)
        remaining = budget - margin - monthly_usage_usd(client)
        if remaining < 0.02:
            raise BudgetExceeded(f"monthly budget US${budget:.2f} reached (remaining {remaining:.2f}); run skipped")
        if remaining < cap_usd:
            logger.warning("%s: run cap lowered from US$%.2f to US$%.2f to stay within budget",
                           label, cap_usd, remaining)
            cap_usd = round(remaining, 2)

    def _call():
        run = client.actor(entry["actor_id"]).call(
            run_input=entry["input"],
            max_items=entry["max_items"],
            max_total_charge_usd=Decimal(str(cap_usd)),
            # Optional platforms.<p>.memory_mbytes: some actors charge their start fee per GB of memory.
            memory_mbytes=cfg["social"]["platforms"][entry["platform"]].get("memory_mbytes"),
            run_timeout=timedelta(minutes=30),
            logger=None,
        )
        if run is None:
            raise RuntimeError("Apify returned no run object")
        status = _status(run)
        items = list(client.dataset(run.default_dataset_id).iterate_items())
        # A run that timed out / hit the charge cap may still have useful items.
        if status != "SUCCEEDED" and not items:
            raise RuntimeError(f"run {run.id} ended with status {status} and no items")
        # Some actors report SUCCEEDED even when the search itself failed (e.g. xquik:
        # "1 target failed ... No results were saved"). Treat that as retryable.
        msg = (getattr(run, "status_message", None) or "").lower()
        # ("12 succeeded, 0 failed" is a normal finish, so "0 failed" alone doesn't count)
        if not items and (re.search(r"(?<!\b0 )failed", msg) or "retry" in msg):
            raise RuntimeError(f"run {run.id}: {getattr(run, 'status_message', '')}")
        return run, status, items

    run, status, items = with_retries(_call, attempts=retry["attempts"],
                                      base_delay=retry["base_delay_s"], logger=logger, label=label)
    logger.info("%s -> %s, %d items", label, status, len(items))
    return {
        "platform": entry["platform"], "label": entry["label"], "queries": entry.get("queries"), "actor_id": entry["actor_id"],
        "run_id": run.id, "status": status, "input": entry["input"],
        "usage_total_usd": float(run.usage_total_usd) if run.usage_total_usd is not None else None,
        "charged_event_counts": dict(run.charged_event_counts or {}),
        "items": items,
    }


# ---------------------------------------------------------------------------
# Normalisers: platform-specific raw item -> project schema
# ---------------------------------------------------------------------------
def _norm_reddit(item: dict, run: dict, collected_at: str) -> dict | None:
    kind = item.get("kind")
    if kind not in ("post", "comment"):
        return None
    is_comment = kind == "comment"
    text = item.get("body") if is_comment else "\n\n".join(
        t for t in (item.get("title"), item.get("body")) if t)
    return {
        "post_id": str(item.get("id")),
        "platform": "reddit",
        "query": item.get("query") or "; ".join(run["input"]["subredditKeywords"]),
        "text": text,
        "created_at": to_iso_utc(item.get("created_utc")),
        "author_hash": hash_author(item.get("author")),
        "url": item.get("canonical_url") or item.get("url"),
        "likes": item.get("score"),
        "replies": None if is_comment else item.get("num_comments"),
        "is_comment": is_comment,
        "collected_at": collected_at,
        # Extra Reddit-only context (NaN for other platforms)
        "subreddit": item.get("subreddit"),
        "parent_post_id": item.get("postId") if is_comment else None,
        "actor_sentiment_score": item.get("sentiment_score"),
        "actor_sentiment_label": item.get("sentiment_label"),
    }


def _norm_twitter(item: dict, run: dict, collected_at: str) -> dict | None:
    # Works for xquik/x-tweet-scraper and apidojo/tweet-scraper output.
    text = item.get("fullText") or item.get("full_text") or item.get("text")
    # xquik labels replies type="reply"; keep them (is_comment=True).
    if item.get("type", "tweet") not in ("tweet", "reply") or not item.get("id") or not text:
        return None  # skip placeholders / diagnostics / run-report rows
    tid = str(item["id"])
    author = item.get("author") or {}
    return {
        "post_id": tid,
        "platform": "twitter",
        "query": (run.get("queries") or [run["label"]])[0],  # query text, without the date-window suffix
        "text": text,
        "created_at": to_iso_utc(item.get("createdAt") or item.get("created_at")),
        "author_hash": hash_author(author.get("username") or author.get("userName")
                                   or author.get("screen_name")),
        # Username-free URL form, so we don't store handles.
        "url": f"https://x.com/i/web/status/{tid}",
        "likes": item.get("likeCount"),
        "replies": item.get("replyCount"),
        "is_comment": bool(item.get("isReply")) or item.get("type") == "reply",
        "collected_at": collected_at,
        "lang": item.get("lang"),
    }


def _norm_threads(item: dict, run: dict, collected_at: str) -> dict | None:
    if item.get("record_type", "post") != "post" or not item.get("post_code"):
        return None
    if (run.get("input") or {}).get("mode") == "user_replies":
        # Rows are the account's replies (e.g. @grabmy answering customers). Keep the customer
        # post being answered, not the official reply.
        code = item.get("replied_to_post_code")
        if not code or not item.get("replied_to_text"):
            return None
        return {
            "post_id": str(code),
            "platform": "threads",
            "query": f"post answered by @{(run['input'].get('usernames') or ['?'])[0]}",
            "text": item.get("replied_to_text"),
            "created_at": to_iso_utc(item.get("replied_to_created_at")),
            "author_hash": hash_author(item.get("replied_to_author_username")),
            "url": f"https://www.threads.com/t/{code}",
            "likes": item.get("replied_to_like_count"),
            "replies": item.get("replied_to_reply_count"),
            "is_comment": False,
            "collected_at": collected_at,
        }
    kws = item.get("search_keywords") or ([item["search_keyword"]] if item.get("search_keyword") else [])
    return {
        "post_id": str(item.get("post_code")),
        "platform": "threads",
        "query": "; ".join(kws) if kws else None,
        "text": item.get("text_content"),
        "created_at": to_iso_utc(item.get("created_at") or item.get("created_at_timestamp")),
        "author_hash": hash_author(item.get("username")),
        # Short-link form without the @username.
        "url": f"https://www.threads.com/t/{item.get('post_code')}",
        "likes": item.get("like_count"),
        "replies": item.get("reply_count"),
        "is_comment": bool(item.get("is_reply")),
        "collected_at": collected_at,
    }


FB_OFFICIAL_NAMES = {"grab", "grab malaysia", "grabmy", "grabfood malaysia", "grabfood"}


def _norm_facebook(item: dict, run: dict, collected_at: str) -> dict | None:
    """apify/facebook-comments-scraper: one row per comment on a Grab Malaysia page post.
    Page posts themselves (posts scraper) are Grab's own content and are not kept."""
    if item.get("error") or not item.get("commentId") or not item.get("text"):
        return None
    author = item.get("author") or {}
    name = str(item.get("profileName") or author.get("name") or "").strip()
    # Grab replying under its own posts: give it the official-account hash so cleaning drops it.
    who = "GrabMY" if name.lower() in FB_OFFICIAL_NAMES else (item.get("profileId") or author.get("id") or name)
    return {
        "post_id": f"fb_{item['commentId']}",
        "platform": "facebook",
        "query": "comment on GrabMY page post",
        "text": item.get("text"),
        "created_at": to_iso_utc(item.get("date")),
        "author_hash": hash_author(who),
        # Comment permalink: post/photo id + comment id, no username.
        "url": item.get("commentUrl"),
        "likes": item.get("likesCount"),
        "replies": item.get("commentsCount"),
        "is_comment": True,
        "parent_post_id": str(item.get("facebookId") or ""),
        "collected_at": collected_at,
    }


def _norm_tiktok(item: dict, run: dict, collected_at: str) -> dict | None:
    """One row per comment on a Grab Malaysia TikTok video.
    Handles clockworks/tiktok-comments-scraper (cid, uniqueId, videoWebUrl, ...) and
    apidojo/tiktok-comments-scraper (id, user.username, postUrl, ...)."""
    cid = item.get("cid") or item.get("id")
    if item.get("type", "comment") != "comment" or not cid or not item.get("text"):
        return None
    user = item.get("user") or {}
    uname = str(item.get("uniqueId") or user.get("username") or "").strip()
    who = "GrabMY" if uname.lower() in {"grabmy", "grabfoodmy"} else (uname or item.get("uid") or user.get("id"))
    video = item.get("videoWebUrl") or item.get("postUrl") or ""
    return {
        "post_id": f"tt_{cid}",
        "platform": "tiktok",
        "query": f"comment on @{item.get('input') or item.get('inputSource') or 'grabmy'} video",
        "text": item.get("text"),
        "created_at": to_iso_utc(item.get("createTimeISO") or item.get("createdAt")),
        "author_hash": hash_author(who),
        # The video link (Grab's own account); the commenter's username is not in it.
        "url": video,
        "likes": item.get("diggCount", item.get("likeCount")),
        "replies": item.get("replyCommentTotal", item.get("replyCount")),
        "is_comment": True,
        "parent_post_id": str(item.get("awemeId") or video.rstrip("/").split("/")[-1]),
        "collected_at": collected_at,
    }


NORMALISERS = {"reddit": _norm_reddit, "twitter": _norm_twitter, "threads": _norm_threads,
               "facebook": _norm_facebook, "tiktok": _norm_tiktok}


def normalise_runs(runs: list[dict], collected_at: str) -> pd.DataFrame:
    rows = []
    for run in runs:
        fn = NORMALISERS[run["platform"]]
        for item in run["items"]:
            row = fn(item, run, collected_at)
            if row:
                rows.append(row)
    df = pd.DataFrame(rows)
    extra = [c for c in df.columns if c not in POST_SCHEMA]
    return df.reindex(columns=POST_SCHEMA + extra)


def renormalise_from_raw(cfg: dict, platform: str, pilot: bool) -> pd.DataFrame:
    """
    Rebuild data/interim/[pilot_]posts_<platform>.csv from the latest saved raw JSON
    (no new scraping, no cost), then rebuild the combined file. Use after fixing a normaliser.
    """
    prefix = "pilot_" if pilot else ""
    raw_dir = project_path(cfg["paths"]["raw"])
    files = sorted(raw_dir.glob(f"{prefix}{platform}_raw_*.json"))
    if not files:
        raise FileNotFoundError(f"No raw file for {prefix}{platform} in {raw_dir}")
    payload = json.loads(files[-1].read_text(encoding="utf-8"))
    posts = normalise_runs(payload["runs"], payload["collected_at"])
    interim = project_path(cfg["paths"]["interim"])
    posts.dropna(axis=1, how="all").to_csv(interim / f"{prefix}posts_{platform}.csv", index=False)
    combine_platform_files(cfg, pilot).to_csv(
        interim / ("posts_raw_combined_pilot.csv" if pilot else "posts_raw_combined.csv"), index=False)
    return posts


def combine_platform_files(cfg: dict, pilot: bool) -> pd.DataFrame:
    """Stack data/interim/[pilot_]posts_<platform>.csv into one table (schema columns first)."""
    prefix = "pilot_" if pilot else ""
    interim = project_path(cfg["paths"]["interim"])
    frames = [pd.read_csv(interim / f"{prefix}posts_{p}.csv")
              for p in cfg["social"]["platforms"] if (interim / f"{prefix}posts_{p}.csv").exists()]
    if not frames:
        return pd.DataFrame(columns=POST_SCHEMA)
    df = pd.concat(frames, ignore_index=True)
    extra = [c for c in df.columns if c not in POST_SCHEMA]
    return df.reindex(columns=POST_SCHEMA + extra)


def collect_social(cfg: dict, logger: logging.Logger, pilot: bool | None = None,
                   platforms: list[str] | None = None, only_labels: list[str] | None = None,
                   append: bool = False) -> dict:
    """
    Run enabled actors (optionally only `platforms`). One failing run never stops the others.
    Returns dict with: posts (DataFrame), summary, cost, failures, files.
    """
    from apify_client import ApifyClient  # imported here so Section A works without it

    pilot = cfg["social"]["pilot_mode"] if pilot is None else pilot
    plan = build_run_plan(cfg, pilot, platforms)
    if only_labels is not None:  # re-run just some runs (e.g. ones that failed)
        plan = [p for p in plan if p["label"] in only_labels]
    clients = {}

    def client_for(platform):
        # Token order: APIFY_TOKEN_<PLATFORM>, then APIFY_TOKEN, then any other APIFY_TOKEN_* in .env.
        if platform not in clients:
            load_dotenv(PROJECT_ROOT / ".env")
            token = (os.getenv(f"APIFY_TOKEN_{platform.upper()}", "").strip()
                     or os.getenv("APIFY_TOKEN", "").strip()
                     or next((v.strip() for k, v in sorted(os.environ.items())
                              if k.startswith("APIFY_TOKEN") and v.strip()), ""))
            if not token:
                raise RuntimeError("No Apify token found. Add APIFY_TOKEN to .env.")
            clients[platform] = ApifyClient(token)
        return clients[platform]
    collected_at = utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = ts_slug()
    prefix = "pilot_" if pilot else ""
    logger.info("Social collection (%s): %d actor runs planned",
                "PILOT" if pilot else "FULL", len(plan))

    runs, failures = [], []
    for entry in plan:
        try:
            runs.append(run_actor(client_for(entry["platform"]), entry, cfg, pilot, logger))
        except BudgetExceeded as exc:
            failures.append({"platform": entry["platform"], "run": entry["label"], "error": str(exc)})
            logger.error("BUDGET: %s | %s: %s", entry["platform"], entry["label"], exc)
        except Exception as exc:  # noqa: BLE001 - log and move on
            failures.append({"platform": entry["platform"], "run": entry["label"], "error": str(exc)})

    # Raw output per platform -> data/raw/ (git-ignored; contains usernames)
    raw_dir = project_path(cfg["paths"]["raw"])
    files = {}
    for platform in sorted({r["platform"] for r in runs}):
        f = raw_dir / f"{prefix}{platform}_raw_{stamp}.json"
        payload = {"collected_at": collected_at, "pilot_mode": pilot,
                   "runs": [r for r in runs if r["platform"] == platform]}
        f.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
        files[platform] = str(f.relative_to(PROJECT_ROOT))

    # One normalised file per platform (so collecting X later doesn't wipe Reddit),
    # then the combined file is rebuilt from every platform file that exists.
    posts = normalise_runs(runs, collected_at)
    interim = project_path(cfg["paths"]["interim"])
    for platform in sorted(set(posts["platform"].dropna())):
        f = interim / f"{prefix}posts_{platform}.csv"
        new = posts[posts["platform"] == platform]
        if append and f.exists():  # add to the existing platform file instead of replacing it
            new = pd.concat([pd.read_csv(f), new], ignore_index=True).drop_duplicates(
                ["post_id", "is_comment"], keep="last")
        new.dropna(axis=1, how="all").to_csv(f, index=False)
        files[f"{platform}_normalised"] = str(f.relative_to(PROJECT_ROOT))
    combined = combine_platform_files(cfg, pilot)
    out = interim / ("posts_raw_combined_pilot.csv" if pilot else "posts_raw_combined.csv")
    combined.to_csv(out, index=False)
    files["combined"] = str(out.relative_to(PROJECT_ROOT))

    # Summary: posts per platform x query
    summary = (posts.groupby(["platform", "query"], dropna=False).size()
               .rename("posts").reset_index().sort_values(["platform", "posts"], ascending=[True, False]))

    # Cost: estimate from items x price, plus what Apify reports for the run
    cost_rows = []
    for r in runs:
        pc = cfg["social"]["platforms"][r["platform"]]
        cost_rows.append({
            "platform": r["platform"], "run": r["label"], "status": r["status"],
            "items": len(r["items"]),
            "estimated_cost_usd": round(len(r["items"]) * pc["price_per_item_usd"] + pc["price_per_start_usd"], 4),
            "apify_reported_usd": r["usage_total_usd"],
        })
    cost = pd.DataFrame(cost_rows)

    for f in failures:
        logger.error("FAILED %s | %s: %s", f["platform"], f["run"], f["error"])
    logger.info("Social collection done: %d posts, %d failed runs", len(posts), len(failures))
    return {"posts": posts, "summary": summary, "cost": cost,
            "failures": pd.DataFrame(failures), "files": files, "plan": plan}


def add_manual_posts(cfg: dict, raw_file: str, platform: str = "threads") -> pd.DataFrame:
    """Add posts the team found by hand (saved as raw JSON in data/raw/, usernames included).

    They are normalised to the same schema (usernames -> author_hash, username-free URL),
    flagged `manually_added=True`, appended to data/interim/posts_<platform>.csv (no duplicates)
    and the combined file is rebuilt. Costs nothing: no Apify run.
    """
    payload = json.loads((PROJECT_ROOT / raw_file).read_text(encoding="utf-8"))
    rows = [{
        "post_id": str(p["post_code"]), "platform": platform, "query": "manual (user-supplied link)",
        "text": p["text"], "created_at": to_iso_utc(p["created_at"]),
        "author_hash": hash_author(p.get("username")),
        "url": f"https://www.threads.com/t/{p['post_code']}",
        "likes": p.get("likes"), "replies": p.get("replies"), "is_comment": False,
        "collected_at": payload["collected_at"], "manually_added": True,
    } for p in payload["posts"]]
    new = pd.DataFrame(rows)
    interim = project_path(cfg["paths"]["interim"])
    path = interim / f"posts_{platform}.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    merged = pd.concat([old, new], ignore_index=True).drop_duplicates(["post_id", "is_comment"], keep="first")
    merged.to_csv(path, index=False)
    combined = combine_platform_files(cfg, pilot=False)
    combined.to_csv(interim / "posts_raw_combined.csv", index=False)
    return new
