from __future__ import annotations

import json
import hashlib
import html
import math
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, quote_plus, urlencode, urljoin, urlparse, urlunparse

import pandas as pd
import requests
import streamlit as st
import altair as alt
from bs4 import BeautifulSoup

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:
    fuzz = None

BUILTIN_SKM_PATH = Path("data/skm_base.csv")
APP_BUILD = "2026-04-28-export-filenames-v25"

MESSE_FRANKFURT_API_BASES = {
    "dev": "https://api-dev.messefrankfurt.com/service/esb_api",
    "tst": "https://api-test.messefrankfurt.com/service/esb_api",
    "prd": "https://api.messefrankfurt.com/service/esb_api",
}

MESSE_FRANKFURT_PUBLIC_API_KEYS = {
    "dev": "jBv3VMiEinag4bCGVCMockK2m9lcbs74BhnFprCq",
    "tst": "ZEmE2V/a8W0FOg6QfYOIdjy3jFU9oF2rPnSfPXu1z+p9XS6J",
    "prd": "LXnMWcYQhipLAS7rImEzmZ3CkrU033FMha9cwVSngG4vbufTsAOCQQ==",
}

SITEMAP_DETAIL_HINTS = (
    "/exhprofiles/",
    "/exhibitors/",
    "/aussteller/",
    "/companies/",
    "/company/",
    "/profiles/",
    "/profile/",
    "/brands/",
    "/brand/",
)

JUNK_NAME_HINTS = (
    "navigation",
    "footer",
    "faq",
    "english",
    "german",
    "companies",
    "kontakt",
    "contact",
    "impressum",
    "privacy",
    "cookie",
    "events",
    "profile der",
    "mehr informationen",
)

# =========================
# SKM matching
# =========================

LEGAL_FORMS = {
    "ag",
    "as",
    "bv",
    "b.v",
    "co",
    "company",
    "corp",
    "corporation",
    "e.k",
    "e.k.",
    "ek",
    "ev",
    "e.v",
    "e.v.",
    "gbr",
    "gmbh",
    "gmbh&co",
    "gmbhco",
    "handelsgesellschaft",
    "inc",
    "kg",
    "limited",
    "llc",
    "ltd",
    "mbh",
    "nv",
    "ohg",
    "plc",
    "sarl",
    "sas",
    "se",
    "sl",
    "spa",
    "srl",
    "ug",
}

NOISE_WORDS = {
    "and",
    "commercial",
    "commerce",
    "deutschland",
    "eu",
    "europe",
    "export",
    "for",
    "fur",
    "für",
    "germany",
    "global",
    "group",
    "handel",
    "handels",
    "holding",
    "import",
    "international",
    "manufacturing",
    "manufacturer",
    "messe",
    "of",
    "online",
    "product",
    "products",
    "service",
    "services",
    "shop",
    "solutions",
    "store",
    "supplier",
    "systems",
    "technologies",
    "technology",
    "the",
    "trade",
    "trading",
    "und",
    "vertrieb",
}

WEAK_TOKENS = {
    "and",
    "commercial",
    "commerce",
    "deutschland",
    "eu",
    "europe",
    "germany",
    "global",
    "group",
    "handel",
    "handels",
    "holding",
    "import",
    "international",
    "manufacturer",
    "messe",
    "online",
    "product",
    "products",
    "service",
    "services",
    "shop",
    "solutions",
    "store",
    "supplier",
    "systems",
    "technologies",
    "technology",
    "the",
    "trade",
    "trading",
    "und",
    "vertrieb",
}


@dataclass
class SkmCandidate:
    skm_name: str
    compare_name: str
    compare_type: str
    normalized: str
    row: Dict[str, Any]


def normalize_company_name(value: Any) -> str:
    """Normalize merchant/exhibitor names for robust SKM matching."""
    if value is None:
        return ""

    text = str(value).strip().lower()
    if not text or text.lower() == "nan":
        return ""

    text = text.replace("&", " and ")
    text = text.replace("+", " and ")
    text = text.replace("@", " at ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("ß", "ss")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token not in LEGAL_FORMS]
    cleaned = [token for token in tokens if token not in NOISE_WORDS]
    if not cleaned:
        cleaned = tokens

    return " ".join(cleaned).strip()


def split_aliases(value: Any) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parts = re.split(r"[;\n\r|/、]+", text)
    return [part.strip() for part in parts if part.strip()]


def build_skm_candidates(
    skm_rows: Sequence[Dict[str, Any]],
    name_col: str,
    alias_cols: Optional[Sequence[str]] = None,
) -> List[SkmCandidate]:
    alias_cols = alias_cols or []
    candidates: List[SkmCandidate] = []

    for row in skm_rows:
        skm_name = str(row.get(name_col, "")).strip()
        if not skm_name or skm_name.lower() == "nan":
            continue

        names: List[Tuple[str, str]] = [(skm_name, "SKM Name")]
        for col in alias_cols:
            for alias in split_aliases(row.get(col)):
                names.append((alias, col))

        seen = set()
        for compare_name, compare_type in names:
            normalized = normalize_company_name(compare_name)
            if not normalized or len(normalized) < 2 or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(
                SkmCandidate(
                    skm_name=skm_name,
                    compare_name=compare_name,
                    compare_type=compare_type,
                    normalized=normalized,
                    row=row,
                )
            )

    return candidates


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return 100.0 * len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _meaningful_tokens(value: str) -> set:
    return {token for token in value.split() if len(token) >= 2 and token not in WEAK_TOKENS}


def _fallback_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    ratio = SequenceMatcher(None, left, right).ratio() * 100.0
    token_score = _token_jaccard(left, right)
    return max(ratio, token_score)


def company_similarity(left: str, right: str) -> float:
    left_norm = normalize_company_name(left)
    right_norm = normalize_company_name(right)
    if not left_norm or not right_norm:
        return 0.0

    if left_norm == right_norm:
        return 100.0

    left_tokens = _meaningful_tokens(left_norm)
    right_tokens = _meaningful_tokens(right_norm)
    if len(left_tokens) >= 2 and len(right_tokens) >= 2:
        if left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
            return 94.0

    if fuzz is not None:
        score = float(max(fuzz.WRatio(left_norm, right_norm), fuzz.token_set_ratio(left_norm, right_norm)))
    else:
        score = _fallback_score(left_norm, right_norm)

    if left_tokens and right_tokens and not (left_tokens & right_tokens) and score < 92.0:
        score = min(score, 74.0)
    if len(left_norm) <= 4 or len(right_norm) <= 4:
        score = score if score >= 94.0 else min(score, 79.0)

    return score


def _best_candidate(exhibitor_name: str, candidates: Sequence[SkmCandidate]) -> Tuple[Optional[SkmCandidate], float]:
    best_candidate: Optional[SkmCandidate] = None
    best_score = 0.0
    exhibitor_norm = normalize_company_name(exhibitor_name)

    for candidate in candidates:
        if not exhibitor_norm or not candidate.normalized:
            continue

        if exhibitor_norm == candidate.normalized:
            score = 100.0
        elif len(exhibitor_norm.split()) >= 2 and len(candidate.normalized.split()) >= 2:
            exhibitor_tokens = _meaningful_tokens(exhibitor_norm)
            candidate_tokens = _meaningful_tokens(candidate.normalized)
            if exhibitor_tokens.issubset(candidate_tokens) or candidate_tokens.issubset(exhibitor_tokens):
                score = 94.0
            else:
                score = company_similarity(exhibitor_norm, candidate.normalized)
        else:
            score = company_similarity(exhibitor_norm, candidate.normalized)

        if score > best_score:
            best_candidate = candidate
            best_score = score

    return best_candidate, round(best_score, 1)


def _build_candidate_indexes(candidates: Sequence[SkmCandidate]) -> Tuple[Dict[str, SkmCandidate], Dict[str, List[int]]]:
    exact_lookup: Dict[str, SkmCandidate] = {}
    token_index: Dict[str, List[int]] = defaultdict(list)

    for index, candidate in enumerate(candidates):
        if candidate.normalized and candidate.normalized not in exact_lookup:
            exact_lookup[candidate.normalized] = candidate

        tokens = _meaningful_tokens(candidate.normalized)
        if not tokens:
            tokens = set(candidate.normalized.split())
        for token in tokens:
            token_index[token].append(index)

    return exact_lookup, token_index


def _candidate_pool_for_exhibitor(
    exhibitor_name: str,
    candidates: Sequence[SkmCandidate],
    exact_lookup: Dict[str, SkmCandidate],
    token_index: Dict[str, List[int]],
) -> Tuple[Optional[SkmCandidate], float, Sequence[SkmCandidate]]:
    exhibitor_norm = normalize_company_name(exhibitor_name)
    if not exhibitor_norm:
        return None, 0.0, []

    exact = exact_lookup.get(exhibitor_norm)
    if exact is not None:
        return exact, 100.0, []

    tokens = _meaningful_tokens(exhibitor_norm)
    if not tokens:
        tokens = set(exhibitor_norm.split())

    pool_indexes = set()
    for token in tokens:
        pool_indexes.update(token_index.get(token, []))

    if pool_indexes:
        return None, 0.0, [candidates[index] for index in pool_indexes]

    # Avoid scanning a 10k+ SKM base for unrelated names; this keeps app runs stable.
    if len(candidates) <= 3000:
        return None, 0.0, candidates
    return None, 0.0, []


def match_exhibitors_to_skm(
    exhibitors: Sequence[Dict[str, Any]],
    skm_rows: Sequence[Dict[str, Any]],
    name_col: str,
    alias_cols: Optional[Sequence[str]] = None,
    threshold: float = 88.0,
    review_margin: float = 8.0,
) -> List[Dict[str, Any]]:
    """Return exhibitors enriched with SKM matching columns."""
    candidates = build_skm_candidates(skm_rows, name_col=name_col, alias_cols=alias_cols)
    exact_lookup, token_index = _build_candidate_indexes(candidates)
    review_threshold = max(0.0, threshold - review_margin)
    output: List[Dict[str, Any]] = []

    for exhibitor in exhibitors:
        row = dict(exhibitor)
        exhibitor_name = str(row.get("exhibitor_name", "")).strip()
        exact_candidate, exact_score, candidate_pool = _candidate_pool_for_exhibitor(
            exhibitor_name,
            candidates,
            exact_lookup,
            token_index,
        )
        if exact_candidate is not None:
            candidate, score = exact_candidate, exact_score
        elif candidate_pool:
            candidate, score = _best_candidate(exhibitor_name, candidate_pool)
        else:
            candidate, score = None, 0.0

        row["normalized_exhibitor_name"] = normalize_company_name(exhibitor_name)
        row["match_score"] = score
        row["match_status"] = "No Match"
        row["match_type"] = ""
        row["skm_name"] = ""
        row["skm_compare_name"] = ""
        row["skm_compare_type"] = ""
        row["normalized_skm_name"] = ""

        if candidate is not None and score >= review_threshold:
            if score >= threshold:
                row["match_status"] = "SKM Match"
            else:
                row["match_status"] = "Needs Review"
            row["match_type"] = "exact" if score == 100.0 else "fuzzy"
            row["skm_name"] = candidate.skm_name
            row["skm_compare_name"] = candidate.compare_name
            row["skm_compare_type"] = candidate.compare_type
            row["normalized_skm_name"] = candidate.normalized
            for key, value in candidate.row.items():
                row[f"skm_{key}"] = value

        output.append(row)

    return output


def summarize_matches(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    summary = {"total": 0, "skm_matches": 0, "review": 0, "unmatched": 0}
    status_rank = {"SKM Match": 3, "Needs Review": 2, "No Match": 1, "": 0}
    best_status_by_exhibitor: Dict[str, str] = {}
    source_total_exhibitors = 0

    for row in rows:
        source_total_exhibitors = max(
            source_total_exhibitors,
            int(row.get("__source_total_exhibitors") or 0),
        )
        exhibitor_key = (
            _clean_text(row.get("exhibitor_uid"))
            or _clean_text(row.get("detail_url"))
            or normalize_company_name(row.get("exhibitor_name", ""))
        )
        if not exhibitor_key:
            continue

        status = _clean_text(row.get("match_status"))
        previous = best_status_by_exhibitor.get(exhibitor_key, "")
        if status_rank.get(status, 0) >= status_rank.get(previous, 0):
            best_status_by_exhibitor[exhibitor_key] = status

    for status in best_status_by_exhibitor.values():
        summary["total"] += 1
        if status == "SKM Match":
            summary["skm_matches"] += 1
        elif status == "Needs Review":
            summary["review"] += 1
        else:
            summary["unmatched"] += 1

    if source_total_exhibitors:
        summary["total"] = source_total_exhibitors
        summary["unmatched"] = max(0, summary["total"] - summary["skm_matches"] - summary["review"])
    return summary

# =========================
# Exhibitor scraping
# =========================

NAME_KEYS = {
    "aussteller",
    "brand",
    "company",
    "companyname",
    "displayname",
    "exhibitor",
    "exhibitorname",
    "firma",
    "firmname",
    "merchant",
    "name",
    "organisation",
    "organization",
    "title",
}

HALL_KEYS = {"hall", "halle", "hallname", "hallnumber", "pavilion"}
BOOTH_KEYS = {"booth", "boothnumber", "stand", "standnumber", "standnr", "standort"}
COUNTRY_KEYS = {"country", "land", "countryname"}
WEBSITE_KEYS = {"website", "web", "url", "homepage"}
DETAIL_KEYS = {"detailurl", "detail_url", "href", "link", "profileurl", "slug"}

GENERIC_LINK_TEXT = {
    "all",
    "back",
    "contact",
    "contacts",
    "details",
    "download",
    "exhibitor",
    "exhibitors",
    "home",
    "impressum",
    "kontakt",
    "login",
    "map",
    "mehr",
    "more",
    "next",
    "privacy",
    "search",
    "show",
    "weiter",
}

SHOW_AREA_PHRASES = [
    "Photo, Video & Content Creation",
    "Communication & Connectivity",
    "Fitness & Digital Health",
    "Home & Entertainment",
    "Computing & Gaming",
    "IFA Global Markets",
    "Home Appliances",
    "Reseller Park",
    "Smart Home",
    "IFA Next",
    "Mobility",
    "Audio",
]

COUNTRY_SUFFIXES = [
    "United Arab Emirates",
    "United Kingdom",
    "South Korea",
    "Hong Kong",
    "Saudi Arabia",
    "United States",
    "USA",
    "Germany",
    "China",
    "Taiwan",
    "Türkiye",
    "Turkey",
    "Italy",
    "Spain",
    "France",
    "Poland",
    "Sweden",
    "Netherlands",
    "Singapore",
    "Japan",
]


@dataclass
class ScrapeConfig:
    url: str
    max_pages: int = 100
    page_url_template: str = ""
    item_selector: str = ""
    name_selector: str = ""
    hall_selector: str = ""
    booth_selector: str = ""
    country_selector: str = ""
    website_selector: str = ""
    detail_link_selector: str = ""
    crawl_detail_pages: bool = False
    auto_discover_exhibitor_directory: bool = True
    detail_page_limit: int = 50
    request_delay_seconds: float = 0.6
    timeout_seconds: int = 25
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )


class ScrapeError(RuntimeError):
    pass


LAST_SCRAPE_WARNINGS: List[str] = []


def _require_bs4():
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ScrapeError("Missing beautifulsoup4. Please install dependencies from requirements.txt.") from exc
    return BeautifulSoup


def fetch_html(url: str, config: Optional[ScrapeConfig] = None) -> str:
    try:
        import requests  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ScrapeError("Missing requests. Please install dependencies from requirements.txt.") from exc

    cfg = config or ScrapeConfig(url=url)
    headers = {"User-Agent": cfg.user_agent, "Accept-Language": "en-US,en;q=0.9,de;q=0.8"}
    response = requests.get(url, headers=headers, timeout=cfg.timeout_seconds)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def fetch_html_with_retry(url: str, config: Optional[ScrapeConfig] = None, retries: int = 3) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return fetch_html(url, config)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            last_error = exc
            if status_code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries - 1:
                raise

        time.sleep(1.5 * (attempt + 1))

    if last_error:
        raise last_error
    raise ScrapeError(f"Failed to fetch {url}")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_html(
    scraper_build: str,
    url: str,
    max_pages: int,
    page_url_template: str,
    item_selector: str,
    name_selector: str,
    hall_selector: str,
    booth_selector: str,
    country_selector: str,
    website_selector: str,
    detail_link_selector: str,
    crawl_detail_pages: bool,
    auto_discover_exhibitor_directory: bool,
    detail_page_limit: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    _ = scraper_build
    config = ScrapeConfig(
        url=url,
        max_pages=max_pages,
        page_url_template=page_url_template,
        item_selector=item_selector,
        name_selector=name_selector,
        hall_selector=hall_selector,
        booth_selector=booth_selector,
        country_selector=country_selector,
        website_selector=website_selector,
        detail_link_selector=detail_link_selector,
        crawl_detail_pages=crawl_detail_pages,
        auto_discover_exhibitor_directory=auto_discover_exhibitor_directory,
        detail_page_limit=detail_page_limit,
    )
    rows = scrape_exhibitors(config)
    return rows, list(LAST_SCRAPE_WARNINGS)


def parse_exhibitors_from_html(html: str, base_url: str = "", config: Optional[ScrapeConfig] = None) -> List[Dict[str, Any]]:
    BeautifulSoup = _require_bs4()
    soup = BeautifulSoup(html, "html.parser")
    cfg = config or ScrapeConfig(url=base_url)

    all_rows: List[Dict[str, Any]] = []
    if cfg.item_selector:
        all_rows.extend(_extract_with_custom_selectors(soup, base_url, cfg))

    brand_rows = _extract_from_brand_cards(soup, base_url)
    if brand_rows:
        return _dedupe_exhibitors(all_rows + brand_rows)

    all_rows.extend(_extract_from_json_scripts(soup, base_url))
    all_rows.extend(_extract_from_tables(soup, base_url))
    all_rows.extend(_extract_from_cards(soup, base_url))

    if not all_rows:
        all_rows.extend(_extract_from_links(soup, base_url))

    return _dedupe_exhibitors(all_rows)


def scrape_exhibitors(config: ScrapeConfig) -> List[Dict[str, Any]]:
    LAST_SCRAPE_WARNINGS.clear()
    site_specific_rows = _try_site_specific_exhibitor_fetch(config)
    if site_specific_rows is not None:
        rows = _dedupe_exhibitors(site_specific_rows)
        if _should_accept_scrape_result(rows, "", config, strategy_name="site adapter"):
            if config.crawl_detail_pages:
                rows = enrich_from_detail_pages(rows, config)
            return rows
        LAST_SCRAPE_WARNINGS.append("Site adapter result looked incomplete; continued with fallback strategies")

    urls = _build_page_urls(config)
    all_rows: List[Dict[str, Any]] = []
    seen_page_fingerprints = set()
    first_page_html = ""
    structured_attempted = False

    for index, url in enumerate(urls):
        try:
            html = fetch_html_with_retry(url, config)
        except requests.HTTPError as exc:
            message = f"Skipped {url}: HTTP {exc.response.status_code if exc.response is not None else 'error'}"
            LAST_SCRAPE_WARNINGS.append(message)
            if index == 0:
                raise
            continue
        except requests.RequestException as exc:
            LAST_SCRAPE_WARNINGS.append(f"Skipped {url}: {exc}")
            if index == 0:
                raise
            continue

        if (
            index == 0
            and config.auto_discover_exhibitor_directory
            and not config.page_url_template
            and not _is_likely_exhibitor_directory_url(url)
        ):
            discovered_url = _discover_exhibitor_directory_url(html, url)
            if discovered_url and discovered_url not in urls:
                urls[0] = discovered_url
                url = discovered_url
                config.url = discovered_url
                html = fetch_html_with_retry(url, config)

                discovered_site_rows = _try_site_specific_exhibitor_fetch(config)
                if discovered_site_rows is not None:
                    discovered_site_rows = _dedupe_exhibitors(discovered_site_rows)
                    if _should_accept_scrape_result(discovered_site_rows, "", config, strategy_name="discovered site adapter"):
                        if config.crawl_detail_pages:
                            discovered_site_rows = enrich_from_detail_pages(discovered_site_rows, config)
                        return discovered_site_rows
                    LAST_SCRAPE_WARNINGS.append(
                        "Discovered site adapter result looked incomplete; continued with fallback strategies"
                    )

        if index == 0:
            first_page_html = html
            structured_attempted = True
            structured_rows = _try_embedded_directory_fetch(config, html)
            if structured_rows:
                structured_rows = _dedupe_exhibitors(structured_rows)
                if _should_accept_scrape_result(structured_rows, html, config, strategy_name="embedded api"):
                    if config.crawl_detail_pages:
                        structured_rows = enrich_from_detail_pages(structured_rows, config)
                    return structured_rows
                LAST_SCRAPE_WARNINGS.append("Embedded API result looked incomplete; continued with HTML fallback")

        fingerprint = _page_fingerprint(html)
        if fingerprint in seen_page_fingerprints:
            LAST_SCRAPE_WARNINGS.append(f"Stopped at {url}: repeated page detected")
            break
        seen_page_fingerprints.add(fingerprint)

        page_rows = parse_exhibitors_from_html(html, base_url=url, config=config)
        all_rows.extend(page_rows)

        if not config.page_url_template and index + 1 < config.max_pages:
            next_url = _find_next_url(html, url)
            if not next_url or next_url in urls:
                break
            urls.append(next_url)

        if config.request_delay_seconds:
            time.sleep(config.request_delay_seconds)

    rows = _dedupe_exhibitors(all_rows)
    if _should_try_sitemap_profile_fallback(config, first_page_html, rows):
        sitemap_rows = _try_sitemap_profile_fetch(config)
        if sitemap_rows:
            rows = _dedupe_exhibitors(sitemap_rows)
            if _should_accept_scrape_result(rows, first_page_html, config, strategy_name="sitemap profile") or structured_attempted:
                if config.crawl_detail_pages:
                    rows = enrich_from_detail_pages(rows, config)
                return rows
    if config.crawl_detail_pages:
        rows = enrich_from_detail_pages(rows, config)
    return rows


def _should_accept_scrape_result(
    rows: Sequence[Dict[str, Any]], first_page_html: str, config: ScrapeConfig, strategy_name: str
) -> bool:
    metrics = _scrape_quality_metrics(rows)
    row_count = metrics["rows"]
    unique_exhibitors = metrics["unique_exhibitors"]
    junk_ratio = metrics["junk_ratio"]
    hall_ratio = metrics["hall_ratio"]
    booth_ratio = metrics["booth_ratio"]

    LAST_SCRAPE_WARNINGS.append(
        f"{strategy_name.title()} quality check: {unique_exhibitors} exhibitors / {row_count} rows, "
        f"hall coverage {hall_ratio:.0%}, booth coverage {booth_ratio:.0%}, junk {junk_ratio:.0%}"
    )

    if row_count == 0 or unique_exhibitors == 0:
        return False
    if junk_ratio >= 0.2:
        return False
    if first_page_html and _looks_like_client_rendered_directory(first_page_html) and unique_exhibitors < 50:
        return False
    if _is_likely_exhibitor_directory_url(config.url) and unique_exhibitors < 10:
        return False
    return True


def _scrape_quality_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    unique_exhibitors = set()
    junk_count = 0
    hall_count = 0
    booth_count = 0

    for row in rows:
        exhibitor_key = (
            _clean_text(row.get("exhibitor_uid"))
            or _clean_text(row.get("detail_url"))
            or normalize_company_name(row.get("exhibitor_name", ""))
        )
        if exhibitor_key:
            unique_exhibitors.add(exhibitor_key)

        name = _clean_text(row.get("exhibitor_name", "")).lower()
        if any(token in name for token in JUNK_NAME_HINTS):
            junk_count += 1
        if _clean_text(row.get("hall")):
            hall_count += 1
        if _clean_text(row.get("booth")):
            booth_count += 1

    total_rows = len(rows)
    return {
        "rows": float(total_rows),
        "unique_exhibitors": float(len(unique_exhibitors)),
        "junk_ratio": (junk_count / total_rows) if total_rows else 0.0,
        "hall_ratio": (hall_count / total_rows) if total_rows else 0.0,
        "booth_ratio": (booth_count / total_rows) if total_rows else 0.0,
    }


def _try_embedded_directory_fetch(config: ScrapeConfig, first_page_html: str) -> Optional[List[Dict[str, Any]]]:
    mf_config = _extract_messefrankfurt_directory_config(first_page_html)
    if mf_config:
        return _fetch_messefrankfurt_directory_exhibitors_from_config(config, mf_config)
    return None


def _try_site_specific_exhibitor_fetch(config: ScrapeConfig) -> Optional[List[Dict[str, Any]]]:
    parsed = urlparse(config.url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()

    if "interzoo.com" in host and "aussteller" in path:
        try:
            return _fetch_interzoo_algolia_exhibitors(config)
        except Exception as exc:
            LAST_SCRAPE_WARNINGS.append(f"Interzoo fast path failed, fell back to HTML scraping: {exc}")
            return None

    return None


def _should_try_sitemap_profile_fallback(config: ScrapeConfig, first_page_html: str, rows: Sequence[Dict[str, Any]]) -> bool:
    parsed = urlparse(config.url)
    path = parsed.path.lower()
    host = parsed.netloc.lower()
    row_count = len(rows)

    if any(token in host for token in ["interzoo.com", "messefrankfurt.com"]):
        return False

    if row_count == 0:
        return True

    if path.startswith("/vis/") or "/vis/" in path:
        return row_count < 100

    if _looks_like_client_rendered_directory(first_page_html):
        return row_count < 100

    suspicious_names = 0
    for row in rows[:50]:
        name = _clean_text(row.get("exhibitor_name", "")).lower()
        if any(token in name for token in JUNK_NAME_HINTS):
            suspicious_names += 1
    return row_count < 30 or suspicious_names >= max(3, row_count // 4)


def _fetch_interzoo_algolia_exhibitors(config: ScrapeConfig) -> List[Dict[str, Any]]:
    app_id = "4EB6G0V1NT"
    api_key = "f0416e3d1b38ae3aa789c8750e12bfe5"
    index_name = "prod_website_companies_de-de"
    search_url = f"https://{app_id}-dsn.algolia.net/1/indexes/{index_name}/query"
    headers = {
        "X-Algolia-API-Key": api_key,
        "X-Algolia-Application-Id": app_id,
        "Content-Type": "application/json",
        "User-Agent": config.user_agent,
    }

    hits_per_page = 300
    rows: List[Dict[str, Any]] = []

    countries = _fetch_interzoo_country_facets(search_url, headers, config)
    if not countries:
        countries = [None]

    for country in countries:
        rows.extend(
            _fetch_interzoo_country_bucket(
                search_url=search_url,
                headers=headers,
                config=config,
                country=country,
                hits_per_page=hits_per_page,
            )
        )

    LAST_SCRAPE_WARNINGS.append(
        f"Used Interzoo structured search API for faster scraping across {len(countries)} country bucket(s)"
    )
    return rows


def _fetch_messefrankfurt_directory_exhibitors(config: ScrapeConfig) -> Optional[List[Dict[str, Any]]]:
    first_page_html = fetch_html_with_retry(config.url, config, retries=2)
    mf_config = _extract_messefrankfurt_directory_config(first_page_html)
    if not mf_config:
        return None

    return _fetch_messefrankfurt_directory_exhibitors_from_config(config, mf_config)


def _fetch_messefrankfurt_directory_exhibitors_from_config(
    config: ScrapeConfig, mf_config: Dict[str, Any]
) -> Optional[List[Dict[str, Any]]]:

    event_id = _clean_text(mf_config.get("API_EVENT_ID"))
    env = _clean_text(mf_config.get("ENV") or "prd").lower()
    language = _clean_text(mf_config.get("LANGUAGE") or "de-DE")
    if not event_id:
        return None

    api_base = MESSE_FRANKFURT_API_BASES.get(env, MESSE_FRANKFURT_API_BASES["prd"])
    api_key = MESSE_FRANKFURT_PUBLIC_API_KEYS.get(env, MESSE_FRANKFURT_PUBLIC_API_KEYS["prd"])
    search_url = f"{api_base}/exhibitor-service/api/2.1/public/exhibitor/search"
    page_size = 90

    params = {
        "language": language,
        "q": "",
        "orderBy": "name",
        "pageNumber": 1,
        "pageSize": page_size,
        "showJumpLabels": "true",
        "findEventVariable": event_id,
        "orSearchFallback": "false",
    }
    headers = {"apikey": api_key, "User-Agent": config.user_agent}

    response = requests.get(search_url, params=params, headers=headers, timeout=config.timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise ScrapeError(f"Messe Frankfurt search API returned unsuccessful response for {event_id}")

    result = payload.get("result") or {}
    meta = result.get("metaData") or {}
    total_hits = int(meta.get("hitsTotal") or 0)
    if total_hits <= 0:
        return []

    total_pages = max(1, math.ceil(total_hits / page_size))
    total_pages = min(total_pages, max(1, int(config.max_pages)))
    rows: List[Dict[str, Any]] = []

    rows.extend(_messefrankfurt_rows_from_hits(result.get("hits") or [], config.url, mf_config))

    for page_number in range(2, total_pages + 1):
        params["pageNumber"] = page_number
        response = requests.get(search_url, params=params, headers=headers, timeout=config.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            LAST_SCRAPE_WARNINGS.append(
                f"Messe Frankfurt page {page_number} returned unsuccessful response and was skipped"
            )
            continue

        page_hits = (payload.get("result") or {}).get("hits") or []
        rows.extend(_messefrankfurt_rows_from_hits(page_hits, config.url, mf_config))

        if config.request_delay_seconds:
            time.sleep(min(config.request_delay_seconds, 0.15))

    for row in rows:
        row["__source_total_exhibitors"] = total_hits

    LAST_SCRAPE_WARNINGS.append(
        f"Used Messe Frankfurt exhibitor API for {event_id} across {total_pages} page(s); "
        f"{total_hits} exhibitors expanded into {len(rows)} hall/booth row(s)"
    )
    return rows


def _extract_messefrankfurt_directory_config(html_text: str) -> Optional[Dict[str, Any]]:
    BeautifulSoup = _require_bs4()
    soup = BeautifulSoup(html_text, "html.parser")
    root = soup.select_one("#mf-ex-root[data-config]") or soup.select_one("[data-config][id='mf-ex-root']")
    if not root:
        return None

    raw_config = root.get("data-config") or ""
    raw_config = html.unescape(raw_config)
    if not raw_config:
        return None

    try:
        data = json.loads(raw_config)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    api_url = _clean_text(data.get("API_URL"))
    if "messefrankfurt" not in api_url and "exhibitorsearch" not in api_url:
        return None

    return data


def _messefrankfurt_rows_from_hits(
    hits: Sequence[Dict[str, Any]], source_url: str, mf_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for hit in hits:
        rows.extend(_messefrankfurt_rows_from_hit(hit, source_url, mf_config))
    return rows


def _messefrankfurt_rows_from_hit(
    hit: Dict[str, Any], source_url: str, mf_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    exhibitor = hit.get("exhibitor") or {}
    if not isinstance(exhibitor, dict):
        return []

    name = _clean_text(exhibitor.get("name"))
    if not name:
        return []

    country = _clean_text(((exhibitor.get("address") or {}).get("country") or {}).get("label"))
    detail_url = _build_messefrankfurt_detail_url(exhibitor, source_url, mf_config)
    website = _extract_messefrankfurt_website(exhibitor, source_url)
    show_area = _clean_text(exhibitor.get("shortDescription") or "")
    halls = ((exhibitor.get("exhibition") or {}).get("exhibitionHall")) or []

    if not halls:
        row = _make_row(
            name=name,
            hall="",
            booth="",
            country=country,
            website=website,
            detail_url=detail_url,
            source_url=source_url,
            raw_text=name,
            method="messefrankfurt_es_api",
        )
        row["exhibitor_uid"] = _clean_text(exhibitor.get("id") or exhibitor.get("rewriteId") or detail_url or name)
        if show_area:
            row["show_area"] = show_area
        return [row] if _is_valid_exhibitor(row) else []

    rows: List[Dict[str, Any]] = []
    for hall_info in halls:
        hall_name = _messefrankfurt_hall_name(hall_info)
        stands = (hall_info or {}).get("stand") or []

        if not stands:
            row = _make_row(
                name=name,
                hall=hall_name,
                booth="",
                country=country,
                website=website,
                detail_url=detail_url,
                source_url=source_url,
                raw_text=f"{name} {hall_name} {country}",
                method="messefrankfurt_es_api",
            )
            row["exhibitor_uid"] = _clean_text(exhibitor.get("id") or exhibitor.get("rewriteId") or detail_url or name)
            if show_area:
                row["show_area"] = show_area
            if _is_valid_exhibitor(row):
                rows.append(row)
            continue

        for stand in stands:
            booth = _clean_text((stand or {}).get("name"))
            row = _make_row(
                name=name,
                hall=hall_name,
                booth=booth,
                country=country,
                website=website,
                detail_url=detail_url,
                source_url=source_url,
                raw_text=f"{name} {hall_name} {booth} {country}",
                method="messefrankfurt_es_api",
            )
            row["exhibitor_uid"] = _clean_text(exhibitor.get("id") or exhibitor.get("rewriteId") or detail_url or name)
            if show_area:
                row["show_area"] = show_area
            if _is_valid_exhibitor(row):
                rows.append(row)

    return rows


def _messefrankfurt_hall_name(hall_info: Dict[str, Any]) -> str:
    if not isinstance(hall_info, dict):
        return ""

    category = _clean_text(
        ((((hall_info.get("categoryLabel") or {}).get("labels") or {}).get("de-DE") or {}).get("text"))
        or ((((hall_info.get("categoryLabel") or {}).get("labels") or {}).get("en-GB") or {}).get("text"))
    )
    name = _clean_text(
        ((((hall_info.get("nameLabel") or {}).get("labels") or {}).get("de-DE") or {}).get("text"))
        or ((((hall_info.get("nameLabel") or {}).get("labels") or {}).get("en-GB") or {}).get("text"))
        or hall_info.get("name")
    )

    if category and name and category.lower() not in name.lower():
        return f"{category} {name}"
    return name or category


def _extract_messefrankfurt_website(exhibitor: Dict[str, Any], source_url: str) -> str:
    source_host = urlparse(source_url).netloc.lower()
    candidates: List[str] = []

    href = _clean_text(exhibitor.get("href"))
    if href:
        candidates.append(href)

    for synonym in ((exhibitor.get("exhibition") or {}).get("synonyme") or []):
        homepage = _clean_text((synonym or {}).get("homepage"))
        if homepage:
            candidates.append(homepage)

    for candidate in candidates:
        normalized = candidate
        if normalized and not normalized.startswith(("http://", "https://")) and "." in normalized:
            normalized = f"https://{normalized}"
        parsed = urlparse(normalized)
        if parsed.netloc and parsed.netloc.lower() != source_host and "messefrankfurt.com" not in parsed.netloc.lower():
            return normalized

    return ""


def _build_messefrankfurt_detail_url(
    exhibitor: Dict[str, Any], source_url: str, mf_config: Dict[str, Any]
) -> str:
    rewrite_id = _clean_text(exhibitor.get("rewriteId"))
    if not rewrite_id:
        return ""

    route_template = _clean_text(((mf_config.get("ROUTES") or {}).get("DETAIL_EXHIBITOR")) or "")
    base_path = _clean_text(mf_config.get("BASE_PATH") or "")
    if not route_template:
        return ""

    relative = route_template.replace(":rewriteId", rewrite_id)
    relative = f"{base_path.rstrip('/')}/{relative.lstrip('/')}"
    parsed = urlparse(source_url)
    root_url = f"{parsed.scheme or 'https'}://{parsed.netloc}"
    return urljoin(root_url, relative)


def _try_sitemap_profile_fetch(config: ScrapeConfig) -> Optional[List[Dict[str, Any]]]:
    try:
        sitemap_urls = _discover_sitemaps_from_robots(config.url, config)
        if not sitemap_urls:
            return None

        detail_urls = _collect_profile_urls_from_sitemaps(sitemap_urls, config)
        if len(detail_urls) < 25:
            return None

        rows = _fetch_profile_rows_concurrently(detail_urls, config)
        rows = [row for row in rows if _is_valid_exhibitor(row)]
        if not rows:
            return None

        LAST_SCRAPE_WARNINGS.append(
            f"Used sitemap profile fallback across {len(detail_urls)} exhibitor profile page(s)"
        )
        return rows
    except Exception as exc:
        LAST_SCRAPE_WARNINGS.append(f"Sitemap profile fallback failed, continued with HTML scraping: {exc}")
        return None


def _discover_sitemaps_from_robots(base_url: str, config: ScrapeConfig) -> List[str]:
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme or 'https'}://{parsed.netloc}/robots.txt"
    response = requests.get(
        robots_url,
        headers={"User-Agent": config.user_agent, "Accept-Language": "de-DE,de;q=0.9,en;q=0.8"},
        timeout=config.timeout_seconds,
    )
    response.raise_for_status()

    sitemap_urls: List[str] = []
    for line in response.text.splitlines():
        if line.lower().startswith("sitemap:"):
            url = _clean_text(line.split(":", 1)[1])
            if url:
                sitemap_urls.append(url)

    return list(dict.fromkeys(sitemap_urls))


def _collect_profile_urls_from_sitemaps(sitemap_urls: Sequence[str], config: ScrapeConfig) -> List[str]:
    profile_urls: List[str] = []
    base_host = urlparse(config.url).netloc.lower()

    for sitemap_url in sitemap_urls[:20]:
        try:
            response = requests.get(
                sitemap_url,
                headers={"User-Agent": config.user_agent, "Accept-Language": "de-DE,de;q=0.9,en;q=0.8"},
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
            locs = _parse_sitemap_locs(response.text)
        except Exception:
            continue

        matched = [loc for loc in locs if _looks_like_exhibitor_profile_url(loc, base_host)]
        if matched:
            profile_urls.extend(matched)

    return list(dict.fromkeys(profile_urls))


def _parse_sitemap_locs(xml_text: str) -> List[str]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    locs: List[str] = []
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    for node in root.findall(".//sm:loc", namespace):
        text = _clean_text(node.text)
        if text:
            locs.append(text)
    return locs


def _looks_like_exhibitor_profile_url(url: str, base_host: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc.lower() != base_host:
        return False

    lower = parsed.path.lower()
    if any(hint in lower for hint in SITEMAP_DETAIL_HINTS):
        if any(token in lower for token in ["/directory/", "/search", "/catalogue", "/countries"]):
            return False
        return True
    return False


def _fetch_profile_rows_concurrently(detail_urls: Sequence[str], config: ScrapeConfig) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    errors = 0
    max_workers = min(4, max(1, len(detail_urls)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_profile_row_from_meta, detail_url, config): detail_url for detail_url in detail_urls}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception:
                errors += 1
                continue
            if row:
                rows.append(row)
    if errors:
        LAST_SCRAPE_WARNINGS.append(f"Sitemap profile fetch skipped {errors} page(s) after retries")
    return rows


def _fetch_profile_row_from_meta(detail_url: str, config: ScrapeConfig) -> Optional[Dict[str, Any]]:
    time.sleep(0.03)
    html = fetch_html_with_retry(detail_url, config, retries=4)

    title = _meta_content(html, "title")
    description = _meta_content(html, "description")
    og_description = _meta_property_content(html, "og:description")
    effective_description = description or og_description

    exhibitor_name, location_text = _parse_profile_title(title)
    hall, booth = extract_hall_booth(effective_description)
    country = _extract_country_from_location_text(location_text)
    website = extract_website_from_html(html, detail_url)

    raw_text = " ".join(part for part in [title, effective_description] if part)
    row = _make_row(
        name=exhibitor_name,
        hall=hall,
        booth=booth,
        country=country,
        website=website,
        detail_url=detail_url,
        source_url=config.url,
        raw_text=raw_text,
        method="sitemap_profile",
    )
    return row if _is_valid_exhibitor(row) else None


def _meta_content(html: str, name: str) -> str:
    pattern = re.compile(
        rf'<meta[^>]+name=["\']{re.escape(name)}["\'][^>]+content=["\']([^"\']*)["\']',
        flags=re.IGNORECASE,
    )
    match = pattern.search(html)
    return _clean_text(match.group(1)) if match else ""


def _meta_property_content(html: str, prop: str) -> str:
    pattern = re.compile(
        rf'<meta[^>]+property=["\']{re.escape(prop)}["\'][^>]+content=["\']([^"\']*)["\']',
        flags=re.IGNORECASE,
    )
    match = pattern.search(html)
    return _clean_text(match.group(1)) if match else ""


def _parse_profile_title(title: str) -> Tuple[str, str]:
    text = _clean_text(title)
    if not text:
        return "", ""

    patterns = [
        r"^(?P<name>.+?)\s+aus\s+(?P<location>.+?)\s+auf der\s+.+$",
        r"^(?P<name>.+?)\s+from\s+(?P<location>.+?)\s+at\s+.+$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_text(match.group("name")), _clean_text(match.group("location"))

    stripped = re.sub(r"\s+--\s+.*$", "", text)
    return stripped, ""


def _extract_country_from_location_text(location_text: str) -> str:
    text = _clean_text(location_text)
    if not text:
        return ""

    known_countries = [
        "Germany",
        "Deutschland",
        "Austria",
        "Österreich",
        "Switzerland",
        "Schweiz",
        "Netherlands",
        "Niederlande",
        "Belgium",
        "Belgien",
        "France",
        "Frankreich",
        "Italy",
        "Italien",
        "Spain",
        "Spanien",
        "Poland",
        "Polen",
        "Türkiye",
        "Turkey",
        "China",
        "Japan",
        "Korea",
        "USA",
        "United States",
        "United Kingdom",
        "UK",
        "Luxembourg",
        "Luxemburg",
    ]
    for country in known_countries:
        if re.search(rf"\b{re.escape(country)}\b", text, flags=re.IGNORECASE):
            return country

    if "," in text:
        tail = _clean_text(text.split(",")[-1])
        if len(tail) >= 4:
            return tail
    return ""


def _fetch_interzoo_country_facets(search_url: str, headers: Dict[str, str], config: ScrapeConfig) -> List[str]:
    params = 'query=&hitsPerPage=0&page=0&filters=site:interz AND isExhibitor:Ja&facets=%5B%22country%22%5D'
    response = requests.post(search_url, headers=headers, json={"params": params}, timeout=config.timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    country_facets = ((payload.get("facets") or {}).get("country") or {})
    return [country for country, count in country_facets.items() if count]


def _fetch_interzoo_country_bucket(
    search_url: str,
    headers: Dict[str, str],
    config: ScrapeConfig,
    country: Optional[str],
    hits_per_page: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base_filter = "site:interz AND isExhibitor:Ja"
    scoped_filter = f'{base_filter} AND country:"{country}"' if country else base_filter

    first_params = f"query=&hitsPerPage={hits_per_page}&page=0&filters={quote_plus(scoped_filter)}"
    first_response = requests.post(
        search_url,
        headers=headers,
        json={"params": first_params},
        timeout=config.timeout_seconds,
    )
    first_response.raise_for_status()
    first_payload = first_response.json()
    total_hits = int(first_payload.get("nbHits") or 0)

    payloads = [first_payload]
    extra_pages = max(0, math.ceil(total_hits / hits_per_page) - 1)
    for page in range(1, extra_pages + 1):
        params = f"query=&hitsPerPage={hits_per_page}&page={page}&filters={quote_plus(scoped_filter)}"
        response = requests.post(search_url, headers=headers, json={"params": params}, timeout=config.timeout_seconds)
        response.raise_for_status()
        payloads.append(response.json())
        if config.request_delay_seconds:
            time.sleep(min(config.request_delay_seconds, 0.05))

    for payload in payloads:
        for hit in payload.get("hits") or []:
            rows.extend(_interzoo_rows_from_algolia_hit(hit, config.url))

    return rows


def _interzoo_rows_from_algolia_hit(hit: Dict[str, Any], source_url: str) -> List[Dict[str, Any]]:
    company_name = _clean_text(hit.get("companyName"))
    country = _clean_text(hit.get("country"))
    detail_url = _absolute_url(_clean_text(hit.get("url")), "https://www.interzoo.com")
    website = _clean_text(hit.get("website"))
    booths = hit.get("booth") or []
    show_area = _clean_text(hit.get("productGroupName") or hit.get("mainProductGroup") or "")

    if not booths:
        row = _make_row(
            name=company_name,
            hall="",
            booth="",
            country=country,
            website=website,
            detail_url=detail_url,
            source_url=source_url,
            raw_text=company_name,
            method="algolia_interzoo",
        )
        if show_area:
            row["show_area"] = show_area
        return [row] if _is_valid_exhibitor(row) else []

    rows: List[Dict[str, Any]] = []
    for booth_info in booths:
        hall = _clean_text(booth_info.get("boothHall"))
        booth = _clean_text(booth_info.get("boothNumber"))
        row = _make_row(
            name=company_name,
            hall=hall,
            booth=booth,
            country=country,
            website=website,
            detail_url=detail_url,
            source_url=source_url,
            raw_text=f"{company_name} {hall} {booth} {country}",
            method="algolia_interzoo",
        )
        if show_area:
            row["show_area"] = show_area
        if _is_valid_exhibitor(row):
            rows.append(row)

    return rows


def _page_fingerprint(html: str) -> str:
    normalized = re.sub(r"\s+", " ", html).strip()
    return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()


def _looks_like_client_rendered_directory(html: str) -> bool:
    lower = html.lower()
    return (
        'id="finder-app"' in lower
        or "finder-base-config" in lower
        or 'data-route="directory"' in lower
        or "Diese Seite benötigt JavaScript".lower() in lower
    )


def _is_likely_exhibitor_directory_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower().strip("/")
    query = parsed.query.lower()
    return any(token in path or token in query for token in ["exhibitors", "aussteller", "brands", "directory"])


def _discover_exhibitor_directory_url(html: str, base_url: str) -> str:
    BeautifulSoup = _require_bs4()
    soup = BeautifulSoup(html, "html.parser")
    best_url = ""
    best_score = 0

    for link in soup.find_all("a"):
        href = link.get("href") or ""
        if not href or href.startswith(("#", "mailto:", "tel:")):
            continue

        text = _clean_text(link.get_text(" ")).lower()
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        path_query = f"{parsed.path} {parsed.query}".lower()
        score = 0

        if "exhibitor" in path_query or "aussteller" in path_query:
            score += 8
        if "exhibitor" in text or "aussteller" in text:
            score += 8
        if "aussteller-finden" in path_query or "exhibitor-search" in path_query or "directory" in path_query:
            score += 12
        if "aussteller finden" in text or "find exhibitors" in text:
            score += 12
        if "aussteller-produkte" in path_query or "exhibitor products" in text:
            score += 6
        if "explore all exhibitors" in text or "exhibitors 2025" in text:
            score += 8
        if parsed.path.rstrip("/").lower() == "/exhibitors":
            score += 6
        if "/ausstellen" in path_query or "ausstellerbereich" in path_query or "shop=" in path_query:
            score -= 12
        if parsed.path.lower().startswith("/de/") and not urlparse(base_url).path.lower().startswith("/de/"):
            score -= 4
        if any(bad in path_query for bad in ["application", "apply", "enquiry", "sponsor", "faq"]):
            score -= 10
        if parsed.netloc and parsed.netloc != urlparse(base_url).netloc:
            score -= 3

        if score > best_score:
            best_url = absolute
            best_score = score

    return best_url if best_score >= 8 else ""


def enrich_from_detail_pages(rows: Sequence[Dict[str, Any]], config: ScrapeConfig) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    visited = set()

    for row in rows:
        new_row = dict(row)
        detail_url = str(row.get("detail_url", "") or "")
        if (
            detail_url
            and detail_url not in visited
            and len(visited) < config.detail_page_limit
            and (not row.get("hall") or not row.get("booth"))
        ):
            visited.add(detail_url)
            try:
                html = fetch_html_with_retry(detail_url, config, retries=2)
                text = _html_to_text(html)
                hall, booth = extract_hall_booth(text)
                country = extract_country(text)
                website = extract_website_from_html(html, detail_url)
                if hall and not new_row.get("hall"):
                    new_row["hall"] = hall
                if booth and not new_row.get("booth"):
                    new_row["booth"] = booth
                if country and not new_row.get("country"):
                    new_row["country"] = country
                if website and not new_row.get("website"):
                    new_row["website"] = website
                new_row["detail_crawled"] = True
            except Exception as exc:
                new_row["detail_crawled"] = False
                new_row["detail_error"] = str(exc)

            if config.request_delay_seconds:
                time.sleep(config.request_delay_seconds)

        enriched.append(new_row)

    return enriched


def _build_page_urls(config: ScrapeConfig) -> List[str]:
    if config.page_url_template:
        return [config.page_url_template.format(page=page) for page in range(1, max(1, config.max_pages) + 1)]
    return [config.url]


def _find_next_url(html: str, base_url: str) -> str:
    if _looks_like_client_rendered_directory(html):
        return ""

    BeautifulSoup = _require_bs4()
    soup = BeautifulSoup(html, "html.parser")
    selectors = [
        'a[rel="next"]',
        'a[aria-label*="Next"]',
        'a[aria-label*="Weiter"]',
        ".pagination a.next",
        ".pager a.next",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        href = node.get("href") if node else ""
        if href:
            return urljoin(base_url, href)

    for node in soup.find_all("a"):
        text = _clean_text(node.get_text(" "))
        if text.lower() in {"next", "weiter", ">", "›", "»"}:
            href = node.get("href")
            if href:
                return urljoin(base_url, href)

    current_page = _current_page_number(base_url)
    numeric_candidates: List[Tuple[int, str]] = []
    for node in soup.find_all("a"):
        href = node.get("href") or ""
        text = _clean_text(node.get_text(" "))
        if not href:
            continue

        page_number = None
        if text.isdigit():
            page_number = int(text)
        else:
            parsed_href = urlparse(urljoin(base_url, href))
            qs = parse_qs(parsed_href.query)
            if qs.get("page") and str(qs["page"][0]).isdigit():
                page_number = int(qs["page"][0])

        if page_number is not None and page_number > current_page:
            numeric_candidates.append((page_number, urljoin(base_url, href)))

    if numeric_candidates:
        numeric_candidates.sort(key=lambda item: item[0])
        return numeric_candidates[0][1]

    if _looks_like_paginated_directory(html):
        return _url_with_page(base_url, current_page + 1)
    return ""


def _current_page_number(url: str) -> int:
    qs = parse_qs(urlparse(url).query)
    if qs.get("page") and str(qs["page"][0]).isdigit():
        return int(qs["page"][0])
    return 1


def _url_with_page(url: str, page: int) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query["page"] = [str(page)]
    encoded = urlencode(query, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, encoded, parsed.fragment))


def _looks_like_paginated_directory(html: str) -> bool:
    return bool(re.search(r"\b\d+\s*/\s*\d+\b", html))


def _clean_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key).strip().lower())


def _clean_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text


def _absolute_url(value: str, base_url: str) -> str:
    if not value:
        return ""
    return urljoin(base_url, value)


def _extract_with_custom_selectors(soup: Any, base_url: str, config: ScrapeConfig) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in soup.select(config.item_selector):
        text = _clean_text(item.get_text(" "))
        name = _select_text(item, config.name_selector) if config.name_selector else _guess_name_from_element(item)
        hall = _select_text(item, config.hall_selector) if config.hall_selector else ""
        booth = _select_text(item, config.booth_selector) if config.booth_selector else ""
        country = _select_text(item, config.country_selector) if config.country_selector else ""
        website = _select_link(item, config.website_selector, base_url) if config.website_selector else ""
        detail_url = _select_link(item, config.detail_link_selector, base_url) if config.detail_link_selector else _guess_detail_link(item, base_url)
        parsed_hall, parsed_booth = extract_hall_booth(text)

        row = _make_row(
            name=name,
            hall=hall or parsed_hall,
            booth=booth or parsed_booth,
            country=country or extract_country(text),
            website=website,
            detail_url=detail_url,
            source_url=base_url,
            raw_text=text,
            method="custom_selector",
        )
        if _is_valid_exhibitor(row):
            rows.append(row)
    return rows


def _select_text(item: Any, selector: str) -> str:
    node = item.select_one(selector) if selector else None
    if node is None:
        return ""
    return _clean_text(node.get_text(" "))


def _select_link(item: Any, selector: str, base_url: str) -> str:
    node = item.select_one(selector) if selector else None
    if node is None:
        return ""
    href = node.get("href") or node.get("data-href") or ""
    if not href and node.name == "a":
        href = node.get("href") or ""
    return _absolute_url(href, base_url)


def _extract_from_tables(soup: Any, base_url: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for table in soup.find_all("table"):
        parsed_rows = _parse_table(table)
        for parsed in parsed_rows:
            name = _first_by_keys(parsed, NAME_KEYS)
            text = " ".join(str(value) for value in parsed.values() if value)
            hall, booth = extract_hall_booth(text)
            row = _make_row(
                name=name,
                hall=_first_by_keys(parsed, HALL_KEYS) or hall,
                booth=_first_by_keys(parsed, BOOTH_KEYS) or booth,
                country=_first_by_keys(parsed, COUNTRY_KEYS) or extract_country(text),
                website=_first_by_keys(parsed, WEBSITE_KEYS),
                detail_url=_first_by_keys(parsed, DETAIL_KEYS),
                source_url=base_url,
                raw_text=text,
                method="table",
            )
            row["detail_url"] = _absolute_url(row.get("detail_url", ""), base_url)
            row["website"] = _absolute_url(row.get("website", ""), base_url) if _looks_like_url(row.get("website", "")) else row.get("website", "")
            if _is_valid_exhibitor(row):
                rows.append(row)

    return rows


def _parse_table(table: Any) -> List[Dict[str, str]]:
    header_cells = table.select("thead th")
    if not header_cells:
        first_row = table.find("tr")
        header_cells = first_row.find_all(["th", "td"]) if first_row else []

    headers = [_clean_text(cell.get_text(" ")) for cell in header_cells]
    normalized_headers = [_clean_key(header) for header in headers]
    body_rows = table.select("tbody tr") or table.find_all("tr")

    parsed_rows: List[Dict[str, str]] = []
    for tr in body_rows:
        cells = tr.find_all(["td", "th"])
        if not cells or len(cells) < 2:
            continue
        values = [_clean_text(cell.get_text(" ")) for cell in cells]
        links = [_absolute_url(link.get("href") or "", "") for cell in cells for link in cell.find_all("a")]

        if normalized_headers and len(normalized_headers) == len(values):
            row = {normalized_headers[i]: values[i] for i in range(len(values))}
        else:
            row = {f"col{i + 1}": value for i, value in enumerate(values)}
            if values:
                row["name"] = values[0]

        if links:
            row["link"] = links[0]
        parsed_rows.append(row)

    return parsed_rows


def _extract_from_json_scripts(soup: Any, base_url: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scripts = soup.find_all("script")

    for script in scripts:
        script_type = (script.get("type") or "").lower()
        text = script.string or script.get_text() or ""
        if not text or len(text) > 6_000_000:
            continue
        if "json" not in script_type and not any(token in text.lower() for token in ["exhibitor", "aussteller", "company", "firma"]):
            continue

        data = _safe_json_loads(text)
        if data is None:
            continue
        for found in _walk_json_for_exhibitors(data, base_url):
            if _is_valid_exhibitor(found):
                rows.append(found)

    return rows


def _safe_json_loads(text: str) -> Optional[Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Next.js and similar pages often embed JSON directly; avoid executing JS.
    candidates = re.findall(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    for candidate in candidates[:2]:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _walk_json_for_exhibitors(data: Any, base_url: str) -> Iterable[Dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            yield from _walk_json_for_exhibitors(item, base_url)
        return

    if not isinstance(data, dict):
        return

    cleaned = {_clean_key(key): value for key, value in data.items()}
    name = _first_by_keys(cleaned, NAME_KEYS)
    if name:
        text = _clean_text(" ".join(str(value) for value in cleaned.values() if isinstance(value, (str, int, float))))
        hall, booth = extract_hall_booth(text)
        row = _make_row(
            name=name,
            hall=_first_by_keys(cleaned, HALL_KEYS) or hall,
            booth=_first_by_keys(cleaned, BOOTH_KEYS) or booth,
            country=_first_by_keys(cleaned, COUNTRY_KEYS) or extract_country(text),
            website=_first_by_keys(cleaned, WEBSITE_KEYS),
            detail_url=_first_by_keys(cleaned, DETAIL_KEYS),
            source_url=base_url,
            raw_text=text[:1000],
            method="json",
        )
        row["detail_url"] = _absolute_url(str(row.get("detail_url", "")), base_url)
        yield row

    for value in data.values():
        if isinstance(value, (dict, list)):
            yield from _walk_json_for_exhibitors(value, base_url)


def _extract_from_brand_cards(soup: Any, base_url: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for card in soup.select(".brand-card"):
        name = _select_text(card, ".name") or _guess_name_from_element(card)
        country = _select_text(card, ".country")
        detail_url = _guess_detail_link(card, base_url)
        show_area = _select_text(card, ".show-area")
        text = _clean_text(card.get_text(" "))
        locations = card.select(".location")

        if not locations:
            hall, booth = extract_hall_booth(text)
            row = _make_row(
                name=name,
                hall=hall,
                booth=booth,
                country=country or extract_country(text),
                website="",
                detail_url=detail_url,
                source_url=base_url,
                raw_text=text,
                method="brand_card",
            )
            row["show_area"] = show_area
            if _is_valid_exhibitor(row):
                rows.append(row)
            continue

        for location in locations:
            hall = _select_text(location, ".brand-location-hall")
            booth = _select_text(location, ".brand-location-stand")
            row = _make_row(
                name=name,
                hall=hall,
                booth=booth,
                country=country,
                website="",
                detail_url=detail_url,
                source_url=base_url,
                raw_text=text,
                method="brand_card",
            )
            row["show_area"] = show_area
            if _is_valid_exhibitor(row):
                rows.append(row)

    return rows


def _extract_from_cards(soup: Any, base_url: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    selectors = [
        "article",
        "li[class*='exhibitor']",
        "li[class*='aussteller']",
        "li[class*='company']",
        "li[class*='brand']",
        "div[class*='exhibitor']",
        "div[class*='aussteller']",
        "div[class*='company']",
        "div[class*='result']",
        "div[class*='card']",
        "div[class*='teaser']",
        "div[id*='exhibitor']",
        "div[id*='aussteller']",
    ]

    seen_nodes = set()
    for selector in selectors:
        for item in soup.select(selector):
            if item.select_one(".brand-card") or item.find_parent(class_="brand-card") or "brand-card" in (item.get("class") or []):
                continue
            node_id = id(item)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            text = _clean_text(item.get_text(" "))
            if len(text) < 4 or len(text) > 2500:
                continue

            name = _guess_name_from_element(item)
            hall, booth = extract_hall_booth(text)
            row = _make_row(
                name=name,
                hall=hall,
                booth=booth,
                country=extract_country(text),
                website=_guess_website(item, base_url),
                detail_url=_guess_detail_link(item, base_url),
                source_url=base_url,
                raw_text=text[:1200],
                method="card",
            )
            if _is_valid_exhibitor(row):
                rows.append(row)

    return rows


def _extract_from_links(soup: Any, base_url: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for link in soup.find_all("a"):
        text = _clean_text(link.get_text(" "))
        href = link.get("href") or ""
        if not text or text.lower() in GENERIC_LINK_TEXT:
            continue
        if len(text) < 3 or len(text) > 90:
            continue
        if _looks_like_navigation_link(text, href):
            continue
        href_abs = _absolute_url(href, base_url)
        row = _make_row(
            name=text,
            hall="",
            booth="",
            country="",
            website="",
            detail_url=href_abs,
            source_url=base_url,
            raw_text=text,
            method="link",
        )
        if _is_valid_exhibitor(row):
            rows.append(row)
    return rows


def _guess_name_from_element(item: Any) -> str:
    for selector in [
        "[itemprop='name']",
        "[class*='name']",
        "[class*='title']",
        "h1",
        "h2",
        "h3",
        "h4",
        "strong",
        "a",
    ]:
        node = item.select_one(selector)
        text = _clean_text(node.get_text(" ")) if node else ""
        if _looks_like_company_name(text):
            return text

    lines = [_clean_text(line) for line in item.get_text("\n").splitlines()]
    for line in lines:
        if _looks_like_company_name(line):
            return line
    return ""


def _guess_detail_link(item: Any, base_url: str) -> str:
    for link in item.find_all("a"):
        href = link.get("href") or ""
        text = _clean_text(link.get_text(" "))
        if not href:
            continue
        if "mailto:" in href or "tel:" in href:
            continue
        if any(token in href.lower() for token in ["exhibitor", "aussteller", "company", "detail", "profile"]) or _looks_like_company_name(text):
            return _absolute_url(href, base_url)
    first = item.find("a")
    return _absolute_url(first.get("href") or "", base_url) if first else ""


def _guess_website(item: Any, base_url: str) -> str:
    base_host = urlparse(base_url).netloc
    for link in item.find_all("a"):
        href = link.get("href") or ""
        if not href.startswith(("http://", "https://")):
            continue
        host = urlparse(href).netloc
        if host and host != base_host:
            return href
    return ""


def _first_by_keys(mapping: Dict[str, Any], keys: Iterable[str]) -> str:
    for key, value in mapping.items():
        clean = _clean_key(key)
        if clean in keys or any(clean.endswith(candidate) for candidate in keys):
            text = _clean_text(value)
            if text:
                return text
    return ""


def _make_row(
    name: str,
    hall: str,
    booth: str,
    country: str,
    website: str,
    detail_url: str,
    source_url: str,
    raw_text: str,
    method: str,
) -> Dict[str, Any]:
    parsed_name, parsed_hall, parsed_booth, parsed_country = _parse_compact_exhibitor_line(name or raw_text)
    if parsed_name and (not hall or not booth):
        name = parsed_name
        hall = hall or parsed_hall
        booth = booth or parsed_booth
        country = country or parsed_country

    return {
        "exhibitor_name": _clean_text(name),
        "hall": normalize_hall(hall),
        "booth": normalize_booth(booth),
        "country": _clean_text(country),
        "website": _clean_text(website),
        "detail_url": _absolute_url(_clean_text(detail_url), source_url),
        "source_url": source_url,
        "raw_text": _clean_text(raw_text),
        "extraction_method": method,
    }


def _parse_compact_exhibitor_line(text: str) -> Tuple[str, str, str, str]:
    text = _clean_text(text)
    if not text:
        return "", "", "", ""

    tokens = text.split()
    hall_index = -1
    for index, token in enumerate(tokens):
        cleaned = token.strip(",;/")
        if re.fullmatch(r"(?:H\d{1,2}(?:\.\d)?[A-Za-z]?|CCB|SOM|Hub27|Hub\s*27)", cleaned, flags=re.IGNORECASE):
            hall_index = index
            break

    if hall_index <= 0:
        return "", "", "", ""

    name_part = " ".join(tokens[:hall_index]).strip(" -/")
    hall = tokens[hall_index].strip(",;/")
    booth = ""

    if hall_index + 1 < len(tokens):
        candidate = tokens[hall_index + 1].strip(",;/")
        if re.fullmatch(r"(?:H\d{1,2}(?:\.\d)?[A-Za-z]?|CCB|SOM|Hub27|Hub\s*27)?-?[A-Za-z]?\d{1,4}[A-Za-z]?|H\d{1,2}(?:\.\d)?[A-Za-z]?-\d{1,4}[A-Za-z]?", candidate, flags=re.IGNORECASE):
            booth = candidate

    country = ""
    for suffix in sorted(COUNTRY_SUFFIXES, key=len, reverse=True):
        if text.lower().endswith(suffix.lower()):
            country = suffix
            break

    for phrase in sorted(SHOW_AREA_PHRASES, key=len, reverse=True):
        pattern = re.compile(rf"\s+{re.escape(phrase)}$", flags=re.IGNORECASE)
        name_part = pattern.sub("", name_part).strip(" -/")

    return name_part, normalize_hall(hall), normalize_booth(booth), country


def _is_valid_exhibitor(row: Dict[str, Any]) -> bool:
    name = _clean_text(row.get("exhibitor_name", ""))
    if not _looks_like_company_name(name):
        return False
    normalized = normalize_company_name(name)
    if not normalized or len(normalized) < 2:
        return False
    if normalized in GENERIC_LINK_TEXT:
        return False
    return True


def _looks_like_company_name(text: str) -> bool:
    text = _clean_text(text)
    if not text:
        return False
    lower = text.lower()
    if lower in GENERIC_LINK_TEXT:
        return False
    if len(text) < 2 or len(text) > 140:
        return False
    if re.search(
        r"(privacy|cookie|terms|login|newsletter|ticket|program|agenda|speaker|navigation|footer|companies|english|german|faq|sidebar)",
        lower,
    ):
        return False
    if len(re.findall(r"[A-Za-zÄÖÜäöüß0-9]", text)) < 2:
        return False
    return True


def _looks_like_navigation_link(text: str, href: str) -> bool:
    lower = f"{text} {href}".lower()
    return bool(re.search(r"(cookie|privacy|login|register|ticket|agenda|program|search|filter|language|#)$", lower))


def _looks_like_url(value: str) -> bool:
    return bool(re.match(r"^(https?://|www\.)", str(value or "").strip().lower()))


def extract_hall_booth(text: str) -> Tuple[str, str]:
    text = _clean_text(text)
    hall = ""
    booth = ""

    hall_patterns = [
        r"\b(?:halle|hall)\s*[:#-]?\s*([0-9]{1,2}(?:\.[0-9])?[a-z]?)\b",
        r"\b(?:halle|hall)\s*([a-z][0-9]{0,2})\b",
    ]
    booth_patterns = [
        r"\b(?:stand|booth|standnr\.?|stand\s*no\.?|booth\s*no\.?)\s*[:#-]?\s*([a-z]?\s*[0-9]{1,4}[a-z]?(?:\.[0-9])?)\b",
        r"\b(?:stand|booth)\s*[:#-]?\s*([0-9]{1,2}[a-z]\s*[0-9]{1,4})\b",
    ]

    for pattern in hall_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            hall = f"Halle {match.group(1).strip().upper()}"
            break

    for pattern in booth_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            booth = match.group(1).strip().upper().replace(" ", "")
            break

    # Common compact pattern: Hall 4.1 / Stand B21
    combo = re.search(
        r"\b(?:halle|hall)\s*([0-9]{1,2}(?:\.[0-9])?[a-z]?)\s*(?:,|/|-|\s)+\s*(?:stand|booth)?\s*([a-z]?\s*[0-9]{1,4}[a-z]?)\b",
        text,
        flags=re.IGNORECASE,
    )
    if combo:
        hall = hall or f"Halle {combo.group(1).strip().upper()}"
        booth = booth or combo.group(2).strip().upper().replace(" ", "")

    return hall, booth


def normalize_hall(value: str) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    hall, _ = extract_hall_booth(text)
    if hall:
        return hall
    if re.fullmatch(r"[0-9]{1,2}(?:\.[0-9])?[A-Za-z]?", text):
        return f"Halle {text.upper()}"
    return text


def normalize_booth(value: str) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    _, booth = extract_hall_booth(text)
    if booth:
        return booth
    return text.upper() if re.fullmatch(r"[A-Za-z]?\s*[0-9]{1,4}[A-Za-z]?(?:\.[0-9])?", text) else text


def extract_country(text: str) -> str:
    text = _clean_text(text)
    match = re.search(r"\b(?:country|land)\s*[:#-]?\s*([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\s-]{2,40})\b", text)
    if match:
        return _clean_text(match.group(1))
    return ""


def _html_to_text(html: str) -> str:
    BeautifulSoup = _require_bs4()
    soup = BeautifulSoup(html, "html.parser")
    for node in soup(["script", "style", "noscript"]):
        node.decompose()
    return _clean_text(soup.get_text(" "))


def extract_website_from_html(html: str, base_url: str) -> str:
    BeautifulSoup = _require_bs4()
    soup = BeautifulSoup(html, "html.parser")
    base_host = urlparse(base_url).netloc
    for link in soup.find_all("a"):
        href = link.get("href") or ""
        if not href.startswith(("http://", "https://")):
            continue
        host = urlparse(href).netloc
        if host and host != base_host:
            return href
    return ""


def _dedupe_exhibitors(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_key: Dict[str, Dict[str, Any]] = {}
    method_rank = {
        "custom_selector": 5,
        "brand_card": 5,
        "algolia_interzoo": 5,
        "sitemap_profile": 5,
        "json": 4,
        "table": 4,
        "card": 3,
        "link": 1,
    }

    for row in rows:
        name = normalize_company_name(row.get("exhibitor_name", ""))
        if not name:
            continue
        key = "|".join([name, normalize_hall(row.get("hall", "")), normalize_booth(row.get("booth", ""))])
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = dict(row)
            continue

        current_score = _row_completeness(row) + method_rank.get(str(row.get("extraction_method")), 0)
        existing_score = _row_completeness(existing) + method_rank.get(str(existing.get("extraction_method")), 0)
        if current_score > existing_score:
            best_by_key[key] = dict(row)

    return list(best_by_key.values())


def _row_completeness(row: Dict[str, Any]) -> int:
    fields = ["hall", "booth", "country", "website", "detail_url"]
    return sum(1 for field in fields if row.get(field))

# =========================
# Exporting
# =========================

EXPORT_COLUMNS = [
    "match_status",
    "exhibitor_name",
    "hall",
    "booth",
    "country",
    "show_area",
    "website",
    "detail_url",
    "source_url",
]


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    first = [col for col in EXPORT_COLUMNS if col in df.columns]
    return df[first]


def sort_leads_by_hall(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_cols = [col for col in ["hall", "booth", "exhibitor_name"] if col in df.columns]
    if not sort_cols:
        return df
    return df.sort_values(sort_cols, kind="stable")


def skm_leads(df: pd.DataFrame) -> pd.DataFrame:
    if "match_status" not in df.columns:
        return df.head(0)
    return df[df["match_status"] == "SKM Match"].copy()


def review_leads(df: pd.DataFrame) -> pd.DataFrame:
    if "match_status" not in df.columns:
        return df.head(0)
    return df[df["match_status"] == "Needs Review"].copy()


def hall_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "hall" not in df.columns:
        return pd.DataFrame(columns=["hall", "skm_leads", "countries", "exhibitors"])

    working = df.copy()
    for col in ["country", "exhibitor_name"]:
        if col not in working.columns:
            working[col] = ""
    working["hall"] = working["hall"].fillna("").replace("", "Unknown Hall")
    grouped = (
        working.groupby("hall", dropna=False)
        .agg(
            skm_leads=("exhibitor_name", "count"),
            countries=("country", lambda values: ", ".join(sorted({str(v) for v in values if str(v).strip()}))),
            exhibitors=("exhibitor_name", lambda values: ", ".join(sorted({str(v) for v in values if str(v).strip()}))),
        )
        .reset_index()
    )
    return grouped.sort_values(["skm_leads", "hall"], ascending=[False, True])


def country_summary(df: pd.DataFrame, row_label: str = "lead_rows") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["country", row_label, "unique_halls", "exhibitors"])

    working = df.copy()
    if "country" not in working.columns:
        working["country"] = ""
    if "hall" not in working.columns:
        working["hall"] = ""
    if "exhibitor_name" not in working.columns:
        working["exhibitor_name"] = ""

    working["country"] = working["country"].fillna("").astype(str).str.strip().replace("", "Unknown Country")
    working["hall"] = working["hall"].fillna("").astype(str).str.strip()
    grouped = (
        working.groupby("country", dropna=False)
        .agg(
            **{
                row_label: ("exhibitor_name", "count"),
                "unique_halls": ("hall", lambda values: int(pd.Series([v for v in values if str(v).strip()]).nunique())),
                "exhibitors": ("exhibitor_name", lambda values: ", ".join(sorted({str(v) for v in values if str(v).strip()}))),
            }
        )
        .reset_index()
    )
    return grouped.sort_values([row_label, "country"], ascending=[False, True])


def _country_focus_mask(series: pd.Series, country_key: str) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    if country_key == "germany":
        patterns = ["germany", "deutschland", "federal republic of germany"]
    elif country_key == "china":
        patterns = ["china", "pr china", "people's republic of china", "people s republic of china", "p.r. china"]
    else:
        patterns = [country_key.lower()]
    mask = pd.Series(False, index=series.index)
    for pattern in patterns:
        mask = mask | normalized.str.contains(re.escape(pattern), regex=True)
    return mask


def _focus_country_rows(df: pd.DataFrame, country_key: str) -> pd.DataFrame:
    if df.empty or "country" not in df.columns:
        return df.head(0)
    return df[_country_focus_mask(df["country"], country_key)].copy()


def run_summary_frame(all_df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(all_df)
    skm_df = skm_leads(all_df)
    review_df = review_leads(all_df)
    hall_count = 0
    booth_count = 0
    country_count = 0
    source_url = ""

    if total_rows:
        if "hall" in all_df.columns:
            hall_count = int(all_df["hall"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique())
        if "booth" in all_df.columns:
            booth_count = int(all_df["booth"].fillna("").astype(str).str.strip().ne("").sum())
        if "country" in all_df.columns:
            country_count = int(all_df["country"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique())
        if "source_url" in all_df.columns:
            source_values = [str(v).strip() for v in all_df["source_url"].fillna("").tolist() if str(v).strip()]
            source_url = source_values[0] if source_values else ""

    rows = [
        {"metric": "Run Date", "value": time.strftime("%Y-%m-%d %H:%M:%S")},
        {"metric": "Source URL", "value": source_url},
        {"metric": "Total Exhibitor Rows", "value": total_rows},
        {"metric": "SKM Exhibitor Rows", "value": len(skm_df)},
        {"metric": "Possible Matches", "value": len(review_df)},
        {"metric": "Unique Halls", "value": hall_count},
        {"metric": "Booth-level Rows", "value": booth_count},
        {"metric": "Source Countries", "value": country_count},
    ]
    return pd.DataFrame(rows)


def build_excel_download(all_df: pd.DataFrame) -> bytes:
    output = BytesIO()
    skm_df = sort_leads_by_hall(skm_leads(all_df))
    review_df = sort_leads_by_hall(review_leads(all_df))
    all_sorted = sort_leads_by_hall(all_df)
    summary_df = hall_summary(skm_df)
    skm_country_df = country_summary(skm_df, row_label="skm_rows")
    all_country_df = country_summary(all_sorted, row_label="lead_rows")
    germany_skm_df = order_columns(sort_leads_by_hall(_focus_country_rows(skm_df, "germany")))
    china_skm_df = order_columns(sort_leads_by_hall(_focus_country_rows(skm_df, "china")))
    run_summary_df = run_summary_frame(all_df)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        run_summary_df.to_excel(writer, sheet_name="Run Summary", index=False)
        summary_df.to_excel(writer, sheet_name="SKM by Hall", index=False)
        skm_country_df.to_excel(writer, sheet_name="SKM by Country", index=False)
        all_country_df.to_excel(writer, sheet_name="All by Country", index=False)
        germany_skm_df.to_excel(writer, sheet_name="Germany SKM Leads", index=False)
        china_skm_df.to_excel(writer, sheet_name="China SKM Leads", index=False)
        order_columns(skm_df).to_excel(writer, sheet_name="SKM Exhibitor Leads", index=False)
        if not review_df.empty:
            order_columns(review_df).to_excel(writer, sheet_name="Possible Matches", index=False)
        order_columns(all_sorted).to_excel(writer, sheet_name="All Exhibitor Leads", index=False)
        _autosize_workbook(writer.sheets)

    return output.getvalue()


def _autosize_workbook(sheets: Dict[str, object]) -> None:
    for sheet in sheets.values():
        for column_cells in sheet.columns:
            values = [str(cell.value) for cell in column_cells if cell.value is not None]
            width = min(max([len(value) for value in values] + [12]) + 2, 48)
            column_letter = column_cells[0].column_letter
            sheet.column_dimensions[column_letter].width = width
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

# =========================
# Streamlit app
# =========================

st.set_page_config(
    page_title="TikTok Shop Fair Intel Console",
    page_icon="TS",
    layout="wide",
)


def _reset_source(source) -> None:
    if hasattr(source, "seek"):
        source.seek(0)


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    df = df.fillna("")
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _columns_look_headerless(columns: Sequence[str]) -> bool:
    if not columns:
        return True
    header_keywords = {
        "alias",
        "aliases",
        "brand",
        "company",
        "company name",
        "firma",
        "merchant",
        "merchant name",
        "name",
        "seller",
        "shop",
        "skm",
        "skm name",
    }
    normalized_columns = [str(col).strip().lower() for col in columns]
    if len(normalized_columns) == 1 and normalized_columns[0] not in header_keywords:
        return True
    bad = 0
    for clean in normalized_columns:
        if not clean or clean.lower().startswith("unnamed"):
            bad += 1
    return bad / max(len(columns), 1) >= 0.5


def _assign_headerless_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    names = ["SKM Name"]
    names.extend([f"Alias {idx}" for idx in range(1, len(df.columns))])
    df.columns = names[: len(df.columns)]
    return df


def _read_csv_like(source, sep: str) -> pd.DataFrame:
    _reset_source(source)
    default_df = pd.read_csv(source, sep=sep, dtype=str, encoding="utf-8-sig")
    default_df = _clean_dataframe(default_df)

    if _columns_look_headerless(list(default_df.columns)):
        _reset_source(source)
        raw_df = pd.read_csv(source, sep=sep, header=None, dtype=str, encoding="utf-8-sig")
        raw_df = _clean_dataframe(raw_df)
        return _assign_headerless_columns(raw_df)

    return default_df


def _read_excel_like(source) -> pd.DataFrame:
    _reset_source(source)
    default_df = pd.read_excel(source, dtype=str)
    default_df = _clean_dataframe(default_df)

    if _columns_look_headerless(list(default_df.columns)):
        _reset_source(source)
        raw_df = pd.read_excel(source, header=None, dtype=str)
        raw_df = _clean_dataframe(raw_df)
        return _assign_headerless_columns(raw_df)

    return default_df


def _read_table_source(source, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        return _read_csv_like(source, sep=",")
    if name.endswith(".tsv"):
        return _read_csv_like(source, sep="\t")
    return _read_excel_like(source)


def _read_table(uploaded_file) -> pd.DataFrame:
    return _read_table_source(uploaded_file, uploaded_file.name)


@st.cache_data(ttl=3600, show_spinner=False)
def _read_builtin_skm() -> pd.DataFrame:
    if not BUILTIN_SKM_PATH.exists():
        return pd.DataFrame()
    text = BUILTIN_SKM_PATH.read_text(encoding="utf-8-sig", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return pd.DataFrame(columns=["SKM Name"])
    if lines[0].strip().lower() in {"skm name", "skm", "name", "merchant", "merchant name"}:
        lines = lines[1:]
    return pd.DataFrame({"SKM Name": lines})


def _read_html(uploaded_file) -> str:
    raw = uploaded_file.read()
    for encoding in ["utf-8", "utf-16", "latin-1"]:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _column_guess(columns: Sequence[str], keywords: Sequence[str]) -> str:
    lowered = {col.lower(): col for col in columns}
    for keyword in keywords:
        for lower, original in lowered.items():
            if keyword in lower:
                return original
    return columns[0] if columns else ""


def _safe_records(df: pd.DataFrame) -> List[Dict[str, object]]:
    return df.fillna("").to_dict(orient="records")


def _file_fingerprint(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    position = uploaded_file.tell() if hasattr(uploaded_file, "tell") else 0
    uploaded_file.seek(0)
    digest = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(position)
    return digest


def _cache_key(parts: Sequence[Any]) -> str:
    return hashlib.md5(json.dumps(list(parts), sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _export_file_stem(result_df: pd.DataFrame) -> str:
    source_url = ""
    if "source_url" in result_df.columns:
        source_values = [str(v).strip() for v in result_df["source_url"].fillna("").tolist() if str(v).strip()]
        source_url = source_values[0] if source_values else ""

    host = urlparse(source_url).netloc.lower()
    host = re.sub(r"^www\.", "", host)
    host = re.sub(r"[^a-z0-9]+", "_", host).strip("_")
    if not host:
        host = "fair"

    run_date = time.strftime("%Y-%m-%d")
    return f"{host}_{run_date}"


def _render_downloads(result_df: pd.DataFrame) -> None:
    ordered = order_columns(sort_leads_by_hall(result_df))
    csv_bytes = ordered.to_csv(index=False).encode("utf-8-sig")
    excel_bytes = build_excel_download(ordered)
    file_stem = _export_file_stem(ordered)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download Excel Leads",
            data=excel_bytes,
            file_name=f"{file_stem}_skm_console.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download All Exhibitor CSV",
            data=csv_bytes,
            file_name=f"{file_stem}_all_leads.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _hall_filtered_rows(df: pd.DataFrame, hall: str) -> pd.DataFrame:
    if df.empty or "hall" not in df.columns:
        return df.head(0)
    normalized_hall = df["hall"].fillna("").replace("", "Unknown Hall")
    return df[normalized_hall == hall].copy()


def _booth_coverage(df: pd.DataFrame) -> int:
    if df.empty or "booth" not in df.columns:
        return 0
    return int(df["booth"].fillna("").astype(str).str.strip().ne("").sum())


def _booth_sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    working = df.copy()
    booth_values = working["booth"].fillna("").astype(str).str.strip() if "booth" in working.columns else pd.Series("", index=working.index)
    working["_has_booth"] = booth_values.ne("")
    working["_booth_sort"] = booth_values
    sort_cols = [col for col in ["_has_booth", "hall", "_booth_sort", "exhibitor_name"] if col in working.columns]
    ascending = [False, True, True, True][: len(sort_cols)]
    working = working.sort_values(sort_cols, ascending=ascending, kind="stable")
    return working.drop(columns=[col for col in ["_has_booth", "_booth_sort"] if col in working.columns])


def _render_hall_snapshot_card(title: str, value: int, caption: str) -> None:
    st.markdown(
        f"""
        <div class="hall-stat-card">
            <div class="hall-stat-title">{title}</div>
            <div class="hall-stat-value">{value}</div>
            <div class="hall-stat-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section_header(eyebrow: str, title: str, description: str = "") -> None:
    description_html = f'<div class="section-description">{description}</div>' if description else ""
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-eyebrow">{eyebrow}</div>
            <div class="section-title">{title}</div>
            {description_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_ribbon(result_df: pd.DataFrame, skm_df: pd.DataFrame) -> None:
    total_rows = len(result_df)
    unique_halls = 0
    booth_rows = 0
    skm_share = 0.0

    if total_rows:
        if "hall" in result_df.columns:
            unique_halls = int(result_df["hall"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique())
        if "booth" in result_df.columns:
            booth_rows = int(result_df["booth"].fillna("").astype(str).str.strip().ne("").sum())
        skm_share = (len(skm_df) / total_rows) * 100

    st.markdown(
        f"""
        <div class="summary-ribbon">
            <div class="summary-ribbon-card">
                <div class="summary-ribbon-title">Hall Coverage</div>
                <div class="summary-ribbon-value">{unique_halls}</div>
                <div class="summary-ribbon-caption">Unique halls detected across the captured fair directory.</div>
            </div>
            <div class="summary-ribbon-card">
                <div class="summary-ribbon-title">Booth Visibility</div>
                <div class="summary-ribbon-value">{booth_rows}</div>
                <div class="summary-ribbon-caption">Lead rows already resolved down to booth-level operating detail.</div>
            </div>
            <div class="summary-ribbon-card">
                <div class="summary-ribbon-title">SKM Density</div>
                <div class="summary-ribbon-value">{skm_share:.1f}%</div>
                <div class="summary-ribbon-caption">Share of captured lead rows currently mapped to priority merchants.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_hall_priority_strip(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    top_halls = summary_df.sort_values("skm_leads", ascending=False).head(6).to_dict(orient="records")
    cards = []
    for record in top_halls:
        hall = html.escape(str(record.get("hall", "") or "Unknown Hall"))
        leads = int(record.get("skm_leads", 0) or 0)
        cards.append(
            f"""
            <div class="hall-priority-card">
                <div class="hall-priority-hall">{hall}</div>
                <div class="hall-priority-value">{leads}</div>
                <div class="hall-priority-caption">SKM leads</div>
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="hall-priority-strip">
            {''.join(cards)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_country_priority_strip(summary_df: pd.DataFrame, row_label: str) -> None:
    if summary_df.empty:
        st.info("No country information found for the current fair run.")
        return

    top_countries = summary_df.head(6).to_dict(orient="records")
    cards = []
    for record in top_countries:
        country = html.escape(str(record.get("country", "") or "Unknown Country"))
        rows = int(record.get(row_label, 0) or 0)
        halls = int(record.get("unique_halls", 0) or 0)
        cards.append(
            f"""
            <div class="hall-priority-card">
                <div class="hall-priority-hall">{country}</div>
                <div class="hall-priority-value">{rows}</div>
                <div class="hall-priority-caption">{halls} hall(s)</div>
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="hall-priority-strip">
            {''.join(cards)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_focus_country_card(title: str, skm_rows_df: pd.DataFrame, all_rows_df: pd.DataFrame, total_skm_rows: int, total_all_rows: int) -> None:
    skm_count = len(skm_rows_df)
    all_count = len(all_rows_df)
    hall_count = 0
    if not all_rows_df.empty and "hall" in all_rows_df.columns:
        hall_count = int(all_rows_df["hall"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique())
    skm_share = (skm_count / total_skm_rows * 100) if total_skm_rows else 0.0
    all_share = (all_count / total_all_rows * 100) if total_all_rows else 0.0

    st.markdown(
        f"""
        <div class="focus-country-card">
            <div class="focus-country-topline">{title}</div>
            <div class="focus-country-grid">
                <div>
                    <div class="focus-country-value">{skm_count}</div>
                    <div class="focus-country-label">SKM rows</div>
                </div>
                <div>
                    <div class="focus-country-value">{all_count}</div>
                    <div class="focus-country-label">All leads</div>
                </div>
            </div>
            <div class="focus-country-meta">
                <span class="focus-country-chip">{hall_count} hall(s)</span>
                <span class="focus-country-chip">{skm_share:.1f}% of SKM</span>
                <span class="focus-country-chip">{all_share:.1f}% of all leads</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_lead_cards(df: pd.DataFrame, empty_message: str) -> None:
    if df.empty:
        st.info(empty_message)
        return

    records = _booth_sort_frame(df).to_dict(orient="records")
    rows_to_show = min(len(records), 80)
    card_number = 0
    for start in range(0, rows_to_show, 2):
        cols = st.columns(2)
        for idx, record in enumerate(records[start:start + 2]):
            with cols[idx]:
                card_number += 1
                booth = str(record.get("booth", "") or "").strip()
                hall = str(record.get("hall", "") or "").strip() or "Unknown Hall"
                country = str(record.get("country", "") or "").strip() or "Unknown Country"
                website = str(record.get("website", "") or "").strip()
                detail_url = str(record.get("detail_url", "") or "").strip()
                show_area = str(record.get("show_area", "") or "").strip()
                match_status = str(record.get("match_status", "") or "").strip()
                badge = "SKM" if match_status == "SKM Match" else "Lead"
                location = f"{hall} / {booth}" if booth else hall
                exhibitor_name = str(record.get("exhibitor_name", "") or "").strip()
                safe_badge_class = "lead" if badge == "Lead" else "skm"
                meta_chips = [f'<span class="booth-card-meta-chip">{html.escape(country)}</span>']
                if booth:
                    meta_chips.append(f'<span class="booth-card-meta-chip">Booth {html.escape(booth)}</span>')
                else:
                    meta_chips.append('<span class="booth-card-meta-chip muted">Booth pending</span>')
                if show_area:
                    meta_chips.append(f'<span class="booth-card-meta-chip">{html.escape(show_area)}</span>')

                link_parts = []
                if detail_url:
                    link_parts.append(f'<a class="booth-card-link" href="{html.escape(detail_url, quote=True)}" target="_blank">Detail</a>')
                if website:
                    link_parts.append(f'<a class="booth-card-link" href="{html.escape(website, quote=True)}" target="_blank">Website</a>')
                links_html = f'<div class="booth-card-links">{"".join(link_parts)}</div>' if link_parts else ""

                st.markdown(
                    f"""
                    <div class="booth-card">
                        <div class="booth-card-top">
                            <div class="booth-card-top-left">
                                <span class="booth-card-badge {safe_badge_class}">{html.escape(badge)}</span>
                                <span class="booth-card-index">{card_number:02d}</span>
                            </div>
                            <div class="booth-card-location">{html.escape(location)}</div>
                        </div>
                        <div class="booth-card-name">{html.escape(exhibitor_name)}</div>
                        <div class="booth-card-meta">
                            {"".join(meta_chips)}
                        </div>
                        {links_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if len(records) > rows_to_show:
        st.markdown(
            f'<div class="lead-card-stack-note">Showing the first {rows_to_show} leads in card view. Use the table tabs for the full hall list.</div>',
            unsafe_allow_html=True,
        )


def _render_hall_drilldown(hall: str, skm_rows: pd.DataFrame, all_rows: pd.DataFrame) -> None:
    hall_skm = _booth_sort_frame(_hall_filtered_rows(skm_rows, hall))
    hall_all = _booth_sort_frame(_hall_filtered_rows(all_rows, hall))
    skm_booth_count = _booth_coverage(hall_skm)
    all_booth_count = _booth_coverage(hall_all)
    countries = []
    if not hall_all.empty and "country" in hall_all.columns:
        countries = sorted({str(v) for v in hall_all["country"].fillna("") if str(v).strip()})

    coverage_note = f"{len(hall_skm)} priority merchant row(s) and {len(hall_all)} total lead row(s) are currently mapped into this hall."
    st.markdown(
        f"""
        <div class="hall-drilldown-shell">
            <div class="hall-drilldown-topline">Selected Hall</div>
            <div class="hall-drilldown-title">{hall}</div>
            <div class="hall-drilldown-subtitle">{coverage_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    stat_cols = st.columns(4)
    with stat_cols[0]:
        _render_hall_snapshot_card("SKM Leads", len(hall_skm), "Priority merchants in this hall")
    with stat_cols[1]:
        _render_hall_snapshot_card("All Leads", len(hall_all), "All exhibitors captured for this hall")
    with stat_cols[2]:
        _render_hall_snapshot_card("SKM with Booth", skm_booth_count, "SKM leads with booth positions")
    with stat_cols[3]:
        _render_hall_snapshot_card("All with Booth", all_booth_count, "All leads with booth positions")

    if countries:
        country_chips = "".join([f'<span class="country-chip">{country}</span>' for country in countries[:10]])
        if len(countries) > 10:
            country_chips += f'<span class="country-chip country-chip-more">+{len(countries) - 10} more</span>'
        st.markdown(f'<div class="country-chip-row">{country_chips}</div>', unsafe_allow_html=True)

    hall_tabs = st.tabs(["SKM Booth Board", "All Leads Booth Board", "SKM Table", "All Leads Table"])
    with hall_tabs[0]:
        _render_section_header("Execution View", "SKM Booth Board", "The priority merchant board for in-hall follow-up.")
        _render_lead_cards(hall_skm, "No SKM leads found in this hall.")
    with hall_tabs[1]:
        _render_section_header("Coverage View", "All Leads Booth Board", "The broader hall roster when you want additional sourcing coverage around SKM targets.")
        _render_lead_cards(hall_all, "No exhibitor leads found in this hall.")
    with hall_tabs[2]:
        _render_section_header("Structured View", "SKM Table", "Sortable SKM detail for export checks and manual review.")
        if hall_skm.empty:
            st.info("No SKM leads found in this hall.")
        else:
            st.dataframe(order_columns(hall_skm), use_container_width=True, hide_index=True)
    with hall_tabs[3]:
        _render_section_header("Structured View", "All Leads Table", "The complete hall table for filtering, handoff, and downstream operating work.")
        if hall_all.empty:
            st.info("No exhibitor leads found in this hall.")
        else:
            st.dataframe(order_columns(hall_all), use_container_width=True, hide_index=True)


def _render_hall_map(skm_df: pd.DataFrame, all_df: pd.DataFrame, *, show_header: bool = True) -> None:
    if show_header:
        _render_section_header("Hall Intelligence", "SKM Hall Heatmap", "See where SKM density is concentrated, then drill into one hall at a time.")
    if skm_df.empty:
        st.info("No SKM exhibitor leads found yet.")
        return

    summary_df = hall_summary(skm_df)
    if summary_df.empty:
        st.info("No hall information found for SKM leads.")
        return

    summary_df = summary_df.reset_index(drop=True)
    cols = 6
    summary_df["map_col"] = summary_df.index % cols
    summary_df["map_row"] = summary_df.index // cols
    summary_df["label"] = summary_df["hall"].astype(str) + "\n" + summary_df["skm_leads"].astype(str) + " SKM"
    summary_df["countries_short"] = summary_df["countries"].astype(str).str.slice(0, 120)
    summary_df["exhibitors_short"] = summary_df["exhibitors"].astype(str).str.slice(0, 220)
    halls = summary_df["hall"].tolist()

    base = alt.Chart(summary_df).encode(
        x=alt.X("map_col:O", axis=None, title=None),
        y=alt.Y("map_row:O", axis=None, title=None, sort="ascending"),
        tooltip=[
            alt.Tooltip("hall:N", title="Hall"),
            alt.Tooltip("skm_leads:Q", title="SKM leads"),
            alt.Tooltip("countries_short:N", title="Countries"),
            alt.Tooltip("exhibitors_short:N", title="SKM examples"),
        ],
    )
    heatmap = base.mark_rect(cornerRadius=6, stroke="#ffffff", strokeWidth=2).encode(
        color=alt.Color(
            "skm_leads:Q",
            scale=alt.Scale(scheme="reds"),
            legend=alt.Legend(title="SKM leads"),
        )
    )
    labels = base.mark_text(fontSize=13, fontWeight="bold", color="#252833", lineBreak="\n").encode(
        text="label:N"
    )
    chart_height = max(240, int((summary_df["map_row"].max() + 1) * 95))
    st.altair_chart((heatmap + labels).properties(height=chart_height), use_container_width=True)
    _render_hall_priority_strip(summary_df)

    st.markdown(
        """
        <div class="map-select-shell">
            <div class="map-select-title">Detailed Operating View</div>
            <div class="map-select-caption">Open one hall at a time to move from heatmap signal into booth-level execution.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_hall = st.selectbox("Choose a hall to open the detailed operating view", halls, label_visibility="collapsed")
    _render_hall_drilldown(selected_hall, skm_df, all_df)


def _render_country_intelligence(skm_df: pd.DataFrame, all_df: pd.DataFrame) -> None:
    _render_section_header("Country Intelligence", "Source Country Breakdown", "Classify exhibitors by source country so you can see where the fair supply base is concentrated.")
    skm_country_df = country_summary(skm_df, row_label="skm_rows")
    all_country_df = country_summary(all_df, row_label="lead_rows")
    germany_skm = sort_leads_by_hall(_focus_country_rows(skm_df, "germany"))
    china_skm = sort_leads_by_hall(_focus_country_rows(skm_df, "china"))
    germany_all = sort_leads_by_hall(_focus_country_rows(all_df, "germany"))
    china_all = sort_leads_by_hall(_focus_country_rows(all_df, "china"))

    focus_cols = st.columns(2)
    with focus_cols[0]:
        _render_focus_country_card("Germany Focus", germany_skm, germany_all, len(skm_df), len(all_df))
    with focus_cols[1]:
        _render_focus_country_card("China Focus", china_skm, china_all, len(skm_df), len(all_df))

    _render_country_priority_strip(skm_country_df if not skm_country_df.empty else all_country_df, "skm_rows" if not skm_country_df.empty else "lead_rows")

    country_tabs = st.tabs(["Germany", "China", "SKM by Country", "All Leads by Country"])
    with country_tabs[0]:
        _render_section_header("Focus Country", "Germany", "Priority and total lead coverage for German exhibitors.")
        if germany_skm.empty and germany_all.empty:
            st.info("No Germany-based exhibitors found for this fair.")
        else:
            focus_subtabs = st.tabs(["Germany SKM Leads", "Germany All Leads"])
            with focus_subtabs[0]:
                if germany_skm.empty:
                    st.info("No Germany-based SKM leads found for this fair.")
                else:
                    st.dataframe(order_columns(germany_skm), use_container_width=True, hide_index=True)
            with focus_subtabs[1]:
                if germany_all.empty:
                    st.info("No Germany-based exhibitors found for this fair.")
                else:
                    st.dataframe(order_columns(germany_all), use_container_width=True, hide_index=True)
    with country_tabs[1]:
        _render_section_header("Focus Country", "China", "Priority and total lead coverage for China-based exhibitors.")
        if china_skm.empty and china_all.empty:
            st.info("No China-based exhibitors found for this fair.")
        else:
            focus_subtabs = st.tabs(["China SKM Leads", "China All Leads"])
            with focus_subtabs[0]:
                if china_skm.empty:
                    st.info("No China-based SKM leads found for this fair.")
                else:
                    st.dataframe(order_columns(china_skm), use_container_width=True, hide_index=True)
            with focus_subtabs[1]:
                if china_all.empty:
                    st.info("No China-based exhibitors found for this fair.")
                else:
                    st.dataframe(order_columns(china_all), use_container_width=True, hide_index=True)
    with country_tabs[2]:
        if skm_country_df.empty:
            st.info("No SKM country data found for this fair.")
        else:
            st.dataframe(skm_country_df, use_container_width=True, hide_index=True)
    with country_tabs[3]:
        if all_country_df.empty:
            st.info("No country data found for this fair.")
        else:
            st.dataframe(all_country_df, use_container_width=True, hide_index=True)


def _render_run_summary_panel(result_df: pd.DataFrame) -> None:
    summary_df = run_summary_frame(result_df)
    st.markdown(
        """
        <div class="dashboard-note">
            <div class="dashboard-note-title">Run Summary</div>
            <div class="dashboard-note-body">
                A compact run record for sharing, export checks, and quick verification of the current fair analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def _render_results(result_df: pd.DataFrame) -> None:
    summary = summarize_matches(_safe_records(result_df))
    skm_df = sort_leads_by_hall(skm_leads(result_df))
    review_df = sort_leads_by_hall(review_leads(result_df))
    all_sorted = sort_leads_by_hall(result_df)

    _render_section_header("Overview", "Fair Command Summary", "A concise operating view of fair coverage, priority merchants, and export-ready output.")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Total Exhibitors", summary["total"])
    metric_cols[1].metric("SKM Exhibitor Leads", summary["skm_matches"])
    metric_cols[2].metric("Needs Review", summary["review"])
    _render_summary_ribbon(all_sorted, skm_df)

    action_left, action_right = st.columns([1.2, 1])
    with action_left:
        st.markdown('<div class="summary-actions">', unsafe_allow_html=True)
        _render_downloads(result_df)
        st.markdown("</div>", unsafe_allow_html=True)
    with action_right:
        st.markdown(
            """
            <div class="dashboard-note">
                <div class="dashboard-note-title">Field Operating Mode</div>
                <div class="dashboard-note-body">
                    Start with hall concentration, then move into booth-level follow-up.
                    Use the exports when you need an offline lead sheet for the floor.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _render_section_header("Hall Intelligence", "SKM Hall Map", "Start with hall concentration, then move into the selected hall for booth-level execution.")
    _render_hall_map(skm_df, all_sorted, show_header=False)
    _render_country_intelligence(skm_df, all_sorted)
    _render_section_header("Run Record", "Run Summary", "A compact record of the current fair run, including source URL and coverage totals.")
    _render_run_summary_panel(result_df)

    tabs = ["SKM Exhibitor Leads", "All Exhibitor Leads"]
    if not review_df.empty:
        tabs.insert(1, "Possible Matches")
    _render_section_header("Lead Tables", "Lead Sheets", "Use the structured tables when you need full-list review, filtering, or export checks.")
    rendered_tabs = st.tabs(tabs)
    tab_skm = rendered_tabs[0]
    tab_all = rendered_tabs[-1]
    with tab_skm:
        _render_section_header("Priority Leads", "SKM Exhibitor Leads", "High-confidence SKM merchants, ready for booth-level follow-up.")
        st.dataframe(
            order_columns(skm_df),
            use_container_width=True,
            hide_index=True,
        )
    if not review_df.empty:
        with rendered_tabs[1]:
            _render_section_header("Review Queue", "Possible Matches", "Lower-confidence matches separated from the main SKM operating list.")
            st.caption("These are lower-confidence fuzzy matches. Use them only as a backup review list.")
            st.dataframe(order_columns(review_df), use_container_width=True, hide_index=True)
    with tab_all:
        _render_section_header("Full Coverage", "All Exhibitor Leads", "The complete fair lead list, ordered for review, filtering, and export.")
        st.dataframe(order_columns(all_sorted), use_container_width=True, hide_index=True)


def _inject_app_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(245, 94, 66, 0.040), transparent 22%),
                radial-gradient(circle at top left, rgba(17, 24, 39, 0.018), transparent 16%),
                linear-gradient(180deg, #fafbfd 0%, #ffffff 22%, #ffffff 100%);
            color: #1f2330;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        .block-container {
            padding-top: 1.2rem;
            max-width: 1280px;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7f8fb 0%, #f2f4f7 100%);
            border-right: 1px solid rgba(25, 28, 38, 0.055);
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.05rem;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stCaption {
            color: #596274;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid rgba(25, 28, 38, 0.065);
            border-radius: 14px;
            padding: 16px 16px 14px;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.035);
        }
        div[data-testid="stMetric"] label {
            color: #6b7280 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #1f2330;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }
        button[kind="secondary"], button[kind="primary"] {
            border-radius: 12px !important;
            min-height: 42px;
            font-weight: 600 !important;
            transition: transform 0.14s ease, box-shadow 0.18s ease, border-color 0.18s ease !important;
        }
        button[kind="primary"] {
            background: linear-gradient(135deg, #f25f45 0%, #e8563e 100%) !important;
            border: 1px solid rgba(214, 78, 54, 0.78) !important;
            box-shadow: 0 12px 24px rgba(242, 95, 69, 0.18) !important;
        }
        button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 28px rgba(242, 95, 69, 0.24) !important;
        }
        button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.98) !important;
            border: 1px solid rgba(25, 28, 38, 0.075) !important;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.03) !important;
        }
        button[kind="secondary"]:hover {
            transform: translateY(-1px);
            border-color: rgba(164, 71, 51, 0.16) !important;
        }
        div[data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(25, 28, 38, 0.06);
            border-radius: 16px;
            padding: 8px;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.024);
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-testid="stNumberInputContainer"] > div {
            border-radius: 12px !important;
            border-color: rgba(25, 28, 38, 0.08) !important;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.02);
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(25, 28, 38, 0.06);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.022);
            background: rgba(255,255,255,0.94);
        }
        div[data-baseweb="tab-list"] {
            gap: 6px;
            padding: 4px;
            background: rgba(247, 248, 250, 0.96);
            border: 1px solid rgba(25, 28, 38, 0.05);
            border-radius: 12px;
        }
        button[role="tab"] {
            border-radius: 9px !important;
            padding: 8px 12px !important;
            transition: all 0.18s ease;
        }
        button[role="tab"][aria-selected="true"] {
            background: #ffffff !important;
            box-shadow: 0 3px 10px rgba(15, 23, 42, 0.06);
        }
        .radar-hero {
            background:
                linear-gradient(135deg, rgba(255,255,255,0.985) 0%, rgba(255,248,244,0.97) 55%, rgba(252,245,242,0.96) 100%);
            border: 1px solid rgba(25, 28, 38, 0.055);
            border-radius: 22px;
            padding: 28px 30px 24px;
            box-shadow: 0 18px 44px rgba(15, 23, 42, 0.05);
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
        }
        .radar-hero::after {
            content: "";
            position: absolute;
            inset: auto -12% -42% auto;
            width: 360px;
            height: 360px;
            background: radial-gradient(circle, rgba(245, 94, 66, 0.09) 0%, rgba(245, 94, 66, 0.0) 72%);
            pointer-events: none;
        }
        .radar-hero::before {
            content: "";
            position: absolute;
            inset: -10% auto auto -6%;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(17, 24, 39, 0.035) 0%, rgba(17, 24, 39, 0) 72%);
            pointer-events: none;
        }
        .radar-eyebrow {
            display: inline-block;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #a44733;
            background: rgba(164, 71, 51, 0.085);
            border-radius: 999px;
            padding: 5px 10px;
            margin-bottom: 14px;
            position: relative;
            z-index: 1;
        }
        .radar-hero h1 {
            margin: 0 0 8px 0;
            color: #171a23;
            font-size: 2.15rem;
            line-height: 1.04;
            position: relative;
            z-index: 1;
        }
        .radar-hero p {
            margin: 0;
            max-width: 820px;
            color: #4e5667;
            font-size: 0.98rem;
            line-height: 1.52;
            position: relative;
            z-index: 1;
        }
        .radar-status-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin: 14px 0 2px 0;
            position: relative;
            z-index: 1;
        }
        .radar-status-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 11px;
            border-radius: 999px;
            background: rgba(255,255,255,0.80);
            border: 1px solid rgba(25, 28, 38, 0.06);
            color: #475063;
            font-size: 0.84rem;
            line-height: 1;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.72);
        }
        .radar-status-chip strong {
            color: #111827;
            font-weight: 700;
        }
        .radar-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin: 16px 0 10px 0;
            position: relative;
            z-index: 1;
        }
        .radar-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(25, 28, 38, 0.055);
            border-radius: 14px;
            padding: 15px 15px 13px;
            min-height: 126px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.7), 0 8px 18px rgba(15,23,42,0.025);
            backdrop-filter: blur(4px);
        }
        .radar-card h3 {
            margin: 0 0 9px 0;
            font-size: 0.96rem;
            color: #1f2330;
        }
        .radar-card p,
        .radar-card li {
            color: #5a6170;
            font-size: 0.94rem;
            line-height: 1.45;
            margin: 0;
        }
        .radar-card ul {
            margin: 0;
            padding-left: 18px;
        }
        .radar-card li + li {
            margin-top: 6px;
        }
        .radar-note {
            margin-top: 8px;
            padding: 12px 14px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.93);
            border: 1px solid rgba(25, 28, 38, 0.06);
            color: #4d5565;
            font-size: 0.9rem;
            line-height: 1.45;
            position: relative;
            z-index: 1;
        }
        .radar-note strong {
            color: #252833;
        }
        .section-header {
            margin: 18px 0 10px 0;
        }
        .section-eyebrow {
            color: #7b818f;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 4px;
        }
        .section-title {
            color: #1f2330;
            font-size: 1.15rem;
            font-weight: 700;
            line-height: 1.25;
        }
        .section-description {
            color: #677081;
            font-size: 0.9rem;
            line-height: 1.45;
            margin-top: 3px;
        }
        .dashboard-note {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid rgba(25, 28, 38, 0.06);
            border-radius: 16px;
            padding: 16px 16px 14px;
            min-height: 104px;
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.035);
        }
        .dashboard-note-title {
            color: #1f2330;
            font-size: 0.96rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .dashboard-note-body {
            color: #5d6575;
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .summary-ribbon {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 10px 0 6px 0;
        }
        .summary-ribbon-card {
            background: rgba(255, 255, 255, 0.98);
            border: 1px solid rgba(25, 28, 38, 0.055);
            border-radius: 16px;
            padding: 14px 15px 12px;
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.03);
        }
        .summary-ribbon-title {
            color: #6b7280;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 8px;
        }
        .summary-ribbon-value {
            color: #1f2330;
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 6px;
        }
        .summary-ribbon-caption {
            color: #5d6575;
            font-size: 0.86rem;
            line-height: 1.35;
        }
        .summary-actions {
            margin: 10px 0 6px 0;
        }
        .focus-country-card {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.99) 0%, rgba(252,247,245,0.98) 100%);
            border: 1px solid rgba(25, 28, 38, 0.06);
            border-radius: 18px;
            padding: 16px 16px 14px;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.04);
            position: relative;
            overflow: hidden;
            min-height: 154px;
        }
        .focus-country-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 4px;
            background: linear-gradient(180deg, #f25f45 0%, rgba(242, 95, 69, 0.22) 100%);
        }
        .focus-country-topline {
            color: #1f2330;
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 12px;
        }
        .focus-country-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 12px;
        }
        .focus-country-value {
            color: #1f2330;
            font-size: 1.42rem;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 4px;
        }
        .focus-country-label {
            color: #6b7280;
            font-size: 0.82rem;
            line-height: 1.25;
        }
        .focus-country-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .focus-country-chip {
            display: inline-flex;
            align-items: center;
            padding: 6px 9px;
            border-radius: 999px;
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(25, 28, 38, 0.06);
            color: #566072;
            font-size: 0.8rem;
            line-height: 1;
        }
        .console-panel {
            background: rgba(255, 255, 255, 0.975);
            border: 1px solid rgba(25, 28, 38, 0.06);
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.04);
        }
        .console-panel-tight {
            padding: 12px 12px 8px;
        }
        .console-toolbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 12px;
        }
        .console-toolbar-title {
            color: #1f2330;
            font-size: 0.98rem;
            font-weight: 700;
        }
        .console-toolbar-caption {
            color: #667085;
            font-size: 0.84rem;
            line-height: 1.4;
        }
        .hall-drilldown-shell {
            margin-top: 12px;
            padding: 16px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.98);
            border: 1px solid rgba(25, 28, 38, 0.06);
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.04);
        }
        .hall-drilldown-topline {
            color: #7b818f;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 6px;
        }
        .hall-drilldown-title {
            color: #1f2330;
            font-size: 1.6rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 8px;
        }
        .hall-drilldown-subtitle {
            color: #5d6575;
            font-size: 0.92rem;
            line-height: 1.45;
            margin-bottom: 12px;
            max-width: 860px;
        }
        .country-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 4px 0 6px 0;
        }
        .country-chip {
            display: inline-flex;
            align-items: center;
            padding: 7px 10px;
            border-radius: 999px;
            background: #f7f8fb;
            border: 1px solid rgba(25, 28, 38, 0.055);
            color: #566072;
            font-size: 0.82rem;
            line-height: 1;
        }
        .country-chip-more {
            background: rgba(245, 94, 66, 0.06);
            color: #a44733;
            border-color: rgba(164, 71, 51, 0.10);
        }
        .hall-priority-strip {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 10px;
            margin: 12px 0 14px 0;
        }
        .hall-priority-card {
            background: rgba(255, 255, 255, 0.98);
            border: 1px solid rgba(25, 28, 38, 0.055);
            border-radius: 14px;
            padding: 12px 12px 10px;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.03);
            position: relative;
            overflow: hidden;
            transition: transform 0.16s ease, box-shadow 0.18s ease;
        }
        .hall-priority-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 3px;
            background: linear-gradient(180deg, #f25f45 0%, #e8563e 100%);
            opacity: 0.9;
        }
        .hall-priority-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 28px rgba(15, 23, 42, 0.05);
        }
        .hall-priority-hall {
            color: #1f2330;
            font-size: 0.88rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 10px;
        }
        .hall-priority-value {
            color: #1f2330;
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 5px;
        }
        .hall-priority-caption {
            color: #667085;
            font-size: 0.78rem;
            line-height: 1.2;
        }
        .booth-card {
            background: rgba(255, 255, 255, 0.985);
            border: 1px solid rgba(25, 28, 38, 0.06);
            border-radius: 16px;
            padding: 14px 14px 12px;
            box-shadow: 0 14px 28px rgba(15, 23, 42, 0.035);
            min-height: 188px;
            transition: transform 0.16s ease, box-shadow 0.18s ease, border-color 0.18s ease;
            position: relative;
            overflow: hidden;
        }
        .booth-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 3px;
            background: linear-gradient(180deg, rgba(242, 95, 69, 0.88) 0%, rgba(242, 95, 69, 0.18) 100%);
            opacity: 0.85;
        }
        .booth-card:hover {
            transform: translateY(-2px);
            border-color: rgba(164, 71, 51, 0.14);
            box-shadow: 0 18px 32px rgba(15, 23, 42, 0.055);
        }
        .booth-card-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 12px;
        }
        .booth-card-top-left {
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .booth-card-badge {
            display: inline-flex;
            align-items: center;
            padding: 5px 9px;
            border-radius: 999px;
            background: rgba(245, 94, 66, 0.08);
            color: #a44733;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .booth-card-badge.lead {
            background: rgba(17, 24, 39, 0.055);
            color: #4b5563;
        }
        .booth-card-index {
            color: #98a2b3;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
        }
        .booth-card-location {
            color: #667085;
            font-size: 0.8rem;
            font-weight: 600;
            text-align: right;
        }
        .booth-card-name {
            color: #1f2330;
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 10px;
        }
        .booth-card-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 12px;
        }
        .booth-card-meta-chip {
            display: inline-flex;
            align-items: center;
            padding: 6px 9px;
            border-radius: 999px;
            background: #f7f8fb;
            border: 1px solid rgba(25, 28, 38, 0.05);
            color: #566072;
            font-size: 0.8rem;
            line-height: 1;
        }
        .booth-card-meta-chip.muted {
            background: rgba(17, 24, 39, 0.04);
            color: #667085;
        }
        .booth-card-links {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .booth-card-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 34px;
            padding: 0 12px;
            border-radius: 10px;
            background: #ffffff;
            border: 1px solid rgba(25, 28, 38, 0.07);
            color: #1f2330 !important;
            font-size: 0.82rem;
            font-weight: 600;
            text-decoration: none !important;
        }
        .booth-card-link:hover {
            border-color: rgba(164, 71, 51, 0.18);
            color: #a44733 !important;
        }
        .lead-card-stack-note {
            color: #7b818f;
            font-size: 0.82rem;
            line-height: 1.4;
            margin-top: 10px;
        }
        .map-select-shell {
            margin-top: 12px;
            padding: 14px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.98);
            border: 1px solid rgba(25, 28, 38, 0.06);
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.03);
        }
        .map-select-title {
            color: #1f2330;
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .map-select-caption {
            color: #667085;
            font-size: 0.84rem;
            line-height: 1.4;
            margin-bottom: 10px;
        }
        .build-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.76);
            border: 1px solid rgba(25, 28, 38, 0.06);
            color: #5f6778;
            font-size: 0.8rem;
        }
        .build-chip strong {
            color: #1f2330;
        }
        .hall-stat-card {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(25, 28, 38, 0.06);
            border-radius: 14px;
            padding: 13px 14px 11px;
            min-height: 90px;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.032);
            margin-bottom: 10px;
        }
        .hall-stat-title {
            color: #6b7280;
            font-size: 0.84rem;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .hall-stat-value {
            color: #1f2330;
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 7px;
        }
        .hall-stat-caption {
            color: #5a6170;
            font-size: 0.85rem;
            line-height: 1.35;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(25, 28, 38, 0.06);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.03);
        }
        @media (max-width: 900px) {
            .radar-grid {
                grid-template-columns: 1fr;
            }
            .summary-ribbon {
                grid-template-columns: 1fr;
            }
            .hall-priority-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .radar-hero {
                padding: 22px 18px;
            }
            .radar-hero h1 {
                font-size: 1.72rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_onboarding(has_builtin_skm: bool) -> None:
    built_in_copy = (
        "The built-in SKM base is already loaded by default, so most runs only need an exhibitor URL."
        if has_builtin_skm
        else "If the built-in SKM base is unavailable, upload an SKM Excel or CSV file first."
    )
    st.markdown(
        f"""
        <section class="radar-hero">
            <div class="radar-eyebrow">Operations Brief</div>
            <h1>TikTok Shop Fair Intel Console</h1>
            <p>
                Track priority merchants across trade fairs, locate them by hall and booth,
                and export a clean operating list for on-site outreach. {built_in_copy}
            </p>
            <div class="radar-status-row">
                <div class="radar-status-chip"><strong>SKM Base</strong> Built in</div>
                <div class="radar-status-chip"><strong>Mode</strong> Fair intelligence</div>
                <div class="radar-status-chip"><strong>Output</strong> Hall and booth leads</div>
            </div>
            <div class="radar-grid">
                <div class="radar-card">
                    <h3>Directory Input</h3>
                    <ul>
                        <li>Paste the exhibitor directory URL from the fair website.</li>
                        <li>Keep the built-in SKM base turned on unless you intentionally want a different list.</li>
                        <li>Use uploaded HTML only as a fallback for JavaScript-heavy pages.</li>
                    </ul>
                </div>
                <div class="radar-card">
                    <h3>Operating Flow</h3>
                    <ul>
                        <li>Open a fair directory URL and click <strong>Scrape and Match</strong>.</li>
                        <li>Review <strong>Hall Map</strong> first to see where SKM density is highest.</li>
                        <li>Download Excel when you need a field-ready lead list by hall.</li>
                    </ul>
                </div>
                <div class="radar-card">
                    <h3>Run Expectations</h3>
                    <ul>
                        <li>Some fairs finish in seconds, others take longer because they split data across many buckets.</li>
                        <li>Please wait patiently while the app scrapes and matches; a longer fair run can still be normal.</li>
                        <li>Use the scrape warnings panel as a quick health check after the run.</li>
                    </ul>
                </div>
            </div>
            <div class="radar-note">
                <strong>Tip:</strong> Let the first run finish before retrying. Successful fair runs are cached,
                so repeat analysis is much faster.
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_app_css()

    with st.sidebar:
        st.header("1. SKM List")
        st.caption("The built-in SKM base is used by default.")
        has_builtin_skm = BUILTIN_SKM_PATH.exists()
        use_builtin_skm = st.checkbox("Use built-in SKM base", value=has_builtin_skm, disabled=not has_builtin_skm)
        skm_upload = None

        skm_df = None
        name_col = ""
        alias_cols: List[str] = []
        threshold = 88
        review_margin = 0

        with st.expander("Advanced SKM and matching settings"):
            threshold = st.slider("Match threshold", min_value=70, max_value=100, value=88, step=1)
            review_margin = st.slider("Manual review range", min_value=0, max_value=20, value=0, step=1)
            if use_builtin_skm:
                st.caption("Optional: turn off built-in SKM above to upload a different SKM file.")
            else:
                skm_upload = st.file_uploader("Upload different SKM Excel/CSV", type=["xlsx", "xls", "csv", "tsv"])

        if use_builtin_skm and has_builtin_skm:
            try:
                skm_df = _read_builtin_skm()
                st.success(f"Loaded built-in SKM base: {len(skm_df)} rows")
                columns = list(skm_df.columns)
                name_col = "SKM Name" if "SKM Name" in columns else columns[0]
                alias_cols = [col for col in columns if col != name_col and col.lower().startswith("alias")]
            except Exception as exc:
                st.error(f"Failed to read built-in SKM base: {exc}")
        elif skm_upload is not None:
            try:
                skm_df = _read_table(skm_upload)
                skm_df.columns = [str(col).strip() for col in skm_df.columns]
                st.success(f"Loaded {len(skm_df)} SKM rows")
                columns = list(skm_df.columns)
                guessed = _column_guess(columns, ["skm", "merchant", "company", "brand", "name", "firma"])
                name_col = st.selectbox("SKM company name column", columns, index=columns.index(guessed))
                alias_cols = st.multiselect(
                    "Alias / brand / shop name columns",
                    [col for col in columns if col != name_col],
                    default=[col for col in columns if col.lower() in {"alias", "aliases", "brand", "shop"}],
                )
            except Exception as exc:
                st.error(f"Failed to read SKM file: {exc}")

        st.header("2. Scraping Settings")
        max_pages = st.number_input("Maximum pages to scrape", min_value=1, max_value=200, value=100, step=1)
        auto_discover_exhibitor_directory = st.checkbox("Auto-detect exhibitor directory from homepage", value=True)
        crawl_detail_pages = st.checkbox("Crawl exhibitor detail pages", value=False)
        detail_page_limit = st.number_input("Detail page limit", min_value=1, max_value=500, value=50, step=10)

        with st.expander("Advanced scraping settings"):
            page_url_template = st.text_input("Pagination URL template", placeholder="https://example.com/exhibitors?page={page}")
            item_selector = st.text_input("Exhibitor card selector", placeholder=".exhibitor-card")
            name_selector = st.text_input("Company name selector", placeholder=".exhibitor-name")
            hall_selector = st.text_input("Hall selector", placeholder=".hall")
            booth_selector = st.text_input("Booth selector", placeholder=".booth")
            country_selector = st.text_input("Country selector", placeholder=".country")
            website_selector = st.text_input("Website selector", placeholder="a.website")
            detail_link_selector = st.text_input("Detail page link selector", placeholder="a.detail")

        st.markdown(
            f'<div class="build-chip"><strong>Build</strong> {APP_BUILD}</div>',
            unsafe_allow_html=True,
        )

    _render_onboarding(has_builtin_skm)

    left, right = st.columns([2, 1])
    with left:
        url = st.text_input("Exhibitor directory URL", placeholder="https://www.example-messe.de/exhibitors")
    with right:
        st.caption("Optional fallback for JavaScript-heavy sites")
        html_upload = st.file_uploader("Upload exhibitor page HTML", type=["html", "htm"])

    can_run = skm_df is not None and bool(name_col) and (bool(url) or html_upload is not None)
    run = st.button("Scrape and Match", type="primary", disabled=not can_run, use_container_width=True)

    if not can_run:
        st.info("Upload an SKM Excel/CSV file, then enter an exhibitor URL or upload exhibitor page HTML.")

    result_state_key = "last_result_df"
    warning_state_key = "last_scrape_warnings"
    signature_state_key = "last_run_signature"

    if run:
        config = ScrapeConfig(
            url=url,
            max_pages=int(max_pages),
            page_url_template=page_url_template.strip(),
            item_selector=item_selector.strip(),
            name_selector=name_selector.strip(),
            hall_selector=hall_selector.strip(),
            booth_selector=booth_selector.strip(),
            country_selector=country_selector.strip(),
            website_selector=website_selector.strip(),
            detail_link_selector=detail_link_selector.strip(),
            crawl_detail_pages=crawl_detail_pages,
            auto_discover_exhibitor_directory=auto_discover_exhibitor_directory,
            detail_page_limit=int(detail_page_limit),
        )
        run_signature = _cache_key(
            [
                url,
                int(max_pages),
                page_url_template,
                item_selector,
                name_selector,
                hall_selector,
                booth_selector,
                country_selector,
                website_selector,
                detail_link_selector,
                crawl_detail_pages,
                auto_discover_exhibitor_directory,
                int(detail_page_limit),
                threshold,
                review_margin,
                name_col,
                alias_cols,
                "builtin_skm" if use_builtin_skm else _file_fingerprint(skm_upload),
                _file_fingerprint(html_upload),
            ]
        )

        try:
            with st.status("Scraping exhibitors...", expanded=True) as status:
                if html_upload is not None:
                    html = _read_html(html_upload)
                    exhibitors = parse_exhibitors_from_html(html, base_url=url, config=config)
                    scrape_warnings = []
                else:
                    exhibitors, scrape_warnings = cached_fetch_html(
                        scraper_build=APP_BUILD,
                        url=url,
                        max_pages=int(max_pages),
                        page_url_template=page_url_template.strip(),
                        item_selector=item_selector.strip(),
                        name_selector=name_selector.strip(),
                        hall_selector=hall_selector.strip(),
                        booth_selector=booth_selector.strip(),
                        country_selector=country_selector.strip(),
                        website_selector=website_selector.strip(),
                        detail_link_selector=detail_link_selector.strip(),
                        crawl_detail_pages=crawl_detail_pages,
                        auto_discover_exhibitor_directory=auto_discover_exhibitor_directory,
                        detail_page_limit=int(detail_page_limit),
                    )
                status.write(f"Found {len(exhibitors)} exhibitor candidates")
                if scrape_warnings:
                    status.write(f"Skipped {len(scrape_warnings)} page(s) after retry. Results may be slightly incomplete.")

                matched = match_exhibitors_to_skm(
                    exhibitors=exhibitors,
                    skm_rows=_safe_records(skm_df),
                    name_col=name_col,
                    alias_cols=alias_cols,
                    threshold=float(threshold),
                    review_margin=float(review_margin),
                )
                status.write("SKM matching complete")
                status.update(label="Complete", state="complete", expanded=False)

            if not matched:
                st.warning("No exhibitors were found. Try uploading page HTML or entering CSS selectors in Advanced settings.")
                return

            result_df = pd.DataFrame(matched)
            st.session_state[result_state_key] = result_df
            st.session_state[warning_state_key] = scrape_warnings
            st.session_state[signature_state_key] = run_signature

        except ScrapeError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)

    if result_state_key in st.session_state:
        scrape_warnings = st.session_state.get(warning_state_key, [])
        if scrape_warnings:
            with st.expander("Scrape warnings"):
                for warning in scrape_warnings[:25]:
                    st.warning(warning)
                if len(scrape_warnings) > 25:
                    st.warning(f"...and {len(scrape_warnings) - 25} more skipped pages.")
        _render_results(st.session_state[result_state_key])


if __name__ == "__main__":
    main()
