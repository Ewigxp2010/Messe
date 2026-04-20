from __future__ import annotations

import json
import hashlib
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

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
    for row in rows:
        summary["total"] += 1
        status = row.get("match_status")
        if status == "SKM Match":
            summary["skm_matches"] += 1
        elif status == "Needs Review":
            summary["review"] += 1
        else:
            summary["unmatched"] += 1
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
    urls = _build_page_urls(config)
    all_rows: List[Dict[str, Any]] = []

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
                html = fetch_html_with_retry(url, config)

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
    if config.crawl_detail_pages:
        rows = enrich_from_detail_pages(rows, config)
    return rows


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
        if "explore all exhibitors" in text or "exhibitors 2025" in text:
            score += 8
        if parsed.path.rstrip("/").lower() == "/exhibitors":
            score += 6
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
    if re.search(r"(privacy|cookie|terms|login|newsletter|ticket|program|agenda|speaker)", lower):
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
    method_rank = {"custom_selector": 5, "brand_card": 5, "json": 4, "table": 4, "card": 3, "link": 1}

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


def build_excel_download(all_df: pd.DataFrame) -> bytes:
    output = BytesIO()
    skm_df = sort_leads_by_hall(skm_leads(all_df))
    review_df = sort_leads_by_hall(review_leads(all_df))
    all_sorted = sort_leads_by_hall(all_df)
    summary_df = hall_summary(skm_df)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="SKM by Hall", index=False)
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
    page_title="TikTok Shop SKM Exhibition Radar",
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


def _render_downloads(result_df: pd.DataFrame) -> None:
    ordered = order_columns(sort_leads_by_hall(result_df))
    csv_bytes = ordered.to_csv(index=False).encode("utf-8-sig")
    excel_bytes = build_excel_download(ordered)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download Excel Leads",
            data=excel_bytes,
            file_name="tiktok_shop_skm_exhibition_matches.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download All Exhibitor CSV",
            data=csv_bytes,
            file_name="all_exhibitor_leads.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _render_hall_map(skm_df: pd.DataFrame) -> None:
    st.subheader("SKM Hall Heatmap")
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

    selected_hall = st.selectbox("Select a hall to view SKM merchant details", halls)
    normalized_hall = skm_df["hall"].fillna("").replace("", "Unknown Hall")
    hall_rows = skm_df[normalized_hall == selected_hall]
    st.markdown(f"**{selected_hall}: {len(hall_rows)} SKM lead(s)**")
    st.dataframe(order_columns(sort_leads_by_hall(hall_rows)), use_container_width=True, hide_index=True)


def _render_results(result_df: pd.DataFrame) -> None:
    summary = summarize_matches(_safe_records(result_df))
    metric_cols = st.columns(3)
    metric_cols[0].metric("Total Exhibitors", summary["total"])
    metric_cols[1].metric("SKM Exhibitor Leads", summary["skm_matches"])
    metric_cols[2].metric("Needs Review", summary["review"])

    _render_downloads(result_df)

    skm_df = sort_leads_by_hall(skm_leads(result_df))
    review_df = sort_leads_by_hall(review_leads(result_df))

    tabs = ["Hall Map", "SKM Exhibitor Leads", "All Exhibitor Leads"]
    if not review_df.empty:
        tabs.insert(2, "Possible Matches")
    rendered_tabs = st.tabs(tabs)
    tab_map = rendered_tabs[0]
    tab_skm = rendered_tabs[1]
    tab_all = rendered_tabs[-1]
    with tab_map:
        _render_hall_map(skm_df)
    with tab_skm:
        st.dataframe(
            order_columns(skm_df),
            use_container_width=True,
            hide_index=True,
        )
    if not review_df.empty:
        with rendered_tabs[2]:
            st.caption("These are lower-confidence fuzzy matches. Use them only as a backup review list.")
            st.dataframe(order_columns(review_df), use_container_width=True, hide_index=True)
    with tab_all:
        st.dataframe(order_columns(sort_leads_by_hall(result_df)), use_container_width=True, hide_index=True)


def main() -> None:
    st.title("TikTok Shop SKM Exhibition Radar")
    st.caption(
        "Upload your Strategic Key Merchant list, enter an exhibitor directory URL, "
        "and identify priority merchants with hall and booth information."
    )

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
