from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:
    fuzz = None


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
    review_threshold = max(0.0, threshold - review_margin)
    output: List[Dict[str, Any]] = []

    for exhibitor in exhibitors:
        row = dict(exhibitor)
        exhibitor_name = str(row.get("exhibitor_name", "")).strip()
        candidate, score = _best_candidate(exhibitor_name, candidates)

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


@dataclass
class ScrapeConfig:
    url: str
    max_pages: int = 1
    page_url_template: str = ""
    item_selector: str = ""
    name_selector: str = ""
    hall_selector: str = ""
    booth_selector: str = ""
    country_selector: str = ""
    website_selector: str = ""
    detail_link_selector: str = ""
    crawl_detail_pages: bool = False
    detail_page_limit: int = 50
    request_delay_seconds: float = 0.25
    timeout_seconds: int = 25
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )


class ScrapeError(RuntimeError):
    pass


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
    headers = {"User-Agent": cfg.user_agent, "Accept-Language": "de-DE,de;q=0.9,en;q=0.8"}
    response = requests.get(url, headers=headers, timeout=cfg.timeout_seconds)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def parse_exhibitors_from_html(html: str, base_url: str = "", config: Optional[ScrapeConfig] = None) -> List[Dict[str, Any]]:
    BeautifulSoup = _require_bs4()
    soup = BeautifulSoup(html, "html.parser")
    cfg = config or ScrapeConfig(url=base_url)

    all_rows: List[Dict[str, Any]] = []
    if cfg.item_selector:
        all_rows.extend(_extract_with_custom_selectors(soup, base_url, cfg))

    all_rows.extend(_extract_from_json_scripts(soup, base_url))
    all_rows.extend(_extract_from_tables(soup, base_url))
    all_rows.extend(_extract_from_cards(soup, base_url))

    if not all_rows:
        all_rows.extend(_extract_from_links(soup, base_url))

    return _dedupe_exhibitors(all_rows)


def scrape_exhibitors(config: ScrapeConfig) -> List[Dict[str, Any]]:
    urls = _build_page_urls(config)
    all_rows: List[Dict[str, Any]] = []

    for index, url in enumerate(urls):
        html = fetch_html(url, config)
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
                html = fetch_html(detail_url, config)
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
    return ""


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
        "div[class*='brand']",
        "div[class*='result']",
        "div[class*='card']",
        "div[class*='teaser']",
        "div[id*='exhibitor']",
        "div[id*='aussteller']",
    ]

    seen_nodes = set()
    for selector in selectors:
        for item in soup.select(selector):
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
    method_rank = {"custom_selector": 5, "json": 4, "table": 4, "card": 3, "link": 1}

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
    "match_score",
    "skm_name",
    "skm_compare_name",
    "skm_compare_type",
    "exhibitor_name",
    "hall",
    "booth",
    "country",
    "website",
    "detail_url",
    "source_url",
    "extraction_method",
]


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    first = [col for col in EXPORT_COLUMNS if col in df.columns]
    remaining = [col for col in df.columns if col not in first and col != "raw_text"]
    tail = ["raw_text"] if "raw_text" in df.columns else []
    return df[first + remaining + tail]


def build_excel_download(all_df: pd.DataFrame) -> bytes:
    output = BytesIO()
    matches = all_df[all_df["match_status"] == "SKM Match"].copy() if "match_status" in all_df.columns else all_df.head(0)
    review = all_df[all_df["match_status"] == "Needs Review"].copy() if "match_status" in all_df.columns else all_df.head(0)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        order_columns(matches).to_excel(writer, sheet_name="SKM Matches", index=False)
        order_columns(review).to_excel(writer, sheet_name="Needs Review", index=False)
        order_columns(all_df).to_excel(writer, sheet_name="All Exhibitors", index=False)
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
    bad = 0
    for col in columns:
        clean = str(col).strip()
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


    return _read_table_source(path, path.name)


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


def _render_downloads(result_df: pd.DataFrame) -> None:
    ordered = order_columns(result_df)
    csv_bytes = ordered.to_csv(index=False).encode("utf-8-sig")
    excel_bytes = build_excel_download(ordered)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download Excel Lead List",
            data=excel_bytes,
            file_name="tiktok_shop_skm_exhibition_matches.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download Full CSV",
            data=csv_bytes,
            file_name="tiktok_shop_skm_exhibition_matches.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _render_results(result_df: pd.DataFrame) -> None:
    summary = summarize_matches(_safe_records(result_df))
    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Exhibitors", summary["total"])
    metric_cols[1].metric("SKM Matches", summary["skm_matches"])
    metric_cols[2].metric("Needs Review", summary["review"])
    metric_cols[3].metric("No Match", summary["unmatched"])

    _render_downloads(result_df)

    matches_df = result_df[result_df["match_status"] == "SKM Match"].copy()
    review_df = result_df[result_df["match_status"] == "Needs Review"].copy()

    tab_matches, tab_review, tab_all = st.tabs(["SKM Matches", "Needs Review", "All Exhibitors"])
    with tab_matches:
        st.dataframe(
            order_columns(matches_df).sort_values(["hall", "match_score"], ascending=[True, False])
            if not matches_df.empty
            else matches_df,
            use_container_width=True,
            hide_index=True,
        )
    with tab_review:
        st.dataframe(
            order_columns(review_df).sort_values("match_score", ascending=False) if not review_df.empty else review_df,
            use_container_width=True,
            hide_index=True,
        )
    with tab_all:
        st.dataframe(order_columns(result_df), use_container_width=True, hide_index=True)


def main() -> None:
    st.title("TikTok Shop SKM Exhibition Radar")
    st.caption(
        "Upload your Strategic Key Merchant list, enter an exhibitor directory URL, "
        "and identify priority merchants with hall and booth information."
    )

    with st.sidebar:
        st.header("1. SKM List")
        st.caption("Upload your SKM list here. Excel and CSV files are supported.")
        skm_upload = st.file_uploader("Upload SKM Excel/CSV", type=["xlsx", "xls", "csv", "tsv"])

        skm_df = None
        name_col = ""
        alias_cols: List[str] = []
        threshold = st.slider("Match threshold", min_value=70, max_value=100, value=88, step=1)
        review_margin = st.slider("Manual review range", min_value=0, max_value=20, value=8, step=1)

        if skm_upload is not None:
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
        max_pages = st.number_input("Maximum pages to scrape", min_value=1, max_value=50, value=1, step=1)
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
            detail_page_limit=int(detail_page_limit),
        )

        try:
            with st.status("Scraping exhibitors...", expanded=True) as status:
                if html_upload is not None:
                    html = _read_html(html_upload)
                    exhibitors = parse_exhibitors_from_html(html, base_url=url, config=config)
                else:
                    exhibitors = scrape_exhibitors(config)
                status.write(f"Found {len(exhibitors)} exhibitor candidates")

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
            _render_results(result_df)

        except ScrapeError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)


if __name__ == "__main__":
    main()
