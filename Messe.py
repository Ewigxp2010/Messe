import re
import time
from collections import deque
from io import BytesIO
from urllib.parse import urljoin, urlparse, parse_qs

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Exhibition Exhibitor Extractor",
    page_icon="🧾",
    layout="wide"
)

# =========================================================
# Constants
# =========================================================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

DEFAULT_DIRECTORY_KEYWORDS = (
    "exhibitor, exhibitors, aussteller, directory, participants, "
    "brands, vendor, vendors, company list, catalogue, catalog, "
    "attendee, attendees, participant list, exhibitor list"
)

DEFAULT_DETAIL_KEYWORDS = (
    "exhibitor, brand, company, participant, vendor, profile, listing"
)

DEFAULT_BLOCK_WORDS = (
    "about, about us, apply, apply as exhibitor, enquiry, contact, privacy, "
    "cookie, cookies, terms, login, sign in, register, press, news, jobs, "
    "career, faq, impressum, imprint, legal, media, ticket, tickets"
)

DEFAULT_BLOCK_PAGE_KEYWORDS = (
    "sponsor, sponsors, press, news, media, about, contact, apply, "
    "career, jobs, legal, privacy, ticket, tickets"
)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-()/]{6,}\d)")
HALL_RE = re.compile(r"\bH\d+(?:\.\d+)?\b", re.IGNORECASE)
BOOTH_RE = re.compile(r"\b[A-Z]?\d+(?:\.\d+)?-\d+\b", re.IGNORECASE)


# =========================================================
# Utility
# =========================================================
def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def parse_csv_keywords(text: str) -> list[str]:
    parts = [x.strip().lower() for x in (text or "").split(",")]
    return [x for x in parts if x]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def safe_get(url: str, timeout: int = 20):
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text, response.url


def get_soup(url: str):
    html, final_url = safe_get(url)
    return BeautifulSoup(html, "html.parser"), final_url


def same_domain(url1: str, url2: str) -> bool:
    return urlparse(url1).netloc.lower() == urlparse(url2).netloc.lower()


def is_valid_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def contains_any(text: str, keywords: list[str]) -> bool:
    text = (text or "").lower()
    return any(k in text for k in keywords)


def path_depth(url: str) -> int:
    return len([p for p in urlparse(url).path.split("/") if p])


def has_page_query(url: str) -> bool:
    qs = parse_qs(urlparse(url).query)
    return "page" in qs


def strip_url_fragment(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl()


def normalize_url_for_dedup(url: str) -> str:
    return strip_url_fragment(url).rstrip("/").lower()


def extract_emails(text: str) -> str:
    emails = sorted(set(EMAIL_RE.findall(text or "")))
    return ", ".join(emails[:5])


def extract_phones(text: str) -> str:
    phones = sorted(set(PHONE_RE.findall(text or "")))
    return ", ".join(phones[:5])


def extract_hall(text: str) -> str:
    match = HALL_RE.search(text or "")
    return match.group(0) if match else ""


def extract_booth(text: str) -> str:
    match = BOOTH_RE.search(text or "")
    return match.group(0) if match else ""


def make_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="exhibitors")
    return output.getvalue()


# =========================================================
# Name filtering / scoring
# =========================================================
def looks_like_bad_name(name: str, block_words: list[str]) -> bool:
    name = clean_text(name)
    low = name.lower()

    if not name:
        return True
    if len(name) < 2:
        return True
    if len(name) > 140:
        return True
    if low.isdigit():
        return True
    if re.fullmatch(r"[\d\W_]+", low):
        return True
    if contains_any(low, block_words):
        return True

    junk_patterns = [
        "read more",
        "load more",
        "show more",
        "next page",
        "previous page",
        "page ",
        "view all",
        "all exhibitors",
        "all brands",
        "all participants",
        "search result",
    ]
    if any(p in low for p in junk_patterns):
        return True

    if len(name.split()) > 14:
        return True

    return False


def name_quality_score(name: str, block_words: list[str]) -> int:
    name = clean_text(name)
    low = name.lower()

    if looks_like_bad_name(name, block_words):
        return 0

    score = 0

    if 2 <= len(name) <= 80:
        score += 20
    if len(name.split()) <= 6:
        score += 15
    if re.search(r"\b(gmbh|ag|ug|ltd|llc|inc|sarl|sas|bv|nv|oy|spa|srl)\b", low):
        score += 20
    if re.search(r"[A-Za-z]", name):
        score += 10
    if re.search(r"\d", name):
        score -= 8
    if contains_any(low, block_words):
        score -= 50

    return max(score, 0)


# =========================================================
# Page classification
# =========================================================
def score_directory_page(
    url: str,
    soup: BeautifulSoup,
    directory_keywords: list[str],
    block_page_keywords: list[str]
) -> int:
    url_low = url.lower()
    text_sample = clean_text(soup.get_text(" ", strip=True))[:3000].lower()
    score = 0

    if contains_any(url_low, directory_keywords):
        score += 35
    if contains_any(text_sample, directory_keywords):
        score += 15
    if has_page_query(url):
        score += 10

    if contains_any(url_low, block_page_keywords):
        score -= 50

    # 目录页通常会有很多内部链接
    internal_links = 0
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"])
        if is_valid_http_url(href) and same_domain(url, href):
            internal_links += 1

    if internal_links >= 20:
        score += 10
    if internal_links >= 60:
        score += 5

    return score


def score_detail_page(
    url: str,
    soup: BeautifulSoup,
    detail_keywords: list[str],
    block_page_keywords: list[str]
) -> int:
    url_low = url.lower()
    page_text = clean_text(soup.get_text(" ", strip=True))[:3000].lower()
    score = 0

    if contains_any(url_low, block_page_keywords):
        score -= 40

    if contains_any(url_low, detail_keywords):
        score += 25

    if path_depth(url) >= 2:
        score += 20
    if path_depth(url) >= 3:
        score += 10

    h1 = soup.find("h1")
    if h1 and clean_text(h1.get_text(" ", strip=True)):
        score += 15

    if any(k in page_text for k in ["website", "email", "phone", "hall", "booth", "stand"]):
        score += 10

    return score


def classify_page(
    url: str,
    soup: BeautifulSoup,
    directory_keywords: list[str],
    detail_keywords: list[str],
    block_page_keywords: list[str]
) -> str:
    dir_score = score_directory_page(url, soup, directory_keywords, block_page_keywords)
    det_score = score_detail_page(url, soup, detail_keywords, block_page_keywords)

    if dir_score >= max(30, det_score + 5):
        return "directory"
    if det_score >= max(35, dir_score + 5):
        return "detail"
    return "other"


# =========================================================
# Link helpers
# =========================================================
def score_link_candidate(
    anchor_text: str,
    href: str,
    directory_keywords: list[str],
    detail_keywords: list[str],
    block_words: list[str],
    block_page_keywords: list[str]
) -> int:
    value = f"{anchor_text} {href}".lower()
    score = 0

    if contains_any(value, block_words):
        score -= 50
    if contains_any(href.lower(), block_page_keywords):
        score -= 40
    if contains_any(value, detail_keywords):
        score += 25
    if contains_any(value, directory_keywords):
        score += 10

    depth = path_depth(href)
    if depth >= 2:
        score += 10
    if depth >= 3:
        score += 5

    if has_page_query(href):
        score -= 20

    if href.lower().endswith((".pdf", ".jpg", ".jpeg", ".png", ".webp", ".svg", ".zip")):
        score -= 100

    return score


def is_next_page_anchor(anchor_text: str, href: str) -> bool:
    text = clean_text(anchor_text).lower()
    href_low = href.lower()

    next_words = [
        "next", "next page", "weiter", "more", "older", ">", "›", "»"
    ]
    if text in next_words or any(w == text for w in next_words):
        return True

    if "page=" in href_low:
        return True

    return False


# =========================================================
# Detail page enrichment
# =========================================================
def extract_external_website(soup: BeautifulSoup, page_url: str) -> str:
    candidates = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        href_abs = urljoin(page_url, href)

        if not is_valid_http_url(href_abs):
            continue

        if not same_domain(page_url, href_abs):
            text = clean_text(a.get_text(" ", strip=True)).lower()
            score = 0
            if text in {"website", "visit website", "official website", "web", "site"}:
                score += 30
            if "http" in href.lower():
                score += 10
            candidates.append((score, href_abs))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def extract_company_name_from_detail(
    soup: BeautifulSoup,
    fallback_anchor_text: str,
    block_words: list[str]
) -> tuple[str, int]:
    h1 = soup.find("h1")
    if h1:
        name = clean_text(h1.get_text(" ", strip=True))
        if not looks_like_bad_name(name, block_words):
            return name, 90

    if soup.title:
        title_text = clean_text(soup.title.get_text(" ", strip=True))
        title_candidate = title_text.split("|")[0].split(" - ")[0].strip()
        if not looks_like_bad_name(title_candidate, block_words):
            return title_candidate, 80

    meta_og = soup.find("meta", attrs={"property": "og:title"})
    if meta_og and meta_og.get("content"):
        og_candidate = clean_text(meta_og["content"])
        if not looks_like_bad_name(og_candidate, block_words):
            return og_candidate, 75

    fallback_anchor_text = clean_text(fallback_anchor_text)
    if not looks_like_bad_name(fallback_anchor_text, block_words):
        return fallback_anchor_text, 55

    return "", 0


def enrich_from_detail_page(
    detail_url: str,
    anchor_text: str,
    block_words: list[str]
):
    soup, resolved = get_soup(detail_url)
    page_text = clean_text(soup.get_text(" ", strip=True))

    company_name, confidence = extract_company_name_from_detail(
        soup=soup,
        fallback_anchor_text=anchor_text,
        block_words=block_words
    )

    if not company_name:
        return None

    website = extract_external_website(soup, resolved)
    email = extract_emails(page_text)
    phone = extract_phones(page_text)
    hall = extract_hall(page_text)
    booth = extract_booth(page_text)

    return {
        "company_name": company_name,
        "website": website,
        "email": email,
        "phone": phone,
        "hall": hall,
        "booth": booth,
        "detail_link": resolved,
        "source_page": "",
        "confidence_score": confidence,
        "extraction_method": "detail_page"
    }


# =========================================================
# Fallback extraction from list-like pages
# =========================================================
def extract_name_from_list_item(container, block_words: list[str]) -> str:
    preferred_tags = ["h1", "h2", "h3", "h4", "strong", "b", "a", "span", "div"]

    for tag_name in preferred_tags:
        for tag in container.find_all(tag_name):
            text = clean_text(tag.get_text(" ", strip=True))
            if not looks_like_bad_name(text, block_words):
                score = name_quality_score(text, block_words)
                if score >= 25:
                    return text

    fallback = clean_text(container.get_text(" ", strip=True))
    if not looks_like_bad_name(fallback, block_words):
        return fallback[:120]

    return ""


def extract_candidates_without_detail_links(
    page_url: str,
    block_words: list[str]
):
    soup, resolved = get_soup(page_url)
    rows = []

    selectors = [
        "article",
        "li",
        "tr",
        ".card",
        ".item",
        ".listing",
        ".result",
        ".company",
        ".brand",
        ".participant",
        ".exhibitor",
        ".vendor",
    ]

    seen_names = set()

    for sel in selectors:
        for el in soup.select(sel):
            name = extract_name_from_list_item(el, block_words)
            if not name:
                continue

            name_norm = name.strip().lower()
            if name_norm in seen_names:
                continue
            seen_names.add(name_norm)

            raw_text = clean_text(el.get_text(" ", strip=True))
            rows.append({
                "company_name": name,
                "website": "",
                "email": extract_emails(raw_text),
                "phone": extract_phones(raw_text),
                "hall": extract_hall(raw_text),
                "booth": extract_booth(raw_text),
                "detail_link": "",
                "source_page": resolved,
                "confidence_score": 45,
                "extraction_method": "list_page_fallback"
            })

    if not rows:
        return []

    df = pd.DataFrame(rows)
    df["name_norm"] = df["company_name"].str.lower().str.strip()
    df = df.drop_duplicates(subset=["name_norm"], keep="first")
    df = df.drop(columns=["name_norm"])
    return df.to_dict(orient="records")


# =========================================================
# Directory-page traversal
# =========================================================
def discover_initial_directory_pages(
    start_url: str,
    directory_keywords: list[str],
    block_page_keywords: list[str],
    max_candidates: int = 20
):
    soup, final_url = get_soup(start_url)
    candidates = []
    seen = set()

    page_type = classify_page(
        final_url,
        soup,
        directory_keywords=directory_keywords,
        detail_keywords=[],
        block_page_keywords=block_page_keywords
    )

    # 起始页总是纳入，但打分不同
    start_score = 10 if page_type == "directory" else 3
    candidates.append({"url": final_url, "text": "start_page", "score": start_score})
    seen.add(normalize_url_for_dedup(final_url))

    for a in soup.find_all("a", href=True):
        text = clean_text(a.get_text(" ", strip=True))
        href = urljoin(final_url, a["href"])
        href_norm = normalize_url_for_dedup(href)

        if href_norm in seen:
            continue
        seen.add(href_norm)

        if not is_valid_http_url(href):
            continue
        if not same_domain(final_url, href):
            continue

        value = f"{text} {href}".lower()
        score = 0

        if contains_any(value, directory_keywords):
            score += 30
        if has_page_query(href):
            score += 10
        if contains_any(href.lower(), block_page_keywords):
            score -= 40

        if score > 0:
            candidates.append({
                "url": href,
                "text": text,
                "score": score
            })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    unique = []
    urls = set()
    for c in candidates:
        key = normalize_url_for_dedup(c["url"])
        if key not in urls:
            unique.append(c)
            urls.add(key)

    return unique[:max_candidates]


def extract_detail_links_and_next_pages(
    soup: BeautifulSoup,
    current_url: str,
    directory_keywords: list[str],
    detail_keywords: list[str],
    block_words: list[str],
    block_page_keywords: list[str]
):
    detail_links = []
    next_pages = []

    for a in soup.find_all("a", href=True):
        anchor_text = clean_text(a.get_text(" ", strip=True))
        href = urljoin(current_url, a["href"])

        if not is_valid_http_url(href):
            continue
        if not same_domain(current_url, href):
            continue

        href_low = href.lower()

        # 分页链接
        if is_next_page_anchor(anchor_text, href):
            if not contains_any(href_low, block_page_keywords):
                next_pages.append({
                    "url": href,
                    "anchor_text": anchor_text
                })
            continue

        # 候选详情链接
        score = score_link_candidate(
            anchor_text=anchor_text,
            href=href,
            directory_keywords=directory_keywords,
            detail_keywords=detail_keywords,
            block_words=block_words,
            block_page_keywords=block_page_keywords
        )

        if score <= 0:
            continue

        if looks_like_bad_name(anchor_text, block_words):
            # 锚文本差，但路径深，仍可弱保留
            if path_depth(href) < 2:
                continue
            score += 5

        # 不能把明显目录页再当详情页
        if has_page_query(href):
            continue

        detail_links.append({
            "anchor_text": anchor_text,
            "detail_link": href,
            "source_page": current_url,
            "link_score": score
        })

    # 去重
    if detail_links:
        df = pd.DataFrame(detail_links)
        df["u"] = df["detail_link"].map(normalize_url_for_dedup)
        df = df.sort_values("link_score", ascending=False).drop_duplicates("u")
        df = df.drop(columns=["u"])
        detail_links = df.to_dict(orient="records")

    if next_pages:
        df = pd.DataFrame(next_pages)
        df["u"] = df["url"].map(normalize_url_for_dedup)
        df = df.drop_duplicates("u").drop(columns=["u"])
        next_pages = df.to_dict(orient="records")

    return detail_links, next_pages


# =========================================================
# Main scraper
# =========================================================
def scrape_exhibitors_v2(
    start_url: str,
    max_directory_pages: int,
    crawl_detail_pages: bool,
    directory_keywords: list[str],
    detail_keywords: list[str],
    block_words: list[str],
    block_page_keywords: list[str],
    delay_seconds: float = 0.3
):
    logs = []
    all_rows = []
    all_directory_pages = []
    all_detail_links = []

    logs.append("Discovering initial directory page candidates...")
    initial_pages = discover_initial_directory_pages(
        start_url=start_url,
        directory_keywords=directory_keywords,
        block_page_keywords=block_page_keywords,
        max_candidates=max_directory_pages
    )
    logs.append(f"Initial directory candidates found: {len(initial_pages)}")

    queue = deque([x["url"] for x in initial_pages])
    visited_directory_pages = set()
    visited_detail_pages = set()

    # 1) 只遍历 directory page
    while queue and len(visited_directory_pages) < max_directory_pages:
        current_url = queue.popleft()
        current_key = normalize_url_for_dedup(current_url)

        if current_key in visited_directory_pages:
            continue

        try:
            soup, resolved = get_soup(current_url)
            resolved_key = normalize_url_for_dedup(resolved)
            if resolved_key in visited_directory_pages:
                continue

            page_type = classify_page(
                resolved,
                soup,
                directory_keywords=directory_keywords,
                detail_keywords=detail_keywords,
                block_page_keywords=block_page_keywords
            )

            # 只处理目录页
            if page_type != "directory":
                logs.append(f"Skipped non-directory page: {resolved} | type={page_type}")
                continue

            visited_directory_pages.add(resolved_key)
            all_directory_pages.append({
                "url": resolved,
                "page_type": page_type
            })
            logs.append(f"Scanning directory page: {resolved}")

            detail_links, next_pages = extract_detail_links_and_next_pages(
                soup=soup,
                current_url=resolved,
                directory_keywords=directory_keywords,
                detail_keywords=detail_keywords,
                block_words=block_words,
                block_page_keywords=block_page_keywords
            )

            logs.append(f"  Detail-link candidates found: {len(detail_links)}")
            logs.append(f"  Next-page candidates found: {len(next_pages)}")

            if detail_links:
                all_detail_links.extend(detail_links)
            else:
                fallback_rows = extract_candidates_without_detail_links(
                    page_url=resolved,
                    block_words=block_words
                )
                logs.append(f"  Fallback list-page entities found: {len(fallback_rows)}")
                all_rows.extend(fallback_rows)

            # 只把 next page 加入队列，绝不把 detail page 加入目录队列
            for n in next_pages:
                next_url = n["url"]
                next_key = normalize_url_for_dedup(next_url)
                if next_key not in visited_directory_pages:
                    queue.append(next_url)

        except Exception as e:
            logs.append(f"Failed directory page: {current_url} | {e}")

        time.sleep(delay_seconds)

    # 2) 去重 detail links
    if all_detail_links:
        df_links = pd.DataFrame(all_detail_links)
        df_links["u"] = df_links["detail_link"].map(normalize_url_for_dedup)
        df_links = df_links.sort_values("link_score", ascending=False).drop_duplicates("u")
        df_links = df_links.drop(columns=["u"])
        all_detail_links = df_links.to_dict(orient="records")

    logs.append(f"Unique directory pages scanned: {len(all_directory_pages)}")
    logs.append(f"Unique detail links collected: {len(all_detail_links)}")

    # 3) 详情页补全
    if crawl_detail_pages and all_detail_links:
        logs.append("Crawling detail pages for enrichment...")

        for i, link_row in enumerate(all_detail_links, start=1):
            detail_url = link_row["detail_link"]
            detail_key = normalize_url_for_dedup(detail_url)

            if detail_key in visited_detail_pages:
                continue

            try:
                soup, resolved = get_soup(detail_url)

                page_type = classify_page(
                    resolved,
                    soup,
                    directory_keywords=directory_keywords,
                    detail_keywords=detail_keywords,
                    block_page_keywords=block_page_keywords
                )

                # 详情页阶段只接受 detail page
                if page_type != "detail":
                    logs.append(f"  Skipped non-detail page: {resolved} | type={page_type}")
                    visited_detail_pages.add(detail_key)
                    time.sleep(delay_seconds)
                    continue

                # enrich_from_detail_page 里再次请求太浪费，所以这里直接内联提取
                page_text = clean_text(soup.get_text(" ", strip=True))
                company_name, confidence = extract_company_name_from_detail(
                    soup=soup,
                    fallback_anchor_text=link_row.get("anchor_text", ""),
                    block_words=block_words
                )

                if company_name:
                    row = {
                        "company_name": company_name,
                        "website": extract_external_website(soup, resolved),
                        "email": extract_emails(page_text),
                        "phone": extract_phones(page_text),
                        "hall": extract_hall(page_text),
                        "booth": extract_booth(page_text),
                        "detail_link": resolved,
                        "source_page": link_row.get("source_page", ""),
                        "confidence_score": min(
                            100,
                            int(confidence + max(0, link_row.get("link_score", 0) // 5))
                        ),
                        "extraction_method": "detail_page"
                    }
                    all_rows.append(row)

                visited_detail_pages.add(normalize_url_for_dedup(resolved))

                if i % 20 == 0:
                    logs.append(f"  Processed detail pages: {i}/{len(all_detail_links)}")

            except Exception as e:
                logs.append(f"  Failed detail page: {detail_url} | {e}")

            time.sleep(delay_seconds)

    elif all_detail_links:
        logs.append("Detail crawling disabled. Building records from link text only.")
        for link_row in all_detail_links:
            anchor_text = clean_text(link_row.get("anchor_text", ""))
            if looks_like_bad_name(anchor_text, block_words):
                continue

            all_rows.append({
                "company_name": anchor_text,
                "website": "",
                "email": "",
                "phone": "",
                "hall": "",
                "booth": "",
                "detail_link": link_row["detail_link"],
                "source_page": link_row.get("source_page", ""),
                "confidence_score": min(80, 40 + link_row.get("link_score", 0)),
                "extraction_method": "link_text_only"
            })

    # 4) 最终清洗
    if not all_rows:
        return (
            pd.DataFrame(),
            logs,
            pd.DataFrame(all_directory_pages),
            pd.DataFrame(all_detail_links)
        )

    df = pd.DataFrame(all_rows)
    df["company_name"] = df["company_name"].fillna("").map(clean_text)
    df = df[df["company_name"] != ""]
    df = df[~df["company_name"].map(lambda x: looks_like_bad_name(x, block_words))]

    df["name_norm"] = df["company_name"].str.lower().str.strip()
    df = df.sort_values(["confidence_score"], ascending=False)
    df = df.drop_duplicates(subset=["name_norm"], keep="first")
    df = df.drop(columns=["name_norm"])

    desired_cols = [
        "company_name",
        "website",
        "email",
        "phone",
        "hall",
        "booth",
        "detail_link",
        "source_page",
        "confidence_score",
        "extraction_method"
    ]
    for col in desired_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[desired_cols].sort_values(
        by=["confidence_score", "company_name"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return (
        df,
        logs,
        pd.DataFrame(all_directory_pages),
        pd.DataFrame(all_detail_links)
    )


# =========================================================
# UI
# =========================================================
st.title("🧾 Exhibition Exhibitor Extractor V2")
st.caption("Generic exhibitor extractor with page classification, pagination tracking, and detail-page enrichment.")

with st.expander("How this version works", expanded=False):
    st.markdown(
        """
This version uses a stricter workflow:

1. Discover likely **directory pages**
2. Traverse only **directory pages**
3. Follow only **pagination links** during directory traversal
4. Collect **detail-page candidates**
5. Visit only **detail pages** for structured enrichment
6. Clean, deduplicate, and score final exhibitor records

This avoids the previous issue where detail pages were scanned like list pages.
        """
    )

with st.form("extract_form"):
    st.subheader("Input")

    website = st.text_input(
        "Event website URL",
        placeholder="https://www.example-expo.com"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        max_directory_pages = st.number_input(
            "Max directory pages to scan",
            min_value=1,
            max_value=100,
            value=20,
            step=1
        )
    with col2:
        crawl_detail_pages = st.checkbox(
            "Crawl detail pages for enrichment",
            value=True
        )
    with col3:
        delay_seconds = st.number_input(
            "Delay per request (seconds)",
            min_value=0.0,
            max_value=3.0,
            value=0.3,
            step=0.1
        )

    st.subheader("Keyword config")

    directory_keywords_text = st.text_area(
        "Directory keywords (comma separated)",
        value=DEFAULT_DIRECTORY_KEYWORDS,
        height=100
    )

    detail_keywords_text = st.text_area(
        "Detail-page keywords (comma separated)",
        value=DEFAULT_DETAIL_KEYWORDS,
        height=100
    )

    block_words_text = st.text_area(
        "Block words for names/text (comma separated)",
        value=DEFAULT_BLOCK_WORDS,
        height=120
    )

    block_page_keywords_text = st.text_area(
        "Block page keywords for URLs/pages (comma separated)",
        value=DEFAULT_BLOCK_PAGE_KEYWORDS,
        height=100
    )

    submitted = st.form_submit_button("Extract exhibitor list", use_container_width=True)

if submitted:
    if not website.strip():
        st.error("Please enter a website URL.")
    else:
        try:
            start_url = normalize_url(website)
            directory_keywords = parse_csv_keywords(directory_keywords_text)
            detail_keywords = parse_csv_keywords(detail_keywords_text)
            block_words = parse_csv_keywords(block_words_text)
            block_page_keywords = parse_csv_keywords(block_page_keywords_text)

            with st.spinner("Scanning website and extracting exhibitors..."):
                df, logs, directory_pages_df, detail_links_df = scrape_exhibitors_v2(
                    start_url=start_url,
                    max_directory_pages=int(max_directory_pages),
                    crawl_detail_pages=crawl_detail_pages,
                    directory_keywords=directory_keywords,
                    detail_keywords=detail_keywords,
                    block_words=block_words,
                    block_page_keywords=block_page_keywords,
                    delay_seconds=float(delay_seconds)
                )

            st.subheader("Directory pages actually scanned")
            if not directory_pages_df.empty:
                st.dataframe(directory_pages_df, use_container_width=True, height=250)
            else:
                st.info("No directory pages were scanned.")

            st.subheader("Collected detail-link candidates")
            if not detail_links_df.empty:
                st.dataframe(detail_links_df.head(300), use_container_width=True, height=250)
            else:
                st.info("No detail-link candidates were collected.")

            st.subheader("Extraction result")
            if df.empty:
                st.warning("No exhibitor records found.")
            else:
                st.success(f"Extracted {len(df)} exhibitor records.")
                st.dataframe(df, use_container_width=True, height=520)

                csv_data = df.to_csv(index=False).encode("utf-8-sig")
                excel_data = make_excel_bytes(df)

                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "Download CSV",
                        data=csv_data,
                        file_name="exhibitors.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with c2:
                    st.download_button(
                        "Download Excel",
                        data=excel_data,
                        file_name="exhibitors.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

            with st.expander("Run log", expanded=False):
                for line in logs:
                    st.write("-", line)

        except Exception as e:
            st.error(f"Extraction failed: {e}")

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Install")
st.sidebar.code("pip install streamlit requests pandas beautifulsoup4 openpyxl")

st.sidebar.header("Run")
st.sidebar.code("streamlit run app.py")

st.sidebar.header("What changed in V2")
st.sidebar.markdown(
    """
- Directory page vs detail page classification
- Pagination tracking
- Detail pages are no longer scanned as list pages
- Page-level blocking keywords
- Cleaner traversal logic for multi-exhibition use
"""
)
