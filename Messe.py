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
    page_title="Exhibition Exhibitor Extractor V3",
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


@st.cache_data(show_spinner=False, ttl=3600)
def safe_get(url: str, timeout: int = 15):
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
# Name / scoring
# =========================================================
def looks_like_bad_name(name: str, block_words: list[str]) -> bool:
    name = clean_text(name)
    low = name.lower()

    if not name:
        return True
    if len(name) < 2 or len(name) > 140:
        return True
    if low.isdigit():
        return True
    if re.fullmatch(r"[\d\W_]+", low):
        return True
    if contains_any(low, block_words):
        return True

    junk_patterns = [
        "read more", "load more", "show more", "next page", "previous page",
        "page ", "view all", "all exhibitors", "all brands", "all participants",
        "search result"
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

    return max(score, 0)


# =========================================================
# Page classification
# =========================================================
def score_directory_page(url: str, soup: BeautifulSoup, directory_keywords: list[str], block_page_keywords: list[str]) -> int:
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


def score_detail_page(url: str, soup: BeautifulSoup, detail_keywords: list[str], block_page_keywords: list[str]) -> int:
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


def classify_page(url: str, soup: BeautifulSoup, directory_keywords: list[str], detail_keywords: list[str], block_page_keywords: list[str]) -> str:
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
def score_link_candidate(anchor_text: str, href: str, directory_keywords: list[str], detail_keywords: list[str], block_words: list[str], block_page_keywords: list[str]) -> int:
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

    next_words = ["next", "next page", "weiter", "more", "older", ">", "›", "»"]
    if text in next_words:
        return True
    if "page=" in href_low:
        return True
    return False


# =========================================================
# Detail-page enrichment
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


def extract_company_name_from_detail(soup: BeautifulSoup, fallback_anchor_text: str, block_words: list[str]) -> tuple[str, int]:
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


# =========================================================
# Discovery
# =========================================================
def discover_initial_directory_pages(start_url: str, directory_keywords: list[str], block_page_keywords: list[str], max_candidates: int = 15):
    soup, final_url = get_soup(start_url)
    candidates = []
    seen = set()

    page_type = classify_page(
        final_url, soup,
        directory_keywords=directory_keywords,
        detail_keywords=[],
        block_page_keywords=block_page_keywords
    )

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
            candidates.append({"url": href, "text": text, "score": score})

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

        if is_next_page_anchor(anchor_text, href):
            if not contains_any(href_low, block_page_keywords):
                next_pages.append({"url": href, "anchor_text": anchor_text})
            continue

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
            if path_depth(href) < 2:
                continue
            score += 5

        if has_page_query(href):
            continue

        detail_links.append({
            "anchor_text": anchor_text,
            "detail_link": href,
            "source_page": current_url,
            "link_score": score
        })

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
# Fast scan
# =========================================================
def fast_scan(
    start_url: str,
    max_directory_pages: int,
    directory_keywords: list[str],
    detail_keywords: list[str],
    block_words: list[str],
    block_page_keywords: list[str],
    delay_seconds: float = 0.1
):
    logs = []
    all_directory_pages = []
    all_detail_links = []

    initial_pages = discover_initial_directory_pages(
        start_url=start_url,
        directory_keywords=directory_keywords,
        block_page_keywords=block_page_keywords,
        max_candidates=min(max_directory_pages, 15)
    )

    logs.append(f"Initial directory candidates found: {len(initial_pages)}")

    queue = deque([x["url"] for x in initial_pages])
    visited_directory_pages = set()

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

            if page_type != "directory":
                logs.append(f"Skipped non-directory page: {resolved} | type={page_type}")
                continue

            visited_directory_pages.add(resolved_key)
            all_directory_pages.append({"url": resolved, "page_type": page_type})
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

            all_detail_links.extend(detail_links)

            for n in next_pages:
                next_url = n["url"]
                next_key = normalize_url_for_dedup(next_url)
                if next_key not in visited_directory_pages:
                    queue.append(next_url)

        except Exception as e:
            logs.append(f"Failed directory page: {current_url} | {e}")

        time.sleep(delay_seconds)

    if all_detail_links:
        df_links = pd.DataFrame(all_detail_links)
        df_links["u"] = df_links["detail_link"].map(normalize_url_for_dedup)
        df_links = df_links.sort_values("link_score", ascending=False).drop_duplicates("u")
        df_links = df_links.drop(columns=["u"])
        all_detail_links = df_links.to_dict(orient="records")

    rows = []
    for link_row in all_detail_links:
        anchor_text = clean_text(link_row.get("anchor_text", ""))
        if looks_like_bad_name(anchor_text, block_words):
            continue

        rows.append({
            "company_name": anchor_text,
            "website": "",
            "email": "",
            "phone": "",
            "hall": "",
            "booth": "",
            "detail_link": link_row["detail_link"],
            "source_page": link_row.get("source_page", ""),
            "confidence_score": min(80, 40 + link_row.get("link_score", 0)),
            "extraction_method": "fast_scan_link_text"
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    if not df.empty:
        df["company_name"] = df["company_name"].fillna("").map(clean_text)
        df = df[df["company_name"] != ""]
        df = df[~df["company_name"].map(lambda x: looks_like_bad_name(x, block_words))]
        df["name_norm"] = df["company_name"].str.lower().str.strip()
        df = df.sort_values(["confidence_score"], ascending=False).drop_duplicates("name_norm")
        df = df.drop(columns=["name_norm"])
        df = df.sort_values(["confidence_score", "company_name"], ascending=[False, True]).reset_index(drop=True)

    logs.append(f"Unique directory pages scanned: {len(all_directory_pages)}")
    logs.append(f"Unique detail links collected: {len(all_detail_links)}")
    logs.append(f"Fast-scan exhibitor rows: {len(df) if not df.empty else 0}")

    return df, logs, pd.DataFrame(all_directory_pages), pd.DataFrame(all_detail_links)


# =========================================================
# Enrich top N
# =========================================================
def enrich_top_n(
    base_df: pd.DataFrame,
    detail_links_df: pd.DataFrame,
    block_words: list[str],
    directory_keywords: list[str],
    detail_keywords: list[str],
    block_page_keywords: list[str],
    max_detail_pages: int = 50,
    delay_seconds: float = 0.1
):
    logs = []
    if base_df.empty or detail_links_df.empty:
        return base_df.copy(), ["Nothing to enrich."]

    link_map = {}
    for _, row in detail_links_df.iterrows():
        link_map[normalize_url_for_dedup(row["detail_link"])] = row.to_dict()

    enriched_rows = []
    count = 0

    work_df = base_df.copy()
    work_df = work_df.sort_values(["confidence_score"], ascending=False).reset_index(drop=True)

    progress = st.progress(0.0, text="Enriching detail pages...")

    total_candidates = min(len(work_df), max_detail_pages)
    processed = 0

    for _, row in work_df.iterrows():
        if count >= max_detail_pages:
            break

        detail_link = str(row.get("detail_link", "")).strip()
        if not detail_link:
            enriched_rows.append(row.to_dict())
            processed += 1
            continue

        try:
            soup, resolved = get_soup(detail_link)

            page_type = classify_page(
                resolved,
                soup,
                directory_keywords=directory_keywords,
                detail_keywords=detail_keywords,
                block_page_keywords=block_page_keywords
            )

            if page_type != "detail":
                enriched_rows.append(row.to_dict())
                logs.append(f"Skipped non-detail page: {resolved} | type={page_type}")
            else:
                page_text = clean_text(soup.get_text(" ", strip=True))
                company_name, confidence = extract_company_name_from_detail(
                    soup=soup,
                    fallback_anchor_text=str(row.get("company_name", "")),
                    block_words=block_words
                )

                new_row = row.to_dict()
                if company_name:
                    new_row["company_name"] = company_name
                new_row["website"] = extract_external_website(soup, resolved) or new_row.get("website", "")
                new_row["email"] = extract_emails(page_text) or new_row.get("email", "")
                new_row["phone"] = extract_phones(page_text) or new_row.get("phone", "")
                new_row["hall"] = extract_hall(page_text) or new_row.get("hall", "")
                new_row["booth"] = extract_booth(page_text) or new_row.get("booth", "")
                new_row["detail_link"] = resolved
                new_row["confidence_score"] = min(100, max(int(new_row.get("confidence_score", 0)), confidence))
                new_row["extraction_method"] = "detail_page_enriched"
                enriched_rows.append(new_row)
                count += 1

        except Exception as e:
            enriched_rows.append(row.to_dict())
            logs.append(f"Failed detail page: {detail_link} | {e}")

        processed += 1
        progress.progress(min(processed / total_candidates, 1.0), text=f"Enriching detail pages... {processed}/{total_candidates}")
        time.sleep(delay_seconds)

        if processed >= total_candidates:
            break

    # 把未处理部分直接补回
    if processed < len(work_df):
        remainder = work_df.iloc[processed:].to_dict(orient="records")
        enriched_rows.extend(remainder)

    progress.empty()

    df = pd.DataFrame(enriched_rows)
    if not df.empty:
        df["company_name"] = df["company_name"].fillna("").map(clean_text)
        df = df[df["company_name"] != ""]
        df = df[~df["company_name"].map(lambda x: looks_like_bad_name(x, block_words))]
        df["name_norm"] = df["company_name"].str.lower().str.strip()
        df = df.sort_values(["confidence_score"], ascending=False).drop_duplicates("name_norm")
        df = df.drop(columns=["name_norm"])
        df = df.sort_values(["confidence_score", "company_name"], ascending=[False, True]).reset_index(drop=True)

    logs.append(f"Detail pages enriched: {count}")
    return df, logs


# =========================================================
# UI
# =========================================================
st.title("🧾 Exhibition Exhibitor Extractor V3")
st.caption("Fast first-pass scan + optional top-N detail enrichment.")

with st.expander("How V3 works", expanded=False):
    st.markdown(
        """
V3 uses a faster two-step workflow:

1. **Fast Scan**  
   Quickly scans directory pages and builds a base exhibitor list from collected detail links.

2. **Enrich Top N**  
   Optionally visits only the top N detail pages to enrich website, email, phone, hall, and booth fields.

This avoids waiting a long time before seeing any result.
        """
    )

with st.form("extract_form"):
    st.subheader("Input")

    website = st.text_input("Event website URL", placeholder="https://www.example-expo.com")

    c1, c2, c3 = st.columns(3)
    with c1:
        max_directory_pages = st.number_input("Max directory pages to scan", min_value=1, max_value=100, value=10, step=1)
    with c2:
        max_detail_pages = st.number_input("Max detail pages to enrich", min_value=0, max_value=500, value=50, step=10)
    with c3:
        delay_seconds = st.number_input("Delay per request (seconds)", min_value=0.0, max_value=3.0, value=0.1, step=0.1)

    directory_keywords_text = st.text_area("Directory keywords (comma separated)", value=DEFAULT_DIRECTORY_KEYWORDS, height=90)
    detail_keywords_text = st.text_area("Detail-page keywords (comma separated)", value=DEFAULT_DETAIL_KEYWORDS, height=90)
    block_words_text = st.text_area("Block words for names/text (comma separated)", value=DEFAULT_BLOCK_WORDS, height=110)
    block_page_keywords_text = st.text_area("Block page keywords for URLs/pages (comma separated)", value=DEFAULT_BLOCK_PAGE_KEYWORDS, height=90)

    submitted = st.form_submit_button("Run Fast Scan", use_container_width=True)

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

            with st.spinner("Running fast scan..."):
                base_df, logs, directory_pages_df, detail_links_df = fast_scan(
                    start_url=start_url,
                    max_directory_pages=int(max_directory_pages),
                    directory_keywords=directory_keywords,
                    detail_keywords=detail_keywords,
                    block_words=block_words,
                    block_page_keywords=block_page_keywords,
                    delay_seconds=float(delay_seconds)
                )

            st.session_state["base_df"] = base_df
            st.session_state["detail_links_df"] = detail_links_df
            st.session_state["directory_pages_df"] = directory_pages_df
            st.session_state["fast_logs"] = logs
            st.session_state["config"] = {
                "directory_keywords": directory_keywords,
                "detail_keywords": detail_keywords,
                "block_words": block_words,
                "block_page_keywords": block_page_keywords,
                "delay_seconds": float(delay_seconds),
                "max_detail_pages": int(max_detail_pages),
            }

        except Exception as e:
            st.error(f"Fast scan failed: {e}")

# Show fast-scan result
base_df = st.session_state.get("base_df", pd.DataFrame())
detail_links_df = st.session_state.get("detail_links_df", pd.DataFrame())
directory_pages_df = st.session_state.get("directory_pages_df", pd.DataFrame())
fast_logs = st.session_state.get("fast_logs", [])

if not directory_pages_df.empty or not detail_links_df.empty or not base_df.empty:
    st.subheader("Directory pages actually scanned")
    if not directory_pages_df.empty:
        st.dataframe(directory_pages_df, use_container_width=True, height=220)
    else:
        st.info("No directory pages were scanned.")

    st.subheader("Collected detail-link candidates")
    if not detail_links_df.empty:
        st.dataframe(detail_links_df.head(300), use_container_width=True, height=220)
    else:
        st.info("No detail-link candidates were collected.")

    st.subheader("Fast Scan Result")
    if base_df.empty:
        st.warning("No exhibitor records found from fast scan.")
    else:
        st.success(f"Fast scan extracted {len(base_df)} exhibitor records.")
        st.dataframe(base_df, use_container_width=True, height=420)

        csv_data = base_df.to_csv(index=False).encode("utf-8-sig")
        excel_data = make_excel_bytes(base_df)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download Fast Scan CSV", data=csv_data, file_name="exhibitors_fast_scan.csv", mime="text/csv", use_container_width=True)
        with d2:
            st.download_button("Download Fast Scan Excel", data=excel_data, file_name="exhibitors_fast_scan.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    with st.expander("Fast Scan Log", expanded=False):
        for line in fast_logs:
            st.write("-", line)

# Enrichment section
if not base_df.empty and not detail_links_df.empty:
    st.subheader("Optional: Enrich Top N Detail Pages")

    cfg = st.session_state.get("config", {})
    enrich_n = st.number_input(
        "How many detail pages to enrich now",
        min_value=0,
        max_value=500,
        value=int(cfg.get("max_detail_pages", 50)),
        step=10,
        key="enrich_n"
    )

    if st.button("Run Detail Enrichment", use_container_width=True):
        try:
            with st.spinner("Running detail enrichment..."):
                enriched_df, enrich_logs = enrich_top_n(
                    base_df=base_df,
                    detail_links_df=detail_links_df,
                    block_words=cfg["block_words"],
                    directory_keywords=cfg["directory_keywords"],
                    detail_keywords=cfg["detail_keywords"],
                    block_page_keywords=cfg["block_page_keywords"],
                    max_detail_pages=int(enrich_n),
                    delay_seconds=float(cfg["delay_seconds"])
                )

            st.session_state["enriched_df"] = enriched_df
            st.session_state["enrich_logs"] = enrich_logs

        except Exception as e:
            st.error(f"Detail enrichment failed: {e}")

enriched_df = st.session_state.get("enriched_df", pd.DataFrame())
enrich_logs = st.session_state.get("enrich_logs", [])

if not enriched_df.empty:
    st.subheader("Enriched Result")
    st.success(f"Enriched result contains {len(enriched_df)} exhibitor records.")
    st.dataframe(enriched_df, use_container_width=True, height=480)

    csv_data2 = enriched_df.to_csv(index=False).encode("utf-8-sig")
    excel_data2 = make_excel_bytes(enriched_df)

    e1, e2 = st.columns(2)
    with e1:
        st.download_button("Download Enriched CSV", data=csv_data2, file_name="exhibitors_enriched.csv", mime="text/csv", use_container_width=True)
    with e2:
        st.download_button("Download Enriched Excel", data=excel_data2, file_name="exhibitors_enriched.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    with st.expander("Enrichment Log", expanded=False):
        for line in enrich_logs:
            st.write("-", line)

# Sidebar
st.sidebar.header("Recommended workflow")
st.sidebar.markdown(
    """
1. Set **Max directory pages** to 5–15  
2. Run **Fast Scan** first  
3. Check if the base list looks reasonable  
4. Then enrich only **Top 30 / 50 / 100** detail pages
"""
)

st.sidebar.header("Install")
st.sidebar.code("pip install streamlit requests pandas beautifulsoup4 openpyxl")

st.sidebar.header("Run")
st.sidebar.code("streamlit run app.py")
