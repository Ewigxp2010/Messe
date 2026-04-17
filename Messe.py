import re
import time
from io import BytesIO
from urllib.parse import urljoin, urlparse

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
    "brands, vendor, vendors, partner, partners, company list, "
    "catalog, catalogue, exhibitors list, attendee, attendees"
)

DEFAULT_DETAIL_KEYWORDS = (
    "exhibitor, brand, company, participant, vendor, partner, profile"
)

DEFAULT_BLOCK_WORDS = (
    "about, about us, apply, apply as exhibitor, enquiry, contact, privacy, "
    "cookie, cookies, terms, login, sign in, register, press, news, jobs, "
    "career, faq, impressum, imprint, legal, media, ticket, tickets"
)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-()/]{6,}\d)")
HALL_RE = re.compile(r"\bH\d+(?:\.\d+)?\b", re.IGNORECASE)
BOOTH_RE = re.compile(r"\b[A-Z]?\d+(?:\.\d+)?-\d+\b", re.IGNORECASE)


# =========================================================
# Utility functions
# =========================================================
def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def parse_csv_keywords(text: str) -> list[str]:
    items = [x.strip().lower() for x in (text or "").split(",")]
    return [x for x in items if x]


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
# Filtering and scoring
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
    ]
    if any(p in low for p in junk_patterns):
        return True

    # 太像一句话/段落，不像公司名
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


def score_link_candidate(
    anchor_text: str,
    href: str,
    directory_keywords: list[str],
    detail_keywords: list[str],
    block_words: list[str]
) -> int:
    value = f"{anchor_text} {href}".lower()
    score = 0

    if contains_any(value, block_words):
        score -= 50

    if contains_any(value, detail_keywords):
        score += 25

    if contains_any(value, directory_keywords):
        score += 15

    path = urlparse(href).path.lower()

    # 路径层级更深，通常更像详情页
    depth = len([p for p in path.split("/") if p])
    if depth >= 2:
        score += 10
    if depth >= 3:
        score += 5

    if "page=" in href.lower():
        score -= 15

    if path.endswith((".pdf", ".jpg", ".jpeg", ".png", ".webp", ".svg", ".zip")):
        score -= 100

    return score


# =========================================================
# Page discovery
# =========================================================
def discover_directory_pages(
    start_url: str,
    directory_keywords: list[str],
    block_words: list[str],
    max_candidates: int = 20
):
    soup, final_url = get_soup(start_url)
    candidates = []
    seen = set()

    # 首页本身也可作为候选
    candidates.append({
        "url": final_url,
        "text": "homepage",
        "score": 5
    })
    seen.add(final_url)

    for a in soup.find_all("a", href=True):
        text = clean_text(a.get_text(" ", strip=True))
        href = urljoin(final_url, a["href"])

        if href in seen:
            continue
        seen.add(href)

        if not is_valid_http_url(href):
            continue
        if not same_domain(final_url, href):
            continue

        value = f"{text} {href}".lower()
        score = 0

        if contains_any(value, directory_keywords):
            score += 30

        if "page=" in href.lower():
            score += 5

        if contains_any(value, block_words):
            score -= 25

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
        if c["url"] not in urls:
            unique.append(c)
            urls.add(c["url"])

    return unique[:max_candidates]


# =========================================================
# List-page extraction
# =========================================================
def extract_candidate_detail_links_from_page(
    page_url: str,
    directory_keywords: list[str],
    detail_keywords: list[str],
    block_words: list[str]
):
    soup, resolved = get_soup(page_url)
    found = []

    for a in soup.find_all("a", href=True):
        anchor_text = clean_text(a.get_text(" ", strip=True))
        href = urljoin(resolved, a["href"])

        if not is_valid_http_url(href):
            continue
        if not same_domain(resolved, href):
            continue

        score = score_link_candidate(
            anchor_text=anchor_text,
            href=href,
            directory_keywords=directory_keywords,
            detail_keywords=detail_keywords,
            block_words=block_words
        )

        if score <= 0:
            continue

        # 过滤明显是目录页/分页页/无意义页
        if "page=" in href.lower() and not contains_any(href.lower(), detail_keywords):
            continue

        if looks_like_bad_name(anchor_text, block_words):
            # 锚文本不好，但如果链接路径很像详情页，也保留低优先级候选
            path = urlparse(href).path.lower()
            if len([p for p in path.split("/") if p]) < 2:
                continue
            score += 5

        found.append({
            "anchor_text": anchor_text,
            "detail_link": href,
            "source_page": resolved,
            "link_score": score
        })

    if not found:
        return []

    df = pd.DataFrame(found)
    df["url_norm"] = df["detail_link"].str.lower().str.strip()
    df = df.sort_values(["link_score"], ascending=False)
    df = df.drop_duplicates(subset=["url_norm"], keep="first")
    df = df.drop(columns=["url_norm"])

    return df.to_dict(orient="records")


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
    """
    兜底：有些网站没有明显详情页，只能从列表卡片/表格中直接提取名称
    """
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
        ".vendor"
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


def extract_company_name_from_detail(
    soup: BeautifulSoup,
    fallback_anchor_text: str,
    block_words: list[str]
) -> tuple[str, int]:
    # 1) h1
    h1 = soup.find("h1")
    if h1:
        name = clean_text(h1.get_text(" ", strip=True))
        if not looks_like_bad_name(name, block_words):
            return name, 90

    # 2) title
    if soup.title:
        title_text = clean_text(soup.title.get_text(" ", strip=True))
        title_candidate = title_text.split("|")[0].split(" - ")[0].strip()
        if not looks_like_bad_name(title_candidate, block_words):
            return title_candidate, 80

    # 3) og:title
    meta_og = soup.find("meta", attrs={"property": "og:title"})
    if meta_og and meta_og.get("content"):
        og_candidate = clean_text(meta_og["content"])
        if not looks_like_bad_name(og_candidate, block_words):
            return og_candidate, 75

    # 4) fallback anchor text
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
# Main scraper
# =========================================================
def scrape_exhibitors(
    start_url: str,
    max_directory_pages: int,
    crawl_detail_pages: bool,
    directory_keywords: list[str],
    detail_keywords: list[str],
    block_words: list[str],
    delay_seconds: float = 0.3
):
    logs = []
    all_rows = []
    discovered_detail_links = []

    # 1. find likely directory pages
    logs.append("Discovering likely directory pages...")
    directory_pages = discover_directory_pages(
        start_url=start_url,
        directory_keywords=directory_keywords,
        block_words=block_words,
        max_candidates=max_directory_pages
    )
    logs.append(f"Directory page candidates found: {len(directory_pages)}")

    visited_directory_urls = set()

    for item in directory_pages:
        page_url = item["url"]
        if page_url in visited_directory_urls:
            continue
        visited_directory_urls.add(page_url)

        try:
            logs.append(f"Scanning directory/list page: {page_url}")
            links = extract_candidate_detail_links_from_page(
                page_url=page_url,
                directory_keywords=directory_keywords,
                detail_keywords=detail_keywords,
                block_words=block_words
            )
            logs.append(f"  Detail-link candidates found: {len(links)}")

            if links:
                discovered_detail_links.extend(links)
            else:
                # 没有 detail link 时，兜底直接从列表页提取
                fallback_rows = extract_candidates_without_detail_links(
                    page_url=page_url,
                    block_words=block_words
                )
                logs.append(f"  Fallback list-page entities found: {len(fallback_rows)}")
                all_rows.extend(fallback_rows)

        except Exception as e:
            logs.append(f"  Failed to scan list page: {page_url} | {e}")

        time.sleep(delay_seconds)

    # 2. deduplicate detail links
    if discovered_detail_links:
        df_links = pd.DataFrame(discovered_detail_links)
        df_links["url_norm"] = df_links["detail_link"].str.lower().str.strip()
        df_links = df_links.sort_values(["link_score"], ascending=False)
        df_links = df_links.drop_duplicates(subset=["url_norm"], keep="first")
        df_links = df_links.drop(columns=["url_norm"])
        discovered_detail_links = df_links.to_dict(orient="records")

    logs.append(f"Unique detail links after deduplication: {len(discovered_detail_links)}")

    # 3. crawl detail pages
    if crawl_detail_pages and discovered_detail_links:
        logs.append("Crawling detail pages for enrichment...")

        for i, link_row in enumerate(discovered_detail_links, start=1):
            detail_url = link_row["detail_link"]
            anchor_text = link_row.get("anchor_text", "")
            source_page = link_row.get("source_page", "")

            try:
                row = enrich_from_detail_page(
                    detail_url=detail_url,
                    anchor_text=anchor_text,
                    block_words=block_words
                )
                if row:
                    row["source_page"] = source_page
                    # 用链接分数再微调一下总置信度
                    row["confidence_score"] = min(
                        100,
                        int(row["confidence_score"] + max(0, link_row.get("link_score", 0) // 5))
                    )
                    all_rows.append(row)

                if i % 20 == 0:
                    logs.append(f"  Processed detail pages: {i}/{len(discovered_detail_links)}")

            except Exception as e:
                logs.append(f"  Failed detail page: {detail_url} | {e}")

            time.sleep(delay_seconds)

    elif discovered_detail_links:
        # 不爬 detail page，只用 anchor_text 做基础名单
        logs.append("Detail crawling disabled. Building records from link text only.")
        for link_row in discovered_detail_links:
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

    # 4. final cleanup
    if not all_rows:
        return pd.DataFrame(), logs, directory_pages, discovered_detail_links

    df = pd.DataFrame(all_rows)

    # 清洗 company_name
    df["company_name"] = df["company_name"].fillna("").map(clean_text)
    df = df[df["company_name"] != ""]
    df = df[~df["company_name"].map(lambda x: looks_like_bad_name(x, block_words))]

    # 去重：优先保留置信度高的
    df["name_norm"] = df["company_name"].str.lower().str.strip()
    df = df.sort_values(["confidence_score"], ascending=False)
    df = df.drop_duplicates(subset=["name_norm"], keep="first")
    df = df.drop(columns=["name_norm"])

    # 字段顺序
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

    return df, logs, directory_pages, discovered_detail_links


# =========================================================
# UI
# =========================================================
st.title("🧾 Exhibition Exhibitor Extractor")
st.caption("Generic exhibitor list extractor for trade fairs and event websites.")

with st.expander("How it works", expanded=False):
    st.markdown(
        """
This app follows a generic workflow:

1. Discover likely exhibitor/directory pages from the website  
2. Extract likely company/detail links from those pages  
3. Optionally crawl detail pages for structured enrichment  
4. Clean, deduplicate, and score the final exhibitor list

It is designed for **many different trade fair websites**, not for one specific event site.
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
            max_value=50,
            value=10,
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

    st.subheader("Configurable keywords")

    directory_keywords_text = st.text_area(
        "Directory keywords (comma separated)",
        value=DEFAULT_DIRECTORY_KEYWORDS,
        height=100
    )

    detail_keywords_text = st.text_area(
        "Detail-link keywords (comma separated)",
        value=DEFAULT_DETAIL_KEYWORDS,
        height=100
    )

    block_words_text = st.text_area(
        "Block words (comma separated)",
        value=DEFAULT_BLOCK_WORDS,
        height=120
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

            with st.spinner("Scanning website and extracting exhibitors..."):
                df, logs, directory_pages, detail_links = scrape_exhibitors(
                    start_url=start_url,
                    max_directory_pages=int(max_directory_pages),
                    crawl_detail_pages=crawl_detail_pages,
                    directory_keywords=directory_keywords,
                    detail_keywords=detail_keywords,
                    block_words=block_words,
                    delay_seconds=float(delay_seconds)
                )

            st.subheader("Detected directory page candidates")
            if directory_pages:
                st.dataframe(pd.DataFrame(directory_pages), use_container_width=True, height=250)
            else:
                st.info("No likely directory pages were detected.")

            st.subheader("Detected detail-link candidates")
            if detail_links:
                preview_links = pd.DataFrame(detail_links)
                st.dataframe(preview_links.head(200), use_container_width=True, height=250)
            else:
                st.info("No detail-link candidates were detected.")

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
st.sidebar.code(
    "pip install streamlit requests pandas beautifulsoup4 openpyxl"
)

st.sidebar.header("Run")
st.sidebar.code(
    "streamlit run app.py"
)

st.sidebar.header("Tips")
st.sidebar.markdown(
    """
- Start with the main event homepage
- If results are weak, paste the exhibitor directory URL directly
- Turn on detail crawling for better website/email/phone extraction
- Adjust keywords for different fair websites
"""
)
