import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from io import BytesIO
import re
import time

# =====================================================
# Page config
# =====================================================
st.set_page_config(
    page_title="Exhibitor List Extractor",
    page_icon="🧾",
    layout="wide"
)

# =====================================================
# Helpers
# =====================================================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

EXHIBITOR_HINTS = [
    "exhibitor", "exhibitors", "aussteller", "brands", "participants",
    "participants-list", "directory", "list of exhibitors", "attendees",
    "vendor", "vendors", "partner", "partners", "companies", "firmen"
]

NEGATIVE_HINTS = [
    "privacy", "impressum", "imprint", "terms", "agb", "login", "register",
    "cart", "checkout", "contact", "faq", "press", "news", "jobs"
]

NAME_CLASS_HINTS = [
    "company", "brand", "name", "title", "exhibitor", "vendor", "participant"
]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-()/]{6,}\d)")


def normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def fetch_html(url: str, timeout: int = 20):
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text, resp.url


def get_soup(url: str):
    html, final_url = fetch_html(url)
    return BeautifulSoup(html, "html.parser"), final_url


def is_same_domain(base_url: str, candidate_url: str) -> bool:
    return urlparse(base_url).netloc == urlparse(candidate_url).netloc


def score_link(text: str, href: str) -> int:
    value = f"{text} {href}".lower()
    score = 0
    for hint in EXHIBITOR_HINTS:
        if hint in value:
            score += 5
    for bad in NEGATIVE_HINTS:
        if bad in value:
            score -= 3
    if any(x in value for x in ["list", "directory", "catalog", "katalog"]):
        score += 2
    return score


def discover_candidate_pages(base_url: str, soup: BeautifulSoup, max_candidates: int = 12):
    candidates = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        text = a.get_text(" ", strip=True)
        abs_url = urljoin(base_url, href)

        if abs_url in seen:
            continue
        seen.add(abs_url)

        if not is_same_domain(base_url, abs_url):
            continue

        s = score_link(text, href)
        if s > 0:
            candidates.append({
                "url": abs_url,
                "text": text,
                "score": s
            })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates[:max_candidates]


def extract_emails_and_phones(text: str):
    emails = sorted(set(EMAIL_RE.findall(text or "")))
    phones = sorted(set(PHONE_RE.findall(text or "")))
    return ", ".join(emails[:3]), ", ".join(phones[:3])


def clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name or "").strip(" -|\n\t")
    return name.strip()


def looks_like_company_name(text: str) -> bool:
    if not text:
        return False
    text = clean_name(text)
    if len(text) < 2 or len(text) > 120:
        return False
    bad_patterns = [
        "cookie", "privacy", "load more", "see more", "read more", "newsletter",
        "login", "sign in", "register", "terms", "impressum", "contact"
    ]
    low = text.lower()
    if any(b in low for b in bad_patterns):
        return False
    if len(text.split()) > 12:
        return False
    return True


def parse_cards(soup: BeautifulSoup, page_url: str):
    results = []
    seen_names = set()

    selectors = [
        "article", ".card", ".item", ".result", ".listing", ".company", ".brand",
        ".vendor", ".participant", ".exhibitor", ".directory-item", ".grid-item",
        "li", ".search-result"
    ]

    elements = []
    for sel in selectors:
        elements.extend(soup.select(sel))

    for el in elements:
        text = el.get_text(" ", strip=True)
        if len(text) < 3:
            continue

        name = None

        for tag in el.find_all(["h1", "h2", "h3", "h4", "strong", "b", "a", "span", "div"]):
            tag_text = clean_name(tag.get_text(" ", strip=True))
            classes = " ".join(tag.get("class", [])) if tag.get("class") else ""
            class_low = classes.lower()
            if any(h in class_low for h in NAME_CLASS_HINTS) and looks_like_company_name(tag_text):
                name = tag_text
                break

        if not name:
            for tag in el.find_all(["h1", "h2", "h3", "h4", "strong", "b"]):
                tag_text = clean_name(tag.get_text(" ", strip=True))
                if looks_like_company_name(tag_text):
                    name = tag_text
                    break

        if not name:
            first_line = clean_name(text.split("  ")[0])
            if looks_like_company_name(first_line):
                name = first_line

        if not name:
            continue

        if name.lower() in seen_names:
            continue
        seen_names.add(name.lower())

        site = ""
        detail_link = ""
        for a in el.find_all("a", href=True):
            href = urljoin(page_url, a["href"])
            if href.startswith("mailto:"):
                continue
            if urlparse(href).netloc and urlparse(href).netloc != urlparse(page_url).netloc:
                site = href
            else:
                detail_link = href

        email, phone = extract_emails_and_phones(text)

        results.append({
            "company_name": name,
            "source_page": page_url,
            "detail_link": detail_link,
            "website": site,
            "email": email,
            "phone": phone,
            "raw_text": text[:500]
        })

    return results


def parse_table_like(soup: BeautifulSoup, page_url: str):
    results = []
    seen_names = set()

    for row in soup.select("tr"):
        cols = [clean_name(td.get_text(" ", strip=True)) for td in row.find_all(["td", "th"])]
        cols = [c for c in cols if c]
        if not cols:
            continue

        candidate = cols[0]
        if looks_like_company_name(candidate) and candidate.lower() not in seen_names:
            seen_names.add(candidate.lower())
            text = " | ".join(cols)
            email, phone = extract_emails_and_phones(text)
            results.append({
                "company_name": candidate,
                "source_page": page_url,
                "detail_link": "",
                "website": "",
                "email": email,
                "phone": phone,
                "raw_text": text[:500]
            })

    return results


def parse_link_list(soup: BeautifulSoup, page_url: str):
    results = []
    seen_names = set()

    for a in soup.find_all("a", href=True):
        name = clean_name(a.get_text(" ", strip=True))
        href = urljoin(page_url, a["href"])
        if looks_like_company_name(name) and name.lower() not in seen_names:
            low = href.lower()
            if any(h in low for h in EXHIBITOR_HINTS) or any(h in name.lower() for h in ["gmbh", "ag", "ug", "ltd", "inc", "srl", "sas"]):
                seen_names.add(name.lower())
                results.append({
                    "company_name": name,
                    "source_page": page_url,
                    "detail_link": href,
                    "website": "",
                    "email": "",
                    "phone": "",
                    "raw_text": name
                })

    return results


def deduplicate_rows(rows):
    if not rows:
        return []

    df = pd.DataFrame(rows)
    df["company_name_norm"] = df["company_name"].fillna("").str.strip().str.lower()
    df = df[df["company_name_norm"] != ""]
    df = df.drop_duplicates(subset=["company_name_norm"], keep="first")
    df = df.drop(columns=["company_name_norm"])
    return df.to_dict(orient="records")


def scrape_exhibitors(start_url: str, max_pages: int = 6, sleep_sec: float = 0.5):
    logs = []
    all_rows = []

    soup, final_url = get_soup(start_url)
    logs.append(f"Homepage loaded: {final_url}")

    candidate_pages = discover_candidate_pages(final_url, soup, max_candidates=max_pages)

    if not candidate_pages:
        logs.append("No obvious exhibitor pages found on homepage. Trying homepage directly.")
        candidate_pages = [{"url": final_url, "text": "homepage", "score": 0}]

    visited = set()

    for item in candidate_pages:
        page_url = item["url"]
        if page_url in visited:
            continue
        visited.add(page_url)

        try:
            time.sleep(sleep_sec)
            page_soup, resolved = get_soup(page_url)
            logs.append(f"Scanning page: {resolved}")

            rows = []
            rows.extend(parse_cards(page_soup, resolved))
            rows.extend(parse_table_like(page_soup, resolved))
            rows.extend(parse_link_list(page_soup, resolved))

            rows = deduplicate_rows(rows)
            logs.append(f"Found {len(rows)} possible exhibitors on this page.")
            all_rows.extend(rows)
        except Exception as e:
            logs.append(f"Failed to scan {page_url}: {e}")

    all_rows = deduplicate_rows(all_rows)
    df = pd.DataFrame(all_rows)

    if not df.empty:
        df = df[[
            "company_name", "website", "email", "phone", "detail_link", "source_page", "raw_text"
        ]]
        df = df.sort_values(by="company_name", ascending=True).reset_index(drop=True)

    return df, logs, candidate_pages


def make_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="exhibitors")
    return output.getvalue()


# =====================================================
# UI
# =====================================================
st.title("🧾 Exhibition Exhibitor Extractor")
st.caption("Input an exhibition website and auto-build an exhibitor list for BD outreach.")

with st.expander("What this app does", expanded=False):
    st.markdown(
        """
- Enter a trade fair or exhibition website URL
- The app tries to locate the exhibitor / directory / participant pages automatically
- It extracts company names and any visible website / email / phone fields
- You can review the table and download it as CSV or Excel

**Best use case:** websites with a visible HTML exhibitor directory

**Less suitable:** websites that hide exhibitors behind JavaScript apps, login walls, anti-bot tools, or PDF-only catalogs
        """
    )

with st.form("extract_form"):
    c1, c2 = st.columns([4, 1])
    with c1:
        website = st.text_input(
            "Exhibition website URL",
            placeholder="https://www.example-expo.com"
        )
    with c2:
        max_pages = st.number_input("Pages", min_value=1, max_value=15, value=6, step=1)

    submitted = st.form_submit_button("Extract exhibitor list", use_container_width=True)

if submitted:
    if not website.strip():
        st.error("Please enter a website URL.")
    else:
        try:
            with st.spinner("Scanning website and extracting exhibitors..."):
                url = normalize_url(website)
                df, logs, candidate_pages = scrape_exhibitors(url, max_pages=int(max_pages))

            st.subheader("Detected candidate pages")
            if candidate_pages:
                page_df = pd.DataFrame(candidate_pages)
                st.dataframe(page_df, use_container_width=True)
            else:
                st.info("No candidate exhibitor pages were detected.")

            st.subheader("Extraction result")
            if df.empty:
                st.warning(
                    "No exhibitors were extracted. This usually means the website uses JavaScript rendering, a private directory, or an unusual page structure."
                )
            else:
                st.success(f"Extracted {len(df)} exhibitor records.")
                st.dataframe(df, use_container_width=True, height=520)

                csv_data = df.to_csv(index=False).encode("utf-8-sig")
                excel_data = make_excel_bytes(df)

                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        "Download CSV",
                        data=csv_data,
                        file_name="exhibitors.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with d2:
                    st.download_button(
                        "Download Excel",
                        data=excel_data,
                        file_name="exhibitors.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

            with st.expander("Run log", expanded=False):
                for log in logs:
                    st.write("-", log)

        except Exception as e:
            st.error(f"Extraction failed: {e}")


# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("Suggestions")
st.sidebar.markdown(
    """
**For better results:**
1. Use the main event homepage first
2. If extraction is weak, paste the specific exhibitor directory URL directly
3. For some fairs, you may later add custom rules for that site

**Next upgrade ideas:**
- Add Playwright support for JavaScript-heavy websites
- Visit each exhibitor detail page for better contact extraction
- Auto-detect country / booth / category
- Add lead scoring for TikTok Shop fit
- Add German / English / Chinese UI switch
    """
)

st.sidebar.header("Install")
st.sidebar.code(
    "pip install streamlit requests beautifulsoup4 pandas openpyxl"
)

st.sidebar.header("Run")
st.sidebar.code(
    "streamlit run app.py"
)
