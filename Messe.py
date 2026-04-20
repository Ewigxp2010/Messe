from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import streamlit as st

from src.exhibitor_scraper import ScrapeConfig, ScrapeError, parse_exhibitors_from_html, scrape_exhibitors
from src.exporting import build_excel_download, order_columns
from src.skm_matching import match_exhibitors_to_skm, summarize_matches


DEFAULT_SKM_PATH = Path("/Users/bytedance/Downloads/Strategic Seller Lead Categorisation 副本 - Sheet1.csv")
SAMPLE_SKM_PATH = Path("data/sample_skm.csv")


st.set_page_config(
    page_title="TikTok Shop SKM 展会招商雷达",
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


def _read_local_table(path: Path) -> pd.DataFrame:
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
            "下载 Excel 招商表",
            data=excel_bytes,
            file_name="tiktok_shop_skm_exhibition_matches.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "下载 CSV 全量表",
            data=csv_bytes,
            file_name="tiktok_shop_skm_exhibition_matches.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _render_results(result_df: pd.DataFrame) -> None:
    summary = summarize_matches(_safe_records(result_df))
    metric_cols = st.columns(4)
    metric_cols[0].metric("参展商总数", summary["total"])
    metric_cols[1].metric("SKM 命中", summary["skm_matches"])
    metric_cols[2].metric("待人工确认", summary["review"])
    metric_cols[3].metric("未命中", summary["unmatched"])

    _render_downloads(result_df)

    matches_df = result_df[result_df["match_status"] == "SKM命中"].copy()
    review_df = result_df[result_df["match_status"] == "待人工确认"].copy()

    tab_matches, tab_review, tab_all = st.tabs(["SKM 命中", "待人工确认", "全部参展商"])
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
    st.title("TikTok Shop SKM 展会招商雷达")
    st.caption("输入德国展会参展商页面，上传 Strategic Key Merchant 名单，自动找出值得优先招商的参展商。")

    with st.sidebar:
        st.header("SKM 名单")
        skm_upload = st.file_uploader("上传 SKM Excel/CSV", type=["xlsx", "xls", "csv", "tsv"])
        use_local_skm = False
        if DEFAULT_SKM_PATH.exists():
            use_local_skm = st.checkbox("使用你提供的本地 SKM 文件", value=skm_upload is None)
        st.download_button(
            "下载 SKM 样例",
            data=BytesIO(SAMPLE_SKM_PATH.read_bytes()),
            file_name="sample_skm.csv",
            mime="text/csv",
            use_container_width=True,
        )

        skm_df = None
        name_col = ""
        alias_cols: List[str] = []
        threshold = st.slider("匹配阈值", min_value=70, max_value=100, value=88, step=1)
        review_margin = st.slider("人工复核区间", min_value=0, max_value=20, value=8, step=1)

        if skm_upload is not None or use_local_skm:
            try:
                skm_df = _read_table(skm_upload) if skm_upload is not None else _read_local_table(DEFAULT_SKM_PATH)
                skm_df.columns = [str(col).strip() for col in skm_df.columns]
                st.success(f"已读取 {len(skm_df)} 条 SKM")
                columns = list(skm_df.columns)
                guessed = _column_guess(columns, ["skm", "merchant", "company", "brand", "name", "firma"])
                name_col = st.selectbox("SKM 公司名列", columns, index=columns.index(guessed))
                alias_cols = st.multiselect(
                    "Alias/品牌/店铺名列",
                    [col for col in columns if col != name_col],
                    default=[col for col in columns if col.lower() in {"alias", "aliases", "brand", "shop"}],
                )
            except Exception as exc:
                st.error(f"SKM 文件读取失败：{exc}")

        st.header("抓取设置")
        max_pages = st.number_input("最多抓取页数", min_value=1, max_value=50, value=1, step=1)
        crawl_detail_pages = st.checkbox("补抓参展商详情页", value=False)
        detail_page_limit = st.number_input("详情页上限", min_value=1, max_value=500, value=50, step=10)

        with st.expander("高级抓取设置"):
            page_url_template = st.text_input("分页 URL 模板", placeholder="https://example.com/exhibitors?page={page}")
            item_selector = st.text_input("参展商卡片 selector", placeholder=".exhibitor-card")
            name_selector = st.text_input("公司名 selector", placeholder=".exhibitor-name")
            hall_selector = st.text_input("展厅 selector", placeholder=".hall")
            booth_selector = st.text_input("展位 selector", placeholder=".booth")
            country_selector = st.text_input("国家 selector", placeholder=".country")
            website_selector = st.text_input("官网 selector", placeholder="a.website")
            detail_link_selector = st.text_input("详情页链接 selector", placeholder="a.detail")

    left, right = st.columns([2, 1])
    with left:
        url = st.text_input("展会参展商页面 URL", placeholder="https://www.example-messe.de/exhibitors")
    with right:
        html_upload = st.file_uploader("或上传页面 HTML", type=["html", "htm"])

    can_run = skm_df is not None and bool(name_col) and (bool(url) or html_upload is not None)
    run = st.button("开始抓取并匹配", type="primary", disabled=not can_run, use_container_width=True)

    if not can_run:
        st.info("请先上传 SKM 名单，并输入展会 URL 或上传页面 HTML。")

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
            with st.status("正在抓取参展商...", expanded=True) as status:
                if html_upload is not None:
                    html = _read_html(html_upload)
                    exhibitors = parse_exhibitors_from_html(html, base_url=url, config=config)
                else:
                    exhibitors = scrape_exhibitors(config)
                status.write(f"抓取到 {len(exhibitors)} 条候选参展商")

                matched = match_exhibitors_to_skm(
                    exhibitors=exhibitors,
                    skm_rows=_safe_records(skm_df),
                    name_col=name_col,
                    alias_cols=alias_cols,
                    threshold=float(threshold),
                    review_margin=float(review_margin),
                )
                status.write("SKM 匹配完成")
                status.update(label="完成", state="complete", expanded=False)

            if not matched:
                st.warning("没有抓取到参展商。可以尝试上传页面 HTML，或在高级设置里填写 CSS selector。")
                return

            result_df = pd.DataFrame(matched)
            _render_results(result_df)

        except ScrapeError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)


if __name__ == "__main__":
    main()
