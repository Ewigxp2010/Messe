import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="Meeting Growth Visualizer",
    layout="wide"
)

# ======================
# Presets
# ======================
AOV_PRESETS = {
    "Home & Living": {
        "Kitchenware": 39.68,
        "Pet Supplies": 29.10,
        "Sports & Outdoor": 127.82,
        "Textiles & Soft Furnishings": 41.68,
        "Tools & Hardware": 34.52,
        "Luggage & Bags": 90.05,
        "Shoes": 70.03,
        "Menswear & Underwear": 53.44,
        "Modest Fashion": 19.42,
        "Womenswear & Underwear": 22.34,
        "Furniture": 145.52,
        "Home Improvement": 53.59,
        "Home Supplies": 27.34,
    },
    "Electronics": {
        "Phones & Electronics": 55.14,
        "Computers & Office Equipment": 59.85,
        "Household Appliances": 59.79,
        "Automotive & Motorcycle": 41.19,
        "Collectibles": 77.02,
        "Beauty & Personal Care": 34.14,
        "Baby & Maternity": 51.12,
        "Books, Magazines & Audio": 21.11,
    }
}

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

PHASE_KEYS = ["phase1", "phase2", "phase3"]

PHASE_DEFAULTS = {
    "phase1": {"gmv_start": 2_000, "gmv_end": 10_000, "ads_take_rate": 0.00, "samples": 30, "affiliate_share": 0.60},
    "phase2": {"gmv_start": 10_000, "gmv_end": 30_000, "ads_take_rate": 0.05, "samples": 25, "affiliate_share": 0.40},
    "phase3": {"gmv_start": 30_000, "gmv_end": 100_000, "ads_take_rate": 0.10, "samples": 20, "affiliate_share": 0.25},
}

PHASE_COLOR_KEYS = {
    "phase1": {"bg": "#EAF4FF", "badge_bg": "#DCEEFF", "badge_text": "#1D4E89"},
    "phase2": {"bg": "#EEFBEF", "badge_bg": "#DFF5E3", "badge_text": "#1E6B35"},
    "phase3": {"bg": "#FFF4E8", "badge_bg": "#FFE8D6", "badge_text": "#A14B00"},
}

# ======================
# i18n
# ======================
TEXT = {
    "en": {
        "app_title": "Meeting Growth Visualizer",
        "app_caption": "Enhanced Streamlit version for GMV / Cost / Profit modeling",
        "language": "Language",
        "global_inputs": "Global Inputs",
        "num_products": "No. of products",
        "promo_60d": "60-day fee promo",
        "promo_yes": "Yes (5% first ~60 days)",
        "promo_no": "No promo",
        "fulfillment": "Fulfillment €/order",
        "sample_product": "Sample product €/unit",
        "sample_shipping": "Sample shipping €/unit",
        "affiliate_commission": "Affiliate commission",
        "weeks_per_phase": "Weeks / phase",
        "phase_controls": "Phase Controls",
        "phase1": "Phase 1 — Cold Start",
        "phase2": "Phase 2 — Growth",
        "phase3": "Phase 3 — Breakout",
        "ads_rate": "Ads Take Rate",
        "samples_per_week": "Samples / week",
        "affiliate_share": "Affiliate share",
        "product_setup": "Product Setup",
        "product_setup_caption": "Set up each product below. Preset is used as category reference; AOV can still be manually overridden.",
        "generate": "Generate charts",
        "product_mix": "Product Mix Used",
        "charts": "Charts",
        "overall_weekly_trend": "Overall Weekly Trend",
        "cumulative_profit_trend": "Cumulative Profit Trend",
        "phase_by_phase": "Phase-by-Phase Weekly Trend",
        "summary": "Summary",
        "phase_summary": "Phase summary",
        "overall_summary": "Overall summary",
        "break_even_signals": "Break-even Signals",
        "weekly_details": "Weekly Details",
        "download_weekly": "Download weekly details CSV",
        "download_phase": "Download phase summary CSV",
        "total_gmv": "Total GMV",
        "total_profit": "Total Profit",
        "profit_margin": "Profit Margin",
        "first_positive_profit": "First positive weekly profit",
        "cumulative_break_even": "Cumulative break-even",
        "not_reached": "Not reached",
        "weekly_be_label": "Weekly BE",
        "cumulative_be_label": "Cumulative BE",
        "week": "Week",
        "product": "Product",
        "preset": "Preset",
        "sales_share": "Sales Share (%)",
        "sales_share_help": "Does not need to equal 100. The system auto-normalizes.",
        "aov": "AOV (€)",
        "gross_margin": "Gross Margin (%)",
        "fee_type": "Fee Type",
        "electronics_fee": "Electronics (7%)",
        "other_fee": "Other (9%)",
        "family": "Family",
        "preset_category": "Preset Category",
        "platform_fee_default": "Platform Fee Rate Default",
        "input_error": "Input error",
        "ads_take_rate_col": "Ads Take Rate",
        "affiliate_share_col": "Affiliate Share",
        "sales_share_col": "Sales Share",
        "product_block": "Product",
        "gross_margin_help": "Enter gross margin as a percentage, e.g. 40 = 40%",
        "gmv_start": "GMV start",
        "gmv_end": "GMV end",
    },
    "de": {
        "app_title": "Meeting Growth Visualizer",
        "app_caption": "Erweiterte Streamlit-Version für GMV-, Kosten- und Gewinnmodellierung",
        "language": "Sprache",
        "global_inputs": "Globale Eingaben",
        "num_products": "Anzahl Produkte",
        "promo_60d": "60-Tage-Gebührenpromo",
        "promo_yes": "Ja (5% in den ersten ~60 Tagen)",
        "promo_no": "Keine Promo",
        "fulfillment": "Fulfillment €/Bestellung",
        "sample_product": "Sample-Produkt €/Einheit",
        "sample_shipping": "Sample-Versand €/Einheit",
        "affiliate_commission": "Affiliate-Provision",
        "weeks_per_phase": "Wochen / Phase",
        "phase_controls": "Phasensteuerung",
        "phase1": "Phase 1 — Kaltstart",
        "phase2": "Phase 2 — Wachstum",
        "phase3": "Phase 3 — Skalierung",
        "ads_rate": "Ads Take Rate",
        "samples_per_week": "Samples / Woche",
        "affiliate_share": "Affiliate-Anteil",
        "product_setup": "Produkteinstellung",
        "product_setup_caption": "Richte unten jedes Produkt ein. Preset dient als Kategoriereferenz; AOV kann weiterhin manuell überschrieben werden.",
        "generate": "Charts erzeugen",
        "product_mix": "Verwendeter Produktmix",
        "charts": "Charts",
        "overall_weekly_trend": "Gesamter Wochenverlauf",
        "cumulative_profit_trend": "Kumulierter Gewinnverlauf",
        "phase_by_phase": "Wochenverlauf je Phase",
        "summary": "Zusammenfassung",
        "phase_summary": "Phasenübersicht",
        "overall_summary": "Gesamtübersicht",
        "break_even_signals": "Break-even-Signale",
        "weekly_details": "Wöchentliche Details",
        "download_weekly": "Wöchentliche Details als CSV herunterladen",
        "download_phase": "Phasenübersicht als CSV herunterladen",
        "total_gmv": "Gesamt-GMV",
        "total_profit": "Gesamtgewinn",
        "profit_margin": "Gewinnmarge",
        "first_positive_profit": "Erste positive Wochenprofitabilität",
        "cumulative_break_even": "Kumulierter Break-even",
        "not_reached": "Nicht erreicht",
        "weekly_be_label": "Wochen-BE",
        "cumulative_be_label": "Kumulierter BE",
        "week": "Woche",
        "product": "Produkt",
        "preset": "Preset",
        "sales_share": "Umsatzanteil (%)",
        "sales_share_help": "Muss nicht 100 ergeben. Das System normalisiert automatisch.",
        "aov": "AOV (€)",
        "gross_margin": "Bruttomarge (%)",
        "fee_type": "Gebührentyp",
        "electronics_fee": "Elektronik (7%)",
        "other_fee": "Sonstige (9%)",
        "family": "Familie",
        "preset_category": "Preset-Kategorie",
        "platform_fee_default": "Standard-Plattformgebühr",
        "input_error": "Eingabefehler",
        "ads_take_rate_col": "Ads Take Rate",
        "affiliate_share_col": "Affiliate-Anteil",
        "sales_share_col": "Umsatzanteil",
        "product_block": "Produkt",
        "gross_margin_help": "Bruttomarge in Prozent eingeben, z. B. 40 = 40%",
        "gmv_start": "GMV-Start",
        "gmv_end": "GMV-Ende",
    },
    "zh": {
        "app_title": "Meeting Growth Visualizer",
        "app_caption": "用于 GMV / 成本 / 利润建模的增强版 Streamlit 工具",
        "language": "语言",
        "global_inputs": "全局输入",
        "num_products": "产品数量",
        "promo_60d": "60天费率优惠",
        "promo_yes": "是（前约60天 5%）",
        "promo_no": "否",
        "fulfillment": "履约成本 €/订单",
        "sample_product": "样品产品成本 €/件",
        "sample_shipping": "样品运费 €/件",
        "affiliate_commission": "达人佣金",
        "weeks_per_phase": "每阶段周数",
        "phase_controls": "阶段控制",
        "phase1": "阶段 1 — 冷启动",
        "phase2": "阶段 2 — 增长",
        "phase3": "阶段 3 — 爆发",
        "ads_rate": "Ads Take Rate",
        "samples_per_week": "每周样品数",
        "affiliate_share": "达人 GMV 占比",
        "product_setup": "产品设置",
        "product_setup_caption": "请在下方逐个设置产品。Preset 作为类目参考；AOV 仍可手动修改。",
        "generate": "生成图表",
        "product_mix": "使用的产品组合",
        "charts": "图表",
        "overall_weekly_trend": "整体周趋势",
        "cumulative_profit_trend": "累计利润趋势",
        "phase_by_phase": "分阶段周趋势",
        "summary": "汇总",
        "phase_summary": "阶段汇总",
        "overall_summary": "整体汇总",
        "break_even_signals": "Break-even 信号",
        "weekly_details": "每周明细",
        "download_weekly": "下载每周明细 CSV",
        "download_phase": "下载阶段汇总 CSV",
        "total_gmv": "总 GMV",
        "total_profit": "总利润",
        "profit_margin": "利润率",
        "first_positive_profit": "首次单周盈利",
        "cumulative_break_even": "累计 Break-even",
        "not_reached": "未达到",
        "weekly_be_label": "单周 BE",
        "cumulative_be_label": "累计 BE",
        "week": "周",
        "product": "产品",
        "preset": "Preset",
        "sales_share": "销售占比 (%)",
        "sales_share_help": "不需要加起来等于100，系统会自动标准化。",
        "aov": "AOV (€)",
        "gross_margin": "毛利率 (%)",
        "fee_type": "费率类型",
        "electronics_fee": "电子类 (7%)",
        "other_fee": "其他 (9%)",
        "family": "大类",
        "preset_category": "Preset 类目",
        "platform_fee_default": "默认平台费率",
        "input_error": "输入错误",
        "ads_take_rate_col": "Ads Take Rate",
        "affiliate_share_col": "达人占比",
        "sales_share_col": "销售占比",
        "product_block": "产品",
        "gross_margin_help": "请输入百分比，例如 40 = 40%",
        "gmv_start": "GMV 起点",
        "gmv_end": "GMV 终点",
    },
    "nl": {
        "app_title": "Meeting Growth Visualizer",
        "app_caption": "Verbeterde Streamlit-versie voor GMV / kosten / winstmodellering",
        "language": "Taal",
        "global_inputs": "Algemene invoer",
        "num_products": "Aantal producten",
        "promo_60d": "60-dagen fee promo",
        "promo_yes": "Ja (5% gedurende de eerste ~60 dagen)",
        "promo_no": "Geen promo",
        "fulfillment": "Fulfillment €/bestelling",
        "sample_product": "Sample product €/stuk",
        "sample_shipping": "Sample verzending €/stuk",
        "affiliate_commission": "Affiliate commissie",
        "weeks_per_phase": "Weken / fase",
        "phase_controls": "Fase-instellingen",
        "phase1": "Fase 1 — Koude start",
        "phase2": "Fase 2 — Groei",
        "phase3": "Fase 3 — Doorbraak",
        "ads_rate": "Ads Take Rate",
        "samples_per_week": "Samples / week",
        "affiliate_share": "Affiliate aandeel",
        "product_setup": "Productinstellingen",
        "product_setup_caption": "Stel hieronder elk product in. Preset dient als categorieverwijzing; AOV kan nog steeds handmatig worden aangepast.",
        "generate": "Grafieken genereren",
        "product_mix": "Gebruikte productmix",
        "charts": "Grafieken",
        "overall_weekly_trend": "Totale wekelijkse trend",
        "cumulative_profit_trend": "Cumulatieve winsttrend",
        "phase_by_phase": "Wekelijkse trend per fase",
        "summary": "Samenvatting",
        "phase_summary": "Faseoverzicht",
        "overall_summary": "Totaaloverzicht",
        "break_even_signals": "Break-even signalen",
        "weekly_details": "Wekelijkse details",
        "download_weekly": "Download wekelijkse details CSV",
        "download_phase": "Download faseoverzicht CSV",
        "total_gmv": "Totale GMV",
        "total_profit": "Totale winst",
        "profit_margin": "Winstmarge",
        "first_positive_profit": "Eerste positieve weekwinst",
        "cumulative_break_even": "Cumulatieve break-even",
        "not_reached": "Niet bereikt",
        "weekly_be_label": "Wekelijkse BE",
        "cumulative_be_label": "Cumulatieve BE",
        "week": "Week",
        "product": "Product",
        "preset": "Preset",
        "sales_share": "Verkoopaandeel (%)",
        "sales_share_help": "Hoeft niet op te tellen tot 100. Het systeem normaliseert automatisch.",
        "aov": "AOV (€)",
        "gross_margin": "Brutomarge (%)",
        "fee_type": "Fee type",
        "electronics_fee": "Elektronica (7%)",
        "other_fee": "Overig (9%)",
        "family": "Familie",
        "preset_category": "Preset-categorie",
        "platform_fee_default": "Standaard platform fee",
        "input_error": "Invoerfout",
        "ads_take_rate_col": "Ads Take Rate",
        "affiliate_share_col": "Affiliate aandeel",
        "sales_share_col": "Verkoopaandeel",
        "product_block": "Product",
        "gross_margin_help": "Voer brutomarge in als percentage, bijv. 40 = 40%",
        "gmv_start": "GMV start",
        "gmv_end": "GMV einde",
    },
}

# ======================
# Language
# ======================
with st.sidebar:
    lang = st.selectbox(
        "Language",
        options=["en", "de", "zh", "nl"],
        format_func=lambda x: {
            "en": "English",
            "de": "Deutsch",
            "zh": "简体中文",
            "nl": "Nederlands",
        }[x],
        index=0
    )

T = TEXT[lang]

# ======================
# Localized phase helpers
# ======================
def phase_label(phase_key: str) -> str:
    return T[phase_key]

def phase_key_from_label(label: str) -> str:
    reverse_map = {T[k]: k for k in PHASE_KEYS}
    return reverse_map[label]

def build_phase_inputs():
    phase_inputs = []
    for phase_key in PHASE_KEYS:
        defaults = PHASE_DEFAULTS[phase_key]
        phase_inputs.append({
            "key": phase_key,
            "name": phase_label(phase_key),
            "gmv_start": defaults["gmv_start"],
            "gmv_end": defaults["gmv_end"],
            "ads_take_rate": defaults["ads_take_rate"],
            "samples": defaults["samples"],
            "affiliate_share": defaults["affiliate_share"],
        })
    return phase_inputs

# ======================
# Helpers
# ======================
def normalize_sales_shares(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sales Share (%)"] = pd.to_numeric(df["Sales Share (%)"], errors="coerce").fillna(0).clip(lower=0)
    s = df["Sales Share (%)"].sum()
    if s <= 0:
        raise ValueError("At least one product sales share must be > 0.")
    df["SalesShareNorm"] = df["Sales Share (%)"] / s
    return df

def format_eur_axis(ax):
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("€{x:,.0f}"))

def get_product_platform_fee_rate(phase_idx: int, promo_60d: bool, fee_type_series: pd.Series) -> np.ndarray:
    if promo_60d and phase_idx in (0, 1):
        return np.full(len(fee_type_series), 0.05)
    return fee_type_series.to_numpy()

def add_phase_backgrounds(ax, df: pd.DataFrame):
    phase_ranges = (
        df.groupby("Phase", as_index=False)
        .agg(
            start_week=("Global Week", "min"),
            end_week=("Global Week", "max")
        )
    )

    for _, row in phase_ranges.iterrows():
        phase_name = row["Phase"]
        phase_key = phase_key_from_label(phase_name)
        start_week = row["start_week"]
        end_week = row["end_week"]
        color = PHASE_COLOR_KEYS.get(phase_key, {}).get("bg", "#F5F5F5")

        ax.axvspan(
            start_week - 0.5,
            end_week + 0.5,
            color=color,
            alpha=0.8,
            zorder=0
        )

def render_phase_badges(phase_keys):
    html_parts = []
    for pkey in phase_keys:
        style = PHASE_COLOR_KEYS.get(pkey, {"badge_bg": "#EEEEEE", "badge_text": "#333333"})
        html_parts.append(
            f"""
            <span style="
                display:inline-block;
                padding:8px 14px;
                margin-right:10px;
                margin-bottom:8px;
                border-radius:999px;
                background:{style['badge_bg']};
                color:{style['badge_text']};
                font-weight:600;
                font-size:14px;
            ">
                {phase_label(pkey)}
            </span>
            """
        )
    st.markdown("".join(html_parts), unsafe_allow_html=True)

def phase_weekly_series(
    phase_idx: int,
    phase_name: str,
    gmv_start: float,
    gmv_end: float,
    ads_take_rate: float,
    samples_per_week: int,
    fulfill_cost: float,
    sample_product_cost: float,
    sample_ship_cost: float,
    promo_60d: bool,
    weeks_in_phase: int,
    prod_df: pd.DataFrame,
    affiliate_share: float,
    affiliate_commission_rate: float,
    week_offset: int = 0,
) -> pd.DataFrame:
    gmv_series = np.linspace(gmv_start, gmv_end, weeks_in_phase)

    share = prod_df["SalesShareNorm"].to_numpy()
    aov = prod_df["AOV"].to_numpy()
    gross_margin = prod_df["Gross Margin"].to_numpy()
    fee_type = prod_df["Platform Fee Rate Default"]

    sample_allin_unit = float(sample_product_cost) + float(sample_ship_cost)
    product_fee_rates = get_product_platform_fee_rate(phase_idx, promo_60d, fee_type)

    rows = []
    for i in range(weeks_in_phase):
        gmv = float(gmv_series[i])

        affiliate_gmv = gmv * affiliate_share
        non_affiliate_gmv = gmv * (1 - affiliate_share)

        rev = gmv * share
        orders_p = rev / aov
        orders_total = float(np.sum(orders_p))

        cogs_p = rev * (1.0 - gross_margin)
        cogs_total = float(np.sum(cogs_p))

        platform_fee_p = rev * product_fee_rates
        platform_fee_total = float(np.sum(platform_fee_p))

        ads_cost = gmv * ads_take_rate
        samples_cost = float(samples_per_week) * sample_allin_unit
        fulfillment = orders_total * float(fulfill_cost)
        creator_commission = affiliate_gmv * affiliate_commission_rate

        total_cost = (
            cogs_total
            + ads_cost
            + samples_cost
            + fulfillment
            + platform_fee_total
            + creator_commission
        )
        profit = gmv - total_cost

        rows.append({
            "Phase": phase_name,
            "Week in Phase": i + 1,
            "Global Week": week_offset + i + 1,
            "GMV": gmv,
            "Affiliate GMV": affiliate_gmv,
            "Non-affiliate GMV": non_affiliate_gmv,
            "Creator Commission": creator_commission,
            "Platform Fee": platform_fee_total,
            "Ads Cost": ads_cost,
            "Samples Cost": samples_cost,
            "Fulfillment Cost": fulfillment,
            "COGS": cogs_total,
            "Total Cost": total_cost,
            "Profit": profit,
            "Orders (est.)": orders_total,
            "Ads Take Rate": ads_take_rate,
            "Samples / Week": samples_per_week,
            "Affiliate Share": affiliate_share,
        })

    return pd.DataFrame(rows)

def build_phase_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df_all.groupby("Phase", as_index=False)
        .agg({
            "GMV": "sum",
            "Total Cost": "sum",
            "Profit": "sum",
            "Orders (est.)": "sum",
            "Platform Fee": "sum",
            "Ads Cost": "sum",
            "Samples Cost": "sum",
            "Fulfillment Cost": "sum",
            "Creator Commission": "sum",
            "COGS": "sum"
        })
    )
    summary["Profit Margin"] = np.where(summary["GMV"] > 0, summary["Profit"] / summary["GMV"], 0)
    return summary

def build_overall_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    total_gmv = df_all["GMV"].sum()
    total_cost = df_all["Total Cost"].sum()
    total_profit = df_all["Profit"].sum()
    total_orders = df_all["Orders (est.)"].sum()

    return pd.DataFrame([{
        "Total GMV": total_gmv,
        "Total Cost": total_cost,
        "Total Profit": total_profit,
        "Total Orders (est.)": total_orders,
        "Overall Profit Margin": (total_profit / total_gmv) if total_gmv > 0 else 0
    }])

def first_positive_profit_week(df_all: pd.DataFrame):
    tmp = df_all[df_all["Profit"] > 0]
    if len(tmp) == 0:
        return None
    return int(tmp["Global Week"].iloc[0])

def first_cumulative_break_even_week(df_all: pd.DataFrame):
    tmp = df_all.copy()
    tmp["Cumulative Profit"] = tmp["Profit"].cumsum()
    pos = tmp[tmp["Cumulative Profit"] >= 0]
    if len(pos) == 0:
        return None
    return int(pos["Global Week"].iloc[0])

def get_point_by_week(df: pd.DataFrame, week: int, y_col: str):
    match = df[df["Global Week"] == week]
    if match.empty:
        return None
    return float(match.iloc[0][y_col])

def make_chart(
    df: pd.DataFrame,
    title: str,
    weekly_be_week=None,
    annotate_break_even: bool = False
):
    fig, ax = plt.subplots(figsize=(10, 5))

    add_phase_backgrounds(ax, df)

    ax.plot(df["Global Week"], df["GMV"], marker="o", label="GMV", zorder=3)
    ax.plot(df["Global Week"], df["Total Cost"], marker="o", label="Total Cost", zorder=3)
    ax.plot(df["Global Week"], df["Profit"], marker="o", linewidth=2, label="Profit", zorder=4)
    ax.axhline(0, linewidth=1, zorder=2)

    if annotate_break_even and weekly_be_week is not None:
        y_weekly = get_point_by_week(df, weekly_be_week, "Profit")
        if y_weekly is not None:
            ax.scatter([weekly_be_week], [y_weekly], s=80, zorder=5)
            ax.annotate(
                f"{T['weekly_be_label']}: W{weekly_be_week}",
                xy=(weekly_be_week, y_weekly),
                xytext=(8, 8),
                textcoords="offset points"
            )
            ax.axvline(weekly_be_week, linestyle="--", alpha=0.35, zorder=2)

    ax.set_title(title)
    ax.set_xlabel(T["week"])
    ax.set_ylabel("€")
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend()
    format_eur_axis(ax)
    fig.tight_layout()
    return fig

def make_cumulative_profit_chart(df_all: pd.DataFrame, cumulative_be_week=None):
    tmp = df_all.copy()
    tmp["Cumulative Profit"] = tmp["Profit"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5))

    add_phase_backgrounds(ax, tmp)

    ax.plot(tmp["Global Week"], tmp["Cumulative Profit"], marker="o", linewidth=2, label="Cumulative Profit", zorder=3)
    ax.axhline(0, linewidth=1, zorder=2)

    if cumulative_be_week is not None:
        y_cum = get_point_by_week(tmp, cumulative_be_week, "Cumulative Profit")
        if y_cum is not None:
            ax.scatter([cumulative_be_week], [y_cum], s=80, zorder=5)
            ax.annotate(
                f"{T['cumulative_be_label']}: W{cumulative_be_week}",
                xy=(cumulative_be_week, y_cum),
                xytext=(8, 8),
                textcoords="offset points"
            )
            ax.axvline(cumulative_be_week, linestyle="--", alpha=0.35, zorder=2)

    ax.set_title(T["cumulative_profit_trend"])
    ax.set_xlabel("Global Week")
    ax.set_ylabel("€")
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend()
    format_eur_axis(ax)
    fig.tight_layout()
    return fig

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def build_product_df_from_ui(n_products: int) -> pd.DataFrame:
    rows = []

    for i in range(int(n_products)):
        family = st.session_state[f"family_{i}"]
        preset = st.session_state[f"preset_{i}"]

        fee_label = st.session_state[f"fee_type_{i}"]
        fee_rate = 0.07 if fee_label == T["electronics_fee"] else 0.09

        gross_margin_pct = float(st.session_state[f"gross_margin_pct_{i}"])
        gross_margin_decimal = gross_margin_pct / 100.0

        rows.append({
            "Product": st.session_state[f"product_name_{i}"],
            "Family": family,
            "Preset Category": preset,
            "Sales Share (%)": float(st.session_state[f"sales_share_pct_{i}"]),
            "AOV": float(st.session_state[f"aov_{i}"]),
            "Gross Margin": gross_margin_decimal,
            "Platform Fee Rate Default": float(fee_rate),
        })

    df = pd.DataFrame(rows)

    if df["Product"].isna().any() or (df["Product"].astype(str).str.strip() == "").any():
        raise ValueError("Product name cannot be empty.")

    if (df["AOV"] <= 0).any() or df["AOV"].isna().any():
        raise ValueError("AOV must be > 0 for all products.")

    if ((df["Gross Margin"] < 0.05) | (df["Gross Margin"] > 0.90) | df["Gross Margin"].isna()).any():
        raise ValueError("Gross Margin must be between 5% and 90% for all products.")

    df = normalize_sales_shares(df)
    return df

# ======================
# UI
# ======================
st.title(T["app_title"])
st.caption(T["app_caption"])

with st.sidebar:
    st.header(T["global_inputs"])

    n_products = st.number_input(
        T["num_products"],
        min_value=1,
        max_value=26,
        value=3,
        step=1,
    )

    promo_60d = st.radio(
        T["promo_60d"],
        options=[True, False],
        format_func=lambda x: T["promo_yes"] if x else T["promo_no"],
        index=0,
    )

    fulfillment_per_order = st.number_input(
        T["fulfillment"],
        min_value=0.0,
        value=6.0,
        step=0.5,
    )

    sample_product_cost = st.number_input(
        T["sample_product"],
        min_value=0.0,
        value=30.0,
        step=1.0,
    )

    sample_ship_cost = st.number_input(
        T["sample_shipping"],
        min_value=0.0,
        value=5.0,
        step=1.0,
    )

    affiliate_commission_rate = st.slider(
        T["affiliate_commission"],
        min_value=0.0,
        max_value=0.5,
        value=0.15,
        step=0.01,
    )

    weeks_in_phase = st.slider(
        T["weeks_per_phase"],
        min_value=2,
        max_value=8,
        value=4,
        step=1,
    )

    st.header(T["phase_controls"])
    phase_inputs = build_phase_inputs()

    for i, phase in enumerate(phase_inputs):
        st.subheader(phase["name"])

        phase["ads_take_rate"] = st.slider(
            f"{T['ads_rate']} - {phase['name']}",
            min_value=0.0,
            max_value=0.30,
            value=float(phase["ads_take_rate"]),
            step=0.01,
            key=f"ads_{i}"
        )

        phase["samples"] = st.number_input(
            f"{T['samples_per_week']} - {phase['name']}",
            min_value=0,
            value=int(phase["samples"]),
            step=1,
            key=f"samples_{i}"
        )

        phase["affiliate_share"] = st.slider(
            f"{T['affiliate_share']} - {phase['name']}",
            min_value=0.0,
            max_value=1.0,
            value=float(phase["affiliate_share"]),
            step=0.01,
            key=f"aff_{i}"
        )

# ======================
# Product Setup
# ======================
st.subheader(T["product_setup"])
st.caption(T["product_setup_caption"])

for i in range(int(n_products)):
    if f"product_name_{i}" not in st.session_state:
        st.session_state[f"product_name_{i}"] = LETTERS[i]

    if f"family_{i}" not in st.session_state:
        st.session_state[f"family_{i}"] = "Home & Living"

    current_family = st.session_state[f"family_{i}"]
    available_presets = list(AOV_PRESETS[current_family].keys())

    if f"preset_{i}" not in st.session_state or st.session_state[f"preset_{i}"] not in available_presets:
        st.session_state[f"preset_{i}"] = available_presets[0]

    current_preset = st.session_state[f"preset_{i}"]
    default_aov = AOV_PRESETS[current_family][current_preset]

    if f"aov_{i}" not in st.session_state:
        st.session_state[f"aov_{i}"] = float(default_aov)

    if f"sales_share_pct_{i}" not in st.session_state:
        st.session_state[f"sales_share_pct_{i}"] = 100.0 / float(n_products)

    if f"gross_margin_pct_{i}" not in st.session_state:
        st.session_state[f"gross_margin_pct_{i}"] = 40.0

    if f"fee_type_{i}" not in st.session_state:
        st.session_state[f"fee_type_{i}"] = T["other_fee"]

    with st.container(border=True):
        st.markdown(f"**{T['product_block']} {i + 1}**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.text_input(T["product"], key=f"product_name_{i}")

        with col2:
            st.selectbox(T["family"], options=list(AOV_PRESETS.keys()), key=f"family_{i}")

        selected_family = st.session_state[f"family_{i}"]
        updated_presets = list(AOV_PRESETS[selected_family].keys())

        if st.session_state[f"preset_{i}"] not in updated_presets:
            st.session_state[f"preset_{i}"] = updated_presets[0]

        with col3:
            st.selectbox(T["preset"], options=updated_presets, key=f"preset_{i}")

        selected_preset = st.session_state[f"preset_{i}"]
        preset_aov = AOV_PRESETS[selected_family][selected_preset]

        if st.session_state[f"aov_{i}"] <= 0:
            st.session_state[f"aov_{i}"] = float(preset_aov)

        col4, col5, col6 = st.columns(3)

        with col4:
            st.number_input(
                T["sales_share"],
                min_value=0.0,
                step=1.0,
                key=f"sales_share_pct_{i}",
                help=T["sales_share_help"]
            )

        with col5:
            st.number_input(
                T["aov"],
                min_value=0.01,
                step=1.0,
                key=f"aov_{i}"
            )

        with col6:
            st.selectbox(
                T["fee_type"],
                options=[T["electronics_fee"], T["other_fee"]],
                key=f"fee_type_{i}"
            )

        st.number_input(
            T["gross_margin"],
            min_value=5.0,
            max_value=90.0,
            step=1.0,
            key=f"gross_margin_pct_{i}",
            help=T["gross_margin_help"]
        )

generate = st.button(T["generate"], type="primary")

# ======================
# Run
# ======================
if generate:
    try:
        prod_df = build_product_df_from_ui(int(n_products))

        mix_display = prod_df[[
            "Product", "Family", "Preset Category", "Sales Share (%)", "SalesShareNorm",
            "AOV", "Gross Margin", "Platform Fee Rate Default"
        ]].copy()

        mix_display.columns = [
            T["product"],
            T["family"],
            T["preset_category"],
            T["sales_share"],
            "SalesShareNorm",
            T["aov"],
            T["gross_margin"],
            T["platform_fee_default"],
        ]

        mix_display[T["sales_share"]] = mix_display[T["sales_share"]].map(lambda v: f"{v:,.1f}%")
        mix_display["SalesShareNorm"] = mix_display["SalesShareNorm"].map(lambda v: f"{v:.0%}")
        mix_display[T["aov"]] = mix_display[T["aov"]].map(lambda v: f"€{v:,.2f}")
        mix_display[T["gross_margin"]] = mix_display[T["gross_margin"]].map(lambda v: f"{v:.0%}")
        mix_display[T["platform_fee_default"]] = mix_display[T["platform_fee_default"]].map(lambda v: f"{v:.0%}")

        st.subheader(T["product_mix"])
        st.dataframe(mix_display, use_container_width=True)

        all_tables = []
        for idx, phase in enumerate(phase_inputs):
            dfp = phase_weekly_series(
                idx,
                phase["name"],
                phase["gmv_start"],
                phase["gmv_end"],
                phase["ads_take_rate"],
                phase["samples"],
                float(fulfillment_per_order),
                float(sample_product_cost),
                float(sample_ship_cost),
                bool(promo_60d),
                int(weeks_in_phase),
                prod_df,
                float(phase["affiliate_share"]),
                float(affiliate_commission_rate),
                week_offset=idx * int(weeks_in_phase),
            )
            all_tables.append(dfp)

        df_all = pd.concat(all_tables, ignore_index=True)

        phase_summary = build_phase_summary(df_all)
        overall_summary = build_overall_summary(df_all)

        single_week_break_even = first_positive_profit_week(df_all)
        cumulative_break_even = first_cumulative_break_even_week(df_all)

        metric1, metric2, metric3 = st.columns(3)
        with metric1:
            st.metric(T["total_gmv"], f"€{overall_summary.iloc[0]['Total GMV']:,.0f}")
        with metric2:
            st.metric(T["total_profit"], f"€{overall_summary.iloc[0]['Total Profit']:,.0f}")
        with metric3:
            st.metric(T["profit_margin"], f"{overall_summary.iloc[0]['Overall Profit Margin']:.1%}")

        st.subheader(T["charts"])
        render_phase_badges([p["key"] for p in phase_inputs])

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.pyplot(
                make_chart(
                    df_all,
                    T["overall_weekly_trend"],
                    weekly_be_week=single_week_break_even,
                    annotate_break_even=True
                )
            )

        with chart_col2:
            st.pyplot(
                make_cumulative_profit_chart(
                    df_all,
                    cumulative_be_week=cumulative_break_even
                )
            )

        st.subheader(T["phase_by_phase"])
        phase_tabs = st.tabs([p["name"] for p in phase_inputs])

        for tab, phase in zip(phase_tabs, phase_inputs):
            with tab:
                phase_df = df_all[df_all["Phase"] == phase["name"]].copy()
                phase_weekly_be = None
                tmp_phase_be = phase_df[phase_df["Profit"] > 0]
                if not tmp_phase_be.empty:
                    phase_weekly_be = int(tmp_phase_be["Global Week"].iloc[0])

                render_phase_badges([phase["key"]])

                st.pyplot(
                    make_chart(
                        phase_df,
                        phase["name"],
                        weekly_be_week=phase_weekly_be,
                        annotate_break_even=True
                    )
                )

        phase_summary_fmt = phase_summary.copy()
        for col in [
            "GMV", "Total Cost", "Profit", "Orders (est.)", "Platform Fee",
            "Ads Cost", "Samples Cost", "Fulfillment Cost", "Creator Commission", "COGS"
        ]:
            phase_summary_fmt[col] = phase_summary_fmt[col].map(lambda v: f"{v:,.0f}")
        phase_summary_fmt["Profit Margin"] = phase_summary["Profit Margin"].map(lambda v: f"{v:.1%}")

        overall_summary_fmt = overall_summary.copy()
        overall_summary_fmt["Total GMV"] = overall_summary_fmt["Total GMV"].map(lambda v: f"{v:,.0f}")
        overall_summary_fmt["Total Cost"] = overall_summary_fmt["Total Cost"].map(lambda v: f"{v:,.0f}")
        overall_summary_fmt["Total Profit"] = overall_summary_fmt["Total Profit"].map(lambda v: f"{v:,.0f}")
        overall_summary_fmt["Total Orders (est.)"] = overall_summary_fmt["Total Orders (est.)"].map(lambda v: f"{v:,.0f}")
        overall_summary_fmt["Overall Profit Margin"] = overall_summary["Overall Profit Margin"].map(lambda v: f"{v:.1%}")

        st.subheader(T["summary"])
        sum_col1, sum_col2 = st.columns(2)
        with sum_col1:
            st.markdown(f"**{T['phase_summary']}**")
            st.dataframe(phase_summary_fmt, use_container_width=True)
        with sum_col2:
            st.markdown(f"**{T['overall_summary']}**")
            st.dataframe(overall_summary_fmt, use_container_width=True)

        st.subheader(T["break_even_signals"])
        be_col1, be_col2 = st.columns(2)
        with be_col1:
            if single_week_break_even is not None:
                st.success(f"{T['first_positive_profit']}: {T['week']} {single_week_break_even}")
            else:
                st.warning(f"{T['first_positive_profit']}: {T['not_reached']}")

        with be_col2:
            if cumulative_break_even is not None:
                st.success(f"{T['cumulative_break_even']}: {T['week']} {cumulative_break_even}")
            else:
                st.warning(f"{T['cumulative_break_even']}: {T['not_reached']}")

        st.subheader(T["weekly_details"])
        df_all_display = df_all.copy()
        money_cols = [
            "GMV", "Affiliate GMV", "Non-affiliate GMV", "Creator Commission",
            "Platform Fee", "Ads Cost", "Samples Cost", "Fulfillment Cost",
            "COGS", "Total Cost", "Profit"
        ]
        for col in money_cols:
            df_all_display[col] = df_all_display[col].map(lambda v: f"{v:,.2f}")
        df_all_display["Orders (est.)"] = df_all_display["Orders (est.)"].map(lambda v: f"{v:,.2f}")
        df_all_display[T["ads_take_rate_col"]] = df_all["Ads Take Rate"].map(lambda v: f"{v:.0%}")
        df_all_display[T["affiliate_share_col"]] = df_all["Affiliate Share"].map(lambda v: f"{v:.0%}")
        if "Ads Take Rate" in df_all_display.columns:
            df_all_display = df_all_display.drop(columns=["Ads Take Rate"])

        st.dataframe(df_all_display, use_container_width=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                T["download_weekly"],
                data=to_csv_bytes(df_all),
                file_name="weekly_details.csv",
                mime="text/csv"
            )
        with dl2:
            st.download_button(
                T["download_phase"],
                data=to_csv_bytes(phase_summary),
                file_name="phase_summary.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"{T['input_error']}: {e}")
