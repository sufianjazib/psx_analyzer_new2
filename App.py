import os
import io
import math
import json
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from groq import Groq
except Exception:
    Groq = None

# ============================================================
# PSX Multibagger Engine
# Streamlit + Groq
#
# Data options:
# 1) Upload a CSV/XLSX containing PSX fundamentals.
# 2) Paste a CSV directly into the text box.
#
# The app deliberately does NOT promise or predict a guaranteed
# 10% monthly return. It produces a quantitative score, scenarios,
# buy-zone framework, and risk/thesis-break monitoring.
# ============================================================

st.set_page_config(
    page_title="PSX Multibagger Engine",
    page_icon="📈",
    layout="wide",
)

DEFAULT_COLUMNS_HELP = """
Recommended columns (case-insensitive; aliases are supported):
ticker, company, price, market_cap, revenue, revenue_prev,
eps, eps_prev, eps_3y_ago, eps_5y_ago,
net_profit, net_profit_prev, roe, roic,
operating_cash_flow, free_cash_flow, debt, cash,
ebitda, shares, dividend_yield, pe, pb, ev_ebitda,
promoter_holding, free_float, avg_volume,
capacity_growth, utilization, catalyst_score, governance_score
"""

ALIASES = {
    "symbol": "ticker",
    "code": "ticker",
    "stock": "ticker",
    "name": "company",
    "company_name": "company",
    "last_price": "price",
    "close": "price",
    "market_capitalisation": "market_cap",
    "market_capitalization": "market_cap",
    "sales": "revenue",
    "sales_current": "revenue",
    "sales_previous": "revenue_prev",
    "eps_current": "eps",
    "eps_current_year": "eps",
    "eps_previous": "eps_prev",
    "previous_eps": "eps_prev",
    "profit": "net_profit",
    "pat": "net_profit",
    "profit_after_tax": "net_profit",
    "previous_profit": "net_profit_prev",
    "roe_percent": "roe",
    "roic_percent": "roic",
    "ocf": "operating_cash_flow",
    "cfo": "operating_cash_flow",
    "fcf": "free_cash_flow",
    "net_debt": "debt",
    "cash_and_equivalents": "cash",
    "dividend_yield_percent": "dividend_yield",
    "pe_ratio": "pe",
    "price_earnings": "pe",
    "price_to_book": "pb",
    "ev_ebitda_ratio": "ev_ebitda",
    "sponsor_holding": "promoter_holding",
    "promoter_ownership": "promoter_holding",
    "free_float_percent": "free_float",
    "average_volume": "avg_volume",
}

REQUIRED_MIN = ["ticker", "price", "market_cap", "eps"]


def clean_col(c):
    c = str(c).strip().lower()
    c = c.replace("%", "percent")
    c = c.replace("/", "_")
    c = c.replace("-", "_")
    c = c.replace(" ", "_")
    return ALIASES.get(c, c)


def clean_numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("PKR", "", regex=False)
        .str.replace("Rs.", "", regex=False)
        .str.replace("Rs", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(s, errors="coerce")


def normalize_df(df):
    df = df.copy()
    df.columns = [clean_col(c) for c in df.columns]

    # Remove duplicate columns after normalization.
    df = df.loc[:, ~df.columns.duplicated()]

    numeric_candidates = [
        "price", "market_cap", "revenue", "revenue_prev", "eps", "eps_prev",
        "eps_3y_ago", "eps_5y_ago", "net_profit", "net_profit_prev", "roe",
        "roic", "operating_cash_flow", "free_cash_flow", "debt", "cash",
        "ebitda", "shares", "dividend_yield", "pe", "pb", "ev_ebitda",
        "promoter_holding", "free_float", "avg_volume", "capacity_growth",
        "utilization", "catalyst_score", "governance_score"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = clean_numeric(df[c])

    if "ticker" not in df.columns:
        raise ValueError("CSV must contain a ticker/symbol/code column.")

    for c in REQUIRED_MIN:
        if c not in df.columns:
            raise ValueError(f"CSV is missing required column: {c}")

    # Infer PE when possible.
    if "pe" not in df.columns:
        df["pe"] = np.where(df["eps"] > 0, df["price"] / df["eps"], np.nan)

    # Infer market cap if shares are supplied.
    if "market_cap" not in df.columns and "shares" in df.columns:
        df["market_cap"] = df["price"] * df["shares"]

    return df


def pct_growth(current, previous):
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return np.nan
    return (current / previous - 1) * 100


def cagr(current, old, years):
    if pd.isna(current) or pd.isna(old) or old <= 0 or current <= 0 or years <= 0:
        return np.nan
    return ((current / old) ** (1 / years) - 1) * 100


def score_range(value, thresholds):
    """
    thresholds = [(minimum_value, points), ...]
    Highest qualifying threshold wins.
    """
    if pd.isna(value):
        return 0
    pts = 0
    for minimum, points in thresholds:
        if value >= minimum:
            pts = points
    return pts


def quality_score(r):
    # 30 points
    revenue_growth = pct_growth(r.get("revenue", np.nan), r.get("revenue_prev", np.nan))
    eps_growth = pct_growth(r.get("eps", np.nan), r.get("eps_prev", np.nan))
    roe = r.get("roe", np.nan)
    roic = r.get("roic", np.nan)

    s = 0
    s += score_range(revenue_growth, [(10, 2), (15, 4), (25, 5)])
    s += score_range(eps_growth, [(10, 3), (20, 5), (30, 7)])
    s += score_range(roe, [(12, 2), (18, 4), (20, 5)])
    s += score_range(roic, [(8, 2), (12, 3), (15, 4)])

    # Cash-flow quality: CFO / PAT
    pat = r.get("net_profit", np.nan)
    cfo = r.get("operating_cash_flow", np.nan)
    cfo_pat = np.nan
    if pd.notna(pat) and pat > 0 and pd.notna(cfo):
        cfo_pat = cfo / pat
    s += score_range(cfo_pat, [(0.70, 1), (0.90, 2), (1.10, 3)])

    # Margin improvement: EBITDA / revenue, if available.
    ebitda = r.get("ebitda", np.nan)
    rev = r.get("revenue", np.nan)
    margin = np.nan if pd.isna(ebitda) or pd.isna(rev) or rev == 0 else ebitda / rev
    s += score_range(margin, [(0.10, 1), (0.15, 2), (0.20, 3)])

    # Governance is kept separate but contributes here only if provided.
    gov = r.get("governance_score", np.nan)
    if pd.notna(gov):
        s += np.clip(gov / 10 * 3, 0, 3)

    return min(float(s), 30.0)


def growth_score(r):
    # 35 points
    mcap = r.get("market_cap", np.nan)
    eps = r.get("eps", np.nan)
    eps_prev = r.get("eps_prev", np.nan)
    eps3 = r.get("eps_3y_ago", np.nan)
    eps5 = r.get("eps_5y_ago", np.nan)

    eps_yoy = pct_growth(eps, eps_prev)
    eps_3y = cagr(eps, eps3, 3)
    eps_5y = cagr(eps, eps5, 5)

    s = 0

    # Small/mid-cap optional bonus. The thresholds are intentionally
    # broad so the model doesn't exclude larger companies automatically.
    if pd.notna(mcap):
        if mcap <= 10e9:
            s += 7
        elif mcap <= 20e9:
            s += 6
        elif mcap <= 40e9:
            s += 5
        elif mcap <= 100e9:
            s += 3
        elif mcap <= 300e9:
            s += 1

    s += score_range(eps_yoy, [(10, 2), (20, 5), (30, 8)])

    if pd.notna(eps_3y):
        s += score_range(eps_3y, [(10, 2), (15, 4), (25, 6)])
    elif pd.notna(eps_5y):
        s += score_range(eps_5y, [(10, 1), (15, 3), (25, 5)])

    # Capacity growth / utilization.
    cap = r.get("capacity_growth", np.nan)
    util = r.get("utilization", np.nan)
    s += score_range(cap, [(5, 1), (10, 3), (20, 5)])
    s += score_range(util, [(60, 1), (75, 2)])

    # Catalyst score 0-10 -> up to 4 points.
    catalyst = r.get("catalyst_score", np.nan)
    if pd.notna(catalyst):
        s += np.clip(catalyst / 10 * 4, 0, 4)

    return min(float(s), 35.0)


def risk_valuation_score(r):
    # 35 points
    price = r.get("price", np.nan)
    eps = r.get("eps", np.nan)
    pe = r.get("pe", np.nan)
    roe = r.get("roe", np.nan)
    debt = r.get("debt", np.nan)
    ebitda = r.get("ebitda", np.nan)
    cash = r.get("cash", np.nan)
    cfo = r.get("operating_cash_flow", np.nan)
    pat = r.get("net_profit", np.nan)
    div_yield = r.get("dividend_yield", np.nan)

    s = 0

    # Valuation 12
    if pd.notna(pe):
        if 0 < pe <= 6:
            s += 12
        elif pe <= 8:
            s += 10
        elif pe <= 10:
            s += 8
        elif pe <= 14:
            s += 5
        elif pe <= 20:
            s += 2
    else:
        # If PE unavailable, don't fabricate a score.
        s += 0

    # Balance sheet 8: debt / EBITDA proxy
    if pd.notna(debt) and pd.notna(ebitda) and ebitda > 0:
        leverage = debt / ebitda
        if leverage <= 0:
            s += 8
        elif leverage <= 1:
            s += 7
        elif leverage <= 2:
            s += 5
        elif leverage <= 3:
            s += 3
        elif leverage <= 4:
            s += 1
    elif pd.notna(debt) and pd.notna(cash):
        net_debt = debt - cash
        if net_debt <= 0:
            s += 7
        elif debt > 0:
            s += 3
    else:
        # Unknown is not treated as safe.
        s += 0

    # Cash-flow quality 6
    if pd.notna(cfo) and pd.notna(pat) and pat > 0:
        ratio = cfo / pat
        if ratio >= 1.1:
            s += 6
        elif ratio >= 0.9:
            s += 4
        elif ratio >= 0.7:
            s += 2

    # Dividend 3 (not required for multibagger status)
    if pd.notna(div_yield):
        if div_yield >= 12:
            s += 3
        elif div_yield >= 6:
            s += 2
        elif div_yield > 0:
            s += 1

    # Governance 6
    gov = r.get("governance_score", np.nan)
    if pd.notna(gov):
        s += np.clip(gov / 10 * 6, 0, 6)

    return min(float(s), 35.0)


def calculate_scores(df):
    out = df.copy()
    out["quality_score"] = out.apply(quality_score, axis=1)
    out["growth_score"] = out.apply(growth_score, axis=1)
    out["risk_value_score"] = out.apply(risk_valuation_score, axis=1)
    out["multibagger_score"] = (
        out["quality_score"] + out["growth_score"] + out["risk_value_score"]
    ).round(1)

    def status(x):
        if x >= 85:
            return "MULTIBAGGER CANDIDATE"
        if x >= 75:
            return "WATCHLIST"
        if x >= 65:
            return "MONITOR"
        return "NO MULTIBAGGER STATUS"

    out["status"] = out["multibagger_score"].apply(status)

    # Fundamental hard gates.
    out["hard_gate"] = (
        (out["eps"] > 0)
        & (out["price"] > 0)
        & (out["market_cap"] > 0)
    )

    # Buy setup score: not a return prediction.
    out["buy_setup_score"] = (
        out["quality_score"] * 0.35
        + out["growth_score"] * 0.20
        + out["risk_value_score"] * 0.25
    )

    # Catalyst/technical fields, if supplied.
    if "catalyst_score" in out.columns:
        out["buy_setup_score"] += np.clip(out["catalyst_score"].fillna(0) / 10 * 10, 0, 10)
    if "price_change_1m" in out.columns:
        out["buy_setup_score"] += np.clip(
            out["price_change_1m"].fillna(0).apply(lambda x: 5 if 0 < x <= 15 else 0), 0, 5
        )

    out["buy_setup_score"] = out["buy_setup_score"].clip(0, 100).round(1)

    def buy_label(x, gate):
        if not gate:
            return "NO BUY — DATA/GATE FAIL"
        if x >= 80:
            return "STRONG BUY SETUP"
        if x >= 70:
            return "BUY / ACCUMULATE SETUP"
        if x >= 60:
            return "WATCH"
        return "NO BUY SETUP"

    out["buy_signal"] = [
        buy_label(score, gate) for score, gate in zip(out["buy_setup_score"], out["hard_gate"])
    ]

    return out.sort_values(["multibagger_score", "buy_setup_score"], ascending=False)


def intrinsic_value(r, terminal_pe=10.0, eps_growth=20.0, years=3):
    eps = r.get("eps", np.nan)
    if pd.isna(eps) or eps <= 0:
        return np.nan
    future_eps = eps * (1 + eps_growth / 100) ** years
    return future_eps * terminal_pe


def scenario_table(r, years=3):
    eps = r.get("eps", np.nan)
    price = r.get("price", np.nan)
    pe = r.get("pe", np.nan)

    if pd.isna(eps) or eps <= 0 or pd.isna(price):
        return pd.DataFrame()

    # Growth assumptions are scenario assumptions, not predictions.
    scenarios = [
        ("Bear", 8.0, 10.0),
        ("Base", 18.0, 10.0),
        ("Bull", 30.0, 12.0),
    ]

    rows = []
    for name, growth, terminal_pe in scenarios:
        future_eps = eps * (1 + growth / 100) ** years
        future_value = future_eps * terminal_pe
        upside = (future_value / price - 1) * 100
        rows.append(
            {
                "Scenario": name,
                "Assumed EPS CAGR": f"{growth:.0f}%",
                "Terminal P/E": terminal_pe,
                f"EPS in {years}Y": round(future_eps, 2),
                f"Value in {years}Y": round(future_value, 2),
                "Potential change": round(upside, 1),
            }
        )
    return pd.DataFrame(rows)


def groq_analysis(row, scenario_df, api_key, model):
    if not api_key or Groq is None:
        return None

    client = Groq(api_key=api_key)

    data = row.to_dict()
    # Keep prompt reasonably small.
    safe_data = {}
    for k, v in data.items():
        if isinstance(v, (np.integer, np.floating)):
            v = float(v)
        if pd.notna(v) if not isinstance(v, str) else True:
            safe_data[k] = v

    prompt = f"""
You are a Pakistan Stock Exchange equity research analyst.
Analyze the supplied company using a neutral, evidence-based framework.

IMPORTANT:
- Do not guarantee returns.
- Do not claim that a stock must hit a target.
- Do not say it will definitely return 10% per month.
- Clearly distinguish reported data from assumptions.
- Identify missing data.
- Focus on earnings quality, valuation, catalyst, balance sheet, cash flow,
  execution risks, and what would invalidate the thesis.

Company data:
{json.dumps(safe_data, default=str, indent=2)}

Scenario model:
{scenario_df.to_dict(orient="records") if not scenario_df.empty else "Unavailable"}

Return these headings:
1. Investment thesis
2. Earnings drivers
3. Valuation
4. Catalysts
5. Key risks
6. Thesis-break triggers
7. Data gaps
8. What to monitor next quarter
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a rigorous PSX equity research analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------- UI ----------------
st.title("📈 PSX Multibagger Engine")
st.caption(
    "Quantitative PSX screening + valuation scenarios + Groq research notes. "
    "No return guarantee: scores are decision-support signals, not promises."
)

with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="You can also set GROQ_API_KEY in Streamlit secrets/environment variables.",
    )

    model = st.selectbox(
        "Groq model",
        [
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ],
        index=0,
    )

    st.divider()
    st.markdown("### Score gates")
    st.write("85+ → Multibagger Candidate")
    st.write("75–84 → Watchlist")
    st.write("65–74 → Monitor")
    st.write("<65 → No Multibagger Status")

    st.divider()
    st.info(
        "The app does not promise 10% monthly returns or guaranteed multibaggers. "
        "Historical back-testing should be used before treating the score as a strategy."
    )

tab1, tab2, tab3 = st.tabs(["📂 Data", "🔎 Screen", "🧠 Research"])

with tab1:
    st.subheader("Load PSX data")
    uploaded = st.file_uploader(
        "Upload PSX fundamentals CSV or Excel",
        type=["csv", "xlsx", "xls"],
    )

    pasted = st.text_area(
        "Or paste CSV data",
        height=180,
        placeholder="ticker,company,price,market_cap,eps,eps_prev,revenue,revenue_prev,roe,roic,...",
    )

    st.markdown("**Recommended fields:**")
    st.code(DEFAULT_COLUMNS_HELP)

    df = None
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.success(f"Loaded {len(df):,} rows from {uploaded.name}")
        except Exception as e:
            st.error(f"Could not read the file: {e}")

    elif pasted.strip():
        try:
            df = pd.read_csv(io.StringIO(pasted))
            st.success(f"Loaded {len(df):,} rows from pasted CSV.")
        except Exception as e:
            st.error(f"Could not parse pasted CSV: {e}")

    if df is not None:
        try:
            df = normalize_df(df)
            st.session_state["raw_df"] = df
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(str(e))

    st.markdown(
        """
**Important:** The app does not scrape PSX automatically by default. This avoids
silently depending on an undocumented endpoint and lets you control the data source.
You can export/supply PSX data and run the model locally or on Streamlit Cloud.
"""
    )

with tab2:
    st.subheader("Run the Multibagger Engine")

    if "raw_df" not in st.session_state:
        st.warning("Load a CSV/Excel file or paste CSV data in the Data tab first.")
    else:
        raw = st.session_state["raw_df"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            min_mcap = st.number_input(
                "Minimum market cap (PKR bn)",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )
        with col2:
            max_mcap = st.number_input(
                "Maximum market cap (PKR bn)",
                min_value=0.0,
                value=500.0,
                step=5.0,
            )
        with col3:
            min_score = st.slider("Minimum Multibagger Score", 0, 100, 65)
        with col4:
            only_gate = st.checkbox("Only hard-gate pass", value=True)

        work = raw.copy()
        work = work[
            (work["market_cap"] >= min_mcap * 1e9)
            & (work["market_cap"] <= max_mcap * 1e9)
        ]

        scored = calculate_scores(work)
        if only_gate:
            scored = scored[scored["hard_gate"]]

        scored = scored[scored["multibagger_score"] >= min_score]

        display_cols = [
            "ticker", "company", "price", "market_cap",
            "eps", "roe", "pe",
            "quality_score", "growth_score", "risk_value_score",
            "multibagger_score", "buy_setup_score",
            "status", "buy_signal",
        ]
        display_cols = [c for c in display_cols if c in scored.columns]

        st.markdown("### Results")
        st.dataframe(
            scored[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        csv = scored.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download scored stocks CSV",
            data=csv,
            file_name="psx_multibagger_scores.csv",
            mime="text/csv",
        )

        st.markdown("### Top candidates")
        top = scored.head(10)
        if top.empty:
            st.info("No companies meet the selected filters. Relax the filters or add more data.")
        else:
            cols = st.columns(min(5, len(top)))
            for i, (_, r) in enumerate(top.iterrows()):
                with cols[i % len(cols)]:
                    st.metric(
                        label=str(r["ticker"]),
                        value=f'{r["multibagger_score"]:.1f}/100',
                        delta=str(r["buy_signal"]),
                    )

with tab3:
    st.subheader("Company-level research")

    if "raw_df" not in st.session_state:
        st.warning("Load data first.")
    else:
        scored = calculate_scores(st.session_state["raw_df"])
        tickers = scored["ticker"].astype(str).tolist()

        if not tickers:
            st.info("No tickers available.")
        else:
            selected = st.selectbox("Select stock", tickers)
            r = scored[scored["ticker"].astype(str) == selected].iloc[0]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Multibagger Score", f'{r["multibagger_score"]:.1f}/100')
            c2.metric("Quality", f'{r["quality_score"]:.1f}/30')
            c3.metric("Growth", f'{r["growth_score"]:.1f}/35')
            c4.metric("Risk/Value", f'{r["risk_value_score"]:.1f}/35')
            c5.metric("Buy Setup", f'{r["buy_setup_score"]:.1f}/100')

            st.markdown(f"### {r['ticker']} — {r.get('company', '')}")
            st.write(f"**Status:** {r['status']}  |  **Buy signal:** {r['buy_signal']}")

            scenarios = scenario_table(r, years=3)
            if not scenarios.empty:
                st.markdown("### 3-year scenario framework")
                st.dataframe(scenarios, use_container_width=True, hide_index=True)
                st.caption(
                    "Scenario values are mathematical illustrations using assumed EPS CAGR "
                    "and terminal P/E. They are not forecasts or guarantees."
                )

            st.markdown("### Thesis-break checklist")
            st.markdown(
                """
- Earnings growth materially below the thesis for multiple reporting periods.
- Major expansion/capacity project delayed, cancelled, or economics deteriorate.
- Leverage rises beyond the company's sustainable capacity.
- Operating cash flow persistently diverges from reported profit.
- Governance/accounting issue materially changes the investment case.
- Valuation becomes disconnected from achievable earnings.
"""
            )

            if st.button("🧠 Generate Groq research note", type="primary"):
                if not api_key:
                    st.warning("Enter a Groq API key in the sidebar first.")
                elif Groq is None:
                    st.error("Groq package is unavailable. Reinstall requirements.txt.")
                else:
                    with st.spinner("Generating research note..."):
                        try:
                            note = groq_analysis(r, scenarios, api_key, model)
                            st.markdown(note)
                        except Exception as e:
                            st.error(f"Groq request failed: {e}")

st.divider()
st.caption(
    f"PSX Multibagger Engine | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
    "For research/decision support only."
)
