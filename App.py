import os
import re
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="PSX Multibagger Engine",
    page_icon="📈",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. HELPER FUNCTIONS & COLUMN ALIASING
# -----------------------------------------------------------------------------

COLUMN_ALIASES = {
    "ticker": ["ticker", "symbol", "code", "scrip"],
    "company": ["company", "company_name", "name"],
    "price": ["price", "close", "current_price", "last_price"],
    "market_cap": ["market_cap", "mcap", "market_capitalization"],
    "revenue": ["revenue", "sales", "turnover"],
    "revenue_prev": ["revenue_prev", "sales_prev", "sales_last_year"],
    "eps": ["eps", "earnings_per_share"],
    "eps_prev": ["eps_prev", "eps_last_year"],
    "eps_3y_ago": ["eps_3y_ago", "eps_3y"],
    "eps_5y_ago": ["eps_5y_ago", "eps_5y"],
    "net_profit": ["net_profit", "pat", "net_income"],
    "net_profit_prev": ["net_profit_prev", "pat_prev"],
    "roe": ["roe", "roe_percent", "return_on_equity"],
    "roic": ["roic", "roic_percent"],
    "operating_cash_flow": ["operating_cash_flow", "ocf", "cash_from_operations"],
    "free_cash_flow": ["free_cash_flow", "fcf"],
    "debt": ["debt", "total_debt", "liabilities"],
    "cash": ["cash", "cash_and_equivalents"],
    "ebitda": ["ebitda"],
    "dividend_yield": ["dividend_yield", "div_yield", "yield"],
    "pe": ["pe", "p_e", "pe_ratio"],
    "capacity_growth": ["capacity_growth", "capacity_expansion"],
    "utilization": ["utilization", "capacity_utilization"],
    "catalyst_score": ["catalyst_score", "catalyst"],
    "governance_score": ["governance_score", "governance"]
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names based on common alias mapping."""
    df_cols = {str(c).strip().lower(): c for c in df.columns}
    rename_dict = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df_cols:
                rename_dict[df_cols[alias]] = standard_name
                break
    return df.rename(columns=rename_dict)

# -----------------------------------------------------------------------------
# 2. PSX WEB DATA SCRAPER
# -----------------------------------------------------------------------------

@st.cache_data(ttl=900)  # Cache results for 15 minutes
def fetch_psx_web_data(symbol: str) -> dict:
    """
    Directly scrapes live stock quote and stat data from the PSX Portal.
    """
    symbol = symbol.strip().upper()
    url = f"https://dps.psx.com.pk/company/{symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }

    try:
        response = requests.get(url, headers=headers, timeout=12)
        if response.status_code != 200:
            return {"error": f"Symbol '{symbol}' not found or PSX Portal unavailable."}

        soup = BeautifulSoup(response.text, "html.parser")
        html_text = response.text

        # 1. Parse Stock Price
        price = None
        price_elem = soup.find("div", class_="quote__close") or soup.find("div", class_="stats_value")
        if price_elem:
            p_match = re.search(r"[\d,]+\.?\d*", price_elem.text)
            if p_match:
                price = float(p_match.group(0).replace(",", ""))

        # 2. Helper to extract metrics via Regex
        def extract_metric(patterns, text_source):
            for pat in patterns:
                match = re.search(pat, text_source, re.IGNORECASE)
                if match:
                    val_str = match.group(1).replace(",", "").strip()
                    try:
                        return float(val_str)
                    except ValueError:
                        continue
            return None

        mcap = extract_metric([
            r'Market\s*Cap(?:italisation)?\s*<\/div>\s*<div[^>]*>\s*Rs\.?\s*([\d,]+)',
            r'Market\s*Cap[^\d]*([\d,]+)'
        ], html_text)

        eps = extract_metric([
            r'EPS\s*\(TTM\)\s*<\/div>\s*<div[^>]*>\s*Rs\.?\s*(-?[\d\.]+)',
            r'EPS[^\d]*(-?[\d\.]+)'
        ], html_text)

        pe = extract_metric([
            r'P\/E\s*Ratio\s*<\/div>\s*<div[^>]*>\s*([\d\.]+)',
            r'P\/E[^\d]*([\d\.]+)'
        ], html_text)

        dy = extract_metric([
            r'Div(?:idend)?\s*Yield\s*<\/div>\s*<div[^>]*>\s*([\d\.]+)%',
            r'Yield[^\d]*([\d\.]+)'
        ], html_text) or 0.0

        if pe is None and price and eps and eps > 0:
            pe = round(price / eps, 2)

        return {
            "ticker": symbol,
            "price": price,
            "market_cap": mcap,
            "eps": eps,
            "pe": pe,
            "dividend_yield": dy,
            "source": "PSX Data Portal"
        }

    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {str(e)}"}

# -----------------------------------------------------------------------------
# 3. MULTIBAGGER EVALUATION ENGINE
# -----------------------------------------------------------------------------

def evaluate_multibagger(row: pd.Series) -> dict:
    """Calculates Business Quality, Multibagger Potential, and Risk/Valuation scores safely."""
    
    def safe_val(key, default=None):
        val = row.get(key, None)
        if pd.isna(val) or val is None or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    mcap = safe_val("market_cap", 0.0)
    pe = safe_val("pe", 0.0)
    dy = safe_val("dividend_yield", 0.0)
    eps = safe_val("eps", 0.0)
    
    revenue = safe_val("revenue", None)
    revenue_prev = safe_val("revenue_prev", None)
    eps_prev = safe_val("eps_prev", None)
    roe = safe_val("roe", None)
    roic = safe_val("roic", None)
    ocf = safe_val("operating_cash_flow", None)
    pat = safe_val("net_profit", None)
    ebitda = safe_val("ebitda", None)
    debt = safe_val("debt", None)
    cash = safe_val("cash", None)
    cap_growth = safe_val("capacity_growth", None)
    utilization = safe_val("utilization", None)
    catalyst = safe_val("catalyst_score", 5.0)
    gov = safe_val("governance_score", 8.0)

    has_full_data = any(v is not None for v in [revenue, roe, roic, ocf, debt])

    if has_full_data:
        bq_score = 0.0
        if revenue and revenue_prev and revenue_prev > 0:
            rev_g = ((revenue - revenue_prev) / revenue_prev) * 100
            bq_score += 5.0 if rev_g >= 15 else (3.0 if rev_g >= 10 else 1.0)
        else: bq_score += 2.5

        if eps > 0 and eps_prev and eps_prev > 0:
            eps_g = ((eps - eps_prev) / eps_prev) * 100
            bq_score += 5.0 if eps_g >= 15 else (3.0 if eps_g >= 8 else 1.0)
        else: bq_score += 2.5

        bq_score += 5.0 if (roe and roe >= 20) else (3.0 if (roe and roe >= 15) else 1.0)
        bq_score += 5.0 if (roic and roic >= 18) else (3.0 if (roic and roic >= 12) else 1.0)
        
        if pat and ocf and pat > 0:
            bq_score += 5.0 if (ocf / pat) >= 1.0 else (3.0 if (ocf / pat) >= 0.7 else 1.0)
        else: bq_score += 2.5

        if revenue and ebitda and revenue > 0:
            bq_score += 5.0 if (ebitda / revenue) * 100 >= 20 else 3.0
        else: bq_score += 2.5

        mp_score = 0.0
        mp_score += 7.0 if 0 < mcap <= 25_000_000_000 else (4.5 if mcap <= 75_000_000_000 else 2.0)
        mp_score += 7.0 if (cap_growth and cap_growth >= 15) else 4.0
        mp_score += 7.0 if (utilization and utilization >= 75) else 4.0
        mp_score += min(max(catalyst, 0.0), 7.0) + 7.0

        rv_score = 0.0
        rv_score += 8.0 if 0 < pe <= 8 else (5.0 if pe <= 14 else 2.0)
        rv_score += 8.0 if (debt == 0 or (cash and debt and cash > debt)) else 4.0
        rv_score += 7.0 if dy >= 8.0 else (4.0 if dy >= 4.0 else 1.0)
        rv_score += min(max(gov, 0.0), 12.0)

    else:
        # Dynamic Scaling for Scraped Web Summaries
        if 0 < pe <= 5: bq_score = 36.0
        elif 5 < pe <= 8: bq_score = 30.0
        elif 8 < pe <= 12: bq_score = 24.0
        elif 12 < pe <= 20: bq_score = 18.0
        else: bq_score = 10.0

        if dy >= 10.0: rv_score = 28.0
        elif dy >= 6.0: rv_score = 22.0
        elif dy >= 3.0: rv_score = 15.0
        else: rv_score = 8.0

        if 0 < mcap <= 30_000_000_000: mp_score = 26.0
        elif 30_000_000_000 < mcap <= 100_000_000_000: mp_score = 22.0
        elif mcap > 100_000_000_000: mp_score = 18.0
        else: mp_score = 15.0

    total_score = round(bq_score + mp_score + rv_score, 2)
    
    if total_score >= 85: status = "Multibagger Candidate"
    elif total_score >= 75: status = "Watchlist"
    elif total_score >= 65: status = "Monitor"
    else: status = "No Multibagger Status"

    buy_setup = round(min(100.0, (total_score * 0.7) + (min(dy, 12) * 1.5) + (7 if 0 < pe < 10 else 2)), 2)

    return {
        "bq_score": round(bq_score, 2),
        "mp_score": round(mp_score, 2),
        "rv_score": round(rv_score, 2),
        "total_score": total_score,
        "status": status,
        "buy_setup_score": buy_setup
    }

# -----------------------------------------------------------------------------
# 4. GROQ RESEARCH NOTE GENERATOR
# -----------------------------------------------------------------------------

def generate_groq_research_note(api_key: str, company_data: dict, scores: dict) -> str:
    """Generates equity research note using Groq API."""
    try:
        client = Groq(api_key=api_key)
        prompt = f"""
        Act as a Senior Equity Analyst covering the Pakistan Stock Exchange (PSX).
        Write a concise research note for the following company analyzed under the 100-Point Multibagger Framework:

        Company Metadata & Metrics:
        {json.dumps(company_data, indent=2, default=str)}

        Engine Scores:
        - Total Multibagger Score: {scores['total_score']}/100
        - Classification: {scores['status']}
        - Business Quality Score: {scores['bq_score']}/30
        - Multibagger Potential Score: {scores['mp_score']}/35
        - Risk & Valuation Score: {scores['rv_score']}/35
        - Buy Setup Score: {scores['buy_setup_score']}/100

        Provide:
        1. Executive Summary & Investment Thesis
        2. Financial Highlights & Growth Drivers
        3. Key Risks (Macro, Currency, Sector Specific to Pakistan)
        4. Valuation Context & Final Verdict
        """
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=800
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating research note via Groq: {str(e)}"

# -----------------------------------------------------------------------------
# 5. STREAMLIT APPLICATION USER INTERFACE
# -----------------------------------------------------------------------------

def main():
    st.title("🇵🇰 PSX Multibagger Engine")
    st.markdown("Screen Pakistan Stock Exchange companies using a 100-point multibagger framework and Groq AI research.")

    st.sidebar.header("⚙️ Configuration")
    api_key_env = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    groq_key = st.sidebar.text_input("Groq API Key", value=api_key_env, type="password", help="Enter key or set GROQ_API_KEY in Secrets")
    
    st.sidebar.markdown("---")
    data_source = st.sidebar.radio("Data Source Option", ["Upload File / Paste CSV", "Direct Fetch from PSX Portal"])

    df_raw = None

    if data_source == "Upload File / Paste CSV":
        st.subheader("📁 Data Input")
        upload_tab, paste_tab = st.tabs(["Upload CSV / Excel", "Paste Raw Data"])

        with upload_tab:
            uploaded_file = st.file_uploader("Upload CSV, XLSX, or XLS file", type=["csv", "xlsx", "xls"])
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df_raw = pd.read_csv(uploaded_file)
                    else:
                        df_raw = pd.read_excel(uploaded_file)
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        with paste_tab:
            pasted_text = st.text_area("Paste CSV content here", height=150, placeholder="ticker,price,market_cap,eps\nOGDC,120.5,518000000000,28.4")
            if pasted_text:
                try:
                    from io import StringIO
                    df_raw = pd.read_csv(StringIO(pasted_text))
                except Exception as e:
                    st.error(f"Error parsing pasted CSV: {e}")

    else:
        st.subheader("🌐 Direct PSX Web Data Fetch")
        symbol_input = st.text_input("Enter PSX Symbol (e.g., OGDC, SYS, LUCK, HUBC):", value="HUBC")
        if st.button("Fetch Live Data from PSX Portal"):
            with st.spinner("Scraping PSX Web Portal..."):
                psx_data = fetch_psx_web_data(symbol_input)
                if "error" in psx_data:
                    st.error(psx_data["error"])
                else:
                    st.success("Successfully fetched data!")
                    df_raw = pd.DataFrame([psx_data])

    if df_raw is not None and not df_raw.empty:
        df_clean = standardize_columns(df_raw)
        
        req_cols = ["ticker", "price", "market_cap", "eps"]
        missing = [col for col in req_cols if col not in df_clean.columns]

        if missing:
            st.warning(f"Missing required columns: {missing}. Standardized columns available: {list(df_clean.columns)}")
        else:
            evaluated_list = []
            for _, row in df_clean.iterrows():
                scores = evaluate_multibagger(row)
                full_row = {**row.to_dict(), **scores}
                evaluated_list.append(full_row)

            results_df = pd.DataFrame(evaluated_list)

            st.markdown("---")
            st.subheader("📊 Multibagger Screening Results")

            col1, col2, col3, col4 = st.columns(4)
            total_count = len(results_df)
            candidates = len(results_df[results_df["status"] == "Multibagger Candidate"])
            watchlist = len(results_df[results_df["status"] == "Watchlist"])
            
            col1.metric("Total Screened", total_count)
            col2.metric("Multibagger Candidates", candidates)
            col3.metric("Watchlist", watchlist)
            col4.metric("Avg Score", f"{results_df['total_score'].mean():.1f}/100")

            display_cols = ["ticker", "price", "market_cap", "eps", "total_score", "status", "buy_setup_score", "bq_score", "mp_score", "rv_score"]
            avail_display_cols = [c for c in display_cols if c in results_df.columns]
            
            st.dataframe(
                results_df[avail_display_cols].sort_values(by="total_score", ascending=False),
                use_container_width=True
            )

            st.markdown("---")
            st.subheader("🤖 Groq AI Equity Research Note")

            selected_ticker = st.selectbox("Select ticker for research note:", results_df["ticker"].unique())
            
            if st.button("Generate Research Note"):
                if not groq_key:
                    st.error("Please provide a Groq API Key in the sidebar or via Streamlit Secrets.")
                else:
                    selected_data = results_df[results_df["ticker"] == selected_ticker].iloc[0].to_dict()
                    scores_data = {
                        "bq_score": selected_data["bq_score"],
                        "mp_score": selected_data["mp_score"],
                        "rv_score": selected_data["rv_score"],
                        "total_score": selected_data["total_score"],
                        "status": selected_data["status"],
                        "buy_setup_score": selected_data["buy_setup_score"]
                    }
                    
                    with st.spinner(f"Generating Groq research note for {selected_ticker}..."):
                        note = generate_groq_research_note(groq_key, selected_data, scores_data)
                        st.markdown(note)

    else:
        st.info("Please upload a dataset, paste CSV data, or fetch a symbol directly to begin screening.")

if __name__ == "__main__":
    main()
