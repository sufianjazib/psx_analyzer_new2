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
    Directly scrapes live stock quote data from the PSX Data Portal (dps.psx.com.pk).
    """
    symbol = symbol.strip().upper()
    url = f"https://dps.psx.com.pk/company/{symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return {"error": f"Symbol '{symbol}' not found or PSX Portal unavailable."}

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Parse Price
        price_elem = soup.find("div", class_="quote__close")
        price = float(price_elem.text.replace("Rs.", "").replace(",", "").strip()) if price_elem else None

        # Parse Stats Table / Key Indicators
        stats = {}
        for item in soup.find_all("div", class_="stats_item"):
            label_elem = item.find("div", class_="stats_label")
            value_elem = item.find("div", class_="stats_value")
            if label_elem and value_elem:
                lbl = label_elem.text.strip().lower()
                val = value_elem.text.strip().replace(",", "")
                stats[lbl] = val

        # Extract specific metrics with fallbacks
        def safe_float(val_str):
            try:
                clean = re.sub(r'[^\d.-]', '', str(val_str))
                return float(clean) if clean else None
            except:
                return None

        mcap = safe_float(stats.get("market cap", stats.get("mcap")))
        eps = safe_float(stats.get("eps", stats.get("eps (ttm)")))
        pe = safe_float(stats.get("p/e", stats.get("pe ratio")))
        div_yield = safe_float(stats.get("dividend yield", stats.get("div yield")))

        return {
            "ticker": symbol,
            "price": price,
            "market_cap": mcap,
            "eps": eps,
            "pe": pe,
            "dividend_yield": div_yield,
            "source": "Direct PSX Portal Scraping"
        }
    except Exception as e:
        return {"error": f"Failed to scrape PSX portal: {str(e)}"}

# -----------------------------------------------------------------------------
# 3. MULTIBAGGER EVALUATION ENGINE (100-POINT FRAMEWORK)
# -----------------------------------------------------------------------------

def evaluate_multibagger(row: pd.Series) -> dict:
    """Calculates Business Quality, Multibagger Potential, and Risk/Valuation scores."""
    
    # --- Category 1: Business Quality (30 Points) ---
    bq_score = 0.0
    
    # Revenue Growth (>15% = 5 pts, >10% = 3 pts)
    rev = row.get("revenue", 0)
    rev_prev = row.get("revenue_prev", 0)
    if rev > 0 and rev_prev > 0:
        rev_growth = ((rev - rev_prev) / rev_prev) * 100
        if rev_growth >= 15: bq_score += 5.0
        elif rev_growth >= 10: bq_score += 3.0
    else:
        bq_score += 2.5 # Neutral fallback

    # EPS Growth (>15% = 5 pts)
    eps = row.get("eps", 0)
    eps_prev = row.get("eps_prev", 0)
    if eps > 0 and eps_prev > 0:
        eps_growth = ((eps - eps_prev) / eps_prev) * 100
        if eps_growth >= 15: bq_score += 5.0
        elif eps_growth >= 8: bq_score += 3.0
    else:
        bq_score += 2.5

    # ROE (>20% = 5 pts, >15% = 3 pts)
    roe = row.get("roe", 0)
    if roe >= 20: bq_score += 5.0
    elif roe >= 15: bq_score += 3.0

    # ROIC (>18% = 5 pts)
    roic = row.get("roic", 0)
    if roic >= 18: bq_score += 5.0
    elif roic >= 12: bq_score += 3.0

    # OCF vs Net Profit Quality (>1.0 = 5 pts)
    ocf = row.get("operating_cash_flow", 0)
    pat = row.get("net_profit", 0)
    if pat > 0 and ocf > 0:
        if (ocf / pat) >= 1.0: bq_score += 5.0
        elif (ocf / pat) >= 0.7: bq_score += 3.0
    else:
        bq_score += 2.5

    # EBITDA Margin (>20% = 5 pts)
    ebitda = row.get("ebitda", 0)
    if rev > 0 and ebitda > 0:
        ebitda_margin = (ebitda / rev) * 100
        if ebitda_margin >= 20: bq_score += 5.0
        elif ebitda_margin >= 12: bq_score += 3.0
    else:
        bq_score += 2.5


    # --- Category 2: Multibagger Potential (35 Points) ---
    mp_score = 0.0

    # Market Cap Scalability (Small/Mid-cap preference for PSX)
    mcap = row.get("market_cap", 0)
    if 0 < mcap <= 25_000_000_000: # <25 Billion PKR
        mp_score += 7.0
    elif 25_000_000_000 < mcap <= 75_000_000_000:
        mp_score += 4.5
    else:
        mp_score += 2.0

    # EPS CAGR (3Y or 5Y)
    eps_3y = row.get("eps_3y_ago", 0)
    if eps > 0 and eps_3y > 0:
        eps_cagr = ((eps / eps_3y) ** (1/3) - 1) * 100
        if eps_cagr >= 20: mp_score += 7.0
        elif eps_cagr >= 12: mp_score += 4.0
    else:
        mp_score += 3.5

    # Capacity Expansion & Utilization
    cap_growth = row.get("capacity_growth", 0)
    utilization = row.get("utilization", 0)
    if cap_growth >= 15: mp_score += 7.0
    elif cap_growth >= 5: mp_score += 4.0
    else: mp_score += 2.0

    if utilization >= 75: mp_score += 7.0
    elif utilization >= 50: mp_score += 4.0
    else: mp_score += 2.0

    # Catalyst Score (Direct Input 1-7)
    catalyst = row.get("catalyst_score", 4)
    mp_score += min(max(catalyst, 0), 7.0)


    # --- Category 3: Risk & Valuation (35 Points) ---
    rv_score = 0.0

    # Price to Earnings (P/E)
    pe = row.get("pe", 0)
    if 0 < pe <= 8: rv_score += 8.0  # Very attractive for PSX
    elif 8 < pe <= 14: rv_score += 5.0
    else: rv_score += 2.0

    # Debt / Cash Health
    debt = row.get("debt", 0)
    cash = row.get("cash", 0)
    if debt == 0 or (cash > debt): rv_score += 8.0
    elif ebitda > 0 and (debt / ebitda) < 2.0: rv_score += 5.0
    else: rv_score += 2.0

    # Dividend Yield
    dy = row.get("dividend_yield", 0)
    if dy >= 8.0: rv_score += 7.0
    elif dy >= 4.0: rv_score += 4.0
    else: rv_score += 1.0

    # Governance Score (Direct Input 1-12)
    gov = row.get("governance_score", 8)
    rv_score += min(max(gov, 0), 12.0)

    # --- Total Score & Classification ---
    total_score = round(bq_score + mp_score + rv_score, 2)
    
    if total_score >= 85:
        status = "Multibagger Candidate"
    elif total_score >= 75:
        status = "Watchlist"
    elif total_score >= 65:
        status = "Monitor"
    else:
        status = "No Multibagger Status"

    # Separate Buy Setup Score (1-100 scale calculation)
    buy_setup = round(min(100.0, (total_score * 0.7) + (min(dy, 12) * 1.5) + (7 if pe > 0 and pe < 10 else 2)), 2)

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
    """Generates an equity research note using Groq API."""
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

    # Sidebar: Setup & API Config
    st.sidebar.header("⚙️ Configuration")
    api_key_env = os.environ.get("GROQ_API_KEY", "")
    groq_key = st.sidebar.text_input("Groq API Key", value=api_key_env, type="password", help="Enter key or set GROQ_API_KEY env variable")
    
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
        symbol_input = st.text_input("Enter PSX Symbol (e.g., OGDC, SYS, LUCK, ENGRO):", value="OGDC")
        if st.button("Fetch Live Data from PSX Portal"):
            with st.spinner("Scraping PSX Web Portal..."):
                psx_data = fetch_psx_web_data(symbol_input)
                if "error" in psx_data:
                    st.error(psx_data["error"])
                else:
                    st.success("Successfully fetched data!")
                    df_raw = pd.DataFrame([psx_data])

    # Processing and Display Logic
    if df_raw is not None and not df_raw.empty:
        df_clean = standardize_columns(df_raw)
        
        # Verify Minimum Required Columns
        req_cols = ["ticker", "price", "market_cap", "eps"]
        missing = [col for col in req_cols if col not in df_clean.columns]

        if missing:
            st.warning(f"Missing required columns: {missing}. Standardized columns available: {list(df_clean.columns)}")
        else:
            # Process Scores
            evaluated_list = []
            for _, row in df_clean.iterrows():
                scores = evaluate_multibagger(row)
                full_row = {**row.to_dict(), **scores}
                evaluated_list.append(full_row)

            results_df = pd.DataFrame(evaluated_list)

            st.markdown("---")
            st.subheader("📊 Multibagger Screening Results")

            # Score Summary Cards
            col1, col2, col3, col4 = st.columns(4)
            total_count = len(results_df)
            candidates = len(results_df[results_df["status"] == "Multibagger Candidate"])
            watchlist = len(results_df[results_df["status"] == "Watchlist"])
            
            col1.metric("Total Screened", total_count)
            col2.metric("Multibagger Candidates", candidates)
            col3.metric("Watchlist", watchlist)
            col4.metric("Avg Score", f"{results_df['total_score'].mean():.1f}/100")

            # Display Primary Table
            display_cols = ["ticker", "price", "market_cap", "eps", "total_score", "status", "buy_setup_score", "bq_score", "mp_score", "rv_score"]
            avail_display_cols = [c for c in display_cols if c in results_df.columns]
            
            st.dataframe(
                results_df[avail_display_cols].sort_values(by="total_score", ascending=False),
                use_container_width=True
            )

            # --- AI Research Note Generation Section ---
            st.markdown("---")
            st.subheader("🤖 Groq AI Equity Research Note")

            selected_ticker = st.selectbox("Select ticker for research note:", results_df["ticker"].unique())
            
            if st.button("Generate Research Note"):
                if not groq_key:
                    st.error("Please provide a Groq API Key in the sidebar or via GROQ_API_KEY environment variable.")
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
        st.info("Please upload a dataset, paste CSV data, or fetch a symbol directly from the sidebar to begin screening.")

if __name__ == "__main__":
    main()
