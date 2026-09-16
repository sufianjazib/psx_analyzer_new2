# PSX Multibagger Engine

A Streamlit app for screening Pakistan Stock Exchange companies using a 100-point
multibagger framework and generating optional Groq-powered equity research notes.

## What it does

### 1. Business Quality — 30 points
- Revenue growth
- EPS growth
- ROE
- ROIC
- Operating cash flow vs net profit
- EBITDA margin
- Optional governance input

### 2. Multibagger Potential — 35 points
- Market-cap scalability
- EPS growth / acceleration
- Historical EPS CAGR
- Capacity expansion
- Utilization
- Catalyst score

### 3. Risk & Valuation — 35 points
- P/E
- Debt / EBITDA or debt/cash information
- Operating cash-flow quality
- Dividend yield
- Optional governance score

## Score interpretation

- **85–100:** Multibagger Candidate
- **75–84:** Watchlist
- **65–74:** Monitor
- **Below 65:** No Multibagger Status

The app also calculates a separate **Buy Setup Score**. It is a signal for
screening and further research, not a promise of a return.

## Important limitation

The application does **not** guarantee 10% monthly returns and does not claim that
a multibagger "must hit." A guaranteed 10% monthly return would require an
unrealistic level of certainty.

The scenario model is explicitly an assumption-based valuation exercise.

Before using the score as an investment strategy, add historical PSX data and
back-test the rules across multiple years.

## Data input

The app currently supports:

- CSV
- XLSX
- XLS
- pasted CSV

Minimum required columns:

```text
ticker
price
market_cap
eps
```

Recommended columns:

```text
ticker,company,price,market_cap,revenue,revenue_prev,
eps,eps_prev,eps_3y_ago,eps_5y_ago,
net_profit,net_profit_prev,roe,roic,
operating_cash_flow,free_cash_flow,debt,cash,
ebitda,shares,dividend_yield,pe,pb,ev_ebitda,
promoter_holding,free_float,avg_volume,
capacity_growth,utilization,catalyst_score,governance_score
```

The app accepts several common aliases such as `symbol` for `ticker`, `sales` for
`revenue`, `pat` for `net_profit`, `roe_percent` for `roe`, etc.

### Units

- `market_cap`: PKR
- `price`: PKR/share
- `eps`: PKR/share
- `revenue`, `net_profit`, `operating_cash_flow`, `free_cash_flow`,
  `debt`, `cash`, `ebitda`: use the same currency unit consistently
- `roe`, `roic`, `dividend_yield`, `capacity_growth`, `utilization`: enter as
  percentage numbers, e.g. `22.5`, not `0.225`

## Groq API

Create a Groq API key and either:

### Option A — enter it in the sidebar

Run the app and paste the key into the password field.

### Option B — environment variable

Linux/macOS:

```bash
export GROQ_API_KEY="your_key_here"
streamlit run App.py
```

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_key_here"
streamlit run App.py
```

For Streamlit Cloud, add `GROQ_API_KEY` in the app's Secrets settings.

Never commit your API key to GitHub.

## Run locally

Install Python 3.10+.

```bash
pip install -r requirements.txt
streamlit run App.py
```

Then open the Streamlit URL shown in the terminal.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Put these files in the repository:
   - `App.py`
   - `requirements.txt`
   - `README.md`
3. Open Streamlit Community Cloud.
4. Select the GitHub repository and `App.py`.
5. Add your secret:

```toml
GROQ_API_KEY = "your_key_here"
```

6. Deploy.

## Suggested next development phase

For a production-quality PSX research product, add:

1. Automated PSX data ingestion.
2. Historical quarterly/annual financial database.
3. 3/5/10-year back-testing.
4. Corporate-action adjustment.
5. Sector-specific valuation models.
6. Separate earnings-acceleration model.
7. Price/volume confirmation.
8. Management/promoter transaction monitoring.
9. PSX announcements/news ingestion.
10. Portfolio tracking and thesis-break alerts.

The most important upgrade is **back-testing**. The model should only be promoted
to a live strategy after measuring how companies scoring 85+ historically behaved
over 3, 6, 12, 24 and 36 months.

## Disclaimer

This software is for research and educational decision support. It is not
personalized investment advice and does not guarantee profits or returns.
