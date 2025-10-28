import os
import pandas as pd
from datetime import datetime
from data_fetch import fetch_etf_data
from indicators import compute_basic_indicators
from strategies.simple_rule_strategy import SimpleRuleStrategy
from strategies.weighted_strategy import WeightedStrategy
from advisor import AdvisorEngine
from datetime import datetime
import os

PORTFOLIO_FILE = "portfolio.csv"
OUTPUT_DIR = "dashboards"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PortfolioTracker:
    def __init__(self, strategy="weighted"):
        self.strategy = WeightedStrategy() if strategy.lower() == "weighted" else SimpleRuleStrategy()
        self.data = {}  # store {ticker: processed dataframe} for reuse

    def load_portfolio(self):
        """Load and sanitize portfolio from CSV (Ticker, Quantity, Buyprice, Buydate)."""
        if not os.path.exists(PORTFOLIO_FILE):
            print(f"⚠️ No portfolio file found at {PORTFOLIO_FILE}. Creating empty portfolio.")
            df = pd.DataFrame(columns=["Ticker", "Quantity", "Buyprice", "Buydate"])
            df.to_csv(PORTFOLIO_FILE, index=False)
            print(f"df after load is {df}")
            return df

        df = pd.read_csv(PORTFOLIO_FILE)

        # --- Normalize column names ---
        df.columns = (
            df.columns.str.strip()
            .str.replace(" ", "", regex=False)
            .str.replace("_", "", regex=False)
            .str.title()
        )

        # --- Ensure all required columns exist ---
        for col in ["Ticker", "Quantity", "Buyprice", "Buydate"]:
            if col not in df.columns:
                df[col] = None

        print("✅ Loaded portfolio:")
        print(df.head())
        print("Columns:", df.columns.tolist())
        return df


    def evaluate_positions(self):
        """Fetch data, compute indicators, evaluate strategy, and generate advisory insights."""
        portfolio_df = self.load_portfolio()
        results = []
        advisor = AdvisorEngine()
        if portfolio_df.empty:
            print("⚠️ Portfolio file is empty. Nothing to evaluate.")
            return pd.DataFrame()

        for _, row in portfolio_df.iterrows():
            ticker = row["Ticker"]
            qty = row["Quantity"]
            buy_price = row["Buyprice"]

            buy_date = row.get("Buydate", "")

            # --- Fetch ETF data ---
            df = fetch_etf_data(ticker, period="1y", interval="1d")
            if df.empty or "Close" not in df.columns:
                print(f"⚠️ Skipping {ticker}: no data fetched.")
                continue

            # --- Compute Indicators ---
            df = compute_basic_indicators(df)
            if df.empty:
                print(f"⚠️ Skipping {ticker}: indicators unavailable.")
                continue

            # Store for possible later use
            self.data[ticker] = df

            try:
                # --- Evaluate Strategy ---
                signal, confidence, trend, score = self.strategy.evaluate(df)
                last_close = float(df.iloc[-1]["Close"])
                pnl = ((last_close - buy_price) / buy_price) * 100

                # --- Generate Full Advisory ---
                # ticker, df, signal, confidence, trend, pnl, score
                advice = advisor.generate_advice(
                    ticker=ticker,
                    df=df,
                    signal=signal,
                    confidence=confidence,
                    trend=trend,
                    pnl=pnl,
                    score=score
                )

                # --- Collect results ---
                results.append({
                    "Ticker": ticker,
                    "Qty": qty,
                    "Buyprice": round(buy_price, 2),
                    "CurrentPrice": round(last_close, 2),
                    "PnL%": round(pnl, 2),
                    "Buydate": buy_date,
                    "Signal": signal,
                    "Confidence": f"{confidence}%",
                    "Trend": trend,
                    "Score": round(score, 2),
                    "Reason": advice["Reason"],
                    "Action": advice["Action"],
                    "NextRange": advice["NextRange"],
                    "PortfolioImpact": advice["PortfolioImpact"]
                })

            except Exception as e:
                print(f"❌ Error processing {ticker}: {e}")
                continue

        if not results:
            print("⚠️ No valid portfolio data found.")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # --- Suggested Action Helper ---
        def get_suggested_action(row):
            if row["Signal"] in ["STRONG BUY", "BUY"]:
                return "Hold / Accumulate"
            elif row["Signal"] in ["STRONG WAIT", "WAIT"]:
                if row["PnL%"] > 10:
                    return "Book Profit"
                elif row["PnL%"] < -10:
                    return "Cut Loss"
                else:
                    return "Hold"
            elif row["Signal"] == "AVOID":
                return "Exit"
            else:
                return "Review"

        df["SuggestedAction"] = df.apply(get_suggested_action, axis=1)

        df.sort_values(by=["SuggestedAction", "PnL%"], ascending=[True, False], inplace=True)
        df.reset_index(drop=True, inplace=True)

        print("✅ Portfolio evaluation complete.")
        return df


    def generate_dashboard(self, df):
        """Generate portfolio dashboard using enhanced HTML template with tooltips."""
        if df.empty:
            print("⚠️ Nothing to display in portfolio dashboard.")
            return None

        # Ensure the dashboards directory exists
        os.makedirs("dashboards", exist_ok=True)
        html_path = os.path.join("dashboards", "portfolio_dashboard.html")

        # --- Portfolio Summary ---
        df["CurrentValue"] = df["Qty"] * df["CurrentPrice"]
        total_value = df["CurrentValue"].sum()
        total_invested = (df["Qty"] * df["Buyprice"]).sum()
        total_pnl = ((total_value - total_invested) / total_invested) * 100
        pnl_color = "#00a65a" if total_pnl >= 0 else "#e74c3c"

        accumulate_or_hold = df["Action"].str.contains("Accumulate|Hold", case=False, na=False)
        exit_or_cut = df["Action"].str.contains("Exit|Cut", case=False, na=False)
        buy_hold_count = accumulate_or_hold.sum()
        exit_count = exit_or_cut.sum()

        # --- Prepare display columns ---
        display_cols = [
            "Ticker", "Qty", "Buyprice", "CurrentPrice", "PnL%",
            "Signal", "Confidence", "Trend", "Score", "Action", "NextRange"
        ]

        df_display = df[display_cols].copy()

        # --- Build table manually with tooltips ---
        table_html = """
        <table class="portfolio-table">
            <thead>
                <tr>
                    """ + "".join([f"<th>{col}</th>" for col in display_cols]) + """
                </tr>
            </thead>
            <tbody>
        """

        for _, row in df_display.iterrows():
            tooltip = (
                f"<b>{row['Ticker']}</b><br>"
                f"<b>Signal:</b> {row['Signal']} ({row['Confidence']})<br>"
                f"<b>Reason:</b> {df.loc[_]['Reason']}<br>"
                f"<b>Action:</b> {df.loc[_]['Action']}<br>"
                f"<b>Next 7D Range:</b> {df.loc[_]['NextRange']}<br>"
                f"<b>Portfolio Impact:</b> {df.loc[_]['PortfolioImpact']}"
            )
            table_html += "<tr>"
            for col in display_cols:
                if col == "Signal":
                    table_html += f"""
                        <td>
                            <div class="tooltip">{row[col]}
                                <span class="tooltiptext">{tooltip}</span>
                            </div>
                        </td>
                    """
                else:
                    table_html += f"<td>{row[col]}</td>"
            table_html += "</tr>"

        table_html += "</tbody></table>"

        # --- HTML Structure ---
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>ETF Portfolio Dashboard</title>
            <style>
                body {{
                    font-family: 'Segoe UI', sans-serif;
                    margin: 40px;
                    background-color: #f8fafc;
                    color: #222;
                }}
                h1 {{
                    text-align: center;
                    color: #1e3a8a;
                }}
                .summary {{
                    text-align: center;
                    margin-bottom: 25px;
                    font-size: 1.2em;
                }}
                .portfolio-table {{
                    width: 100%;
                    border-collapse: collapse;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                }}
                .portfolio-table th {{
                    background-color: #2563eb;
                    color: white;
                    padding: 10px;
                    text-align: left;
                    font-size: 14px;
                }}
                .portfolio-table td {{
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                    background: #fff;
                    font-size: 14px;
                    text-align: center;
                }}
                .portfolio-table tr:hover {{
                    background-color: #f1f5f9;
                }}
                .tooltip {{
                    position: relative;
                    display: inline-block;
                    cursor: help;
                }}
                .tooltip .tooltiptext {{
                    visibility: hidden;
                    width: 280px;
                    background-color: #111827;
                    color: #fff;
                    text-align: left;
                    border-radius: 6px;
                    padding: 10px;
                    position: absolute;
                    z-index: 1;
                    bottom: 125%;
                    left: 50%;
                    margin-left: -140px;
                    opacity: 0;
                    transition: opacity 0.3s;
                    white-space: normal;
                    line-height: 1.4;
                    box-shadow: 0 3px 8px rgba(0,0,0,0.25);
                }}
                .tooltip:hover .tooltiptext {{
                    visibility: visible;
                    opacity: 1;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 20px;
                    color: #555;
                }}
            </style>
        </head>
        <body>
            <h1>ETF Portfolio Dashboard</h1>
            <div class="summary">
                <p><b>Total Value:</b> ₹{total_value:,.2f} |
                   <b>Invested:</b> ₹{total_invested:,.2f} |
                   <b>PnL:</b> <span style="color:{pnl_color};">{total_pnl:.2f}%</span></p>
                <p>🟢 Accumulate / Hold: {buy_hold_count} &nbsp;&nbsp; 🔴 Exit / Cut: {exit_count}</p>
            </div>
            {table_html}
            <div class="footer">
                Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
            </div>
        </body>
        </html>
        """

        # --- Save HTML ---
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"💾 Portfolio dashboard saved: {html_path}")
        return html_path


def run_portfolio_tracker():
    tracker = PortfolioTracker(strategy="weighted")
    portfolio_df = tracker.evaluate_positions()
    html_file = tracker.generate_dashboard(portfolio_df)
    return html_file


def add_to_portfolio(ticker, quantity, buy_price, buy_date=None):
    """Append a new ETF holding to the portfolio CSV."""
    try:
        if buy_date is None:
            buy_date = datetime.now().strftime("%Y-%m-%d")

        new_entry = pd.DataFrame([{
            "Ticker": ticker,
            "Quantity": quantity,
            "Buyprice": buy_price,
            "Buydate": buy_date
        }])

        try:
            existing = pd.read_csv(PORTFOLIO_FILE)
            print(existing.head())
            print(existing.columns)
        except FileNotFoundError:
            existing = pd.DataFrame(columns=["Ticker", "Quantity", "Buyprice", "Buydate"])

        updated = pd.concat([existing, new_entry], ignore_index=True)
        updated.to_csv(PORTFOLIO_FILE, index=False)
        print(f"✅ Added {ticker} ({quantity} @ ₹{buy_price}) to portfolio.")
        return True
    except Exception as e:
        print(f"❌ Error adding ETF to portfolio: {e}")
        return False

if __name__ == "__main__":
    run_portfolio_tracker()
