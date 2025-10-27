import os
import pandas as pd
from datetime import datetime
from data_fetch import fetch_etf_data
from indicators import compute_basic_indicators
from strategies.simple_rule_strategy import SimpleRuleStrategy
from strategies.weighted_strategy import WeightedStrategy

PORTFOLIO_FILE = "portfolio.csv"
OUTPUT_DIR = "dashboards"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PortfolioTracker:
    def __init__(self, strategy="weighted"):
        self.strategy = WeightedStrategy() if strategy.lower() == "weighted" else SimpleRuleStrategy()

    def load_portfolio(self):
        """Load portfolio from CSV (Ticker, Quantity, BuyPrice, BuyDate)."""
        if not os.path.exists(PORTFOLIO_FILE):
            print(f"⚠️ No portfolio file found at {PORTFOLIO_FILE}. Creating empty portfolio.")
            df = pd.DataFrame(columns=["Ticker", "Quantity", "BuyPrice", "BuyDate"])
            df.to_csv(PORTFOLIO_FILE, index=False)
            return df
        return pd.read_csv(PORTFOLIO_FILE)

    def evaluate_positions(self):
        """Fetch latest ETF data, compute indicators, evaluate signals, and update portfolio."""
        portfolio = self.load_portfolio()
        results = []

        if portfolio.empty:
            print("⚠️ Portfolio is empty. Add holdings to portfolio.csv")
            return pd.DataFrame()

        print("📈 Evaluating portfolio...")

        for _, row in portfolio.iterrows():
            ticker = row["Ticker"]
            qty = row["Quantity"]
            buy_price = row["BuyPrice"]
            buy_date = row.get("BuyDate", "N/A")

            try:
                df = fetch_etf_data(ticker, period="6mo", interval="1d")
                if df.empty or "Close" not in df.columns:
                    print(f"⚠️ Skipping {ticker}: no data.")
                    continue

                df = compute_basic_indicators(df)
                if df.empty:
                    continue

                # Strategy evaluation
                signal, confidence, trend, score = self.strategy.evaluate(df)
                last_close = float(df.iloc[-1]["Close"])
                pnl = ((last_close - buy_price) / buy_price) * 100

                results.append({
                    "Ticker": ticker,
                    "Qty": qty,
                    "BuyPrice": round(buy_price, 2),
                    "CurrentPrice": round(last_close, 2),
                    "PnL%": round(pnl, 2),
                    "BuyDate": buy_date,
                    "Signal": signal,
                    "Confidence": f"{confidence}%",
                    "Trend": trend,
                    "Score": score,
                })

            except Exception as e:
                print(f"❌ Error processing {ticker}: {e}")

        if not results:
            print("⚠️ No valid ETF data for portfolio.")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # --- Determine Suggested Action ---
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
        """Generate a simple HTML dashboard for portfolio."""
        if df.empty:
            print("⚠️ Nothing to display in portfolio dashboard.")
            return None

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        html_path = os.path.join(OUTPUT_DIR, "portfolio_dashboard.html")

        html_content = f"""
        <html>
        <head>
            <title>ETF Portfolio Tracker</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: #f5f7fa;
                    margin: 40px;
                }}
                h1 {{
                    color: #1f77b4;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    background: white;
                }}
                th, td {{
                    border: 1px solid #ccc;
                    padding: 8px;
                    text-align: center;
                }}
                th {{
                    background: #1f77b4;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background: #f9f9f9;
                }}
            </style>
        </head>
        <body>
            <h1>📊 ETF Portfolio Tracker</h1>
            <p><b>Last Updated:</b> {now}</p>
            {df.to_html(index=False, escape=False)}
        </body>
        </html>
        """

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
            "BuyPrice": buy_price,
            "BuyDate": buy_date
        }])

        try:
            existing = pd.read_csv(PORTFOLIO_FILE)
        except FileNotFoundError:
            existing = pd.DataFrame(columns=["Ticker", "Quantity", "BuyPrice", "BuyDate"])

        updated = pd.concat([existing, new_entry], ignore_index=True)
        updated.to_csv(PORTFOLIO_FILE, index=False)
        print(f"✅ Added {ticker} ({quantity} @ ₹{buy_price}) to portfolio.")
        return True
    except Exception as e:
        print(f"❌ Error adding ETF to portfolio: {e}")
        return False

if __name__ == "__main__":
    run_portfolio_tracker()
