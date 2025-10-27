import pandas as pd
import yfinance as yf
from datetime import datetime
import os

PORTFOLIO_FILE = "portfolio.csv"

class PortfolioTracker:
    def __init__(self):
        # Ensure portfolio file exists
        if not os.path.exists(PORTFOLIO_FILE):
            pd.DataFrame(columns=["Ticker", "Quantity", "BuyPrice", "BuyDate"]).to_csv(PORTFOLIO_FILE, index=False)

    def load_portfolio(self):
        """Load saved portfolio CSV."""
        try:
            df = pd.read_csv(PORTFOLIO_FILE)
            if df.empty:
                print("⚠️ Portfolio is empty.")
            return df
        except Exception as e:
            print(f"❌ Error loading portfolio: {e}")
            return pd.DataFrame()

    def evaluate_positions(self):
        """Fetch latest prices and calculate PnL for all held ETFs."""
        portfolio = self.load_portfolio()
        if portfolio.empty:
            return pd.DataFrame()

        results = []
        for _, row in portfolio.iterrows():
            ticker, qty, buy_price, buy_date = (
                row["Ticker"],
                float(row["Quantity"]),
                float(row["BuyPrice"]),
                row.get("BuyDate", "Unknown")
            )

            try:
                data = yf.download(ticker, period="5d", interval="1d", progress=False)
                if data.empty:
                    print(f"⚠️ No data for {ticker}")
                    continue
                current_price = data["Close"].iloc[-1]
                pnl = (current_price - buy_price) * qty
                pnl_pct = ((current_price - buy_price) / buy_price) * 100

                if pnl_pct > 15:
                    action = "💰 Consider Selling (Take Profit)"
                elif pnl_pct < -10:
                    action = "⚠️ Review / Cut Loss"
                else:
                    action = "✅ Hold"

                results.append({
                    "Ticker": ticker,
                    "BuyDate": buy_date,
                    "BuyPrice": round(buy_price, 2),
                    "Quantity": int(qty),
                    "CurrentPrice": round(current_price, 2),
                    "PnL (₹)": round(pnl, 2),
                    "PnL%": round(pnl_pct, 2),
                    "SuggestedAction": action
                })
            except Exception as e:
                print(f"❌ Error processing {ticker}: {e}")

        if not results:
            print("⚠️ No valid positions found.")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df.sort_values(by=["SuggestedAction", "PnL%"], ascending=[True, False], inplace=True)
        return df

    def write_portfolio_dashboard(self, df):
        """Generate HTML dashboard for portfolio overview."""
        if df.empty:
            html = """
            <html><body><h2>⚠️ No holdings found in your portfolio.</h2></body></html>
            """
            path = "portfolio_dashboard.html"
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            return path

        total_invested = (df["BuyPrice"] * df["Quantity"]).sum()
        total_value = (df["CurrentPrice"] * df["Quantity"]).sum()
        total_pnl = total_value - total_invested
        total_pnl_pct = (total_pnl / total_invested) * 100 if total_invested > 0 else 0

        summary_html = f"""
        <div style='margin:15px;'>
            <h2>📊 Portfolio Summary</h2>
            <p><b>Total Invested:</b> ₹{total_invested:,.2f}</p>
            <p><b>Current Value:</b> ₹{total_value:,.2f}</p>
            <p><b>Total P&L:</b> ₹{total_pnl:,.2f} ({total_pnl_pct:.2f}%)</p>
        </div>
        """

        df_html = df.to_html(escape=False, index=False, justify="center")

        html_page = f"""
        <html>
        <head>
            <title>ETF Portfolio Tracker</title>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 30px;
                    background-color: #f8f9fa;
                    color: #222;
                }}
                h1, h2 {{
                    color: #1f77b4;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-top: 20px;
                }}
                th, td {{
                    padding: 8px 10px;
                    border-bottom: 1px solid #ccc;
                    text-align: center;
                }}
                th {{
                    background-color: #1f77b4;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f1f1f1;
                }}
            </style>
        </head>
        <body>
            <h1>📈 ETF Portfolio Tracker</h1>
            {summary_html}
            <h2>Holdings</h2>
            {df_html}
        </body>
        </html>
        """

        path = "portfolio_dashboard.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_page)

        print(f"💾 Portfolio dashboard saved: {path}")
        return path


def add_to_portfolio(ticker, quantity, buy_price, buy_date=None):
    """Append new ETF purchase to portfolio file."""
    buy_date = buy_date or datetime.now().strftime("%Y-%m-%d")

    new_entry = pd.DataFrame([{
        "Ticker": ticker,
        "Quantity": quantity,
        "BuyPrice": buy_price,
        "BuyDate": buy_date
    }])

    if os.path.exists(PORTFOLIO_FILE):
        portfolio = pd.read_csv(PORTFOLIO_FILE)
        portfolio = pd.concat([portfolio, new_entry], ignore_index=True)
    else:
        portfolio = new_entry

    portfolio.to_csv(PORTFOLIO_FILE, index=False)
    print(f"✅ Added {ticker} ({quantity} @ ₹{buy_price}) on {buy_date}")
    return True
