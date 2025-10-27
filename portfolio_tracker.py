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
            df.rename(columns=lambda x: x.strip().title().replace(" ", ""), inplace=True)
            df.to_csv(PORTFOLIO_FILE, index=False)
            return df
        file= pd.read_csv(PORTFOLIO_FILE)
        print(file.head())
        print(file.columns)
        return file

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
                df = fetch_etf_data(ticker, period="1y", interval="1d")
                if df.empty or "Close" not in df.columns:
                    print(f"⚠️ Skipping {ticker}: no data.")
                    continue

                print(f"Fetched {ticker}: {len(df)} rows, cols={df.columns.tolist()}")

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

        # --- Portfolio Summary ---
        df["CurrentValue"] = df["Qty"] * df["CurrentPrice"]
        total_value = df["CurrentValue"].sum()
        total_invested = (df["Qty"] * df["BuyPrice"]).sum()
        total_pnl = ((total_value - total_invested) / total_invested) * 100

        # Use mutually exclusive buckets
        accumulate_or_hold = df["SuggestedAction"].str.contains("Accumulate|Hold", case=False, na=False)
        exit_or_cut = df["SuggestedAction"].str.contains("Exit|Cut", case=False, na=False)

        buy_hold_count = accumulate_or_hold.sum()
        exit_count = exit_or_cut.sum()

        html_content = f"""
        <html>
        <head>
            <title>ETF Portfolio Tracker</title>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Inter', 'Segoe UI', Roboto, Arial, sans-serif;
                    background: #f4f6f9;
                    margin: 40px;
                    color: #333;
                }}
                .card {{
                    background: white;
                    border-radius: 14px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
                    padding: 28px;
                }}
                h1 {{
                    color: #1f77b4;
                    font-weight: 700;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-bottom: 6px;
                }}
                h1::before {{
                    content: '📊';
                    font-size: 28px;
                }}
                .last-updated {{
                    font-size: 14px;
                    color: #666;
                    margin-bottom: 22px;
                }}
                .summary {{
                    display: flex;
                    gap: 25px;
                    flex-wrap: wrap;
                    margin-bottom: 25px;
                }}
                .summary-card {{
                    background: #f8fbff;
                    border: 1px solid #dbe9ff;
                    border-radius: 10px;
                    padding: 15px 20px;
                    flex: 1;
                    min-width: 200px;
                    text-align: center;
                    box-shadow: 0 3px 8px rgba(0,0,0,0.03);
                }}
                .summary-card h3 {{
                    margin: 0;
                    color: #1f77b4;
                    font-size: 15px;
                    margin-bottom: 6px;
                }}
                .summary-card p {{
                    margin: 0;
                    font-size: 18px;
                    font-weight: 700;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    background: white;
                    border-radius: 10px;
                    overflow: hidden;
                    font-size: 14px;
                }}
                th {{
                    background: #1f77b4;
                    color: white;
                    text-transform: uppercase;
                    font-size: 12.5px;
                    letter-spacing: 0.4px;
                    padding: 10px;
                }}
                td {{
                    padding: 10px;
                    text-align: center;
                    border-bottom: 1px solid #eee;
                    vertical-align: middle;
                    white-space: nowrap;
                }}
                tr:hover {{
                    background: #f1f7ff;
                }}
                .gain {{
                    color: #00a65a;
                    font-weight: 600;
                }}
                .loss {{
                    color: #e74c3c;
                    font-weight: 600;
                }}
                .bar {{
                    display: inline-block;
                    height: 6px;
                    border-radius: 4px;
                    margin-left: 6px;
                }}
                .signal-buy {{
                    background: #e8f5e9;
                    color: #2e7d32;
                    padding: 4px 10px;
                    border-radius: 6px;
                    font-weight: 600;
                }}
                .signal-wait {{
                    background: #fff3e0;
                    color: #ef6c00;
                    padding: 4px 10px;
                    border-radius: 6px;
                    font-weight: 600;
                }}
                .signal-avoid {{
                    background: #ffebee;
                    color: #c62828;
                    padding: 4px 10px;
                    border-radius: 6px;
                    font-weight: 600;
                }}
                .trend-up {{
                    color: #00a65a;
                    font-weight: 600;
                }}
                .trend-down {{
                    color: #e74c3c;
                    font-weight: 600;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    font-size: 13px;
                    color: #999;
                }}
                @media (max-width: 768px) {{
                    body {{ margin: 10px; }}
                    table, th, td {{ font-size: 12px; }}
                    .summary {{ flex-direction: column; }}
                }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>ETF Portfolio Tracker</h1>
                <div class="last-updated"><b>Last Updated:</b> {now}</div>
        
                <div class="summary">
                    <div class="summary-card">
                        <h3>💰 Total Portfolio Value</h3>
                        <p>₹{total_value:,.2f}</p>
                    </div>
                    <div class="summary-card">
                        <h3>📈 Total PnL%</h3>
                        <p style="color:{'#00a65a' if total_pnl>=0 else '#e74c3c'};">{total_pnl:.2f}%</p>
                    </div>
                    <div class="summary-card">
                        <h3>🟢 Accumulate / Hold</h3>
                        <p>{buy_hold_count}</p>
                    </div>
                    <div class="summary-card">
                        <h3>🔴 Exit / Cut Loss</h3>
                        <p>{exit_count}</p>
                    </div>
                </div>
        
                <table>
                    <thead>
                        <tr>
                            {"".join(f"<th>{col}</th>" for col in df.columns)}
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(
                    "<tr>" +
                    "".join(
                        f"<td class='"
                        + (
                            "gain" if col=="PnL%" and val>0 else
                            "loss" if col=="PnL%" and val<0 else
                            ""
                        ) + "'>"
                        + (
                            # --- Signal badge ---
                            f"<span class='signal-buy'>{val}</span>" if col=="Signal" and "BUY" in str(val) else
                            f"<span class='signal-wait'>{val}</span>" if col=="Signal" and "WAIT" in str(val) else
                            f"<span class='signal-avoid'>{val}</span>" if col=="Signal" and "AVOID" in str(val) else
                            # --- PnL% bar ---
                            (f"{val:.2f}% <span class='bar' style='width:{min(abs(val)*2,100)}px;background:{'#00a65a' if val>0 else '#e74c3c'}'></span>" if col=="PnL%" else
                             # --- Trend emoji ---
                             f"<span class='trend-up'>📈 Up</span>" if col=="Trend" and 'Up' in str(val) else
                             f"<span class='trend-down'>📉 Down</span>" if col=="Trend" and 'Down' in str(val) else
                             str(val))
                        )
                        + "</td>"
                        for col, val in zip(df.columns, row)
                    ) +
                    "</tr>"
                    for _, row in df.iterrows()
                )}
                    </tbody>
                </table>
        
                <div class="footer">© 2025 ETF Screener Pro — Powered by yFinance</div>
            </div>
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
            print(existing.head())
            print(existing.columns)
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
