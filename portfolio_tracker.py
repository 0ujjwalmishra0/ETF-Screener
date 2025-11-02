import os
import pandas as pd
from datetime import datetime
from data_fetch import fetch_etf_data
from indicators import compute_basic_indicators
from strategies.simple_rule_strategy import SimpleRuleStrategy
from strategies.weighted_strategy import WeightedStrategy
from strategies.smart_hybrid_strategy import SmartHybridStrategy
from advisor import AdvisorEngine
from datetime import datetime
from utils.notify import send_push_notification, send_mac_notification
import pandas as pd
import os
import html,re

PORTFOLIO_FILE = "portfolio.csv"
OUTPUT_DIR = "dashboards"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PortfolioTracker:
    def __init__(self, strategy):
        self.strategy = strategy
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
                #for weighted
                signal, confidence, trend, score = self.strategy.evaluate(df)
                # for ml
                # signal, confidence, trend, score, breakdown, model_prob = self.strategy.evaluate(df)
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
                    "PortfolioImpact": advice["PortfolioImpact"],
                    "Advice": advice.get("Advice", "")
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



    def generate_dashboard(self, results):
        """
        Generates the clean, sortable, tooltip-enabled portfolio dashboard.
        Removes Reason, Action, and PortfolioImpact columns.
        """
        df = pd.DataFrame(results)

        # Drop unwanted columns
        for col in ["Reason", "Advice","Action", "PortfolioImpact"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        def format_signal_with_tooltip(row):
            signal = str(row.get("Signal", "")).upper().strip()
            advice_raw = str(row.get("Advice", "") or "")

            # Clean and normalize
            advice_clean = html.unescape(advice_raw).strip()

            cls = (
                "signal-buy" if signal in ["BUY", "STRONG BUY"]
                else "signal-wait" if signal in ["WAIT", "STRONG WAIT"]
                else "signal-avoid"
            )

            return f"""<div class="tooltip-wrapper"><div class="{cls}">{signal}</span><div class="tooltip-box">{advice_clean}</div></div>"""


        def format_pnl(val):
            try:
                val = float(val)
            except:
                return val
            color = "#00a65a" if val > 0 else "#e74c3c" if val < 0 else "#555"
            sign = "+" if val > 0 else "" if val == 0 else ""
            return f'<span style="color:{color};font-weight:600;">{sign}{val:.2f}%</span>'

        def format_range(val):
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return f'<span class="tooltip-wrapper">₹{val[0]:.2f} - ₹{val[1]:.2f}<span class="tooltip-box">Expected next price zone</span></span>'
            return str(val) if val else "-"

        def format_trend(val):
            v = str(val).strip().lower()
            if v == "up":
                return '<span class="trend-up">Up</span>'
            elif v == "down":
                return '<span class="trend-down">Down</span>'
            return v

        # Apply formatting
        if "Signal" in df.columns:
            df["Signal"] = df.apply(format_signal_with_tooltip, axis=1)
        if "PnL%" in df.columns:
            df["PnL%"] = df["PnL%"].apply(format_pnl)
        if "NextRange" in df.columns:
            df["NextRange"] = df["NextRange"].apply(format_range)
        if "Trend" in df.columns:
            df["Trend"] = df["Trend"].apply(format_trend)

        # Compute summary
        total_value = sum(df["CurrentPrice"].astype(float) * df["Qty"].astype(float))
        total_invested = sum(df["Buyprice"].astype(float) * df["Qty"].astype(float))
        total_pnl = ((total_value - total_invested) / total_invested) * 100
        pnl_color = "#00a65a" if total_pnl >= 0 else "#e74c3c"
        buy_hold_count = df["Signal"].str.contains("BUY|WAIT", case=False).sum()
        exit_count = df["Signal"].str.contains("SELL|EXIT", case=False).sum()

        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Generate HTML table
        table_html = df.to_html(index=False, escape=False, border=0)

        # Load template
        with open("dashboards/portfolio_dashboard_template.html") as f:
            html_template = f.read()

        html_output = (
            html_template
            .replace("{{table_html}}", table_html)
            .replace("{{last_updated}}", last_updated)
            .replace("{{total_value}}", f"{total_value:,.2f}")
            .replace("{{total_pnl}}", f"{total_pnl:.2f}")
            .replace("{{pnl_color}}", pnl_color)
            .replace("{{buy_hold_count}}", str(buy_hold_count))
            .replace("{{exit_count}}", str(exit_count))
        )

        # Save
        output_path = "dashboards/portfolio_dashboard.html"
        with open(output_path, "w") as f:
            f.write(html_output)

        print(f"✅ Dashboard updated: {output_path}")
        return output_path


    def check_for_new_strong_buys(self,df):
        strong_buy_file = "data/strong_buy_cache.csv"
        os.makedirs("data", exist_ok=True)

        new_buys = df[df["Signal"].str.contains("STRONG BUY", case=False, na=False)]
        if os.path.exists(strong_buy_file):
            old_buys = pd.read_csv(strong_buy_file)
            old_tickers = set(old_buys["Ticker"].tolist())
        else:
            old_tickers = set()

        new_tickers = set(new_buys["Ticker"].tolist())
        diff = new_tickers - old_tickers

        if diff:
            message = f"{', '.join(diff)}"
            send_mac_notification("🔥New STRONG BUYs detected", message)

        new_buys.to_csv(strong_buy_file, index=False)




def run_portfolio_tracker():
    tracker = PortfolioTracker(strategy=WeightedStrategy())
    portfolio_df = tracker.evaluate_positions()
    tracker.check_for_new_strong_buys(portfolio_df)
    html_file = tracker.generate_dashboard(portfolio_df)
    return html_file


def add_to_portfolio(ticker, quantity, buy_price, buy_date=None):
    """Append a new ETF holding to the portfolio CSV."""
    try:
        if buy_date is None:
            buy_date = datetime.now().strftime("%dd-%mm-%YYYY")

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
