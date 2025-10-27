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
                reason = AdvisorEngine().get_reason(signal, trend, pnl)

                results.append({
                    "Ticker": ticker,
                    "Qty": qty,
                    "BuyPrice": round(buy_price, 2),
                    "CurrentPrice": round(last_close, 2),
                    "PnL%": round(pnl, 2),
                    "BuyDate": buy_date,
                    "Signal": signal,
                    "SignalReason": reason,
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
        """Generate portfolio dashboard (manually render table rows to avoid pandas html quirks)."""
        import html  # for escaping
        if df is None or df.empty:
            print("⚠️ Nothing to display in portfolio dashboard.")
            return None

        template_path = os.path.join("dashboards", "portfolio_dashboard_template.html")
        output_path = os.path.join("dashboards", "portfolio_dashboard.html")

        # read base template (which should contain placeholders as in earlier file)
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # --- Portfolio summary ---
        df["CurrentValue"] = df["Qty"] * df["CurrentPrice"]
        total_value = df["CurrentValue"].sum()
        total_invested = (df["Qty"] * df["BuyPrice"]).sum()
        total_pnl = ((total_value - total_invested) / total_invested) * 100 if total_invested != 0 else 0.0
        pnl_color = "#00a65a" if total_pnl >= 0 else "#e74c3c"

        accumulate_or_hold = df["SuggestedAction"].str.contains("Accumulate|Hold", case=False, na=False)
        exit_or_cut = df["SuggestedAction"].str.contains("Exit|Cut", case=False, na=False)
        buy_hold_count = int(accumulate_or_hold.sum())
        exit_count = int(exit_or_cut.sum())

        # --- helper functions ---
        def clean_text(x):
            """Remove escaped/newline artifacts and trim; ensure string (no backslash-n left)."""
            if x is None:
                return ""
            s = str(x)
            s = s.replace("\\n", " ").replace("\n", " ").replace("\r", " ").strip()
            # collapse multiple spaces
            while "  " in s:
                s = s.replace("  ", " ")
            return s

        def badge_class_for(signal_text):
            s = (signal_text or "").upper()
            if "BUY" in s:
                return "signal-buy"
            if "WAIT" in s or "HOLD" in s:
                return "signal-wait"
            if "SELL" in s or "AVOID" in s:
                return "signal-avoid"
            return "signal-wait"

        def html_escape(s):
            """Escape text for insertion into HTML content (not attributes)."""
            return html.escape(str(s), quote=False)

        # --- Build table rows manually (controlled) ---
        rows_html = []
        for _, row in df.iterrows():
            ticker = html_escape(clean_text(row.get("Ticker", "")))
            qty = html_escape(clean_text(row.get("Qty", "")))
            buy_price = row.get("BuyPrice", "")
            current_price = row.get("CurrentPrice", "")
            pnl_val = row.get("PnL%", 0.0)
            buy_date = html_escape(clean_text(row.get("BuyDate", "")))

            # New columns
            action = html_escape(clean_text(row.get("SuggestedAction", "—")))
            score = html_escape(clean_text(row.get("Score", "—")))
            current_value = f"₹{float(row.get('CurrentValue', 0)):.2f}"

            # Signal & reason
            signal_raw = clean_text(row.get("Signal", ""))
            # If your Signal column already contains HTML (from earlier), ensure we extract plain text:
            # remove any HTML tags if present (simple approach)
            # but we assume it's plain text; still sanitize
            signal_text = html_escape(signal_raw)
            reason_text = clean_text(row.get("SignalReason", ""))  # keep plain text
            reason_text = html_escape(reason_text)
            conf = html_escape(clean_text(row.get("Confidence", "")))

            # choose badge class
            badge_class = badge_class_for(signal_text)

            pnl_color_cell = "gain" if float(pnl_val) > 0 else "loss" if float(pnl_val) < 0 else ""

            # build the tooltip-wrapper and badge (tooltip text is reason_text)
            # Use data-* attribute as well as inner tooltip div to be robust
            signal_cell_html = (
                f'<div class="tooltip-wrapper" data-reason="{reason_text}">'
                f'  <span class="{badge_class}">{signal_text}</span>'
                f'  <div class="tooltip-box" aria-hidden="true">{reason_text}</div>'
                f'</div>'
            )

            row_html = (
                "<tr>"
                f"<td>{ticker}</td>"
                f"<td>{qty}</td>"
                f"<td>₹{buy_price:.2f}</td>"
                f"<td>₹{current_price:.2f}</td>"
                f"<td class='{pnl_color_cell}'>{pnl_val:.2f}%</td>"
                f"<td style='white-space:nowrap'>{buy_date}</td>"
                f"<td>{signal_cell_html}</td>"
                f"<td>{conf}</td>"
                f"<td>{action}</td>"
                f"<td>{score}</td>"
                f"<td>{current_value}</td>"
                "</tr>"
            )
            rows_html.append(row_html)

        table_rows_combined = "\n".join(rows_html)

        # --- Inject rows into the template's tbody (replace {{table_html}} placeholder) ---
        # Our template uses {{table_html}} placeholder - we will replace that with table markup.
        # Build the full table markup here
        # --- table markup ---
        table_full_html = f"""
        <table class="table">
          <thead>
            <tr>
              <th>TICKER</th><th>QTY</th><th>BUYPRICE</th><th>CURRENTPRICE</th>
              <th>PNL%</th><th>BUYDATE</th><th>SIGNAL</th>
              <th>CONFIDENCE</th><th>ACTION</th><th>SCORE</th><th>CURRENT VALUE</th>
            </tr>
          </thead>
          <tbody>
            {table_rows_combined}
          </tbody>
        </table>
        """

        rendered = (
            template
            .replace("{{last_updated}}", now)
            .replace("{{total_value}}", f"{total_value:,.2f}")
            .replace("{{total_pnl}}", f"{total_pnl:.2f}")
            .replace("{{pnl_color}}", pnl_color)
            .replace("{{buy_hold_count}}", str(buy_hold_count))
            .replace("{{exit_count}}", str(exit_count))
            .replace("{{table_html}}", table_full_html)
        )

        # --- Write final file ---
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered)

        print(f"💾 Portfolio dashboard saved: {output_path}")
        return output_path


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
