import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import webbrowser
from datetime import datetime, timedelta

from data_fetch import fetch_etf_data
from indicators import compute_basic_indicators
from strategies.simple_rule_strategy import SimpleRuleStrategy
from strategies.weighted_strategy import WeightedStrategy

TICKERS = [
    "NIFTYBEES.NS", "JUNIORBEES.NS", "MID150BEES.NS",
    "BANKBEES.NS", "ICICINXT50.NS", "MON100.NS",
    "MAFANG.NS", "HNGSNGBEES.NS", "ICICITECH.NS",
    "GOLDBEES.NS", "SILVERBEES.NS", "ICICIPHARM.NS",
    "MOM100.NS", "N100.NS", "KOTAKNV20.NS"
]


def backtest_strategy(strategy, period_years=5, capital_per_etf=100000):
    results = []
    end = datetime.today()
    start = end - timedelta(days=period_years * 365)

    for ticker in TICKERS:
        try:
            df = fetch_etf_data(ticker, start=start, end=end)
            df = compute_basic_indicators(df)

            if df.empty or len(df) < 200:
                continue

            signals = []
            for i in range(len(df)):
                subset = df.iloc[: i + 1]
                signal, _, _, _ = strategy.evaluate(subset)
                signals.append(signal)

            df["Signal"] = signals

            position = 0
            cash = capital_per_etf
            shares = 0
            entry_price = 0

            for i in range(1, len(df)):
                price = df["Close"].iloc[i]
                signal = df["Signal"].iloc[i]

                # BUY / STRONG BUY → Enter position if not already in
                if signal in ["BUY", "STRONG BUY"] and position == 0:
                    shares = cash / price
                    cash = 0
                    position = 1
                    entry_price = price

                # WAIT / STRONG WAIT / AVOID → Exit position
                elif signal in ["WAIT", "STRONG WAIT", "AVOID"] and position == 1:
                    cash = shares * price
                    shares = 0
                    position = 0

            # Final value = remaining cash + open position value
            final_value = cash + (shares * df["Close"].iloc[-1])

            total_return = (final_value - capital_per_etf) / capital_per_etf * 100
            cagr = ((final_value / capital_per_etf) ** (1 / period_years) - 1) * 100

            results.append({
                "Ticker": ticker,
                "Period": f"{period_years}Y",
                "Final Value": round(final_value, 2),
                "Total Return %": round(total_return, 2),
                "CAGR %": round(cagr, 2),
            })

        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")

    return pd.DataFrame(results)


def run_backtest():
    strategies = {
        "SimpleRule": SimpleRuleStrategy(),
        "Weighted": WeightedStrategy()
    }

    all_results = []

    for name, strategy in strategies.items():
        for years in [3, 5, 10]:
            df = backtest_strategy(strategy, years)
            df["Strategy"] = name
            all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)

    # Aggregate by Strategy and Period
    summary = (
        final_df.groupby(["Strategy", "Period"])
        .agg({"CAGR %": "mean", "Total Return %": "mean"})
        .reset_index()
    )

    # --- Plotly Interactive Visualization ---
    fig = go.Figure()

    for strat in summary["Strategy"].unique():
        df_strat = summary[summary["Strategy"] == strat]
        fig.add_trace(
            go.Bar(
                x=df_strat["Period"],
                y=df_strat["CAGR %"],
                name=f"{strat} (CAGR%)",
                text=df_strat["CAGR %"].round(2).astype(str) + "%",
                textposition="auto"
            )
        )

    fig.update_layout(
        title="📊 Backtest Results: SimpleRule vs Weighted Strategy",
        xaxis_title="Backtest Period",
        yaxis_title="Avg CAGR (%)",
        barmode="group",
        template="plotly_dark",
        width=900,
        height=600,
    )

    # Save interactive HTML
    output_html = "backtest_results.html"
    pio.write_html(fig, file=output_html, auto_open=False)

    print("\n✅ Backtesting complete!")
    print(final_df.head())
    print(f"Interactive HTML saved → {output_html}")

    webbrowser.open(output_html)


if __name__ == "__main__":
    run_backtest()
