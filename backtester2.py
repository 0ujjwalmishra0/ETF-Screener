import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from strategies.simple_rule_strategy import SimpleRuleStrategy
from strategies.weighted_strategy import WeightedStrategy
from data_fetch import fetch_etf_data
from indicators import compute_basic_indicators

TICKERS = [
    "NIFTYBEES.NS", "JUNIORBEES.NS", "MID150BEES.NS",
    "BANKBEES.NS", "ICICINXT50.NS", "MON100.NS",
    "MAFANG.NS", " ̰HNGSNGBEES.NS", "ICICITECH.NS",
    "GOLDBEES.NS", "SILVERBEES.NS", "ICICIPHARM.NS"
]

INITIAL_CAPITAL = 1_00_000  # per ETF


# --- Utility Functions -----------------------------------------------------
def compute_cagr(df):
    """Compute CAGR of cumulative returns DataFrame."""
    if df is None or df.empty:
        return 0.0
    start_val = df.iloc[0]
    end_val = df.iloc[-1]
    n_years = len(df) / 12
    return (end_val / start_val) ** (1 / n_years) - 1


def compute_period_return(df, years):
    """Compute return over the given number of years."""
    if df is None or df.empty or len(df) < 12:
        return 0.0
    months = years * 12
    if len(df) < months:
        months = len(df)
    start_val = df.iloc[-months]
    end_val = df.iloc[-1]
    return (end_val / start_val) - 1


# --- Core Backtesting Logic ------------------------------------------------
def simulate_strategy(strategy, years=10):
    portfolio = []
    print(f"\n🔁 Backtesting '{strategy.__class__.__name__}' for {years} years (monthly rebalancing)...")

    for ticker in TICKERS:
        try:
            df = fetch_etf_data(ticker, period=f"{years}y", interval="1d")
            if df.empty or "Close" not in df.columns:
                print(f"⚠️ Skipping {ticker}: no valid data.")
                continue

            if not isinstance(df.index, pd.DatetimeIndex):
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df.set_index("Date", inplace=True)
                else:
                    print(f"⚠️ Skipping {ticker}: no Date column found.")
                    continue

            df= compute_basic_indicators(df)
            # --- Apply Strategy ---
            evaluated = strategy.evaluate(df)

            # Handle multiple return types (tuple, df, str)
            if isinstance(evaluated, tuple):
                signal = evaluated[0]
                df["Signal"] = 1 if "BUY" in str(signal).upper() else 0
            elif isinstance(evaluated, pd.DataFrame):
                df = evaluated
                if "Signal" not in df.columns:
                    df["Signal"] = 0
            elif isinstance(evaluated, str):
                df["Signal"] = 1 if "BUY" in evaluated.upper() else 0
            else:
                df["Signal"] = 0

            # --- Convert to monthly ---
            df_monthly = df.resample("ME").last().dropna(subset=["Close"])

            if len(df_monthly) < 12:
                print(f"⚠️ Skipping {ticker}: insufficient monthly data ({len(df_monthly)} months).")
                continue

            df_monthly["Position"] = df_monthly["Signal"].astype(int)
            df_monthly["Returns"] = df_monthly["Close"].pct_change()
            df_monthly["StrategyRet"] = df_monthly["Position"].shift(1) * df_monthly["Returns"]

            df_monthly["EquityCurve"] = INITIAL_CAPITAL * (1 + df_monthly["StrategyRet"].fillna(0)).cumprod()
            portfolio.append(df_monthly["EquityCurve"])

        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
            continue

    if not portfolio:
        print(f"⚠️ No valid ETFs found for '{strategy.__class__.__name__}'.")
        return None

    portfolio_df = pd.concat(portfolio, axis=1).fillna(method="ffill")
    portfolio_df["PortfolioValue"] = portfolio_df.mean(axis=1)
    return portfolio_df["PortfolioValue"]


# --- Combined Comparison ---------------------------------------------------
def run_backtest():
    print("🔁 Backtesting both strategies with ₹1L per ETF...")

    simple_df = simulate_strategy(SimpleRuleStrategy(), years=10)
    weighted_df = simulate_strategy(WeightedStrategy(), years=10)

    if simple_df is None or weighted_df is None:
        print("⚠️ One or both strategies failed to produce valid data.")
        return

    # --- Compute Results ---
    summary = []
    for years in [1, 3, 5, 10]:
        simple_return = compute_period_return(simple_df, years)
        weighted_return = compute_period_return(weighted_df, years)
        summary.append((years, simple_return, weighted_return))

    # --- Print Results ---
    print("\n📊 Strategy Performance Summary:")
    print("──────────────────────────────────────────────")
    print(f"{'Period':<10}{'Simple CAGR':<15}{'Weighted CAGR'}")
    print("──────────────────────────────────────────────")
    for years, s_ret, w_ret in summary:
        s_str = f"{s_ret*100:.2f}%" if s_ret else "—"
        w_str = f"{w_ret*100:.2f}%" if w_ret else "—"
        print(f"{str(years)+' Year':<10}{s_str:<15}{w_str}")
    print("──────────────────────────────────────────────")


# --- Main -----------------------------------------------------------------
if __name__ == "__main__":
    run_backtest()
