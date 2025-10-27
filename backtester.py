#!/usr/bin/env python3
import os
import pickle
import webbrowser
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

from data_fetch import fetch_etf_data, fetch_cached
from indicators import compute_basic_indicators
from strategies.simple_rule_strategy import SimpleRuleStrategy
from strategies.weighted_strategy import WeightedStrategy

# -------------------- CONFIG --------------------
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

TICKERS = [
    "NIFTYBEES.NS", "JUNIORBEES.NS", "MID150BEES.NS",
    "BANKBEES.NS", "ICICINXT50.NS", "MON100.NS",
    "MAFANG.NS", "HNGSNGBEES.NS", "ICICITECH.NS",
    "GOLDBEES.NS", "SILVERBEES.NS", "ICICIPHARM.NS",
    "MOM100.NS"
]

CAPITAL_PER_ETF = 100_000  # ₹1 L per ETF
YEARS_TO_RUN = 10          # default backtest horizon (10y)
MIN_MONTHS_REQUIRED = 6   # skip ETFs with < 12 monthly points


# -------------------- UTILITIES --------------------
def cache_path(ticker: str, years: int):
    """Return path for pickle cache file for ticker+years"""
    safe = ticker.replace("/", "_").replace("\\", "_")
    return os.path.join(CACHE_DIR, f"{safe}__{years}y.pkl")


def save_cache(df: pd.DataFrame, path: str):
    """Save DataFrame to pickle safely"""
    try:
        pd.to_pickle(df, path)
    except Exception:
        # fallback to pickle module
        with open(path, "wb") as f:
            pickle.dump(df, f)


def load_cache(path: str):
    """Load cached DataFrame if possible"""
    try:
        return pd.read_pickle(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return pd.DataFrame()


def ensure_datetime_index(df: pd.DataFrame):
    """Make sure df has a DatetimeIndex. Accepts Date column or index as date-like."""
    if df is None or df.empty:
        return pd.DataFrame()

    # if df has a Date column, use it
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date")
        except Exception:
            pass

    # if index is not datetime, try converting index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            pass

    # drop rows with invalid index
    df = df[~df.index.isnull()]

    return df.sort_index()


def coerce_numeric_close(df: pd.DataFrame):
    """Ensure Close column exists and is numeric; try to guess common column names if not present."""
    if df is None or df.empty:
        return pd.DataFrame()

    # if multiindex columns (yfinance sometimes), flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close_candidates = [c for c in df.columns if c.lower() in ("close", "adj close", "adjusted close", "adjclose")]
    if "Close" not in df.columns:
        if close_candidates:
            df["Close"] = df[close_candidates[0]]
        else:
            # no close column found
            return pd.DataFrame()

    # coerce to numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    # drop NaN closes
    df = df.dropna(subset=["Close"])
    return df


def compute_cagr_from_series(series: pd.Series, years: float):
    """Compute CAGR from a series of portfolio values (assumes chronological order)."""
    if series is None or len(series) < 2 or years <= 0:
        return 0.0
    start = series.iloc[0]
    end = series.iloc[-1]
    if start <= 0 or end <= 0:
        return 0.0
    return (end / start) ** (1.0 / years) - 1.0


def compute_drawdown(series: pd.Series):
    """Compute max drawdown (negative fraction)."""
    if series is None or series.empty:
        return 0.0
    cum = series / series.iloc[0]
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())  # negative value or 0


def safe_eval_strategy(strategy, df_slice: pd.DataFrame):
    """
    Call strategy.evaluate(df_slice) and return (signal, conf, trend, score)
    Accepts either tuple/list or dict returns from evaluate.
    """
    try:
        res = strategy.evaluate(df_slice)
    except Exception as e:
        # If evaluate expects a DataFrame shaped differently, try passing last N rows
        try:
            res = strategy.evaluate(df_slice.copy())
        except Exception:
            # fallback to wait
            return ("WAIT", 0, "➡️ Side", 0)

    # Unpack different possible return types
    if isinstance(res, dict):
        signal = res.get("Signal") or res.get("signal") or "WAIT"
        conf = res.get("Confidence") or res.get("confidence") or 0
        trend = res.get("Trend") or res.get("trend") or "➡️ Side"
        score = res.get("Score") or res.get("score") or 0
        return (str(signal), conf, trend, score)

    if isinstance(res, (list, tuple)):
        # try to fill up to 4 values
        vals = list(res) + [None] * (4 - len(res))
        signal = vals[0] if vals[0] is not None else "WAIT"
        conf = vals[1] if vals[1] is not None else 0
        trend = vals[2] if vals[2] is not None else "➡️ Side"
        score = vals[3] if vals[3] is not None else 0
        return (str(signal), conf, trend, score)

    # Unexpected type
    return (str(res), 0, "➡️ Side", 0)

# -------------------- SIMULATOR --------------------
def simulate_strategy(strategy, label: str, years: int = YEARS_TO_RUN):
    """
    Runs monthly rebalancing per ETF using given strategy.
    Returns (portfolio_series, etf_metrics_df)
    portfolio_series: pandas Series indexed by month-end timestamps (average across ETFs)
    etf_metrics_df: DataFrame indexed by ETF ticker with metrics (1y/3y/5y/10y cagr and drawdown)
    """
    print(f"\n🔁 Backtesting '{label}' strategy for {years} years (monthly rebalancing)...")
    per_etf_monthly_series = []   # list of pd.Series (month-index -> value)
    etf_metrics = []

    total_tickers = len(TICKERS)
    for t in TICKERS:
        try:
            df = fetch_etf_data(t, years)
            if df.empty or "Close" not in df.columns:
                print(f"⚠️ Skipping {t}: no 'Close' data after fetch/clean.")
                continue

            # compute indicators (ensure function returns dataframe with indicators)
            try:
                df = compute_basic_indicators(df)
            except Exception:
                # if indicators fail, proceed with cleaned df
                pass

            # Resample to month-end; we keep last available row per month
            monthly = df.resample("ME").last().dropna(subset=["Close"])
            if monthly.empty or len(monthly) < MIN_MONTHS_REQUIRED:
                print(f"⚠️ Skipping {t}: insufficient monthly data ({len(monthly)} months).")
                continue

            # simulate monthly rebalancing:
            # - start with CAPITAL_PER_ETF
            # - for each month i>=1, evaluate strategy on data up to that month-end
            # - if signal contains BUY -> value *= (price_i / price_{i-1}); otherwise value stays same
            values = []
            value = float(CAPITAL_PER_ETF)
            # store first month value
            values.append(value)

            # iterate months by index
            months = list(monthly.index)
            for i in range(1, len(months)):
                month_end = months[i]
                # sub_df: all daily data up to this month_end
                sub_df = df.loc[:month_end]
                signal, conf, trend, score = safe_eval_strategy(strategy, sub_df)

                # compute monthly return from monthly series
                prev_price = monthly["Close"].iloc[i - 1]
                cur_price = monthly["Close"].iloc[i]
                if prev_price <= 0 or cur_price <= 0:
                    mret = 0.0
                else:
                    mret = cur_price / prev_price - 1.0

                # invest only when signal indicates BUY
                if isinstance(signal, str) and "BUY" in signal.upper():
                    value = value * (1 + mret)
                # otherwise hold cash (value unchanged)
                values.append(value)

            # create series with month index; align length
            ser = pd.Series(values, index=months[: len(values)])
            per_etf_monthly_series.append(ser)

            # compute metrics from this series (use available length to determine years)
            months_len = len(ser)
            years_available = max(months_len / 12.0, 1.0)

            cagr_10y = compute_cagr_from_series(ser, min(years_available, years))
            # compute for 1/3/5 using available months; if insufficient, compute based on available span
            cagr_1y = compute_cagr_from_series(ser.tail(min(12, len(ser))), 1.0) if len(ser) >= 2 else 0.0
            cagr_3y = compute_cagr_from_series(ser.tail(min(36, len(ser))), min(3.0, years_available)) if len(ser) >= 2 else 0.0
            cagr_5y = compute_cagr_from_series(ser.tail(min(60, len(ser))), min(5.0, years_available)) if len(ser) >= 2 else 0.0
            drawdown = compute_drawdown(ser)

            etf_metrics.append({
                "ETF": t,
                f"{label} 1Y": cagr_1y,
                f"{label} 3Y": cagr_3y,
                f"{label} 5Y": cagr_5y,
                f"{label} 10Y": cagr_10y,
                f"{label} Drawdown": drawdown
            })

            print(f"  ✓ {t}: months={len(ser)} final_val={ser.iloc[-1]:.0f}")

        except Exception as ex:
            print(f"❌ Error processing {t}: {ex}")

    # if nothing processed, return empty
    if not per_etf_monthly_series:
        print(f"⚠️ No valid ETFs found for '{label}'.")
        return pd.Series(dtype=float), pd.DataFrame()

    # align all series to union of dates, forward-fill missing, then compute average portfolio (equal-weighted)
    combined_df = pd.concat(per_etf_monthly_series, axis=1)
    combined_df = combined_df.sort_index().ffill().bfill()
    # compute equal-weighted portfolio value per month:
    portfolio_series = combined_df.mean(axis=1)

    # metrics DF
    etf_df = pd.DataFrame(etf_metrics).set_index("ETF").sort_index()

    return portfolio_series, etf_df


# -------------------- DASHBOARD (single HTML) --------------------
def build_dashboard(simple_series, weighted_series, simple_metrics, weighted_metrics, out_file="backtest_dashboard.html"):
    # align and inner-join dates so both series plot together
    df = pd.concat([simple_series.rename("Simple"), weighted_series.rename("Weighted")], axis=1, join="inner").dropna()
    if df.empty:
        print("⚠️ No overlapping dates between strategies to plot.")
        return

    # normalize to 100
    df_norm = df / df.iloc[0] * 100.0
    df_norm.index = pd.to_datetime(df_norm.index)

    # portfolio line chart
    fig_port = go.Figure()
    fig_port.add_trace(go.Scatter(x=df_norm.index, y=df_norm["Simple"], mode="lines", name="Simple", line=dict(color="blue")))
    fig_port.add_trace(go.Scatter(x=df_norm.index, y=df_norm["Weighted"], mode="lines", name="Weighted", line=dict(color="orange", dash="dot")))
    fig_port.update_layout(title="Portfolio (Indexed to 100)", xaxis_title="Date", yaxis_title="Indexed Value", template="plotly_white", height=520)

    # Merge ETF metrics side-by-side for table
    merge_df = pd.merge(simple_metrics, weighted_metrics, left_index=True, right_index=True, how="outer", suffixes=("_Simple", "_Weighted")).fillna(0.0)
    # Format numeric values as percentages for display
    display_df = merge_df.copy()
    for col in display_df.columns:
        if display_df[col].dtype.kind in "fc":
            display_df[col] = (display_df[col] * 100).round(2)  # convert to percent and round

    # Create Plotly table (interactive)
    header_vals = list(display_df.reset_index().columns)
    cell_vals = [display_df.reset_index()[c].astype(str).tolist() for c in display_df.reset_index().columns]

    table_fig = go.Figure(data=[go.Table(
        header=dict(values=header_vals, fill_color="#2a3f5f", font=dict(color="white", size=12), align="center"),
        cells=dict(values=cell_vals, fill_color=["#f7fbff"] * len(header_vals), align="center")
    )])
    table_fig.update_layout(title="ETF Breakdown (values shown as % where applicable)", height=600)

    # Build single HTML with both plots (include plotly js once)
    port_html = pio.to_html(fig_port, include_plotlyjs="cdn", full_html=False)
    table_html = pio.to_html(table_fig, include_plotlyjs=False, full_html=False)

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Backtest Dashboard</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; margin: 16px; background: #fbfcfd; }}
        .tabs {{ display:flex; gap: 8px; margin-bottom: 12px; }}
        .tabbtn {{ padding: 8px 14px; border-radius: 6px; border: 1px solid #d0d7de; cursor: pointer; background: #fff; }}
        .tabbtn.active {{ background: #2a3f5f; color: white; }}
        .pane {{ display: none; }}
        .pane.active {{ display: block; }}
      </style>
    </head>
    <body>
      <h2>ETF Backtest Dashboard — Simple vs Weighted</h2>
      <div class="tabs">
        <button class="tabbtn active" onclick="show('pane1', this)">Portfolio Overview</button>
        <button class="tabbtn" onclick="show('pane2', this)">ETF Breakdown</button>
      </div>

      <div id="pane1" class="pane active">
        {port_html}
      </div>

      <div id="pane2" class="pane">
        {table_html}
      </div>

      <script>
        function show(id, btn) {{
          document.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
          document.querySelectorAll('.tabbtn').forEach(b => b.classList.remove('active'));
          document.getElementById(id).classList.add('active');
          btn.classList.add('active');
        }}
      </script>
    </body>
    </html>
    """

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Dashboard saved to: {out_file}")
    webbrowser.open("file://" + os.path.abspath(out_file))

# -------------------- PRINT SUMMARY --------------------
# --- Helper to compute CAGR safely ---
def period_return(df, years):
    if df is None or len(df) < 2:
        return None

    value_col = None
    if isinstance(df, pd.Series):
        df = df.to_frame(name="PortfolioValue")
        value_col = "PortfolioValue"
    else:
        for c in ["PortfolioValue", "Portfolio", "Value", "Equity"]:
            if c in df.columns:
                value_col = c
                break

    if value_col is None or df[value_col].isnull().all():
        return None

    start_value = df[value_col].iloc[0]
    end_value = df[value_col].iloc[-1]
    if start_value <= 0 or end_value <= 0:
        return None

    total_months = len(df)
    total_years = total_months / 12
    if total_years < years:
        return None

    cagr = ((end_value / start_value) ** (1 / total_years) - 1) * 100
    return cagr


# --- Print table summary in console ---
def print_summary(simple_df, weighted_df):
    print("\n📊 Strategy Performance Summary:")
    print("──────────────────────────────────────────────")
    print(f"{'Period':<10}{'Simple CAGR':<15}{'Weighted CAGR':<15}")
    print("──────────────────────────────────────────────")

    for years in [1, 3, 5, 10]:
        simple = period_return(simple_df, years)
        weighted = period_return(weighted_df, years)
        if simple is not None and weighted is not None:
            print(f"{years} Year   {simple:>8.2f}%        {weighted:>8.2f}%")
        else:
            print(f"{years} Year   {'—':>8}          {'—':>8}")

    print("──────────────────────────────────────────────")


# --- Final combined dashboard builder ---
def compare_strategies_and_build_dashboard():
    print("🔁 Running backtest comparison...\n")

    # Run both strategies
    simple_df, simple_cagr = simulate_strategy(SimpleRuleStrategy(), "Simple")
    weighted_df, weighted_cagr = simulate_strategy(WeightedStrategy(), "Weighted")

    # --- Normalize Series to DataFrame if needed ---
    if isinstance(simple_df, pd.Series):
        simple_df = simple_df.to_frame(name="PortfolioValue")
    if isinstance(weighted_df, pd.Series):
        weighted_df = weighted_df.to_frame(name="PortfolioValue")

    # --- Empty validation ---
    if simple_df is None or weighted_df is None or simple_df.empty or weighted_df.empty:
        print("⚠️ Unable to build dashboard: one or both strategy results are empty.")
        return

    # --- Identify portfolio column safely ---
    for df in [simple_df, weighted_df]:
        if not isinstance(df, pd.DataFrame):
            continue
        col = next((c for c in df.columns if c in ["PortfolioValue", "Portfolio", "Value", "Equity"]), None)
        if col:
            df[col] = df[col] / df[col].iloc[0] * 100
            df.rename(columns={col: "PortfolioValue"}, inplace=True)
        else:
            # if missing, fallback to index values
            df["PortfolioValue"] = (df / df.iloc[0] * 100).values

    # --- Align dates ---
    portfolio_df = pd.DataFrame({
        "Date": simple_df.index,
        "Simple": simple_df["PortfolioValue"].values,
        "Weighted": weighted_df["PortfolioValue"].reindex(simple_df.index, method="ffill").values
    }).dropna()

    # --- Build interactive comparison chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_df["Date"], y=portfolio_df["Simple"],
                             mode="lines", name="Simple Strategy", line=dict(color="blue", width=2)))
    fig.add_trace(go.Scatter(x=portfolio_df["Date"], y=portfolio_df["Weighted"],
                             mode="lines", name="Weighted Strategy", line=dict(color="green", width=2)))

    fig.update_layout(
        title="📈 Simple vs Weighted Strategy Performance (Indexed to 100)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0, y=1.05, orientation="h", bgcolor="rgba(255,255,255,0.7)")
    )

    html_file = "backtest_dashboard.html"
    fig.write_html(html_file, include_plotlyjs="cdn")
    print(f"✅ Dashboard saved to: {html_file}")

    # --- Print performance summary in console ---
    print_summary(simple_df, weighted_df)


if __name__ == "__main__":
    compare_strategies_and_build_dashboard()
