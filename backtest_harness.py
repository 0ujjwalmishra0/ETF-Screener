"""
backtest_harness.py

Walk-forward backtest harness for SmartHybridStrategy.

Usage:
    python backtest_harness.py \
        --data /Users/ujjwal/Documents/Investment/etf_screener/portfolio.csv \
        --ticker CPSEETF.NS \
        --train-window 500 \
        --test-window 60 \
        --retrain-every 60 \
        --initial-capital 100000

CSV must contain columns: Date, Open, High, Low, Close, Volume  (Date parseable)
"""

import os
import argparse
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Import your strategy
from strategies.smart_hybrid_strategy import SmartHybridStrategy
from strategies.weighted_strategy import WeightedStrategy

# ----------------------
# Utils / Metrics
# ----------------------
def compute_cagr(equity_series, days_per_year=252):
    # equity_series: daily equity indexed by date (or integer)
    returns = equity_series.pct_change().fillna(0)
    total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1.0
    num_years = len(equity_series) / days_per_year
    if num_years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / num_years) - 1

def max_drawdown(equity_series):
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    return drawdown.min()

def sharpe_ratio(returns, risk_free=0.0, days_per_year=252):
    # returns: daily returns series
    if returns.std() == 0:
        return 0.0
    ann_mean = returns.mean() * days_per_year
    ann_std = returns.std() * np.sqrt(days_per_year)
    return (ann_mean - risk_free) / (ann_std + 1e-9)

# ----------------------
# Simple trading sim
# ----------------------
def simulate_trades(df_test, strategy, initial_cash=100000, position_size_pct=1.0,
                    verbose=False):
    """
    Simple long-only simulator:
     - When signal == BUY or STRONG BUY: go long (use full position_size_pct of capital).
     - When signal == WAIT: hold existing position
     - When signal in (STRONG WAIT, AVOID): exit to cash
     - Execution: next-day Open price (if available) otherwise Close.
    Returns:
      equity_ts (pd.Series), trades (list of dict)
    """
    cash = initial_cash
    position = 0.0  # number of shares
    equity_list = []
    trades = []
    last_price = None

    # We'll query strategy.evaluate on rolling window of history: to be called externally.
    for idx, row in df_test.iterrows():
        date = row["Date"] if "Date" in row else idx
        prev_close = row["Close"]  # the strategy will be computed beforehand by calling evaluate on slice including historical context.
        # Strategy.evaluate expects full df for context; here we assume df_test contains rolling context rows where row['_signal'] exists
        signal = row.get("_signal", None)
        # determine execution price (use next bar's open if simulation has access to it)
        exec_price = row.get("Open", row.get("Close"))

        # Decide
        if signal is None:
            # treat as hold
            pass

        # Entry
        if signal in ("BUY", "STRONG BUY"):
            # Buy using position_size_pct fraction of equity
            equity_val = cash + position * exec_price
            target_alloc = equity_val * position_size_pct
            target_shares = target_alloc / exec_price if exec_price > 0 else 0
            # If already long, do partial top-up or adjust
            buy_qty = max(0, target_shares - position)
            cost = buy_qty * exec_price
            if cost > 0 and cash >= cost:
                cash -= cost
                position += buy_qty
                trades.append({"date": date, "side": "BUY", "price": exec_price, "qty": buy_qty, "cash": cash})
                if verbose:
                    print(f"{date} BUY {buy_qty:.2f} @ {exec_price:.2f}")

        # Exit
        if signal in ("STRONG WAIT", "AVOID"):
            if position > 0:
                proceeds = position * exec_price
                cash += proceeds
                trades.append({"date": date, "side": "SELL", "price": exec_price, "qty": position, "cash": cash})
                if verbose:
                    print(f"{date} SELL {position:.2f} @ {exec_price:.2f}")
                position = 0.0

        # Equity mark-to-market
        equity = cash + position * exec_price
        equity_list.append({"Date": date, "Equity": equity})

    eq_df = pd.DataFrame(equity_list).set_index("Date")["Equity"]
    return eq_df, trades

# ----------------------
# Walk-forward engine
# ----------------------
def walk_forward_backtest(df_full, strategy_cls=SmartHybridStrategy,
                          train_window=500, test_window=60, retrain_every=None,
                          initial_capital=100000, position_size_pct=1.0, model_auto_train=True,
                          save_dir="backtest_output"):
    """
    df_full: historical DataFrame with Date, Open, High, Low, Close, Volume
    train_window: rows used to train model initially
    test_window: rows per walk-forward test chunk
    retrain_every: when to retrain ML model within walk-forward (in days). If None, retrain once at start.
    model_auto_train: if True will call strategy.train_model() to train model on each training window.
    """
    os.makedirs(save_dir, exist_ok=True)
    df_full = df_full.copy().reset_index(drop=True)
    # Ensure Date column
    if "Date" not in df_full.columns:
        df_full["Date"] = pd.to_datetime(df_full.index)
    else:
        df_full["Date"] = pd.to_datetime(df_full["Date"])

    n = len(df_full)
    start = 0
    results = []
    all_trades = []

    idx = train_window
    while idx + test_window <= n:
        train_df = df_full.iloc[max(0, idx - train_window): idx].reset_index(drop=True)
        test_df = df_full.iloc[idx: idx + test_window].reset_index(drop=True)

        strat = strategy_cls()
        # Optionally train ML model on train_df
        if model_auto_train:
            try:
                print(f"[{train_df['Date'].iloc[0].date()} -> {train_df['Date'].iloc[-1].date()}] Training model...")
                res = strat.train_model(train_df, save_path=strat.model_path, use_lightgbm=True)
                print("  train result:", res)
            except Exception as e:
                print("  model train failed:", e)

        # Now we need to run strategy on an augmented df that includes both train (for context) + test rows incrementally
        # For each row in test_df, compute evaluate using full history up to that row
        combined = pd.concat([train_df, test_df], ignore_index=True)
        # We'll walk through test rows and compute signal for each day (using combined up to that day)
        signals = []
        for j in range(len(train_df), len(combined)):
            window = combined.iloc[: j + 1].reset_index(drop=True)
            try:
                out = strat.evaluate(window)
                # evaluate returns: signal, confidence, trend, score, breakdown, model_prob
                signal = out[0]
                confidence = out[1]
                score = out[3]
            except Exception as e:
                print("Evaluate error:", e)
                signal = None
                confidence = 50
                score = 0.0
            row = combined.iloc[j].to_dict()
            row["_signal"] = signal
            row["_confidence"] = confidence
            row["_score"] = score
            signals.append(row)

        # simulate trades on signals (test period only)
        test_signals_df = pd.DataFrame(signals)
        eq_series, trades = simulate_trades(test_signals_df, strat, initial_cash=initial_capital, position_size_pct=position_size_pct)
        # metrics
        daily_returns = eq_series.pct_change().fillna(0)
        cagr = compute_cagr(eq_series)
        mdd = max_drawdown(eq_series)
        sr = sharpe_ratio(daily_returns)
        wins = sum(1 for t in trades if t["side"] == "SELL" and t["price"] > 0)  # crude
        results.append({
            "train_start": train_df["Date"].iloc[0],
            "train_end": train_df["Date"].iloc[-1],
            "test_start": test_df["Date"].iloc[0],
            "test_end": test_df["Date"].iloc[-1],
            "num_trades": len(trades),
            "cagr": cagr,
            "sharpe": sr,
            "max_drawdown": mdd,
            "final_equity": float(eq_series.iloc[-1]) if len(eq_series)>0 else float(initial_capital)
        })

        # persist trades
        for t in trades:
            t["train_start"] = train_df["Date"].iloc[0]
            t["test_start"] = test_df["Date"].iloc[0]
            all_trades.append(t)

        # Move window forward
        idx += test_window

    results_df = pd.DataFrame(results)
    trades_df = pd.DataFrame(all_trades)
    results_df.to_csv(os.path.join(save_dir, "walkforward_results.csv"), index=False)
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(save_dir, "walkforward_trades.csv"), index=False)
    print("Backtest complete. Results saved to", save_dir)
    return results_df, trades_df

# ----------------------
# CLI
# ----------------------
def load_csv_data(path):
    df = pd.read_csv(path, parse_dates=["Buydate"])
    df = df.sort_values("Buydate").reset_index(drop=True)
    # if ticker:
    #     # if file contains multiple tickers, filter (optional)
    #     if "Ticker" in df.columns:
    #         df = df[df["Ticker"] == ticker].reset_index(drop=True)
    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",help="CSV OHLCV path",default="/Users/ujjwal/Documents/Investment/etf_screener/portfolio.csv")
    p.add_argument("--ticker", default=None)
    p.add_argument("--train-window", type=int, default=500)
    p.add_argument("--test-window", type=int, default=60)
    p.add_argument("--retrain-every", type=int, default=None)
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--position-size", type=float, default=1.0)
    p.add_argument("--save-dir", default="backtest_output")
    args = p.parse_args()

    df_all = load_csv_data(args.data)
    res, trades = walk_forward_backtest(df_all,
                                        strategy_cls=SmartHybridStrategy,
                                        train_window=args.train_window,
                                        test_window=args.test_window,
                                        retrain_every=args.retrain_every,
                                        initial_capital=args.initial_capital,
                                        position_size_pct=args.position_size,
                                        model_auto_train=True,
                                        save_dir=args.save_dir)
    print(res.head())
