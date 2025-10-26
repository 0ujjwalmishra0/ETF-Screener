import pandas as pd, webbrowser
from data_fetch import fetch_etf_data
from indicators import compute_basic_indicators
from output_writer import write_outputs
from sparkline_generator import create_sparkline
from performance_charts import create_performance_chart

TICKERS = [
    "NIFTYBEES.NS", "JUNIORBEES.NS", "MID150BEES.NS",
    "BANKBEES.NS", "ICICINXT50.NS", "MON100.NS",
    "MAFANG.NS", "HNGSNGBEES.NS", "ICICITECH.NS",
    "GOLDBEES.NS", "SILVERBEES.NS", "ICICIPHARM.NS",
    "MOM100.NS", "N100.NS", "KOTAKNV20.NS"
]

def evaluate_signal(rsi, ma50, ma200, price, zscore):
    """
    Evaluate ETF Buy/Wait signal using multiple weighted indicators:
    RSI (momentum), Moving Averages (trend), Price position, and Z-Score (valuation).
    """

    score = 0

    # --- RSI-based momentum ---
    # Lower RSI (<40) → oversold → potential rebound
    # Very high RSI (>70) → overbought → caution
    if rsi < 40:
        score += 2
    elif 40 <= rsi < 50:
        score += 1
    elif rsi > 70:
        score -= 1

    # --- Trend-based logic (MA crossovers) ---
    # Short-term trend > long-term → bullish
    if ma50 > ma200:
        score += 1
    else:
        score -= 1

    # --- Price vs MA50 confirmation ---
    # If price above MA50 → strength confirmation
    if price > ma50:
        score += 1
    else:
        score -= 1

    # --- Z-Score (valuation bias) ---
    # Z-score < -2 → deeply undervalued (strong buy)
    # Z-score > 2 → overbought
    if zscore < -2:
        score += 2
    elif -2 <= zscore < -1:
        score += 1
    elif zscore > 2:
        score -= 2
    elif 1 < zscore <= 2:
        score -= 1

    # --- Confidence scaling ---
    confidence = int((abs(score) / 6) * 100)
    confidence = min(confidence, 100)

    # --- Trend Label ---
    if price > ma50 > ma200:
        trend = "📈 Up"
    elif price < ma200 and ma50 < ma200:
        trend = "📉 Down"
    else:
        trend = "➡️ Side"

    # --- Final Signal ---
    if score >= 4:
        signal = "STRONG BUY"
    elif 2 <= score < 4:
        signal = "BUY"
    elif -1 <= score < 2:
        signal = "WAIT"
    elif -3 <= score < -1:
        signal = "STRONG WAIT"
    else:
        signal = "AVOID"

    return signal, confidence, trend, score


def run_once():
    results = []

    for t in TICKERS:
        try:
            df = fetch_etf_data(t)
            if df.empty:
                print(f"⚠️  No price data for {t}")
                continue

            df = compute_basic_indicators(df)
            if df.empty:
                print(f"⚠️  Not enough history for {t}")
                continue

            last = df.iloc[-1]

            # Graceful fallback for missing indicators
            rsi = float(last.get("RSI", 50))
            ma50 = float(last.get("50DMA", last["Close"]))
            ma200 = float(last.get("200DMA", last["Close"]))
            price = float(last["Close"])
            zscore = float(last.get("ZScore", 0))

            signal, confidence, trend, score = evaluate_signal(rsi, ma50, ma200, price,zscore)
            trend_up = "Up" in trend

            # Sparkline (last 30 days)
            spark_path = create_sparkline(df["Close"].tail(30).tolist(), t.replace(".NS", ""), trend_up)
            # Full-year performance chart
            chart_path = create_performance_chart(df.tail(250), t)

            results.append({
                "Ticker": t,
                "Sparkline": f"<img src='{spark_path}' width='80' height='25'>",
                "Chart": f"<img src='{chart_path}' width='200' height='100'>",
                "Close": round(price, 2),
                "RSI": round(rsi, 2),
                "50DMA": round(ma50, 2),
                "200DMA": round(ma200, 2),
                "ZScore": round(zscore, 2),
                "Signal": signal,
                "Score": score,
                "Confidence": f"{confidence}%",
            })

            print(f"{signal:>12} {trend:>8} → {t}: ₹{price:.2f} | RSI={rsi:.2f} | Conf={confidence}%")

        except Exception as e:
            print(f"❌ Error processing {t}: {e}")

    if not results:
        print("⚠️ No ETFs processed successfully.")
        return

    # Convert results → DataFrame
    out = pd.DataFrame(results)

    # Ensure 'Score' column exists even if missing for some
    if "Score" not in out.columns:
        out["Score"] = 0
    if "Signal" not in out.columns:
        out["Signal"] = "WAIT"

    # Sorting by signal and score
    signal_order = {"STRONG BUY": 1, "BUY": 2, "WAIT": 3, "STRONG WAIT": 4, "AVOID": 5}
    out["SignalRank"] = out["Signal"].map(signal_order).fillna(99)
    out = out.sort_values(by=["SignalRank", "Score"], ascending=[True, False]).reset_index(drop=True)
    out = out.drop(columns=["SignalRank"])

    # Print summary
    print("\n📊 ETF Summary by Signal:")
    print(out["Signal"].value_counts().to_string())

    # Pass results to output writer
    html_file = write_outputs(out)

    # ✅ Auto-open the HTML dashboard
    webbrowser.open(html_file)

    print("\n✅ Screener complete. Dashboard auto-opened in browser.")
